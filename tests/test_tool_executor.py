"""Offline tests for deterministic validation, approval, and retry control."""

from __future__ import annotations

import json
from uuid import UUID

from encore.approval import ApprovalGate
from encore.contracts import fail, ok
from encore.tool_catalog import IMPLEMENTED_TOOL_SPECS
from encore.tool_executor import ApprovalDecision, ConsoleApprovalHandler, ToolExecutor


SPECS = {spec.name: spec for spec in IMPLEMENTED_TOOL_SPECS}
RECORDING = "5b11f4ce-0000-4000-8000-000000000002"
PLAYLIST = "aaaaaaaa-0000-4000-8000-000000000001"


def executor_for(name, handler, **kwargs):
    """Build an executor around one real catalog schema and one fake handler."""
    return ToolExecutor({name: handler}, specs=(SPECS[name],), **kwargs)


def test_valid_json_string_is_detached_validated_and_executed():
    calls = []

    def handler(count=25):
        calls.append(count)
        return ok({"count": count})

    report = executor_for("get_recent_listens", handler).execute(
        "get_recent_listens", json.dumps({"count": 4})
    )
    assert report.result.success and report.result.data == {"count": 4}
    assert report.attempts == 1 and calls == [4]


def test_unknown_tool_is_rejected_before_handler_lookup():
    report = executor_for("get_recent_listens", lambda **_: ok({})).execute(
        "filesystem_delete", {}
    )
    assert report.result.error_category == "input"
    assert report.attempts == 0 and "Unknown tool" in report.result.error


def test_schema_rejects_extra_and_out_of_range_arguments_before_execution():
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return ok({})

    executor = executor_for("get_recent_listens", handler)
    too_large = executor.execute("get_recent_listens", {"count": 101})
    invented = executor.execute(
        "get_recent_listens", {"count": 5, "approved": True}
    )
    assert too_large.result.error_category == "input"
    assert invented.result.error_category == "input"
    assert calls == []


def test_non_object_and_nonportable_json_are_rejected():
    executor = executor_for("get_recent_listens", lambda **_: ok({}))
    array = executor.execute("get_recent_listens", "[]")
    nan = executor.execute("get_recent_listens", '{"count": NaN}')
    assert array.result.error_category == "input"
    assert nan.result.error_category == "input"
    assert array.attempts == nan.attempts == 0


def test_transient_read_retries_twice_with_full_jitter_then_succeeds():
    results = [fail("transient", "busy"), fail("transient", "busy"), ok({"done": True})]
    sleeps = []

    def handler(count=25):
        return results.pop(0)

    executor = executor_for(
        "get_recent_listens",
        handler,
        sleep=sleeps.append,
        uniform=lambda low, high: high,
    )
    report = executor.execute("get_recent_listens", {"count": 1})
    assert report.result.success and report.attempts == 3
    assert report.retry_delays_s == (1.0, 2.0)
    assert sleeps == [1.0, 2.0]


def test_input_or_resource_failure_is_never_retried():
    calls = []

    def handler(count=25):
        calls.append(count)
        return fail("resource", "nothing found")

    report = executor_for("get_recent_listens", handler).execute(
        "get_recent_listens", {"count": 1}
    )
    assert report.attempts == 1 and len(calls) == 1
    assert report.retry_delays_s == ()


def test_write_without_approval_handler_fails_closed():
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return ok({})

    report = executor_for("set_feedback", handler).execute(
        "set_feedback", {"recording_mbid": RECORDING, "rating": "love"}
    )
    assert report.result.error_category == "resource"
    assert report.approval_requested and not report.approved
    assert report.attempts == 0 and calls == []


def test_approved_write_receives_permit_outside_model_arguments():
    gate = ApprovalGate()
    seen = {}

    def approval(request):
        seen["request"] = request
        return ApprovalDecision("yes")

    def handler(recording_mbid, rating, *, permit=None):
        arguments = {"recording_mbid": recording_mbid, "rating": rating}
        seen["permit_valid"] = gate.validate(
            permit, "set_feedback", arguments
        )
        return ok({"changed": True})

    executor = executor_for(
        "set_feedback",
        handler,
        approval_gate=gate,
        approval_handler=approval,
    )
    arguments = {"recording_mbid": RECORDING, "rating": "love"}
    report = executor.execute("set_feedback", arguments)
    assert report.result.success and report.approved
    assert seen["permit_valid"] is True
    assert seen["request"].arguments == arguments
    assert "permit" not in seen["request"].arguments


def test_declined_write_never_calls_handler():
    calls = []
    executor = executor_for(
        "delete_playlist",
        lambda **kwargs: calls.append(kwargs) or ok({}),
        approval_handler=lambda request: ApprovalDecision("no"),
    )
    report = executor.execute("delete_playlist", {"playlist_mbid": PLAYLIST})
    assert report.result.error_category == "resource"
    assert not report.approved and report.attempts == 0 and calls == []


def test_invalid_logical_count_and_broken_approval_callback_fail_cleanly():
    executor = executor_for(
        "delete_playlist",
        lambda **kwargs: ok({}),
        approval_handler=lambda request: (_ for _ in ()).throw(EOFError()),
    )
    invalid_count = executor.execute(
        "delete_playlist", {"playlist_mbid": PLAYLIST}, logical_count=0
    )
    broken_ui = executor.execute("delete_playlist", {"playlist_mbid": PLAYLIST})
    assert invalid_count.result.error_category == "input"
    assert broken_ui.result.error_category == "fatal"
    assert "EOFError" in broken_ui.result.error


def test_high_volume_write_requires_and_accepts_exact_second_phrase():
    mbids = [str(UUID(int=index + 1)) for index in range(51)]
    seen_phrase = []

    def approval(request):
        seen_phrase.append(request.high_volume_phrase)
        return ApprovalDecision("yes", request.high_volume_phrase or "")

    executor = executor_for(
        "add_playlist_items",
        lambda playlist_mbid, recording_mbids, *, permit=None: ok(
            {"added": len(recording_mbids)}
        ),
        approval_handler=approval,
    )
    report = executor.execute(
        "add_playlist_items",
        {"playlist_mbid": PLAYLIST, "recording_mbids": mbids},
    )
    assert report.result.success and report.high_volume_confirmation_required
    assert seen_phrase == ["CONFIRM ADD_PLAYLIST_ITEMS 51 TRACKS"]


def test_transient_account_write_is_not_automatically_retried():
    calls = []

    def handler(recording_mbid, rating, *, permit=None):
        calls.append((recording_mbid, rating))
        return fail("transient", "uncertain response")

    report = executor_for(
        "set_feedback",
        handler,
        approval_handler=lambda request: ApprovalDecision("yes"),
    ).execute("set_feedback", {"recording_mbid": RECORDING, "rating": "love"})
    assert report.attempts == 1 and len(calls) == 1
    assert report.retry_delays_s == ()


def test_unexpected_handler_exception_becomes_sanitized_fatal_result():
    def broken(count=25):
        raise RuntimeError("private implementation detail")

    report = executor_for("get_recent_listens", broken).execute(
        "get_recent_listens", {"count": 1}
    )
    assert report.result.error_category == "fatal"
    assert "RuntimeError" in report.result.error
    assert "private implementation detail" not in report.result.error


def test_console_approval_displays_preview_and_collects_both_responses():
    prompts = []
    outputs = []
    responses = iter(["yes", "CONFIRM DELETE_PLAYLIST 2 PLAYLISTS"])
    ui = ConsoleApprovalHandler(
        read=lambda prompt: prompts.append(prompt) or next(responses),
        write=outputs.append,
    )
    request = ApprovalGate().prepare(
        "delete_playlist", {"playlist_mbid": PLAYLIST}, logical_count=2
    )
    decision = ui(request)
    assert decision.response == "yes"
    assert decision.high_volume_response == request.high_volume_phrase
    assert request.summary in outputs[0]
    assert len(prompts) == 2
