"""Offline tests for Encore's bounded, one-action-at-a-time agent loop."""

from __future__ import annotations

import json

from encore.agent_loop import AgentLoop, FORCE_FINAL_ANSWER
from encore.contracts import POLICY_HALT_PREFIX, fail, ok
from encore.memory_store import ArchiveStore, PersonalMemoryStore
from encore.model_client import ModelResponse
from encore.router import RouteDecision
from encore.tool_executor import ExecutionReport
from encore.working_memory import WorkingMemoryService


SESSION = "11111111-1111-4111-8111-111111111111"


def tool_message(name, arguments, call_id="call-1"):
    """Build one OpenAI-compatible assistant function proposal."""
    # JSON strings exercise the same boundary used by real providers.
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def final_message(
    *,
    status="answered",
    answer="Done.",
    confidence="high",
    sources=None,
    call_id="final-1",
):
    """Build one valid final_answer control call."""
    # A fresh list prevents one test mutating another test's source data.
    payload = {
        "status": status,
        "answer": answer,
        "confidence": confidence,
        "sources": list(sources or []),
    }
    return tool_message("final_answer", payload, call_id)


def compression_message():
    """Build the forced structured result used by loop/compressor integration tests."""
    return tool_message(
        "compress_working_memory",
        {
            "summary": "Several recording searches were completed.",
            "retained_tool_evidence": [
                {"tool_call_id": "call-0", "fact": "The first search succeeded."}
            ],
        },
        "compression-1",
    )


class FakeModel:
    """Queue responses and retain detached call arguments for assertions."""

    def __init__(self, *messages):
        # Tests supply assistant messages in exact expected call order.
        self.messages = list(messages)
        # Captured calls expose tiers, schemas, choices, and reconstructed history.
        self.calls = []

    def chat(self, messages, **kwargs):
        # Copy message dictionaries enough to prevent later list mutation hiding bugs.
        self.calls.append(([dict(message) for message in messages], dict(kwargs)))
        # Each response uses a recognizable but otherwise irrelevant fake model ID.
        return ModelResponse(self.messages.pop(0), "fake-model", None, None)


class FakeExecutor:
    """Return queued execution reports without invoking any account or network API."""

    def __init__(self, *reports):
        # Reports represent already deterministic executor outcomes.
        self.reports = list(reports)
        # Calls prove malformed/parallel model output never reached execution.
        self.calls = []

    def execute(self, name, arguments):
        # Store the raw provider arguments to verify no loop-side rewriting occurs.
        self.calls.append((name, arguments))
        return self.reports.pop(0)


def report(result, *, attempts=1):
    """Construct the common no-approval execution report used in loop tests."""
    return ExecutionReport(result, attempts, ())


def catalog_route(request, client):
    """Route tests to one harmless read tool plus the automatic final control."""
    return RouteDecision(("catalog",))


def test_router_receives_bounded_recent_dialogue_for_title_follow_up():
    captured = []

    def capture_route(request, client):
        captured.append(request)
        return RouteDecision(("catalog",))

    model = FakeModel(
        final_message(call_id="cheap-final"),
        final_message(answer="I will use the supplied track."),
    )
    prior = (
        {"role": "user", "content": "Create a bluegrass playlist."},
        {
            "role": "assistant",
            "content": "Please provide a specific track title so I can continue.",
        },
        {"role": "tool", "content": "large untrusted result"},
    )

    AgentLoop(model, FakeExecutor(), "RULES", route=capture_route).run(
        "Hill Country Highway",
        prior_conversation=prior,
    )

    assert len(captured) == 1
    assert "Create a bluegrass playlist" in captured[0]
    assert "Please provide a specific track title" in captured[0]
    assert "<CURRENT_REQUEST>\nHill Country Highway\n</CURRENT_REQUEST>" in captured[0]
    assert "large untrusted result" not in captured[0]


def test_explicit_prior_session_summary_forces_memory_search_tool():
    def memory_route(request, client):
        return RouteDecision(("memory",))

    model = FakeModel(
        tool_message(
            "search_past_sessions",
            {"query": "prior sessions"},
            "sessions-1",
        ),
        final_message(call_id="cheap-final"),
        final_message(answer="Here is the session history."),
    )
    result = AgentLoop(
        model,
        FakeExecutor(
            report(ok({"sessions": [{"summary": {"topics": ["music"]}}]}))
        ),
        "RULES",
        route=memory_route,
    ).run("Give me a summary of prior sessions")

    first_step_tools = {
        schema["function"]["name"] for schema in model.calls[0][1]["tools"]
    }
    assert first_step_tools == {"search_past_sessions"}
    history_parameters = model.calls[0][1]["tools"][0]["function"]["parameters"]
    assert set(history_parameters["properties"]) == {"query"}
    assert model.calls[0][1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_past_sessions"},
    }
    assert result.status == "answered" and result.tool_calls == 1


def test_new_factual_request_cannot_finalize_from_stale_prior_dialogue(tmp_path):
    def recommendation_route(request, client):
        return RouteDecision(("listening", "catalog", "discovery"))

    model = FakeModel(
        # This is the faulty behavior observed live: immediately finalize while
        # mentally continuing the preceding Spotify export request.
        final_message(call_id="stale-final"),
        tool_message("get_recent_listens", {"count": 25}, "listens-1"),
        final_message(call_id="cheap-final"),
        final_message(answer="Try Molly Tuttle based on your recent listens."),
    )
    executor = FakeExecutor(
        report(ok({"listens": [{"track": "Hill Country Highway"}]}))
    )
    prior = (
        {"role": "user", "content": "Save this playlist to Spotify"},
        {"role": "assistant", "content": "The playlist was exported to Spotify."},
    )

    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        result = AgentLoop(
            model,
            executor,
            "RULES",
            route=recommendation_route,
            working_memory=WorkingMemoryService(
                model,
                personal,
                ArchiveStore(tmp_path / "archives"),
                "alice",
                SESSION,
            ),
        ).run(
            "Suggest a bluegrass artist I haven't listened to",
            prior_conversation=prior,
        )

    assert result.status == "answered" and result.tool_calls == 1
    assert executor.calls[0][0] == "get_recent_listens"
    assert result.steps == 3
    final_context = model.calls[-1][0]
    assert final_context[-1]["content"] == (
        "<CURRENT_OBJECTIVE_REMINDER>\n"
        "Suggest a bluegrass artist I haven't listened to\n"
        "</CURRENT_OBJECTIVE_REMINDER>"
    )


def test_read_action_then_signal_uses_expensive_validated_final_answer():
    # Cheap action, cheap termination signal, then expensive user-facing answer.
    model = FakeModel(
        tool_message("search_recordings", {"title": "Black"}),
        final_message(answer="cheap text is ignored", call_id="cheap-final"),
        final_message(
            answer="I found Black by Pearl Jam.",
            confidence="high",
            sources=["MusicBrainz"],
        ),
    )
    executor = FakeExecutor(report(ok({"recordings": [{"title": "Black"}]})))
    loop = AgentLoop(model, executor, "SAFETY RULES", route=catalog_route)

    result = loop.run("Find Black by Pearl Jam")

    assert result.status == "answered"
    assert result.answer == "I found Black by Pearl Jam."
    assert result.sources == ("MusicBrainz",)
    assert result.steps == 2 and result.tool_calls == 1
    # Router count plus two cheap decisions plus one expensive final call.
    assert result.model_calls == 4
    assert all(message.get("role") != "tool" for message in result.conversation)
    assert model.calls[0][1]["tool_choice"] == "required"
    assert [call[1]["tier"] for call in model.calls] == [
        "cheap",
        "cheap",
        "expensive",
    ]
    assert model.calls[-1][1]["tool_choice"] == FORCE_FINAL_ANSWER
    # A fresh bounded evidence digest—not stale tool-call protocol—is used for
    # finalization, and the cheap final payload is ignored.
    final_history = model.calls[-1][0]
    assert not any(turn.get("role") == "tool" for turn in final_history)
    assert any(
        "<UNTRUSTED_CURRENT_TOOL_EVIDENCE>" in turn.get("content", "")
        for turn in final_history
    )
    assert "cheap text is ignored" not in json.dumps(final_history)


def test_route_exposes_selected_domain_tools_plus_final_answer():
    model = FakeModel(final_message(), final_message())
    loop = AgentLoop(model, FakeExecutor(), "RULES", route=catalog_route)

    result = loop.run("hello")

    names = [tool["function"]["name"] for tool in model.calls[0][1]["tools"]]
    assert names == [
        "search_recordings",
        "lookup_artist",
        "get_top_recordings_for_artist",
        "verify_recordings",
        "final_answer",
    ]
    assert result.route.tools == tuple(names[:-1])


def test_model_cannot_execute_tool_outside_routed_capability():
    model = FakeModel(
        tool_message("export_playlist", {
            "playlist_mbid": "aaaaaaaa-0000-4000-8000-000000000001",
            "service": "spotify",
        }),
        final_message(call_id="cheap-final"),
        final_message(answer="The unrouted export was not attempted."),
    )
    executor = FakeExecutor()

    result = AgentLoop(
        model,
        executor,
        "RULES",
        route=catalog_route,
    ).run("Find a recording")

    assert result.status == "answered"
    assert executor.calls == []
    assert "was not offered" in model.calls[1][0][-1]["content"]


def test_prose_or_parallel_calls_consume_steps_but_never_execute():
    # First output has no tool call; second violates the one-call invariant.
    parallel = tool_message("search_recordings", {"title": "A"})
    parallel["tool_calls"].append(
        tool_message("search_recordings", {"title": "B"}, "call-2")["tool_calls"][0]
    )
    model = FakeModel(
        {"role": "assistant", "content": "I can answer directly."},
        parallel,
        final_message(call_id="cheap-final"),
        final_message(answer="Recovered."),
    )
    executor = FakeExecutor()
    loop = AgentLoop(model, executor, "RULES", route=catalog_route)

    result = loop.run("Find something")

    assert result.status == "answered" and result.steps == 3
    assert executor.calls == []
    # Corrections are plain user turns, never unmatched assistant tool calls.
    second_call_history = model.calls[1][0]
    assert "exactly one complete tool call" in second_call_history[-1]["content"]


def test_duplicate_successful_read_is_not_executed_twice():
    repeated = tool_message("search_recordings", {"title": "Black"}, "repeat-2")
    model = FakeModel(
        tool_message("search_recordings", {"title": "Black"}, "original"),
        repeated,
        final_message(call_id="cheap-final"),
        final_message(answer="I used the original search result."),
    )
    executor = FakeExecutor(report(ok({"recordings": [{"title": "Black"}]})))

    result = AgentLoop(model, executor, "RULES", route=catalog_route).run(
        "Find Black"
    )

    assert result.status == "answered" and result.tool_calls == 1
    assert len(executor.calls) == 1
    duplicate_history = model.calls[2][0]
    assert "already completed" in duplicate_history[-1]["content"]
    replayed = json.loads(duplicate_history[-2]["content"])
    assert replayed["success"] is True


def test_playlist_lookup_count_variation_is_the_same_proposal():
    first = AgentLoop._proposal_key(
        "find_playlists", {"query": "Encore Eval Bluegrass", "count": 10}
    )
    second = AgentLoop._proposal_key(
        "find_playlists", {"query": "Encore Eval Bluegrass", "count": 100}
    )

    assert first == second


def test_second_duplicate_read_forces_finalization_without_budget_exhaustion():
    model = FakeModel(
        tool_message("search_recordings", {"title": "Black"}, "original"),
        tool_message("search_recordings", {"title": "Black"}, "repeat-1"),
        tool_message("search_recordings", {"title": "Black"}, "repeat-2"),
        final_message(answer="The original result is enough."),
    )
    executor = FakeExecutor(report(ok({"recordings": [{"title": "Black"}]})))

    result = AgentLoop(model, executor, "RULES", route=catalog_route).run(
        "Find Black"
    )

    assert result.status == "answered"
    assert result.steps == 3 and result.tool_calls == 1
    assert len(executor.calls) == 1


def test_second_successful_lb_radio_call_reuses_first_batch_and_moves_on():
    model = FakeModel(
        tool_message("lb_radio", {"prompt": "tag:(bluegrass)", "mode": "medium"}),
        tool_message(
            "lb_radio",
            {"prompt": "tag:(bluegrass)", "mode": "easy"},
            "radio-2",
        ),
        tool_message(
            "verify_recordings",
            {"recording_mbids": ["11111111-1111-4111-8111-111111111111"]},
            "verify-1",
        ),
        final_message(call_id="cheap-final"),
        final_message(answer="One discovery batch was enough."),
    )
    executor = FakeExecutor(
        report(ok({"tracks": [{"recording_mbid": SESSION}]})),
        report(ok({"verified_recording_mbids": [SESSION]})),
    )

    result = AgentLoop(
        model,
        executor,
        "RULES",
        route=lambda request, client: RouteDecision(("catalog", "discovery")),
    ).run("Find and verify a bluegrass recording")

    assert result.status == "answered" and result.tool_calls == 2
    assert [name for name, _ in executor.calls] == ["lb_radio", "verify_recordings"]
    correction_history = model.calls[2][0]
    assert "do not run discovery again" in correction_history[-1]["content"]
    assert json.loads(correction_history[-2]["content"])["success"] is True
    second_step_tools = {
        schema["function"]["name"] for schema in model.calls[1][1]["tools"]
    }
    assert second_step_tools == {"verify_recordings"}
    assert model.calls[1][1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "verify_recordings"},
    }


def test_finalizer_receives_bounded_current_tool_evidence_digest():
    model = FakeModel(
        tool_message("search_recordings", {"title": "Black"}),
        final_message(call_id="cheap-final"),
        final_message(answer="Black was found."),
    )
    executor = FakeExecutor(report(ok({"recordings": [{"title": "Black"}]})))

    result = AgentLoop(model, executor, "RULES", route=catalog_route).run("Find Black")

    assert result.status == "answered"
    final_history = model.calls[-1][0]
    digests = [
        message["content"]
        for message in final_history
        if isinstance(message.get("content"), str)
        and "<UNTRUSTED_CURRENT_TOOL_EVIDENCE>" in message["content"]
    ]
    assert len(digests) == 1
    assert "search_recordings" in digests[0] and "Black" in digests[0]
    assert '"result":{"success":true' in digests[0]
    assert "result_json" not in digests[0]
    assert not any(message.get("tool_calls") for message in final_history)


def test_stuck_final_answer_is_always_low_confidence():
    model = FakeModel(
        final_message(call_id="cheap-final"),
        final_message(status="stuck", answer="Could not finish.", confidence="high"),
    )

    result = AgentLoop(model, FakeExecutor(), "RULES", route=catalog_route).run(
        "Find something"
    )

    assert result.status == "stuck" and result.confidence == "low"


def test_repeated_invalid_outputs_get_six_corrections_before_finalizing():
    model = FakeModel(
        {"role": "assistant", "content": "invalid one"},
        {"role": "assistant", "content": "invalid two"},
        {"role": "assistant", "content": "invalid three"},
        {"role": "assistant", "content": "invalid four"},
        {"role": "assistant", "content": "invalid five"},
        {"role": "assistant", "content": "invalid six"},
        final_message(answer="I could not gather tool evidence."),
    )

    result = AgentLoop(model, FakeExecutor(), "RULES", route=catalog_route).run(
        "Find something"
    )

    assert result.status == "answered" and result.steps == 6
    assert result.tool_calls == 0 and len(model.calls) == 7


def test_exhausted_transient_and_declined_write_force_stuck_status():
    transient_model = FakeModel(
        tool_message("search_recordings", {"title": "Black"}),
        final_message(status="answered", answer="Not actually retrieved."),
    )
    transient = AgentLoop(
        transient_model,
        FakeExecutor(report(fail("transient", "HTTP 503"), attempts=3)),
        "RULES",
        route=catalog_route,
    ).run("Find Black")

    def library_route(request, client):
        return RouteDecision(("library",))

    declined_report = ExecutionReport(
        fail("resource", "User declined the proposed account write."),
        0,
        (),
        approval_requested=True,
        approved=False,
    )
    declined_model = FakeModel(
        tool_message(
            "export_playlist",
            {
                "playlist_mbid": "aaaaaaaa-0000-4000-8000-000000000001",
                "service": "spotify",
            },
        ),
        final_message(status="answered", answer="Not exported."),
    )
    declined = AgentLoop(
        declined_model,
        FakeExecutor(declined_report),
        "RULES",
        route=library_route,
    ).run("Export this playlist")

    assert transient.status == "stuck" and transient.steps == 1
    assert declined.status == "stuck" and declined.steps == 1


def test_declined_write_is_saved_before_terminal_exit(tmp_path):
    def library_route(request, client):
        return RouteDecision(("library",))

    preview = "Export playlist aaaaaaaa-0000-4000-8000-000000000001 to spotify."
    declined_report = ExecutionReport(
        fail("resource", "User declined the proposed account write."),
        0,
        (),
        approval_requested=True,
        approved=False,
        approval_summary=preview,
    )
    model = FakeModel(
        tool_message(
            "export_playlist",
            {
                "playlist_mbid": "aaaaaaaa-0000-4000-8000-000000000001",
                "service": "spotify",
            },
        ),
        final_message(status="stuck", answer="The export was declined."),
    )
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        working = WorkingMemoryService(
            model,
            personal,
            ArchiveStore(tmp_path / "archives"),
            "alice",
            SESSION,
        )
        AgentLoop(
            model,
            FakeExecutor(declined_report),
            "RULES",
            route=library_route,
            working_memory=working,
        ).run("Export this playlist")
        pending = personal.get_hot("alice", "pending_approval")

    assert pending == {
        "tool": "export_playlist",
        "preview": preview,
        "status": "declined",
    }


def test_bulk_and_named_unmarked_deletes_are_policy_halts():
    bulk_model = FakeModel(final_message(status="answered", answer="Refused."))
    bulk = AgentLoop(
        bulk_model,
        FakeExecutor(),
        "RULES",
        route=catalog_route,
    ).run("Delete every playlist in my account")

    def library_route(request, client):
        return RouteDecision(("library",))

    unmarked_model = FakeModel(
        tool_message("find_playlists", {"query": "Fall2026", "count": 10}),
        final_message(status="answered", answer="Refused."),
    )
    unmarked = AgentLoop(
        unmarked_model,
        FakeExecutor(
            report(
                ok(
                    {
                        "playlists": [
                            {"title": "Fall2026", "encore_managed": False}
                        ]
                    }
                )
            )
        ),
        "RULES",
        route=library_route,
    ).run("Delete my unmarked playlist named Fall2026")

    assert bulk.status == "policy_halt" and bulk.steps == 0
    assert bulk.tool_calls == 0
    assert unmarked.status == "policy_halt" and unmarked.tool_calls == 1


def test_step_limit_forces_budget_status_over_model_success_claim():
    # Two malformed decisions exhaust this deliberately tiny test budget.
    model = FakeModel(
        {"role": "assistant", "content": "no call"},
        {"role": "assistant", "content": "still no call"},
        final_message(status="answered", answer="I finished anyway."),
    )
    loop = AgentLoop(
        model,
        FakeExecutor(),
        "RULES",
        route=catalog_route,
        max_steps=2,
    )

    result = loop.run("Do work")

    assert result.status == "budget_exceeded"
    assert result.confidence == "low" and result.steps == 2


def test_policy_marker_and_ordinary_fatal_have_distinct_stop_statuses():
    # Safety refusals are marked in code, while unexpected fatal errors are stuck.
    policy_model = FakeModel(
        tool_message("search_recordings", {"title": "unsafe"}),
        final_message(status="answered", answer="Stopped safely."),
    )
    policy_executor = FakeExecutor(
        report(fail("fatal", f"{POLICY_HALT_PREFIX} test refusal"))
    )
    policy_result = AgentLoop(
        policy_model,
        policy_executor,
        "RULES",
        route=catalog_route,
    ).run("unsafe request")

    fatal_model = FakeModel(
        tool_message("search_recordings", {"title": "broken"}),
        final_message(status="answered", answer="Not really done."),
    )
    fatal_executor = FakeExecutor(report(fail("fatal", "internal failure")))
    fatal_result = AgentLoop(
        fatal_model,
        fatal_executor,
        "RULES",
        route=catalog_route,
    ).run("broken request")

    assert policy_result.status == "policy_halt"
    assert fatal_result.status == "stuck"
    assert policy_result.confidence == fatal_result.confidence == "low"


def test_malformed_expensive_final_falls_back_without_another_model_call():
    model = FakeModel(
        final_message(call_id="cheap-final"),
        {"role": "assistant", "content": "invalid final prose"},
    )
    result = AgentLoop(
        model,
        FakeExecutor(),
        "RULES",
        route=catalog_route,
    ).run("hello")

    assert result.status == "stuck" and result.confidence == "low"
    assert "valid final response" in result.answer
    assert "invalid final_answer payload" in result.diagnostic
    assert len(model.calls) == 2


def test_malformed_expensive_final_preserves_latest_tool_evidence():
    model = FakeModel(
        tool_message("search_recordings", {"title": "Black"}),
        final_message(call_id="cheap-final"),
        {"role": "assistant", "content": "invalid final prose"},
    )
    result = AgentLoop(
        model,
        FakeExecutor(report(ok({"recordings": [{"title": "Black"}]}))),
        "RULES",
        route=catalog_route,
    ).run("Find Black")

    assert result.status == "stuck"
    assert "latest tool evidence" in result.answer
    assert "search_recordings" in result.answer and "Black" in result.answer


def test_expensive_http_failure_reports_status_without_response_details():
    class FakeResponse:
        status_code = 429

    class ProviderFailure(Exception):
        response = FakeResponse()

    class FailingFinalModel(FakeModel):
        def chat(self, messages, **kwargs):
            self.calls.append(([dict(message) for message in messages], dict(kwargs)))
            if kwargs["tier"] == "expensive":
                raise ProviderFailure("private provider response body")
            return ModelResponse(self.messages.pop(0), "fake-model", None, None)

    model = FailingFinalModel(final_message(call_id="cheap-final"))
    result = AgentLoop(
        model,
        FakeExecutor(),
        "RULES",
        route=catalog_route,
    ).run("hello")

    assert result.status == "stuck"
    assert result.diagnostic == (
        "Expensive finalization model request failed (HTTP 429). "
        "Check the configured expensive model and provider availability."
    )
    assert "private provider response body" not in result.diagnostic


def test_low_confidence_and_impossible_claims_are_reconciled():
    low_model = FakeModel(
        final_message(call_id="cheap-final"),
        final_message(status="answered", confidence="low"),
    )
    low_result = AgentLoop(
        low_model,
        FakeExecutor(),
        "RULES",
        route=catalog_route,
    ).run("hello")

    invented_model = FakeModel(
        final_message(call_id="cheap-final"),
        final_message(status="policy_halt", confidence="high"),
    )
    invented_result = AgentLoop(
        invented_model,
        FakeExecutor(),
        "RULES",
        route=catalog_route,
    ).run("hello")

    assert low_result.status == "answered_low_confidence"
    assert invented_result.status == "stuck"
    assert invented_result.confidence == "low"


def test_every_main_model_call_has_one_fresh_system_message_only():
    model = FakeModel(
        {"role": "assistant", "content": "missing call"},
        final_message(call_id="cheap-final"),
        final_message(),
    )
    AgentLoop(model, FakeExecutor(), "CRITICAL SAFETY", route=catalog_route).run(
        "hello"
    )

    for messages, _ in model.calls:
        system_turns = [turn for turn in messages if turn["role"] == "system"]
        assert len(system_turns) == 1
        assert system_turns[0]["content"].startswith("CRITICAL SAFETY")


def test_router_exception_uses_safe_all_tool_fallback():
    def broken_route(request, client):
        raise ConnectionError("router unavailable")

    model = FakeModel(final_message(call_id="cheap-final"), final_message())
    result = AgentLoop(
        model,
        FakeExecutor(),
        "RULES",
        route=broken_route,
    ).run("hello")

    assert result.route.used_fallback is True
    assert result.status == "answered"


def test_loop_compresses_after_eight_tool_results_without_spending_action_step(tmp_path):
    action_messages = [
        tool_message("search_recordings", {"title": f"Track {index}"}, f"call-{index}")
        for index in range(8)
    ]
    model = FakeModel(
        *action_messages,
        compression_message(),
        final_message(call_id="cheap-final"),
        final_message(answer="Searches completed."),
    )
    executor = FakeExecutor(
        *[
            report(ok({"recordings": [{"title": "x" * 300}]}))
            for _ in range(8)
        ]
    )
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        archives = ArchiveStore(tmp_path / "archives")
        working = WorkingMemoryService(
            model,
            personal,
            archives,
            "alice",
            SESSION,
        )
        result = AgentLoop(
            model,
            executor,
            "RULES",
            route=catalog_route,
            working_memory=working,
        ).run("Find several recordings")
        hot_count = personal.get_hot("alice", "tool_call_count")

    assert result.status == "answered" and result.steps == 9
    assert result.tool_calls == 8 and hot_count == 8
    # Router + 9 action decisions + compression + expensive finalization.
    assert result.model_calls == 12
    # The current objective plus six old assistant/tool pairs leave active context.
    assert archives.count_user("alice").messages == 13
    # Compression is a lifecycle call between the eighth action and final signal.
    assert model.calls[8][1]["tool_choice"]["function"]["name"] == (
        "compress_working_memory"
    )
    # Final context is constraints, objective, hot facts, then compressed conversation.
    final_context = model.calls[-1][0]
    assert final_context[0]["role"] == "system"
    assert final_context[1]["content"].startswith("<CURRENT_OBJECTIVE>")
    assert final_context[2]["content"].startswith("<UNTRUSTED_KEY_FACTS>")
    assert any(
        message.get("content", "").startswith("<UNTRUSTED_WORKING_MEMORY_SUMMARY>")
        for message in final_context
    )


def test_compression_failure_preserves_completed_work_and_allows_finalization(tmp_path):
    action_messages = [
        tool_message("search_recordings", {"title": f"Track {index}"}, f"call-{index}")
        for index in range(8)
    ]
    model = FakeModel(
        *action_messages,
        {"role": "assistant", "content": "invalid compression prose"},
        final_message(call_id="cheap-final"),
        final_message(status="answered", answer="Searches completed."),
    )
    executor = FakeExecutor(*[report(ok({"recordings": []})) for _ in range(8)])
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        archives = ArchiveStore(tmp_path / "archives")
        result = AgentLoop(
            model,
            executor,
            "RULES",
            route=catalog_route,
            working_memory=WorkingMemoryService(
                model,
                personal,
                archives,
                "alice",
                SESSION,
            ),
        ).run("Find several recordings")

    assert result.status == "answered" and result.confidence == "high"
    assert result.steps == 9 and result.tool_calls == 8
    # Eight actions, failed compression, cheap stop, and finalization.
    assert len(model.calls) == 11
    assert archives.count_user("alice").messages == 0
