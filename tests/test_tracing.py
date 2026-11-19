"""Offline tests for secret-minimized, session-scoped JSONL tracing."""

from __future__ import annotations

import hashlib
import json
import stat
from datetime import datetime, timezone

import pytest

from encore.agent_loop import AgentLoop
from encore.contracts import ok
from encore.model_client import ModelResponse
from encore.router import RouteDecision
from encore.tool_executor import ExecutionReport
from encore.tracing import TraceWriteError, TraceWriter, TracingModelClient


SESSION_ID = "11111111-1111-4111-8111-111111111111"
REQUEST_ID = "22222222-2222-4222-8222-222222222222"
FIXED_TIME = datetime(2026, 10, 28, 18, 30, tzinfo=timezone.utc)


def read_events(path):
    """Decode every physical JSONL line for focused structural assertions."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def tool_message(name, arguments, call_id):
    """Build one provider-shaped function call for the integrated loop test."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def final_message(call_id, answer="Done."):
    """Build one schema-valid final_answer function call."""
    return tool_message(
        "final_answer",
        {
            "status": "answered",
            "answer": answer,
            "confidence": "high",
            "sources": ["MusicBrainz"],
        },
        call_id,
    )


class QueueModel:
    """Return deterministic model metadata and queued assistant messages."""

    def __init__(self, *messages):
        self.messages = list(messages)

    def chat(self, messages, **kwargs):
        return ModelResponse(self.messages.pop(0), "exact/test-model-v1", 12, 4)


class OneExecutor:
    """Capture one dispatch and return a successful approval-bearing report."""

    def __init__(self):
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return ExecutionReport(
            ok({"private_result": "not traced"}),
            2,
            (0.25,),
            approval_requested=True,
            approved=True,
            high_volume_confirmation_required=False,
        )


def catalog_route(request, client):
    """Avoid a router model call while retaining a realistic validated route."""
    return RouteDecision(("catalog",))


def test_writer_uses_hashed_user_directory_uuid_file_and_private_mode(tmp_path):
    with TraceWriter(
        tmp_path / "traces",
        "student@example.edu",
        session_id=SESSION_ID,
        now=lambda: FIXED_TIME,
    ) as trace:
        trace.start_request(REQUEST_ID)
        trace.write("route", domains=["catalog"], tools=[], used_fallback=False)

    expected_key = hashlib.sha256(b"student@example.edu").hexdigest()[:16]
    assert trace.path == tmp_path / "traces" / expected_key / f"{SESSION_ID}.jsonl"
    event = read_events(trace.path)[0]
    assert event["ts"] == "2026-10-28T18:30:00+00:00"
    assert event["session_id"] == SESSION_ID
    assert event["request_id"] == REQUEST_ID
    # Permission assertion uses only the low nine POSIX mode bits.
    assert stat.S_IMODE(trace.path.stat().st_mode) == 0o600


def test_allowlist_rejects_unknown_fields_and_nested_secrets_are_redacted(tmp_path):
    with TraceWriter(tmp_path, "student", session_id=SESSION_ID) as trace:
        trace.write(
            "tool_call",
            phase="proposed",
            tool="search_recordings",
            args={
                "title": "Black",
                "authorization": "Bearer should-never-appear",
                "nested": {"api-key": "also-secret"},
            },
        )
        with pytest.raises(ValueError, match="unsupported fields"):
            trace.write("route", domains=[], tools=[], used_fallback=False, token="x")

    raw = trace.path.read_text()
    event = json.loads(raw)
    assert "should-never-appear" not in raw and "also-secret" not in raw
    assert event["args"]["authorization"] == "[REDACTED]"
    assert event["args"]["nested"]["api-key"] == "[REDACTED]"


def test_model_wrapper_records_exact_model_usage_latency_but_not_messages(tmp_path):
    model = QueueModel({"role": "assistant", "content": "private answer"})
    ticks = iter([10.0, 10.125])
    with TraceWriter(tmp_path, "student", session_id=SESSION_ID) as trace:
        wrapped = TracingModelClient(model, trace, monotonic=lambda: next(ticks))
        wrapped.chat(
            [{"role": "user", "content": "private prompt"}],
            tier="cheap",
        )

    raw = trace.path.read_text()
    event = json.loads(raw)
    assert event["model"] == "exact/test-model-v1"
    assert event["tokens_in"] == 12 and event["tokens_out"] == 4
    assert event["latency_ms"] == 125.0
    assert "private prompt" not in raw and "private answer" not in raw


def test_model_failure_records_only_exception_type_then_reraises(tmp_path):
    class BrokenModel:
        def chat(self, messages, **kwargs):
            raise RuntimeError("secret endpoint and credential detail")

    with TraceWriter(tmp_path, "student", session_id=SESSION_ID) as trace:
        wrapped = TracingModelClient(BrokenModel(), trace, monotonic=lambda: 1.0)
        with pytest.raises(RuntimeError, match="credential detail"):
            wrapped.chat([], tier="expensive")

    raw = trace.path.read_text()
    event = json.loads(raw)
    assert event["success"] is False and event["error"] == "RuntimeError"
    assert "credential detail" not in raw


def test_agent_loop_writes_route_model_tool_approval_and_final_events(tmp_path):
    model = QueueModel(
        tool_message("search_recordings", {"title": "Black"}, "action-1"),
        final_message("cheap-final"),
        final_message("expensive-final", "Found it."),
    )
    executor = OneExecutor()
    with TraceWriter(tmp_path, "student", session_id=SESSION_ID) as trace:
        result = AgentLoop(
            model,
            executor,
            "SAFETY",
            route=catalog_route,
            trace=trace,
        ).run("Find Black")

    events = read_events(trace.path)
    names = [event["event"] for event in events]
    assert names == [
        "route",
        "llm_call",
        "tool_call",
        "tool_call",
        "approval",
        "llm_call",
        "llm_call",
        "final",
    ]
    proposed = next(event for event in events if event.get("phase") == "proposed")
    completed = next(event for event in events if event.get("phase") == "completed")
    approval = next(event for event in events if event["event"] == "approval")
    assert proposed["args"] == {"title": "Black"}
    assert completed["attempts"] == 2 and completed["retry_delays_s"] == [0.25]
    assert approval["approved"] is True
    assert events[-1]["status"] == result.status == "answered"
    # Tool result payloads and the raw user request are intentionally absent.
    raw = trace.path.read_text()
    assert "private_result" not in raw and "Find Black" not in raw


def test_closed_trace_stops_before_any_tool_execution(tmp_path):
    trace = TraceWriter(tmp_path, "student", session_id=SESSION_ID)
    trace.close()
    executor = OneExecutor()
    loop = AgentLoop(
        QueueModel(final_message("unused")),
        executor,
        "SAFETY",
        route=catalog_route,
        trace=trace,
    )

    with pytest.raises(TraceWriteError, match="closed"):
        loop.run("do not run invisibly")
    assert executor.calls == []


def test_trace_failure_inside_routing_is_not_swallowed_as_router_fallback(tmp_path):
    trace = TraceWriter(tmp_path, "student", session_id=SESSION_ID)
    trace.close()
    model = QueueModel(final_message("router-response"))

    def model_using_route(request, client):
        # Closed writer fails after the wrapped model returns its response.
        client.chat([], tier="cheap")
        return RouteDecision(("none",))

    loop = AgentLoop(
        model,
        OneExecutor(),
        "SAFETY",
        route=model_using_route,
        trace=trace,
    )

    with pytest.raises(TraceWriteError, match="closed"):
        loop.run("must remain auditable")
    # The response queue proves no cheap action or expensive final call followed.
    assert model.messages == []
