"""Offline tests for bounded compression, archival, and context ordering."""

from __future__ import annotations

import json

from encore.memory_store import ArchiveStore, PersonalMemoryStore
from encore.model_client import ModelResponse
from encore.tracing import TraceWriter
from encore.working_memory import WorkingMemoryService


SESSION = "11111111-1111-4111-8111-111111111111"


def compression_message(*, retained_id="call-0"):
    """Return one valid forced working-memory compression call."""
    payload = {
        "summary": "The user is building a focused instrumental playlist.",
        "retained_tool_evidence": [
            {"tool_call_id": retained_id, "fact": "A candidate recording was found."}
        ],
    }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "compress-1",
                "type": "function",
                "function": {
                    "name": "compress_working_memory",
                    "arguments": json.dumps(payload),
                },
            }
        ],
    }


class QueueModel:
    """Return queued provider messages and capture complete calls."""

    def __init__(self, *messages):
        self.messages = list(messages)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return ModelResponse(self.messages.pop(0), "cheap-compressor-test", 200, 40)


def tool_history(count=8, *, payload_size=200):
    """Build valid assistant/tool pairs with intentionally bulky raw results."""
    messages = []
    for index in range(count):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "search_recordings",
                                "arguments": json.dumps({"title": f"Track {index}"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "search_recordings",
                    "content": json.dumps({"result": "x" * payload_size}),
                },
            ]
        )
    return messages


def test_below_both_triggers_is_a_model_free_archive_free_noop(tmp_path):
    model = QueueModel()
    history = tool_history(1, payload_size=10)
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        archives = ArchiveStore(tmp_path / "archives")
        service = WorkingMemoryService(model, personal, archives, "alice", SESSION)
        result = service.maybe_compress(
            history,
            tool_results_since_compression=1,
        )

    assert not result.triggered and not result.compressed
    assert list(result.history) == history
    assert model.calls == [] and archives.count_user("alice").messages == 0


def test_eight_tool_results_compress_old_units_and_archive_raw_turns(tmp_path):
    model = QueueModel(compression_message())
    history = tool_history(8)
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        archives = ArchiveStore(tmp_path / "archives")
        service = WorkingMemoryService(model, personal, archives, "alice", SESSION)
        result = service.maybe_compress(
            history,
            tool_results_since_compression=8,
        )

    assert result.triggered and result.compressed
    assert result.reasons == ("tool_result_threshold",)
    # Six old assistant/tool units means twelve raw messages leave context.
    assert result.archived_messages == 12
    assert archives.count_user("alice").messages == 12
    # One summary plus the two most recent proposal/result pairs remain active.
    assert len(result.history) == 5
    assert result.history[0]["content"].startswith("<UNTRUSTED_WORKING_MEMORY_SUMMARY>")
    assert result.history[-1]["tool_call_id"] == "call-7"
    assert result.tokens_after < result.tokens_before
    assert model.calls[0][1]["tier"] == "cheap"
    assert model.calls[0][1]["tool_choice"]["function"]["name"] == (
        "compress_working_memory"
    )


def test_token_threshold_independently_compresses_one_large_unit(tmp_path):
    model = QueueModel(
        compression_message(retained_id="call-0")
    )
    history = tool_history(1, payload_size=2_000)
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        service = WorkingMemoryService(
            model,
            personal,
            ArchiveStore(tmp_path / "archives"),
            "alice",
            SESSION,
            token_budget=400,
            compression_ratio=0.5,
        )
        result = service.maybe_compress(
            history,
            tool_results_since_compression=1,
        )

    assert result.compressed and result.reasons == ("token_threshold",)
    assert result.archived_messages == 2 and len(result.history) == 1


def test_invalid_compression_keeps_history_and_does_not_archive(tmp_path):
    model = QueueModel({"role": "assistant", "content": "unstructured summary"})
    history = tool_history(8)
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        archives = ArchiveStore(tmp_path / "archives")
        result = WorkingMemoryService(
            model,
            personal,
            archives,
            "alice",
            SESSION,
        ).maybe_compress(history, tool_results_since_compression=8)

    assert result.triggered and not result.compressed
    assert result.error == "Working-memory compression failed (ValueError)."
    assert list(result.history) == history
    assert archives.count_user("alice").messages == 0


def test_frontload_orders_objective_hot_facts_memories_then_conversation(tmp_path):
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        personal.set_hot("alice", "active_playlist", {"title": "Focus"})
        personal.set_hot("alice", "last_recordings", [{"title": "Black"}])
        service = WorkingMemoryService(
            QueueModel(),
            personal,
            ArchiveStore(tmp_path / "archives"),
            "alice",
            SESSION,
        )
        # Noted during this session, so it is a current fact.
        personal.note_preference("alice", "No Christmas music", "user said it")
        ordered = service.frontload(
            "Make it more upbeat",
            [{"role": "assistant", "content": "Recent conversation"}],
            retrieved_memories=({"memory_id": "past-1", "summary": "Earlier mix"},),
        )

    assert [message["role"] for message in ordered] == [
        "user",
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert ordered[0]["content"].startswith("<CURRENT_OBJECTIVE>")
    assert ordered[1]["content"].startswith("<UNTRUSTED_KEY_FACTS>")
    assert ordered[2]["content"].startswith("<UNTRUSTED_RETRIEVED_MEMORIES>")
    assert ordered[-1]["content"] == (
        "<CURRENT_OBJECTIVE_REMINDER>\n"
        "Make it more upbeat\n"
        "</CURRENT_OBJECTIVE_REMINDER>"
    )
    assert "No Christmas music" in ordered[1]["content"]


def test_trace_records_counts_and_model_but_not_private_compression_text(tmp_path):
    model = QueueModel(compression_message())
    history = tool_history(8)
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        with TraceWriter(tmp_path / "traces", "alice", session_id=SESSION) as trace:
            trace.start_request()
            result = WorkingMemoryService(
                model,
                personal,
                ArchiveStore(tmp_path / "archives"),
                "alice",
                SESSION,
                trace=trace,
            ).maybe_compress(history, tool_results_since_compression=8)

    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert result.compressed
    assert [event["event"] for event in events] == ["llm_call", "compress"]
    assert events[-1]["quality_passed"] is True
    raw_trace = trace.path.read_text()
    assert "instrumental study music" not in raw_trace
    assert "candidate recording" not in raw_trace
