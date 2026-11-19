"""Offline tests for mandatory end-of-session summarization and quality checks."""

from __future__ import annotations

import json

from encore.episodic_memory import EpisodicWriteResult
from encore.model_client import ModelResponse
from encore.session_memory import QUALITY_PASS_THRESHOLD, SessionMemoryService
from encore.tracing import TraceWriter


SESSION = "11111111-1111-4111-8111-111111111111"


def function_message(name, arguments, call_id="call-1"):
    """Build one OpenAI-compatible forced structured response."""
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


def summary_payload(**overrides):
    """Return the six required summary fields with one explicit preference."""
    payload = {
        "decisions": ["Created a focus playlist"],
        "preferences": ["Prefers instrumental study music"],
        "open_threads": ["May export later"],
        "key_facts": ["Playlist has 12 tracks"],
        "errors": [],
        "topics": ["focus", "playlist"],
    }
    payload.update(overrides)
    return payload


def quality_payload(**overrides):
    """Return a passing quality audit covering the default preference once."""
    payload = {
        "consistency_score": QUALITY_PASS_THRESHOLD,
        "unsupported_claims": [],
        "preference_checks": [
            {
                "preference": "Prefers instrumental study music",
                "supported_by_explicit_user_statement": True,
                "evidence": "User said: I prefer instrumental study music.",
            }
        ],
        "notes": "All claims are supported.",
    }
    payload.update(overrides)
    return payload


class QueueModel:
    """Return queued responses while retaining complete prompts for inspection."""

    def __init__(self, *messages):
        self.messages = list(messages)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return ModelResponse(self.messages.pop(0), "cheap-test-model", 20, 5)


class RecordingEpisodicStore:
    """Capture storage arguments without Chroma or embedding execution."""

    def __init__(self, *, merged=False, fail=False):
        self.calls = []
        self.merged = merged
        self.fail = fail

    def store_summary(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.fail:
            raise RuntimeError("private database detail")
        return EpisodicWriteResult("memory-1", self.merged, 0.95 if self.merged else None)


def session_messages():
    """Provide one explicit preference and one successful tool observation."""
    return [
        {"role": "system", "content": "procedural prompt must not be archived"},
        {
            "role": "user",
            "content": "I prefer instrumental study music; make a focus playlist.",
        },
        {"role": "tool", "content": '{"success":true,"track_count":12}'},
        {"role": "assistant", "content": "The playlist was created."},
    ]


def test_two_forced_cheap_calls_store_verified_structured_summary():
    model = QueueModel(
        function_message("summarize_session", summary_payload(), "summary-1"),
        function_message("check_session_summary", quality_payload(), "quality-1"),
    )
    episodic = RecordingEpisodicStore()
    result = SessionMemoryService(model, episodic).summarize_and_store(
        "alice",
        SESSION,
        session_messages(),
        final_status="answered",
    )

    assert result.stored and result.quality.passed
    assert result.memory_id == "memory-1" and not result.merged
    assert [call[1]["tier"] for call in model.calls] == ["cheap", "cheap"]
    assert model.calls[0][1]["tool_choice"]["function"]["name"] == "summarize_session"
    assert model.calls[1][1]["tool_choice"]["function"]["name"] == "check_session_summary"
    stored_kwargs = episodic.calls[0][1]
    assert stored_kwargs["quality_passed"] is True
    assert stored_kwargs["quality_score"] == QUALITY_PASS_THRESHOLD
    assert stored_kwargs["quality_issues"] == ()


def test_system_turn_is_excluded_and_transcript_is_marked_untrusted():
    model = QueueModel(
        function_message("summarize_session", summary_payload()),
        function_message("check_session_summary", quality_payload()),
    )
    SessionMemoryService(model, RecordingEpisodicStore()).summarize_and_store(
        "alice",
        SESSION,
        session_messages(),
        final_status="stuck",
    )

    summary_input = model.calls[0][0][1]["content"]
    assert summary_input.startswith("<UNTRUSTED_SESSION_TRANSCRIPT>")
    assert "procedural prompt must not be archived" not in summary_input
    assert '"final_status":"stuck"' in summary_input
    # The quality call receives the same transcript plus a separately delimited candidate.
    assert "<UNTRUSTED_CANDIDATE_SUMMARY>" in model.calls[1][0][2]["content"]


def test_low_score_or_unsupported_claim_stores_flagged_memory_for_audit():
    model = QueueModel(
        function_message("summarize_session", summary_payload()),
        function_message(
            "check_session_summary",
            quality_payload(
                consistency_score=0.70,
                unsupported_claims=["The playlist was exported"],
            ),
        ),
    )
    episodic = RecordingEpisodicStore()
    result = SessionMemoryService(model, episodic).summarize_and_store(
        "alice", SESSION, session_messages(), final_status="answered"
    )

    assert result.stored and not result.quality.passed
    assert "The playlist was exported" in result.quality.issues
    assert any("below 0.85" in issue for issue in result.quality.issues)
    assert episodic.calls[0][1]["quality_passed"] is False


def test_every_preference_must_have_one_explicit_supported_evidence_check():
    two_preferences = summary_payload(
        preferences=[
            "Prefers instrumental study music",
            "Dislikes Christmas music",
        ]
    )
    duplicate_check = quality_payload(
        preference_checks=[
            quality_payload()["preference_checks"][0],
            quality_payload()["preference_checks"][0],
        ]
    )
    model = QueueModel(
        function_message("summarize_session", two_preferences),
        function_message("check_session_summary", duplicate_check),
    )
    episodic = RecordingEpisodicStore()
    result = SessionMemoryService(model, episodic).summarize_and_store(
        "alice", SESSION, session_messages(), final_status="answered"
    )

    assert result.stored and not result.quality.passed
    assert any("every preference exactly once" in issue for issue in result.quality.issues)


def test_invalid_quality_output_is_stored_at_zero_weighted_quality_not_blessed():
    model = QueueModel(
        function_message("summarize_session", summary_payload()),
        {"role": "assistant", "content": "looks fine"},
    )
    episodic = RecordingEpisodicStore()
    result = SessionMemoryService(model, episodic).summarize_and_store(
        "alice", SESSION, session_messages(), final_status="answered"
    )

    assert result.stored and result.quality.score == 0.0
    assert not result.quality.passed
    assert "Quality check unavailable" in result.quality.issues[0]
    assert episodic.calls[0][1]["quality_passed"] is False


def test_invalid_or_oversized_summary_never_enters_episodic_store():
    malformed_model = QueueModel({"role": "assistant", "content": "summary prose"})
    malformed_store = RecordingEpisodicStore()
    malformed = SessionMemoryService(
        malformed_model, malformed_store
    ).summarize_and_store(
        "alice", SESSION, session_messages(), final_status="answered"
    )

    long_items = [" ".join(["word"] * 16) + f" {index}" for index in range(20)]
    oversized_model = QueueModel(
        function_message(
            "summarize_session",
            summary_payload(decisions=long_items),
        )
    )
    oversized_store = RecordingEpisodicStore()
    oversized = SessionMemoryService(
        oversized_model, oversized_store
    ).summarize_and_store(
        "alice", SESSION, session_messages(), final_status="answered"
    )

    assert not malformed.stored and malformed_store.calls == []
    assert not oversized.stored and oversized_store.calls == []
    assert "exceeds 300 words" not in (oversized.error or "")  # Error is sanitized.


def test_storage_failure_returns_sanitized_loud_result():
    model = QueueModel(
        function_message("summarize_session", summary_payload()),
        function_message("check_session_summary", quality_payload()),
    )
    result = SessionMemoryService(
        model, RecordingEpisodicStore(fail=True)
    ).summarize_and_store(
        "alice", SESSION, session_messages(), final_status="answered"
    )

    assert not result.stored and result.quality.passed
    assert result.error == "Episodic storage failed (RuntimeError)."
    assert "private database detail" not in result.error


def test_trace_records_two_model_calls_and_memory_write_without_transcript(tmp_path):
    model = QueueModel(
        function_message("summarize_session", summary_payload()),
        function_message("check_session_summary", quality_payload()),
    )
    with TraceWriter(tmp_path / "traces", "alice") as trace:
        trace.start_request()
        result = SessionMemoryService(
            model,
            RecordingEpisodicStore(),
            trace=trace,
        ).summarize_and_store(
            "alice", SESSION, session_messages(), final_status="answered"
        )

    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert result.stored
    assert [event["event"] for event in events] == [
        "llm_call",
        "llm_call",
        "memory_write",
    ]
    assert events[-1]["operation"] == "insert_summary"
    assert events[-1]["record_count"] == 1
    raw = trace.path.read_text()
    assert "instrumental study music" not in raw
    assert "make a focus playlist" not in raw
