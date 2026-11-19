"""Offline tests for conservative, deletion-last episodic retention."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from encore.episodic_memory import RetentionCandidate, StructuredSessionSummary
from encore.memory_store import PersonalMemoryStore
from encore.model_client import ModelResponse
from encore.retention import RetentionPlan, RetentionService


NOW = datetime(2026, 10, 28, 18, 0, tzinfo=timezone.utc)
SESSION_1 = "11111111-1111-4111-8111-111111111111"
SESSION_2 = "22222222-2222-4222-8222-222222222222"
PREFERENCE = "Prefers instrumental study music"


def structured(preference=PREFERENCE):
    """Build a complete structured episode containing one preference."""
    return StructuredSessionSummary.from_dict(
        {
            "decisions": ["Created a focus playlist"],
            "preferences": [preference],
            "open_threads": [],
            "key_facts": [],
            "errors": [],
            "topics": ["focus"],
        }
    )


def candidate(memory_id, session_id, *, passed=True, preference=PREFERENCE):
    """Create one expired, code-shaped retention candidate."""
    return RetentionCandidate(
        memory_id=memory_id,
        summary=structured(preference),
        timestamp=(NOW - timedelta(days=100)).timestamp(),
        quality_passed=passed,
        session_ids=(session_id,),
        reasons=("older_than_ttl",),
    )


def forced_message(preferences):
    """Create one valid forced consolidation response."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "retention-1",
                "type": "function",
                "function": {
                    "name": "consolidate_episodic_memory",
                    "arguments": json.dumps(
                        {
                            "retention_summary": "Reviewed old focus sessions.",
                            "standing_preferences": preferences,
                        }
                    ),
                },
            }
        ],
    }


def proposal(*source_ids, preference=PREFERENCE):
    """Return candidate preference text; Python finds its supporting records."""
    return preference


class QueueModel:
    """Return one queued response and retain prompts for privacy assertions."""

    def __init__(self, *messages):
        self.messages = list(messages)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return ModelResponse(self.messages.pop(0), "cheap-test-model", 30, 10)


class RecordingEpisodic:
    """Return stable candidates and record exact deletion requests."""

    def __init__(self, candidates):
        self.candidates = tuple(candidates)
        self.plan_calls = []
        self.delete_calls = []

    def retention_candidates(self, user_id, **kwargs):
        self.plan_calls.append((user_id, kwargs))
        return list(self.candidates)

    def delete_retention_candidates(self, user_id, memory_ids):
        self.delete_calls.append((user_id, memory_ids))
        return len(memory_ids)


def make_plan(*candidates):
    """Construct an already-reviewed plan for explicit application tests."""
    return RetentionPlan("alice", NOW, 50, tuple(candidates))


def test_plan_is_read_only_and_does_not_invoke_model_or_delete(tmp_path):
    episode = candidate("memory-1", SESSION_1)
    episodic = RecordingEpisodic([episode])
    model = QueueModel()
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        plan = RetentionService(
            model,
            episodic,
            personal,
            now=lambda: NOW,
        ).plan("alice")

    assert plan.candidates == (episode,)
    assert model.calls == [] and episodic.delete_calls == []


def test_two_passed_sessions_create_long_term_preference_then_delete(tmp_path):
    first = candidate("memory-1", SESSION_1)
    second = candidate("memory-2", SESSION_2)
    episodic = RecordingEpisodic([first, second])
    model = QueueModel(forced_message([proposal("memory-1", "memory-2")]))
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        result = RetentionService(model, episodic, personal).apply_plan(
            "alice", make_plan(first, second)
        )
        rows = personal.list_preferences("alice")

    assert result.success and result.deleted_count == 2
    assert result.promoted[0].action == "created_long_term"
    assert rows[0]["scope"] == "long_term"
    assert episodic.delete_calls == [("alice", ("memory-1", "memory-2"))]
    # The model sees structured summaries but never a raw conversation transcript.
    assert "UNTRUSTED_RETENTION_CANDIDATES" in model.calls[0][0][1]["content"]


def test_one_session_can_promote_exact_existing_explicit_preference(tmp_path):
    episode = candidate("memory-1", SESSION_1)
    episodic = RecordingEpisodic([episode])
    model = QueueModel(forced_message([proposal("memory-1")]))
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        personal.note_preference("alice", PREFERENCE, "user stated it")
        result = RetentionService(model, episodic, personal).apply_plan(
            "alice", make_plan(episode)
        )
        row = personal.find_preference("alice", PREFERENCE)

    assert result.success and result.promoted[0].action == "promoted_existing"
    assert row is not None and row["scope"] == "long_term"


def test_one_session_without_explicit_sqlite_evidence_is_not_promoted(tmp_path):
    episode = candidate("memory-1", SESSION_1)
    episodic = RecordingEpisodic([episode])
    model = QueueModel(forced_message([proposal("memory-1")]))
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        result = RetentionService(model, episodic, personal).apply_plan(
            "alice", make_plan(episode)
        )
        rows = personal.list_preferences("alice")

    assert result.success and result.promoted == () and rows == []
    assert "Needs two quality-passed sessions" in result.skipped[0].reason
    assert result.deleted_count == 1


def test_flagged_source_is_skipped_but_successful_consolidation_can_expire_it(tmp_path):
    flagged = candidate("memory-1", SESSION_1, passed=False)
    episodic = RecordingEpisodic([flagged])
    model = QueueModel(forced_message([proposal("memory-1")]))
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        result = RetentionService(model, episodic, personal).apply_plan(
            "alice", make_plan(flagged)
        )

    assert result.success and result.promoted == ()
    assert "quality check" in result.skipped[0].reason
    assert result.deleted_count == 1


def test_malformed_or_invented_model_output_keeps_originals(tmp_path):
    episode = candidate("memory-1", SESSION_1)
    episodic = RecordingEpisodic([episode])
    invented = forced_message(
        [proposal("made-up-memory", preference="Invented preference")]
    )
    model = QueueModel(invented)
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        result = RetentionService(model, episodic, personal).apply_plan(
            "alice", make_plan(episode)
        )

    assert not result.success and result.deleted_count == 0
    assert result.error == "Retention consolidation failed (ValidationError)."
    assert episodic.delete_calls == []


def test_failed_preference_promotion_prevents_episode_deletion(tmp_path):
    class FailingPromotionStore(PersonalMemoryStore):
        """Simulate a scoped SQLite promotion failure after validation."""

        def promote_preference(self, user_id, preference_id):
            return False

    episode = candidate("memory-1", SESSION_1)
    episodic = RecordingEpisodic([episode])
    model = QueueModel(forced_message([proposal("memory-1")]))
    with FailingPromotionStore(tmp_path / "personal.sqlite3") as personal:
        personal.note_preference("alice", PREFERENCE, "user stated it")
        result = RetentionService(model, episodic, personal).apply_plan(
            "alice", make_plan(episode)
        )

    assert not result.success and result.deleted_count == 0
    assert result.error == "Retention application failed (RuntimeError)."
    assert episodic.delete_calls == []


def test_stale_or_cross_user_plan_is_never_applied(tmp_path):
    episode = candidate("memory-1", SESSION_1)
    episodic = RecordingEpisodic([])  # Recheck differs from reviewed plan.
    model = QueueModel(forced_message([]))
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        service = RetentionService(model, episodic, personal)
        stale = service.apply_plan("alice", make_plan(episode))
        with pytest.raises(ValueError, match="different user"):
            service.apply_plan("bob", make_plan(episode))

    assert not stale.success and "changed" in (stale.error or "")
    assert model.calls == [] and episodic.delete_calls == []
