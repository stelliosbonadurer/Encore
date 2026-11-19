"""Offline tests for structured, user-filtered episodic Chroma behavior."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from encore.episodic_memory import (
    DEFAULT_EMBEDDING_MODEL_ID,
    DEDUP_SIMILARITY_THRESHOLD,
    EmbeddingModelMismatch,
    EpisodicMemoryStore,
    StructuredSessionSummary,
    expected_collection_metadata,
)
from encore.memory_store import PersonalMemoryStore
from encore.memory_tools import MemoryTools


SESSION_1 = "11111111-1111-4111-8111-111111111111"
SESSION_2 = "22222222-2222-4222-8222-222222222222"
NOW = datetime(2026, 10, 28, 18, 0, tzinfo=timezone.utc)


def summary(**overrides):
    """Create a complete structured summary with concise realistic defaults."""
    payload = {
        "decisions": ["Created a focus playlist"],
        "preferences": ["Prefers instrumental music"],
        "open_threads": [],
        "key_facts": ["Active playlist is Focus"],
        "errors": [],
        "topics": ["focus", "playlist"],
    }
    payload.update(overrides)
    return StructuredSessionSummary.from_dict(payload)


def metadata_for(
    structured,
    *,
    timestamp,
    quality_passed=True,
    quality_weight=1.0,
    quality_score=1.0,
    sessions=None,
):
    """Build one fake Chroma metadata object using the production field shape."""
    return {
        "user_id": "alice",
        "timestamp": timestamp.timestamp(),
        "timestamp_iso": timestamp.isoformat(),
        "topics": ",".join(structured.topics),
        "quality_passed": quality_passed,
        "quality_score": quality_score,
        "quality_weight": quality_weight,
        "verify_before_using": not quality_passed,
        "embedding_model_id": DEFAULT_EMBEDDING_MODEL_ID,
        "summary_json": json.dumps(structured.to_dict()),
        "session_ids_json": json.dumps(sessions or [SESSION_1]),
    }


class FakeCollection:
    """Queue Chroma-shaped query results and capture every storage operation."""

    def __init__(self, *query_results, metadata=None, get_result=None):
        self.metadata = metadata or expected_collection_metadata()
        self.query_results = list(query_results)
        self.get_result = get_result or {"ids": []}
        self.queries = []
        self.upserts = []
        self.gets = []
        self.deletes = []

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return self.query_results.pop(0)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def get(self, **kwargs):
        self.gets.append(kwargs)
        return self.get_result

    def delete(self, **kwargs):
        self.deletes.append(kwargs)


def empty_query():
    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def test_structured_summary_requires_exact_fields_and_nonempty_document():
    with pytest.raises(ValueError, match="fields mismatch"):
        StructuredSessionSummary.from_dict({"topics": ["focus"]})
    empty = StructuredSessionSummary.from_dict(
        {field: [] for field in (
            "decisions",
            "preferences",
            "open_threads",
            "key_facts",
            "errors",
            "topics",
        )}
    )
    with pytest.raises(ValueError, match="at least one"):
        empty.to_document()


def test_embedding_model_mismatch_refuses_startup_before_any_query():
    collection = FakeCollection(
        metadata={
            "schema_version": 1,
            "embedding_model_id": "different-model",
        }
    )
    with pytest.raises(EmbeddingModelMismatch, match="rebuild"):
        EpisodicMemoryStore(collection)
    assert collection.queries == []


def test_new_summary_stores_structured_payload_quality_and_exact_model():
    collection = FakeCollection(empty_query())
    store = EpisodicMemoryStore(collection, now=lambda: NOW)

    result = store.store_summary(
        "alice",
        SESSION_1,
        summary(),
        quality_passed=True,
        quality_score=0.97,
    )

    assert result.merged is False and result.matched_similarity is None
    assert result.memory_id == f"session-{SESSION_1}"
    upsert = collection.upserts[0]
    saved = upsert["metadatas"][0]
    assert saved["user_id"] == "alice"
    assert saved["embedding_model_id"] == DEFAULT_EMBEDDING_MODEL_ID
    assert saved["quality_weight"] == 1.0
    assert json.loads(saved["summary_json"])["topics"] == ["focus", "playlist"]
    # Dedup search is filtered before similarity is considered.
    assert collection.queries[0]["where"] == {"user_id": {"$eq": "alice"}}


def test_near_duplicate_merges_at_point_93_and_conservatively_flags_quality():
    old = summary(preferences=["Prefers instrumental music"])
    previous = metadata_for(old, timestamp=NOW - timedelta(days=2))
    previous["quality_issues_json"] = json.dumps(["Older audit issue"])
    collection = FakeCollection(
        {
            "ids": [[f"session-{SESSION_1}"]],
            "documents": [[old.to_document()]],
            "metadatas": [[previous]],
            # Distance .07 means cosine similarity .93, exactly the threshold.
            "distances": [[1.0 - DEDUP_SIMILARITY_THRESHOLD]],
        }
    )
    store = EpisodicMemoryStore(collection, now=lambda: NOW)
    newer = summary(
        decisions=["Exported the focus playlist"],
        preferences=["Prefers instrumental music", "Avoids vocals"],
    )

    result = store.store_summary(
        "alice",
        SESSION_2,
        newer,
        quality_passed=False,
        quality_score=0.4,
        quality_issues=("New audit issue",),
    )

    assert result.merged is True
    assert result.memory_id == f"session-{SESSION_1}"
    saved = collection.upserts[0]["metadatas"][0]
    merged = json.loads(saved["summary_json"])
    assert merged["decisions"] == [
        "Created a focus playlist",
        "Exported the focus playlist",
    ]
    assert merged["preferences"] == [
        "Prefers instrumental music",
        "Avoids vocals",
    ]
    assert saved["quality_passed"] is False
    assert saved["quality_weight"] == 0.5
    assert json.loads(saved["quality_issues_json"]) == [
        "Older audit issue",
        "New audit issue",
    ]
    assert json.loads(saved["session_ids_json"]) == [SESSION_1, SESSION_2]


def test_below_dedup_threshold_creates_separate_memory():
    old = summary()
    collection = FakeCollection(
        {
            "ids": [[f"session-{SESSION_1}"]],
            "documents": [[old.to_document()]],
            "metadatas": [[metadata_for(old, timestamp=NOW)]],
            "distances": [[0.071]],
        }
    )
    result = EpisodicMemoryStore(collection, now=lambda: NOW).store_summary(
        "alice",
        SESSION_2,
        summary(topics=["running"]),
        quality_passed=True,
        quality_score=1.0,
    )
    assert result.merged is False
    assert result.memory_id == f"session-{SESSION_2}"


def test_search_filters_user_and_90_days_then_reranks_similarity_recency_quality():
    old_relevant = summary(topics=["focus"])
    recent_flagged = summary(topics=["focus", "recent"])
    recent_clean = summary(topics=["study"])
    rows = [
        ("old-relevant", old_relevant, NOW - timedelta(days=80), 0.02, True, 1.0),
        ("recent-flagged", recent_flagged, NOW - timedelta(days=1), 0.0, False, 0.5),
        ("recent-clean", recent_clean, NOW - timedelta(days=2), 0.15, True, 1.0),
    ]
    collection = FakeCollection(
        {
            "ids": [[row[0] for row in rows]],
            "documents": [[row[1].to_document() for row in rows]],
            "metadatas": [[
                metadata_for(
                    row[1],
                    timestamp=row[2],
                    quality_passed=row[4],
                    quality_weight=row[5],
                )
                for row in rows
            ]],
            "distances": [[row[3] for row in rows]],
        }
    )
    store = EpisodicMemoryStore(collection, now=lambda: NOW)

    matches = store.search_sessions("alice", "focus music", 3)

    # Clean recent memory wins; flagged perfect match is halved and ranks last.
    assert [item["memory_id"] for item in matches] == [
        "recent-clean",
        "old-relevant",
        "recent-flagged",
    ]
    assert matches[-1]["annotation"].startswith("VERIFY BEFORE USING")
    where = collection.queries[0]["where"]
    assert where["$and"][0] == {"user_id": {"$eq": "alice"}}
    expected_cutoff = (NOW - timedelta(days=90)).timestamp()
    assert where["$and"][1] == {"timestamp": {"$gte": expected_cutoff}}
    assert collection.queries[0]["n_results"] == 12


def test_forget_counts_ids_and_deletes_only_same_user_filter():
    collection = FakeCollection(get_result={"ids": ["a", "b"]})
    store = EpisodicMemoryStore(collection)

    assert store.forget_user("alice") == 2
    assert collection.gets == [
        {"where": {"user_id": {"$eq": "alice"}}, "include": []}
    ]
    assert collection.deletes == [
        {"where": {"user_id": {"$eq": "alice"}}}
    ]


def test_retention_plan_selects_ttl_and_cap_candidates_without_deleting():
    newest = summary(topics=["newest"])
    second = summary(topics=["second"])
    expired = summary(topics=["expired"])
    records = [
        ("newest", newest, NOW - timedelta(days=1)),
        ("second", second, NOW - timedelta(days=2)),
        ("expired", expired, NOW - timedelta(days=100)),
    ]
    collection = FakeCollection(
        get_result={
            "ids": [record[0] for record in records],
            "metadatas": [
                metadata_for(record[1], timestamp=record[2]) for record in records
            ],
        }
    )
    store = EpisodicMemoryStore(collection, now=lambda: NOW)

    candidates = store.retention_candidates("alice", max_summaries=1)

    # Oldest-first order is stable; the newest record occupies the one-record cap.
    assert [candidate.memory_id for candidate in candidates] == ["expired", "second"]
    assert candidates[0].reasons == ("older_than_ttl", "over_per_user_cap")
    assert candidates[1].reasons == ("over_per_user_cap",)
    assert collection.gets[0]["include"] == ["metadatas"]
    assert collection.deletes == []


def test_retention_delete_intersects_explicit_ids_with_current_user_ownership():
    collection = FakeCollection(get_result={"ids": ["owned-a", "owned-b"]})
    store = EpisodicMemoryStore(collection)

    count = store.delete_retention_candidates(
        "alice",
        ("owned-a", "not-owned", "owned-a"),
    )

    assert count == 1
    assert collection.deletes == [{"ids": ["owned-a"]}]


def test_retention_rejects_naive_simulated_time():
    store = EpisodicMemoryStore(FakeCollection())
    with pytest.raises(ValueError, match="timezone-aware"):
        store.retention_candidates("alice", now=datetime(2026, 10, 28))


def test_memory_tool_uses_vector_backend_and_preserves_staleness_annotation(tmp_path):
    structured = summary()
    collection = FakeCollection(
        {
            "ids": [["memory-1"]],
            "documents": [[structured.to_document()]],
            "metadatas": [[metadata_for(structured, timestamp=NOW)]],
            "distances": [[0.1]],
        }
    )
    episodic = EpisodicMemoryStore(collection, now=lambda: NOW)
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        tools = MemoryTools(personal, episodic, object(), "alice")
        result = tools.search_past_sessions("focus", 1)

    assert result.success
    assert result.data["sessions"][0]["memory_id"] == "memory-1"
    assert result.data["sessions"][0]["date"] == NOW.isoformat()


def test_last_session_query_uses_timestamp_order_not_semantic_similarity(tmp_path):
    older = summary(topics=["highly similar old preference"])
    latest = summary(
        decisions=["Suggested ten new bluegrass songs"],
        topics=["recent recommendations"],
    )
    collection = FakeCollection(
        get_result={
            "ids": ["older", "latest"],
            "metadatas": [
                metadata_for(older, timestamp=NOW - timedelta(days=2)),
                metadata_for(latest, timestamp=NOW - timedelta(minutes=5)),
            ],
        }
    )
    episodic = EpisodicMemoryStore(collection, now=lambda: NOW)
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as personal:
        tools = MemoryTools(personal, episodic, object(), "alice")
        result = tools.search_past_sessions("what did we do last session")

    assert result.success
    assert [row["memory_id"] for row in result.data["sessions"]] == ["latest"]
    assert result.data["sessions"][0]["summary"]["decisions"] == [
        "Suggested ten new bluegrass songs"
    ]
    assert collection.queries == []
    assert collection.gets[0]["include"] == ["metadatas"]
