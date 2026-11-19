"""Chroma-backed structured episodic summaries with measured retrieval policy.

Only end-of-session summaries belong here; raw conversations remain in the
private archive store. Every query and deletion is filtered by ``user_id``.
Collection metadata pins the exact embedding artifact so incompatible vectors
can never be silently mixed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

# Chroma 1.5.9's default ONNX artifact exposes this exact content checksum.
DEFAULT_EMBEDDING_MODEL_ID = (
    "chroma-default/all-MiniLM-L6-v2@sha256:"
    "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
)
# A new schema/embedding choice receives a new collection rather than mutation.
EPISODIC_COLLECTION_NAME = "encore_episodic_v1"
EPISODIC_SCHEMA_VERSION = 1
# These accepted settings are intentionally centralized for evaluation.
DEDUP_SIMILARITY_THRESHOLD = 0.93
SEMANTIC_WEIGHT = 0.80
RECENCY_WEIGHT = 0.20
FLAGGED_QUALITY_WEIGHT = 0.50
EPISODIC_RETENTION_DAYS = 90
SECONDS_PER_DAY = 86_400.0

# The six fields are required by the assignment's end-of-session contract.
SUMMARY_FIELDS = (
    "decisions",
    "preferences",
    "open_threads",
    "key_facts",
    "errors",
    "topics",
)


class EmbeddingModelMismatch(RuntimeError):
    """Raised when an existing collection uses a different vector space."""


class ChromaCollection(Protocol):
    """Small collection surface needed by EpisodicMemoryStore and its tests."""

    # Collection metadata carries schema and embedding-space identity.
    metadata: Mapping[str, Any] | None

    def query(self, **kwargs: Any) -> dict[str, Any]: ...

    def upsert(self, **kwargs: Any) -> None: ...

    def get(self, **kwargs: Any) -> dict[str, Any]: ...

    def delete(self, **kwargs: Any) -> None: ...


def _utc_now() -> datetime:
    """Return the production timezone-aware retrieval/write clock."""
    return datetime.now(timezone.utc)


def _clean_items(field: str, values: Any) -> tuple[str, ...]:
    """Validate and normalize one structured-summary string array."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"summary field '{field}' must be an array")
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"summary field '{field}' must contain only strings")
        normalized = " ".join(value.split())
        # Empty list items add no retrievable evidence and are discarded.
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return tuple(cleaned)


@dataclass(frozen=True)
class StructuredSessionSummary:
    """Validated structured content produced at the end of one session."""

    # Decisions capture actions actually selected or completed in the session.
    decisions: tuple[str, ...]
    # Preferences must originate in explicit user statements or corrections.
    preferences: tuple[str, ...]
    # Open threads make unfinished user intentions retrievable next time.
    open_threads: tuple[str, ...]
    # Key facts retain small session-specific context rather than raw logs.
    key_facts: tuple[str, ...]
    # Errors preserve useful failure lessons for future sessions/evaluation.
    errors: tuple[str, ...]
    # Topics support human inspection and future metadata filtering.
    topics: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StructuredSessionSummary:
        """Reject missing/extra fields and normalize every list deterministically."""
        keys = set(payload)
        expected = set(SUMMARY_FIELDS)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(f"summary fields mismatch; missing={missing}, extra={extra}")
        return cls(
            **{field: _clean_items(field, payload[field]) for field in SUMMARY_FIELDS}
        )

    def to_dict(self) -> dict[str, list[str]]:
        """Return a JSON-compatible copy rather than exposing tuple internals."""
        return {field: list(getattr(self, field)) for field in SUMMARY_FIELDS}

    def to_document(self) -> str:
        """Format labeled natural language for embedding and transparent inspection."""
        lines: list[str] = []
        for field in SUMMARY_FIELDS:
            values = getattr(self, field)
            if values:
                label = field.replace("_", " ").title()
                lines.append(f"{label}: {'; '.join(values)}")
        # An entirely empty structured call is not useful episodic memory.
        if not lines:
            raise ValueError("session summary must contain at least one item")
        return "\n".join(lines)

    def merged_with(self, newer: StructuredSessionSummary) -> StructuredSessionSummary:
        """Union near-duplicate fields while preserving first-seen readable order."""
        merged: dict[str, tuple[str, ...]] = {}
        for field in SUMMARY_FIELDS:
            values = [*getattr(self, field), *getattr(newer, field)]
            merged[field] = tuple(dict.fromkeys(values))
        return StructuredSessionSummary(**merged)


@dataclass(frozen=True)
class EpisodicWriteResult:
    """Describe whether one summary created or merged a Chroma record."""

    # Stable Chroma identifier points to either the new or reused memory.
    memory_id: str
    # True means the 0.93 policy merged rather than multiplying memories.
    merged: bool
    # Similarity is absent for a new independent episode.
    matched_similarity: float | None


@dataclass(frozen=True)
class RetentionCandidate:
    """One episode selected read-only for TTL or per-user-cap consolidation."""

    # Chroma ID becomes an enum value in the forced consolidation schema.
    memory_id: str
    # Structured content is the only evidence exposed to retention logic.
    summary: StructuredSessionSummary
    # Numeric timestamp supports deterministic oldest-first ordering.
    timestamp: float
    # Flagged memories may be summarized but cannot support preference promotion.
    quality_passed: bool
    # Merged episodes retain every originating session for support counting.
    session_ids: tuple[str, ...]
    # Reasons make a retention plan explainable before any deletion occurs.
    reasons: tuple[str, ...]


def expected_collection_metadata(
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
) -> dict[str, Any]:
    """Build immutable-by-convention metadata for a cosine episodic collection."""
    return {
        # Cosine distance supports the accepted 1-distance similarity conversion.
        "hnsw:space": "cosine",
        # Schema version protects stored metadata/JSON interpretation.
        "schema_version": EPISODIC_SCHEMA_VERSION,
        # Full artifact checksum prevents semantically incompatible vector mixing.
        "embedding_model_id": embedding_model_id,
    }


def validate_collection_metadata(
    metadata: Mapping[str, Any] | None,
    embedding_model_id: str,
) -> None:
    """Refuse collection reuse when schema or embedding identity differs."""
    # Copy provider metadata so validation cannot mutate collection state.
    actual = dict(metadata or {})
    # Expected values come from the one authoritative builder above.
    expected = expected_collection_metadata(embedding_model_id)
    for key in ("schema_version", "embedding_model_id"):
        if actual.get(key) != expected[key]:
            raise EmbeddingModelMismatch(
                "Episodic index metadata does not match configured embedding "
                f"space ({key}); rebuild the index before startup."
            )


def open_persistent_episodic_collection(
    path: Path,
    *,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
) -> ChromaCollection:
    """Open/create the production Chroma collection without embedding any text."""
    # Import here keeps non-memory commands and unit tests independent of Chroma.
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    storage = Path(path)
    storage.mkdir(mode=0o700, parents=True, exist_ok=True)
    # PersistentClient stores vectors and metadata below the ignored memory root.
    client = chromadb.PersistentClient(
        path=str(storage),
        # Local memory does not need Chroma product telemetry.
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    # Creating an empty collection does not yet invoke or download the embedding model.
    collection = client.get_or_create_collection(
        name=EPISODIC_COLLECTION_NAME,
        metadata=expected_collection_metadata(embedding_model_id),
        embedding_function=DefaultEmbeddingFunction(),
    )
    # get_or_create does not replace existing metadata; validate after opening.
    validate_collection_metadata(collection.metadata, embedding_model_id)
    return collection


class EpisodicMemoryStore:
    """Deduplicate, retrieve, and delete structured user-scoped session memories."""

    def __init__(
        self,
        collection: ChromaCollection,
        *,
        embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
        now: Callable[[], datetime] = _utc_now,
        dedup_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
        semantic_weight: float = SEMANTIC_WEIGHT,
        recency_weight: float = RECENCY_WEIGHT,
        flagged_quality_weight: float = FLAGGED_QUALITY_WEIGHT,
        retention_days: int = EPISODIC_RETENTION_DAYS,
    ) -> None:
        # Startup mismatch detection occurs before any query or write.
        validate_collection_metadata(collection.metadata, embedding_model_id)
        if not 0.0 <= dedup_threshold <= 1.0:
            raise ValueError("dedup_threshold must be between 0 and 1")
        if semantic_weight < 0 or recency_weight < 0:
            raise ValueError("retrieval weights must be non-negative")
        if abs((semantic_weight + recency_weight) - 1.0) > 1e-9:
            raise ValueError("semantic and recency weights must sum to 1")
        if not 0.0 <= flagged_quality_weight <= 1.0:
            raise ValueError("flagged quality weight must be between 0 and 1")
        if retention_days < 1:
            raise ValueError("retention_days must be positive")
        # Collection is accessed only through the narrow ChromaCollection protocol.
        self.collection = collection
        # The same exact string is copied into every memory's metadata.
        self.embedding_model_id = embedding_model_id
        # Clock injection supports deterministic recency and retention tests.
        self._now = now
        # Policy values remain instance fields so evaluation can report them.
        self.dedup_threshold = dedup_threshold
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.flagged_quality_weight = flagged_quality_weight
        self.retention_days = retention_days

    @staticmethod
    def _first_row(result: Mapping[str, Any]) -> tuple[Any, Any, Any, Any] | None:
        """Extract the first Chroma query row without trusting optional arrays."""
        # Chroma nests arrays once per query text; this store always sends one query.
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        if not ids:
            return None
        # A malformed backend result is a store failure, never partial evidence.
        if not (documents and metadatas and distances):
            raise RuntimeError("episodic query returned incomplete result arrays")
        return ids[0], documents[0], metadatas[0], distances[0]

    @staticmethod
    def _summary_from_metadata(metadata: Mapping[str, Any]) -> StructuredSessionSummary:
        """Decode the authoritative structured payload stored beside its vector."""
        # Structured JSON, rather than embedded prose, is authoritative for retrieval.
        payload = json.loads(str(metadata["summary_json"]))
        if not isinstance(payload, dict):
            raise RuntimeError("stored episodic summary is not an object")
        return StructuredSessionSummary.from_dict(payload)

    def _dedup_match(
        self,
        user_id: str,
        document: str,
    ) -> tuple[str, Mapping[str, Any], float] | None:
        """Find one same-user near duplicate using the accepted cosine threshold."""
        # Metadata filtering prevents any other user's candidate entering Python.
        result = self.collection.query(
            query_texts=[document],
            n_results=1,
            where={"user_id": {"$eq": user_id}},
            include=["documents", "metadatas", "distances"],
        )
        row = self._first_row(result)
        if row is None:
            return None
        memory_id, _, metadata, distance = row
        # Clamp protects ranking from backend floating-point values outside bounds.
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        if similarity < self.dedup_threshold:
            return None
        return str(memory_id), metadata, similarity

    def store_summary(
        self,
        user_id: str,
        session_id: str,
        summary: StructuredSessionSummary,
        *,
        quality_passed: bool,
        quality_score: float,
        quality_issues: tuple[str, ...] = (),
        timestamp: datetime | None = None,
    ) -> EpisodicWriteResult:
        """Insert one summary or merge it into a same-user near duplicate."""
        if not user_id.strip():
            raise ValueError("user_id must not be empty")
        # Canonical UUID text prevents path-like or ambiguous session identifiers.
        canonical_session = str(UUID(session_id))
        if not 0.0 <= quality_score <= 1.0:
            raise ValueError("quality_score must be between 0 and 1")
        # Caller timestamp supports retention simulation; production defaults to UTC now.
        created = timestamp or self._now()
        if created.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        created = created.astimezone(timezone.utc)
        document = summary.to_document()
        # Deduplication occurs before insertion so a repeated run stays bounded.
        match = self._dedup_match(user_id, document)

        if match is None:
            # A new episode ID is deterministically tied to its originating session.
            memory_id = f"session-{canonical_session}"
            merged_summary = summary
            session_ids = [canonical_session]
            final_quality_passed = quality_passed
            final_quality_score = quality_score
            final_quality_issues = list(quality_issues)
            matched_similarity = None
        else:
            # Reuse the existing ID so merge is an upsert, not a second record.
            memory_id, previous_metadata, matched_similarity = match
            previous_summary = self._summary_from_metadata(previous_metadata)
            merged_summary = previous_summary.merged_with(summary)
            # Source session IDs keep a merged memory auditable.
            previous_sessions = json.loads(
                str(previous_metadata.get("session_ids_json", "[]"))
            )
            session_ids = list(dict.fromkeys([*previous_sessions, canonical_session]))
            # Conservative merge: one flagged source keeps the combined memory flagged.
            final_quality_passed = bool(
                previous_metadata.get("quality_passed", False)
            ) and quality_passed
            final_quality_score = min(
                float(previous_metadata.get("quality_score", 0.0)),
                quality_score,
            )
            previous_issues = json.loads(
                str(previous_metadata.get("quality_issues_json", "[]"))
            )
            # A merge must not erase the reason an older memory was flagged.
            final_quality_issues = list(
                dict.fromkeys([*previous_issues, *quality_issues])
            )

        # Metadata remains scalar-only for Chroma portability.
        # Chroma metadata values must be scalar, so arrays/objects become JSON strings.
        metadata = {
            # Raw user_id is required for exact Chroma metadata filtering/deletion.
            "user_id": user_id,
            # Numeric time supports range filters and ISO time supports display.
            "timestamp": created.timestamp(),
            "timestamp_iso": created.isoformat(),
            # Comma-separated topics remain inspectable within scalar metadata.
            "topics": ",".join(merged_summary.topics),
            # Boolean and score retain the audit decision independently.
            "quality_passed": final_quality_passed,
            "quality_score": final_quality_score,
            # Weight is applied only by deterministic code-side re-ranking.
            "quality_weight": (
                1.0 if final_quality_passed else self.flagged_quality_weight
            ),
            # Explicit flag prevents a future consumer treating flagged text as fact.
            "verify_before_using": not final_quality_passed,
            # JSON text keeps multiple audit issues inside Chroma's scalar metadata.
            "quality_issues_json": json.dumps(
                final_quality_issues, allow_nan=False, separators=(",", ":")
            ),
            # Per-record identity supports export/debugging as well as collection check.
            "embedding_model_id": self.embedding_model_id,
            "summary_json": json.dumps(
                merged_summary.to_dict(), allow_nan=False, separators=(",", ":")
            ),
            "session_ids_json": json.dumps(session_ids, separators=(",", ":")),
        }
        # Upsert makes both initial insertion and dedup merge idempotent by ID.
        self.collection.upsert(
            ids=[memory_id],
            documents=[merged_summary.to_document()],
            metadatas=[metadata],
        )
        return EpisodicWriteResult(
            memory_id=memory_id,
            merged=match is not None,
            matched_similarity=matched_similarity,
        )

    def search_sessions(
        self,
        user_id: str,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Retrieve same-user recent candidates, then re-rank by policy in code."""
        if not user_id.strip() or not query.strip():
            return []
        if limit < 1:
            raise ValueError("limit must be positive")
        # One captured now value keeps every candidate's recency comparable.
        now = self._now().astimezone(timezone.utc)
        window_seconds = self.retention_days * SECONDS_PER_DAY
        # The range filter excludes expired episodes before vector results return.
        cutoff = now.timestamp() - window_seconds
        # Pull extra vector candidates so code-side recency/quality can change order.
        candidate_count = max(limit * 4, limit)
        result = self.collection.query(
            query_texts=[query],
            n_results=candidate_count,
            where={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"timestamp": {"$gte": cutoff}},
                ]
            },
            include=["documents", "metadatas", "distances"],
        )
        # Parallel arrays correspond by index and must remain exactly aligned.
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        if not (len(ids) == len(metadatas) == len(distances)):
            raise RuntimeError("episodic query returned misaligned result arrays")

        # Each candidate receives transparent component scores for evaluation.
        ranked: list[dict[str, Any]] = []
        for memory_id, metadata, distance in zip(ids, metadatas, distances):
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            age_s = max(0.0, now.timestamp() - float(metadata["timestamp"]))
            recency = max(0.0, 1.0 - (age_s / window_seconds))
            # Older records without the field default to verified weight for migration.
            quality_weight = float(metadata.get("quality_weight", 1.0))
            score = (
                self.semantic_weight * similarity + self.recency_weight * recency
            ) * quality_weight
            quality_passed = bool(metadata.get("quality_passed", False))
            ranked.append(
                {
                    "memory_id": str(memory_id),
                    "date": str(metadata["timestamp_iso"]),
                    "summary": self._summary_from_metadata(metadata).to_dict(),
                    "topics": [
                        topic for topic in str(metadata.get("topics", "")).split(",") if topic
                    ],
                    "similarity": round(similarity, 6),
                    "recency": round(recency, 6),
                    "score": round(score, 6),
                    "quality_passed": quality_passed,
                    "quality_issues": json.loads(
                        str(metadata.get("quality_issues_json", "[]"))
                    ),
                    "annotation": (
                        None
                        if quality_passed
                        else "VERIFY BEFORE USING: summary failed quality check."
                    ),
                    "embedding_model_id": str(metadata["embedding_model_id"]),
                }
            )
        # Code-side sorting applies recency and quality after Chroma similarity search.
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]

    def recent_sessions(self, user_id: str, limit: int = 1) -> list[dict[str, Any]]:
        """Return same-user episodes by timestamp, without semantic re-ranking."""
        if not user_id.strip():
            return []
        if limit < 1:
            raise ValueError("limit must be positive")
        now = self._now().astimezone(timezone.utc)
        cutoff = now.timestamp() - self.retention_days * SECONDS_PER_DAY
        result = self.collection.get(
            where={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"timestamp": {"$gte": cutoff}},
                ]
            },
            include=["metadatas"],
        )
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        if len(ids) != len(metadatas):
            raise RuntimeError("episodic get returned misaligned result arrays")
        rows = sorted(
            zip(ids, metadatas),
            key=lambda row: float(row[1]["timestamp"]),
            reverse=True,
        )
        recent: list[dict[str, Any]] = []
        for memory_id, metadata in rows[:limit]:
            quality_passed = bool(metadata.get("quality_passed", False))
            recent.append(
                {
                    "memory_id": str(memory_id),
                    "date": str(metadata["timestamp_iso"]),
                    "summary": self._summary_from_metadata(metadata).to_dict(),
                    "topics": [
                        topic
                        for topic in str(metadata.get("topics", "")).split(",")
                        if topic
                    ],
                    "quality_passed": quality_passed,
                    "quality_issues": json.loads(
                        str(metadata.get("quality_issues_json", "[]"))
                    ),
                    "annotation": (
                        None
                        if quality_passed
                        else "VERIFY BEFORE USING: summary failed quality check."
                    ),
                    "embedding_model_id": str(metadata["embedding_model_id"]),
                }
            )
        return recent

    def count_user(self, user_id: str) -> int:
        """Count exact same-user episodic IDs without retrieving private documents."""
        # include=[] retrieves identifiers only, minimizing personal data exposure.
        result = self.collection.get(
            where={"user_id": {"$eq": user_id}},
            include=[],
        )
        return len(result.get("ids") or [])

    def forget_user(self, user_id: str) -> int:
        """Delete every episodic vector for one user and return the exact count."""
        # Count before deletion makes the privacy report exact and testable.
        count = self.count_user(user_id)
        if count:
            self.collection.delete(where={"user_id": {"$eq": user_id}})
        return count

    def retention_candidates(
        self,
        user_id: str,
        *,
        max_summaries: int = 50,
        now: datetime | None = None,
    ) -> list[RetentionCandidate]:
        """Plan TTL/cap removals without mutating the collection."""
        if not user_id.strip():
            raise ValueError("user_id must not be empty")
        if max_summaries < 1:
            raise ValueError("max_summaries must be positive")
        # Reject naïve clocks rather than interpreting them in the machine timezone.
        effective_now = now or self._now()
        if effective_now.tzinfo is None:
            raise ValueError("retention time must be timezone-aware")
        # Normalize once so TTL comparison is stable across caller timezones.
        effective_now = effective_now.astimezone(timezone.utc)
        cutoff = effective_now.timestamp() - (
            self.retention_days * SECONDS_PER_DAY
        )
        # Fetch only same-user scalar metadata; embedded prose is unnecessary here.
        result = self.collection.get(
            where={"user_id": {"$eq": user_id}},
            include=["metadatas"],
        )
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        if len(ids) != len(metadatas):
            raise RuntimeError("episodic retention query returned misaligned arrays")

        # Newest records occupy the allowed cap; older positions exceed it.
        ordered = sorted(
            zip(ids, metadatas),
            key=lambda item: float(item[1]["timestamp"]),
            reverse=True,
        )
        over_cap_ids = {str(item[0]) for item in ordered[max_summaries:]}
        candidates: list[RetentionCandidate] = []
        for memory_id, metadata in zip(ids, metadatas):
            timestamp = float(metadata["timestamp"])
            reasons: list[str] = []
            if timestamp < cutoff:
                reasons.append("older_than_ttl")
            if str(memory_id) in over_cap_ids:
                reasons.append("over_per_user_cap")
            if not reasons:
                continue
            # Structured metadata—not embedding prose—is authoritative here too.
            structured = self._summary_from_metadata(metadata)
            session_ids = json.loads(
                str(metadata.get("session_ids_json", "[]"))
            )
            if not isinstance(session_ids, list) or not all(
                isinstance(item, str) for item in session_ids
            ):
                raise RuntimeError("stored session_ids_json is invalid")
            candidates.append(
                RetentionCandidate(
                    memory_id=str(memory_id),
                    summary=structured,
                    timestamp=timestamp,
                    quality_passed=bool(metadata.get("quality_passed", False)),
                    session_ids=tuple(session_ids),
                    reasons=tuple(reasons),
                )
            )
        # Oldest-first order makes consolidation behavior stable and readable.
        candidates.sort(key=lambda candidate: candidate.timestamp)
        return candidates

    def delete_retention_candidates(
        self,
        user_id: str,
        memory_ids: tuple[str, ...],
    ) -> int:
        """Delete only requested IDs proven to belong to the exact user."""
        if not memory_ids:
            return 0
        # Re-read ownership immediately before destructive mutation.
        owned = self.collection.get(
            where={"user_id": {"$eq": user_id}},
            include=[],
        )
        owned_ids = {str(item) for item in (owned.get("ids") or [])}
        targets = tuple(dict.fromkeys(item for item in memory_ids if item in owned_ids))
        if targets:
            # Explicit IDs prevent deletion expanding beyond the reviewed plan.
            self.collection.delete(ids=list(targets))
        return len(targets)
