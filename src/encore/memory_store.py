"""Persistent personal-memory stores and their deletion-first privacy boundary.

SQLite holds exact, frequently accessed state and explicit preferences. Raw
conversation turns removed by context compression live in separate,
private JSONL archives so they never become semantic episodic memories. Every
personal record is scoped by ``ENCORE_USER_ID`` or its one-way path key.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

# Only these conversation roles belong in raw archives. System constraints are
# reconstructed from version-controlled procedural memory and are never archived.
ARCHIVE_ROLES = frozenset({"user", "assistant", "tool"})
# Preference scope is explicit so session observations cannot silently become
# long-term claims without passing through the future retention policy.
PREFERENCE_SCOPES = frozenset({"session", "long_term"})


def user_storage_key(user_id: str) -> str:
    """Return a stable non-reversible directory key for one non-empty user ID."""
    # Whitespace-only identity would collapse unrelated invalid configurations.
    if not user_id.strip():
        raise ValueError("user_id must not be empty")
    # The full digest makes accidental path-key collisions impractical.
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def _json_text(value: Any) -> str:
    """Serialize portable JSON and reject NaN, Infinity, or custom objects."""
    # Compact output keeps the local store readable without wasting disk space.
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _normalized_preference(value: str) -> tuple[str, str]:
    """Return display text and a case-insensitive exact-deduplication key."""
    # Collapsing whitespace makes repeated model calls deterministic.
    display = " ".join(value.split())
    if not display:
        raise ValueError("preference must not be empty")
    # Casefold handles more Unicode cases than lower() while retaining display text.
    return display, display.casefold()


@dataclass(frozen=True)
class PersonalDeleteCounts:
    """Rows removed from the two user-scoped SQLite tables."""

    hot_state: int
    preferences: int


@dataclass(frozen=True)
class ArchiveDeleteCounts:
    """Private archive artifacts removed for one hashed user directory."""

    files: int
    messages: int


@dataclass(frozen=True)
class ForgetReport:
    """Complete deletion result across every current personal-memory store."""

    hot_state: int
    preferences: int
    episodic_summaries: int
    archive_files: int
    archived_messages: int

    @property
    def total_items(self) -> int:
        """Count logical records while reporting archive files separately."""
        # A file is a container, so it is not added to its contained message count.
        return (
            self.hot_state
            + self.preferences
            + self.episodic_summaries
            + self.archived_messages
        )

    def to_dict(self) -> dict[str, int]:
        """Return a detached JSON-compatible report for CLI/tests/traces."""
        return asdict(self)


class EpisodicForgetter(Protocol):
    """Narrow Chroma deletion contract used by the privacy coordinator."""

    def forget_user(self, user_id: str) -> int: ...


class PersonalMemoryStore:
    """SQLite hot-state and preference storage scoped by raw user identity."""

    def __init__(
        self,
        path: Path,
        *,
        now: Callable[[], float] = time.time,
    ) -> None:
        # The database path is application configuration, never model output.
        self.path = Path(path)
        # Clock injection makes TTL and retention tests instantaneous.
        self._now = now
        # Private directory permissions reduce accidental local disclosure.
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # sqlite3 parameter binding is used for every user-controlled value.
        self._connection = sqlite3.connect(self.path)
        # Named access improves readability and avoids column-order coupling.
        self._connection.row_factory = sqlite3.Row
        # Foreign keys are enabled now even though the first schema is flat.
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()
        # SQLite honors process umask; chmod makes the intended boundary explicit.
        self.path.chmod(0o600)

    def _create_schema(self) -> None:
        """Create the minimal exact-lookup tables without speculative columns."""
        # A transaction ensures startup never exposes half of the required schema.
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hot_state (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    expires_at REAL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, key)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    preference TEXT NOT NULL,
                    normalized_preference TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    scope TEXT NOT NULL CHECK (scope IN ('session', 'long_term')),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE (user_id, normalized_preference)
                )
                """
            )
            # User indexes make retrieval and complete deletion predictable.
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_hot_user ON hot_state(user_id)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_preferences_user "
                "ON preferences(user_id, updated_at DESC)"
            )

    def set_hot(
        self,
        user_id: str,
        key: str,
        value: Any,
        *,
        ttl_s: float | None = None,
    ) -> None:
        """Set one JSON hot-state value, optionally expiring after positive seconds."""
        # Validate identity even though SQLite can store an empty string.
        user_storage_key(user_id)
        # Keys are code-owned names such as active_playlist or last_recording.
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("hot-state key must not be empty")
        # Zero/negative TTL would create a surprising immediately absent write.
        if ttl_s is not None and ttl_s <= 0:
            raise ValueError("ttl_s must be positive when provided")
        # Strict serialization occurs before opening the mutation transaction.
        encoded = _json_text(value)
        now = self._now()
        expires_at = now + ttl_s if ttl_s is not None else None
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO hot_state(user_id, key, value_json, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (user_id, normalized_key, encoded, expires_at, now),
            )

    def get_hot(self, user_id: str, key: str, default: Any = None) -> Any:
        """Return one decoded hot value or default, lazily removing expiration."""
        user_storage_key(user_id)
        row = self._connection.execute(
            "SELECT value_json, expires_at FROM hot_state "
            "WHERE user_id = ? AND key = ?",
            (user_id, key.strip()),
        ).fetchone()
        if row is None:
            return default
        # Expired state is deleted in the same user/key scope before returning.
        if row["expires_at"] is not None and row["expires_at"] <= self._now():
            self.delete_hot(user_id, key)
            return default
        # Stored JSON was strict at write time, so decoding is deterministic.
        return json.loads(row["value_json"])

    def delete_hot(self, user_id: str, key: str) -> int:
        """Delete one exact user/key pair and return the affected-row count."""
        user_storage_key(user_id)
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM hot_state WHERE user_id = ? AND key = ?",
                (user_id, key.strip()),
            )
        return cursor.rowcount

    def note_preference(
        self,
        user_id: str,
        preference: str,
        evidence: str,
        *,
        scope: str = "session",
    ) -> dict[str, Any]:
        """Insert or update one explicit preference without creating duplicates."""
        user_storage_key(user_id)
        display, normalized = _normalized_preference(preference)
        if scope not in PREFERENCE_SCOPES:
            raise ValueError(f"unknown preference scope '{scope}'")
        # Evidence is descriptive data; whitespace normalization aids comparison.
        normalized_evidence = " ".join(evidence.split())
        now = self._now()
        with self._connection:
            # Repetition updates evidence/time but does not auto-promote scope.
            self._connection.execute(
                """
                INSERT INTO preferences(
                    user_id, preference, normalized_preference, evidence,
                    scope, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, normalized_preference) DO UPDATE SET
                    preference = excluded.preference,
                    evidence = CASE
                        WHEN excluded.evidence = '' THEN preferences.evidence
                        ELSE excluded.evidence
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    display,
                    normalized,
                    normalized_evidence,
                    scope,
                    now,
                    now,
                ),
            )
        # Fetching by the exact dedup key returns both new and existing records.
        row = self._connection.execute(
            """
            SELECT id, preference, evidence, scope, created_at, updated_at
            FROM preferences
            WHERE user_id = ? AND normalized_preference = ?
            """,
            (user_id, normalized),
        ).fetchone()
        assert row is not None
        return dict(row)

    def promote_preference(self, user_id: str, preference_id: int) -> bool:
        """Explicitly promote one owned session preference to long-term scope."""
        user_storage_key(user_id)
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE preferences
                SET scope = 'long_term', updated_at = ?
                WHERE user_id = ? AND id = ?
                """,
                (self._now(), user_id, preference_id),
            )
        return cursor.rowcount == 1

    def find_preference(
        self,
        user_id: str,
        preference: str,
    ) -> dict[str, Any] | None:
        """Find one exact normalized preference owned by the requested user."""
        # Validate identity and normalize with the same function used on insertion.
        user_storage_key(user_id)
        _, normalized = _normalized_preference(preference)
        # Exact lookup is intentionally stronger evidence than semantic similarity.
        row = self._connection.execute(
            """
            SELECT id, preference, evidence, scope, created_at, updated_at
            FROM preferences
            WHERE user_id = ? AND normalized_preference = ?
            """,
            (user_id, normalized),
        ).fetchone()
        # Return a detached mapping so callers never hold a live SQLite Row.
        return None if row is None else dict(row)

    def list_preferences(self, user_id: str) -> list[dict[str, Any]]:
        """Return only this user's preferences in most-recently-updated order."""
        user_storage_key(user_id)
        rows = self._connection.execute(
            """
            SELECT id, preference, evidence, scope, created_at, updated_at
            FROM preferences WHERE user_id = ?
            ORDER BY updated_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def count_user(self, user_id: str) -> PersonalDeleteCounts:
        """Count current personal rows without changing either table."""
        user_storage_key(user_id)
        hot = self._connection.execute(
            "SELECT COUNT(*) FROM hot_state WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        preferences = self._connection.execute(
            "SELECT COUNT(*) FROM preferences WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        return PersonalDeleteCounts(hot, preferences)

    def forget_user(self, user_id: str) -> PersonalDeleteCounts:
        """Delete all SQLite personal memory for one exact user atomically."""
        user_storage_key(user_id)
        # Counts and deletes share a transaction so the report matches the mutation.
        with self._connection:
            before = self.count_user(user_id)
            self._connection.execute(
                "DELETE FROM hot_state WHERE user_id = ?", (user_id,)
            )
            self._connection.execute(
                "DELETE FROM preferences WHERE user_id = ?", (user_id,)
            )
        return before

    def close(self) -> None:
        """Close the SQLite connection after its owning session ends."""
        self._connection.close()

    def __enter__(self) -> PersonalMemoryStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class ArchiveStore:
    """Private per-session JSONL storage for raw turns removed from context."""

    def __init__(
        self,
        root: Path,
        *,
        now: Callable[[], float] = time.time,
    ) -> None:
        # Archives remain physically separate from semantic/vector collections.
        self.root = Path(root)
        self._now = now
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _user_directory(self, user_id: str) -> Path:
        """Resolve the one exact hashed directory belonging to this user."""
        return self.root / user_storage_key(user_id)

    def archive_messages(
        self,
        user_id: str,
        session_id: str,
        messages: Iterable[dict[str, Any]],
    ) -> int:
        """Append validated raw conversation messages to one private session file."""
        # UUID validation prevents traversal through a model-controlled filename.
        canonical_session = str(UUID(session_id))
        directory = self._user_directory(user_id)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = directory / f"{canonical_session}.jsonl"
        # Validate and detach every record before opening the destination file.
        prepared: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("archived message must be an object")
            role = message.get("role")
            if role not in ARCHIVE_ROLES:
                raise ValueError(f"role '{role}' may not be archived")
            # Strict round-trip rejects custom objects and non-portable numbers.
            detached = json.loads(_json_text(message))
            envelope = {"archived_at": self._now(), "message": detached}
            prepared.append(_json_text(envelope))
        if not prepared:
            return 0
        # O_NOFOLLOW prevents an existing filename redirecting private writes.
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as archive:
            for line in prepared:
                archive.write(line + "\n")
            archive.flush()
        path.chmod(0o600)
        return len(prepared)

    def count_user(self, user_id: str) -> ArchiveDeleteCounts:
        """Count regular JSONL archive files and valid physical message lines."""
        directory = self._user_directory(user_id)
        if not directory.exists() or not directory.is_dir():
            return ArchiveDeleteCounts(0, 0)
        files = 0
        messages = 0
        # scandir allows an explicit no-symlink policy for deletion accounting.
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.name.endswith(".jsonl") or not entry.is_file(
                    follow_symlinks=False
                ):
                    continue
                files += 1
                with open(entry.path, "r", encoding="utf-8") as archive:
                    messages += sum(1 for line in archive if line.strip())
        return ArchiveDeleteCounts(files, messages)

    def forget_user(self, user_id: str) -> ArchiveDeleteCounts:
        """Delete only regular JSONL files inside this user's exact hashed directory."""
        directory = self._user_directory(user_id)
        counts = self.count_user(user_id)
        if not directory.exists() or not directory.is_dir():
            return counts
        # Never recurse or follow links; only known archive filenames are removed.
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name.endswith(".jsonl") and entry.is_file(
                    follow_symlinks=False
                ):
                    Path(entry.path).unlink()
        # Remove the user directory only when no unrelated artifact remains.
        try:
            directory.rmdir()
        except OSError:
            pass
        return counts


class MemoryPrivacyManager:
    """Coordinate complete user deletion across local personal-memory stores."""

    def __init__(
        self,
        personal: PersonalMemoryStore,
        archives: ArchiveStore,
        episodic: EpisodicForgetter,
    ) -> None:
        self.personal = personal
        self.archives = archives
        self.episodic = episodic

    def forget_user(self, user_id: str) -> ForgetReport:
        """Delete one user's data from every current store and return exact counts."""
        # Each backend is called with the same raw deletion identity.
        personal = self.personal.forget_user(user_id)
        episodic_count = self.episodic.forget_user(user_id)
        archives = self.archives.forget_user(user_id)
        return ForgetReport(
            hot_state=personal.hot_state,
            preferences=personal.preferences,
            episodic_summaries=episodic_count,
            archive_files=archives.files,
            archived_messages=archives.messages,
        )
