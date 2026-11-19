"""Model-facing tools for the three selected persistent memory stores."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from encore.contracts import ToolResult, fail, ok
from encore.tool_catalog import MEMORY_TOOL_SPECS

if TYPE_CHECKING:
    from episodic_memory import EpisodicMemoryStore
    from memory_store import PersonalMemoryStore
    from semantic_memory import SemanticMemoryStore


class MemoryTools:
    """Validate access to episodic, semantic, and preference memory."""

    def __init__(
        self,
        personal: "PersonalMemoryStore",
        episodic: "EpisodicMemoryStore",
        semantic: "SemanticMemoryStore",
        user_id: str | None,
    ) -> None:
        self.personal = personal
        self.episodic = episodic
        self.semantic = semantic
        self.user_id = user_id

    def search_past_sessions(self, query: str, limit: int = 10) -> ToolResult:
        if not query.strip():
            return fail("input", "query must not be empty.")
        if not 1 <= limit <= 10:
            return fail("input", f"limit must be between 1 and 10; got {limit}.")
        if not self.user_id:
            return fail("fatal", "ENCORE_USER_ID is not configured.")
        normalized = " ".join(query.casefold().split())
        chronological = any(
            phrase in normalized
            for phrase in (
                "last session",
                "most recent session",
                "latest session",
                "previous session",
                "what did we do last",
            )
        )
        matches = (
            self.episodic.recent_sessions(self.user_id, 1)
            if chronological
            else self.episodic.search_sessions(self.user_id, query, limit)
        )
        if not matches:
            return fail("resource", "No relevant prior session was found.")
        return ok({"sessions": matches})

    def search_music_knowledge(self, query: str, limit: int = 3) -> ToolResult:
        if not query.strip():
            return fail("input", "query must not be empty.")
        if not 1 <= limit <= 5:
            return fail("input", f"limit must be between 1 and 5; got {limit}.")
        matches = self.semantic.search_knowledge(query, limit)
        if not matches:
            return fail("resource", "No relevant music-knowledge chunk was found.")
        return ok({"chunks": matches})

    def note_preference(self, preference: str, evidence: str = "") -> ToolResult:
        if not preference.strip():
            return fail("input", "preference must not be empty.")
        if len(preference) > 500 or len(evidence) > 500:
            return fail(
                "input",
                "preference and evidence must each be at most 500 characters.",
            )
        if not self.user_id:
            return fail("fatal", "ENCORE_USER_ID is not configured.")
        record = self.personal.note_preference(
            self.user_id,
            preference.strip(),
            evidence.strip(),
            scope="session",
        )
        return ok({"preference": record})

    def registry(self) -> dict[str, Callable[..., ToolResult]]:
        return {spec.name: getattr(self, spec.name) for spec in MEMORY_TOOL_SPECS}
