"""Human-approval and idempotency primitives for account writes.

The model can propose only JSON arguments. A write additionally requires a
``WritePermit`` object created here after human confirmation. Because that
object is never part of a tool schema or model message, the model cannot forge
approval by emitting something like ``{"approved": true}``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from encore.contracts import ToolResult
from encore.playlist_policy import normalized_playlist_title
from encore.tool_catalog import WRITE_TOOL_SPECS

# A frozen set makes membership fast and prevents runtime mutation of policy.
WRITE_TOOL_NAMES = frozenset(spec.name for spec in WRITE_TOOL_SPECS)


def operation_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    """Create a stable, secret-free identity for one exact proposed operation."""
    # Sorting keys makes semantically identical dictionaries hash identically.
    canonical = json.dumps(
        {"tool": tool_name, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    # SHA-256 is used as a stable identifier, not to hide the public arguments.
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApprovalRequest:
    """Exact preview shown to the user before a permit may be issued."""

    # Random ID connects one UI confirmation exchange to its future permit.
    request_id: str
    # Tool and detached arguments are the exact proposal being displayed.
    tool_name: str
    arguments: dict[str, Any]
    # Summary is deterministic code-generated prose, never model-authored prose.
    summary: str
    # Fingerprint prevents argument substitution after approval.
    fingerprint: str
    # Large operations add a second, stronger confirmation step.
    high_volume: bool
    high_volume_phrase: str | None


@dataclass(frozen=True)
class WritePermit:
    """Opaque capability bound to one request and one exact argument payload."""

    request_id: str
    fingerprint: str
    # Object identity proves that this instance came from the active gate. A
    # model can produce JSON strings but cannot reproduce this Python object.
    _issuer_token: object


def _requires_high_volume(
    tool_name: str, arguments: dict[str, Any], logical_count: int
) -> bool:
    # Playlist additions/creation measure tracks; other operations use their
    # domain-specific counts below.
    track_count = len(arguments.get("recording_mbids") or [])
    if tool_name == "create_playlist":
        return track_count > 50 or logical_count > 1
    if tool_name == "add_playlist_items":
        return track_count > 50
    if tool_name == "remove_playlist_items":
        return int(arguments.get("count") or 0) > 20
    if tool_name == "set_feedback":
        return logical_count > 5
    if tool_name in {"export_playlist", "delete_playlist"}:
        return logical_count > 1
    return False


def _high_volume_phrase(
    tool_name: str, arguments: dict[str, Any], logical_count: int
) -> str:
    """Name both the operation and the quantity that triggered escalation."""
    track_count = len(arguments.get("recording_mbids") or [])
    if tool_name == "create_playlist":
        # Creation can independently cross playlist-count and track thresholds.
        parts = ["CONFIRM CREATE_PLAYLIST"]
        if logical_count > 1:
            parts.append(f"{logical_count} PLAYLISTS")
        if track_count > 50:
            parts.append(f"{track_count} TRACKS")
        return " ".join(parts)
    if tool_name == "add_playlist_items":
        return f"CONFIRM ADD_PLAYLIST_ITEMS {track_count} TRACKS"
    if tool_name == "remove_playlist_items":
        return f"CONFIRM REMOVE_PLAYLIST_ITEMS {arguments.get('count', 0)} TRACKS"
    if tool_name == "set_feedback":
        return f"CONFIRM SET_FEEDBACK {logical_count} CHANGES"
    # Export/delete escalation is expressed in playlist units.
    return f"CONFIRM {tool_name.upper()} {logical_count} PLAYLISTS"


def _summary(tool_name: str, arguments: dict[str, Any], logical_count: int) -> str:
    """Produce a deterministic preview without trusting model-authored prose."""
    # Each branch names exact targets/counts without accepting model prose.
    if tool_name == "create_playlist":
        return (
            f"Create private playlist "
            f"'{normalized_playlist_title(str(arguments.get('title', '')))}' with "
            f"{len(arguments.get('recording_mbids') or [])} tracks."
        )
    if tool_name == "add_playlist_items":
        return (
            f"Add {len(arguments.get('recording_mbids') or [])} tracks to playlist "
            f"{arguments.get('playlist_mbid', '')}."
        )
    if tool_name == "remove_playlist_items":
        return (
            f"Remove {arguments.get('count', 0)} tracks starting at index "
            f"{arguments.get('index', 0)} from playlist "
            f"{arguments.get('playlist_mbid', '')}."
        )
    if tool_name == "set_feedback":
        return (
            f"Set feedback for recording {arguments.get('recording_mbid', '')} "
            f"to {arguments.get('rating', '')}."
        )
    if tool_name == "export_playlist":
        return (
            f"Export playlist {arguments.get('playlist_mbid', '')} to "
            f"{arguments.get('service', '')}."
        )
    if tool_name == "delete_playlist":
        return f"Permanently delete playlist {arguments.get('playlist_mbid', '')}."
    return f"Run {logical_count} {tool_name} operation(s)."


class ApprovalGate:
    """Create and validate exact, user-confirmed write capabilities.

    The executor owns calls to ``prepare`` and ``confirm``. Tool handlers can
    only call ``validate``/``complete``; they cannot manufacture confirmation.
    """

    def __init__(self) -> None:
        # This object identity never crosses the JSON/model boundary.
        self._issuer_token = object()
        # Active requests are small and exist only until completion/decline.
        self._active: dict[str, ApprovalRequest] = {}

    def prepare(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        logical_count: int = 1,
    ) -> ApprovalRequest:
        # Never create a permit for a read, memory, or invented tool name.
        if tool_name not in WRITE_TOOL_NAMES:
            raise ValueError(f"{tool_name} is not an account-write tool")
        # Aggregate counts come from loop planning and must be meaningful.
        if logical_count < 1:
            raise ValueError("logical_count must be positive")

        # Round-tripping produces a detached JSON-only copy. Later mutation of
        # the caller's dict therefore cannot alter what the user approved.
        copied = json.loads(json.dumps(arguments, allow_nan=False))
        # Thresholds warn rather than cap, matching the student's decision.
        high_volume = _requires_high_volume(tool_name, copied, logical_count)
        phrase = (
            _high_volume_phrase(tool_name, copied, logical_count)
            if high_volume
            else None
        )
        # Preparing a request does not authorize it or place it in active state.
        return ApprovalRequest(
            request_id=str(uuid4()),
            tool_name=tool_name,
            arguments=copied,
            summary=_summary(tool_name, copied, logical_count),
            fingerprint=operation_fingerprint(tool_name, copied),
            high_volume=high_volume,
            high_volume_phrase=phrase,
        )

    def confirm(
        self,
        request: ApprovalRequest,
        response: str,
        *,
        high_volume_response: str = "",
    ) -> WritePermit | None:
        """Issue a permit only for unambiguous normal and volume confirmation."""
        # Anything except a standalone yes is an unambiguous decline.
        if response.strip().casefold() != "yes":
            return None
        # High-volume text is intentionally case-sensitive and operation-specific.
        if request.high_volume and high_volume_response != request.high_volume_phrase:
            return None
        # Only now does the request become an active executable capability.
        self._active[request.request_id] = request
        # The hidden issuer object cannot be reconstructed from model JSON.
        return WritePermit(request.request_id, request.fingerprint, self._issuer_token)

    def validate(
        self, permit: WritePermit | None, tool_name: str, arguments: dict[str, Any]
    ) -> bool:
        # Reject missing, structurally fake, and foreign-gate permits first.
        if not isinstance(permit, WritePermit) or permit._issuer_token is not self._issuer_token:
            return False
        # An already-completed/unknown request has no active authorization.
        request = self._active.get(permit.request_id)
        # Recompute from handler arguments rather than trusting stored caller data.
        expected = operation_fingerprint(tool_name, arguments)
        return bool(
            request
            and request.tool_name == tool_name
            and request.fingerprint == permit.fingerprint == expected
        )

    def complete(self, permit: WritePermit) -> None:
        """Close a logical operation after success or non-retryable partial work."""
        # pop is idempotent so cleanup remains safe on repeated terminal paths.
        self._active.pop(permit.request_id, None)


class IdempotencyLedger:
    """Remember the latest completed write per target for one session.

    A write is a replay only when it exactly matches the most recent completed
    write on every target it touches. Any newer write on that target, or any
    failed/uncertain attempt, ends the match. Therefore love -> clear -> love
    sends three real requests, while an immediate retry of the same add sends
    one. The ledger is deliberately process-local: cross-session duplicates are
    handled by live-state preflight (title reuse, current feedback score,
    ownership), and nothing personal is persisted outside the memory stores.
    """

    def __init__(self) -> None:
        # target key -> (fingerprint, result) of the latest completed write.
        self._latest: dict[str, tuple[str, ToolResult]] = {}

    def replay(self, targets: list[str], fingerprint: str) -> ToolResult | None:
        """Return the stored result only if it is the latest write on every target."""
        if not targets:
            return None
        entries = [self._latest.get(target) for target in targets]
        if any(entry is None or entry[0] != fingerprint for entry in entries):
            return None
        # All entries share this fingerprint, so any of them holds the result.
        return entries[0][1]  # type: ignore[index]

    def record(self, targets: list[str], fingerprint: str, result: ToolResult) -> None:
        """Make this completed write the latest one on each target it touched."""
        for target in targets:
            self._latest[target] = (fingerprint, result)

    def invalidate(self, targets: list[str]) -> None:
        """Forget targets whose external state may have changed unpredictably."""
        for target in targets:
            self._latest.pop(target, None)
