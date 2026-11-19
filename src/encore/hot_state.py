"""Deterministic hot-state updates derived from executed tool results.

The model never writes hot state. After each executed tool call, the loop asks
this module which code-owned keys the observation should change. Keeping the
mapping pure (no I/O) makes it easy to test and to explain: every key has one
documented source.

``None`` in a returned mapping means "delete this key".
"""

from __future__ import annotations

from typing import Any

from encore.contracts import ToolResult

# Session-scoped keys. They are cleared at the start of every chat session so a
# previous session's active playlist cannot leak into a new conversation.
SESSION_HOT_KEYS = (
    "active_playlist",
    "last_recordings",
    "last_artist",
    "pending_approval",
    "tool_call_count",
)
# Enough positions for "the second song" without flooding the context.
MAX_LAST_RECORDINGS = 10
# Tool name -> (result list key, title field) for recording-bearing reads.
_RECORDING_LISTS = {
    "search_recordings": ("recordings", "title"),
    "verify_recordings": ("verified_recordings", "title"),
    "get_recent_listens": ("listens", "track"),
    "lb_radio": ("tracks", "title"),
    "get_top_recordings_for_artist": ("recordings", "title"),
}
# Successful playlist writes make their target the active playlist.
_ACTIVE_PLAYLIST_WRITES = (
    "create_playlist",
    "add_playlist_items",
    "remove_playlist_items",
    "export_playlist",
)


def _recordings(items: Any, title_field: str) -> list[dict[str, Any]]:
    """Keep the ordered identity fields needed to resolve positional references."""
    if not isinstance(items, list):
        return []
    trimmed = []
    for item in items:
        if not isinstance(item, dict) or not item.get("recording_mbid"):
            continue
        trimmed.append(
            {
                "position": len(trimmed) + 1,
                "title": item.get(title_field),
                "artist": item.get("artist"),
                "recording_mbid": item["recording_mbid"],
            }
        )
        if len(trimmed) == MAX_LAST_RECORDINGS:
            break
    return trimmed


def updates_for_result(
    tool_name: str,
    result: ToolResult,
    active_playlist: Any = None,
) -> dict[str, Any]:
    """Return hot-state changes implied by one successful tool observation."""
    if not result.success or not isinstance(result.data, dict):
        return {}
    data = result.data
    updates: dict[str, Any] = {}

    if tool_name in _RECORDING_LISTS:
        list_key, title_field = _RECORDING_LISTS[tool_name]
        recordings = _recordings(data.get(list_key), title_field)
        if recordings:
            updates["last_recordings"] = recordings

    elif tool_name == "lookup_artist" and data.get("artist_mbid"):
        updates["last_artist"] = {
            "name": data.get("name"),
            "artist_mbid": data["artist_mbid"],
        }

    elif tool_name in _ACTIVE_PLAYLIST_WRITES and data.get("playlist_mbid"):
        updates["active_playlist"] = {
            "playlist_mbid": data["playlist_mbid"],
            "title": data.get("title"),
        }

    elif tool_name == "find_playlists":
        playlists = data.get("playlists")
        # Only an unambiguous single match becomes the referent of "that playlist".
        if isinstance(playlists, list) and len(playlists) == 1:
            match = playlists[0]
            if isinstance(match, dict) and match.get("playlist_mbid"):
                updates["active_playlist"] = {
                    "playlist_mbid": match["playlist_mbid"],
                    "title": match.get("title"),
                }

    elif tool_name == "delete_playlist":
        deleted = data.get("playlist_mbid")
        if (
            isinstance(active_playlist, dict)
            and deleted
            and active_playlist.get("playlist_mbid") == deleted
        ):
            updates["active_playlist"] = None

    return updates


def declined_write(tool_name: str, preview: str) -> dict[str, str]:
    """Describe the last declined write so the user can later reconsider it.

    This is a record, not a permit: re-proposing it still needs fresh approval.
    """
    return {"tool": tool_name, "preview": preview, "status": "declined"}
