"""Shared deterministic naming policy for Encore-created playlists."""

# This logical marker substitutes for service folders, which APIs cannot manage.
PLAYLIST_PREFIX = "[ListenBrainz]"


def normalized_playlist_title(title: str) -> str:
    """Return the exact title shown in previews and sent to ListenBrainz."""
    # Remove accidental surrounding whitespace before displaying or writing.
    cleaned = title.strip()
    # Prefix only once so retries and already-normalized titles remain stable.
    if cleaned and not cleaned.startswith(PLAYLIST_PREFIX):
        return f"{PLAYLIST_PREFIX} {cleaned}"
    # Empty input remains empty so the handler can return a clear input error.
    return cleaned
