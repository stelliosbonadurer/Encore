"""Small, explicit working state shared by tools during one agent session."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionEvidence:
    """Evidence the deterministic executor may trust for later safety checks.

    The model may mention arbitrary MBIDs, but only ``verify_recordings`` writes
    to this set after every requested ID succeeds. Playlist-write tools consult
    the set before making network calls.
    """

    # A set gives constant-time membership checks and naturally removes repeats.
    verified_recording_mbids: set[str] = field(default_factory=set)

    def record_verified(self, mbids: list[str]) -> None:
        """Trust a complete batch only after verify_recordings succeeds."""
        # ``update`` is atomic at the logical batch boundary used by the caller.
        self.verified_recording_mbids.update(mbids)

    def unverified(self, mbids: list[str]) -> list[str]:
        """Preserve request order while returning IDs not yet evidenced."""
        # Returning the actual missing IDs makes the model's recovery call clear.
        return [mbid for mbid in mbids if mbid not in self.verified_recording_mbids]
