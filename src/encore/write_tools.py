"""Approval-gated ListenBrainz account-write tools.

These handlers implement the side effect, while the ReAct loop owns the
conversation that obtains approval. Every public method independently checks
runtime mode, an opaque permit, current account state, and idempotency so router
or prompt mistakes cannot bypass the safety boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from encore.api_client import LB_API, SLOW_READ_TIMEOUT_S, WRITE_TIMEOUT_S, ApiClient
from encore.approval import ApprovalGate, IdempotencyLedger, WritePermit, operation_fingerprint
from encore.config import RuntimePolicy
from encore.contracts import POLICY_HALT_PREFIX, ToolResult, classify_status, fail, ok
from encore.playlist_policy import PLAYLIST_PREFIX, normalized_playlist_title
from encore.session_state import SessionEvidence
from encore.tool_catalog import WRITE_TOOL_SPECS

# JSPF extension controls privacy on playlist creation.
JSPF_EXTENSION = "https://musicbrainz.org/doc/jspf#playlist"
# Sequential, bounded batches make partial progress measurable and recoverable.
ADD_BATCH_SIZE = 100


def _valid_mbid(value: str) -> bool:
    """Reject malformed model identifiers before network or account activity."""
    try:
        # Playlist and recording MBIDs both use UUID syntax.
        UUID(value)
    except (ValueError, AttributeError, TypeError):
        # Null/non-string/malformed input is expected model data, not an exception.
        return False
    # Successful parsing is the local syntactic proof; APIs prove existence later.
    return True


def _identifier_mbid(identifier: Any) -> str | None:
    """Normalize the string-or-list identifier shape used by JSPF."""
    if isinstance(identifier, list):
        identifier = identifier[0] if identifier else None
    if not isinstance(identifier, str):
        return None
    # ListenBrainz identifiers are URLs ending in the relevant UUID.
    candidate = identifier.rstrip("/").rsplit("/", 1)[-1]
    # Refuse arbitrary URL tails before comparing against approved targets.
    return candidate if _valid_mbid(candidate) else None


def _jspf_track(recording_mbid: str) -> dict[str, str]:
    """Build the minimal MusicBrainz-extended JSPF track accepted by LB."""
    # Titles/artists are omitted because identity comes from the verified MBID.
    return {"identifier": f"https://musicbrainz.org/recording/{recording_mbid}"}


def _safe_service_error(payload: Any) -> str | None:
    """Extract bounded public API detail without serializing arbitrary payloads."""
    value: Any = None
    if isinstance(payload, dict):
        value = payload.get("error") or payload.get("message")
    elif isinstance(payload, str):
        value = payload
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned[:300] if cleaned else None


class WriteTools:
    """Execute writes only when all deterministic safeguards agree."""

    def __init__(
        self,
        client: ApiClient,
        listenbrainz_user: str | None,
        policy: RuntimePolicy,
        evidence: SessionEvidence,
        approval_gate: ApprovalGate,
        ledger: IdempotencyLedger,
    ) -> None:
        # Transport remains allowlisted and owns token attachment.
        self.client = client
        # Account identity cannot be supplied or changed by the model.
        self.listenbrainz_user = listenbrainz_user
        # Runtime mode is the outer offline/explicit-flag safety switch.
        self.policy = policy
        # Session evidence distinguishes verified IDs from model assertions.
        self.evidence = evidence
        # Approval gate validates hidden, exact-argument capabilities.
        self.approval_gate = approval_gate
        # Ledger turns a completed identical retry into a local no-op.
        self.ledger = ledger

    def _authorize(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        permit: WritePermit | None,
        targets: list[str],
    ) -> tuple[str, ToolResult | None]:
        """Return a fingerprint and either a denied/replayed result or None."""
        # Recompute identity from the handler's actual arguments.
        fingerprint = operation_fingerprint(tool_name, arguments)

        # Policy and approval come first: a remembered result never bypasses them.
        # A valid permit still cannot override a read-only process policy.
        if not self.policy.writes_enabled:
            return fingerprint, fail(
                "fatal", "Account writes are disabled in this runtime mode."
            )
        # Name and every argument must match the exact displayed proposal.
        if not self.approval_gate.validate(permit, tool_name, arguments):
            return fingerprint, fail(
                "resource",
                "Write approval is missing, expired, or does not match the exact arguments.",
            )
        # An exact repeat of the latest completed write on these targets is a
        # retry; report the earlier outcome instead of mutating the account again.
        previous = self.ledger.replay(targets, fingerprint)
        if previous is not None:
            assert permit is not None  # Proven by validate above.
            self.approval_gate.complete(permit)
            replayed = ok(
                {
                    **(previous.data or {}),
                    "replayed": True,
                    "note": "Identical to the latest completed change on this "
                    "target; nothing was sent again.",
                }
            )
            return fingerprint, replayed
        return fingerprint, None

    def _finish(
        self,
        fingerprint: str,
        targets: list[str],
        permit: WritePermit,
        result: ToolResult,
    ) -> ToolResult:
        """Record success as the latest write on its targets, then close the permit."""
        # Record before closing to minimize duplicate risk if the process crashes.
        self.ledger.record(targets, fingerprint, result)
        # A completed permit cannot authorize another operation.
        self.approval_gate.complete(permit)
        return result

    def _uncertain(
        self, targets: list[str], permit: WritePermit, result: ToolResult
    ) -> ToolResult:
        """Close a permit after an attempted mutation whose outcome is unknown."""
        # External state may have changed, so no earlier write may be replayed.
        self.ledger.invalidate(targets)
        self.approval_gate.complete(permit)
        return result

    def _close_if_terminal(
        self, permit: WritePermit | None, result: ToolResult
    ) -> ToolResult:
        """Allow transient pre-write retry; close permits on terminal outcomes."""
        # Only a pre-write transient outcome may reuse the same approval.
        if permit and result.error_category != "transient":
            self.approval_gate.complete(permit)
        return result

    def _list_owned_playlists(self) -> tuple[ToolResult | None, list[dict[str, Any]]]:
        """Page through the configured user's authoritative owned-playlist list."""
        if not self.listenbrainz_user:
            return fail("input", "LISTENBRAINZ_USER is not configured."), []
        # Start at the first page and advance by the actual returned page size.
        offset = 0
        playlists: list[dict[str, Any]] = []
        while True:
            # This authenticated user endpoint is the ownership authority.
            status, body, _ = self.client.request(
                "GET",
                f"{LB_API}/user/{self.listenbrainz_user}/playlists",
                params={"count": 100, "offset": offset},
                auth=True,
                timeout=SLOW_READ_TIMEOUT_S,
            )
            if status != 200:
                return (
                    fail(
                        classify_status(status),
                        f"Could not prove playlist ownership (HTTP {status}).",
                    ),
                    [],
                )
            # Remove the API's outer wrapper but preserve playlist metadata.
            page = body.get("playlists", [])
            playlists.extend((item.get("playlist") or {}) for item in page)
            # API count controls pagination; fallback supports minimal test fakes.
            total = int(body.get("playlist_count") or len(playlists))
            if not page or len(playlists) >= total:
                return None, playlists
            # Never assume the service returned the requested page size.
            offset += len(page)

    def _fetch_owned_playlist(
        self, playlist_mbid: str
    ) -> tuple[ToolResult | None, dict[str, Any] | None, str]:
        """Prove ownership, then fetch current tracks immediately before mutation."""
        # Impossible IDs should not trigger an account-wide listing request.
        if not _valid_mbid(playlist_mbid):
            return fail("input", "playlist_mbid must be a UUID."), None, ""
        list_error, owned = self._list_owned_playlists()
        if list_error:
            return list_error, None, ""
        # Presence in this user's list is stronger proof than trusting creator text.
        metadata = next(
            (item for item in owned if _identifier_mbid(item.get("identifier")) == playlist_mbid),
            None,
        )
        if metadata is None:
            return (
                fail("resource", "Playlist is not owned by the configured user."),
                None,
                "",
            )
        # Fetch current tracks immediately before any position-sensitive mutation.
        status, body, _ = self.client.request(
            "GET",
            f"{LB_API}/playlist/{playlist_mbid}",
            params={"fetch_metadata": "false"},
            auth=True,
            timeout=SLOW_READ_TIMEOUT_S,
        )
        if status != 200:
            return (
                fail(
                    classify_status(status),
                    f"Could not fetch current playlist state (HTTP {status}).",
                ),
                None,
                "",
            )
        return None, body.get("playlist") or {}, metadata.get("title") or ""

    def create_playlist(
        self,
        title: str,
        recording_mbids: list[str],
        *,
        permit: WritePermit | None = None,
    ) -> ToolResult:
        """Create or safely reuse one exact Encore-marked private playlist."""
        # Keep raw arguments unchanged because the permit fingerprints these values.
        arguments = {"title": title, "recording_mbids": recording_mbids}
        # Shared policy guarantees the preview and external title are identical.
        normalized_title = normalized_playlist_title(title)
        # The Encore title is the only target known before the playlist exists.
        targets = [f"title:{normalized_title}"]
        # Authorization precedes even harmless preflight so every write path agrees.
        fingerprint, denied = self._authorize(
            "create_playlist", arguments, permit, targets
        )
        if denied:
            return denied
        assert permit is not None  # Proven by _authorize; never model-controlled.

        if not normalized_title:
            return self._close_if_terminal(permit, fail("input", "title must not be empty."))
        if len(normalized_title) > 200:
            return self._close_if_terminal(
                permit, fail("input", "prefixed playlist title exceeds 200 characters.")
            )
        # Reject accidental repeats rather than silently changing requested order.
        if len(set(recording_mbids)) != len(recording_mbids):
            return self._close_if_terminal(
                permit, fail("input", "recording_mbids must not contain duplicates.")
            )
        if any(not _valid_mbid(mbid) for mbid in recording_mbids):
            return self._close_if_terminal(
                permit, fail("input", "every recording_mbid must be a UUID.")
            )
        # A plausible model-generated UUID is not sufficient evidence for a write.
        missing = self.evidence.unverified(recording_mbids)
        if missing:
            return self._close_if_terminal(
                permit,
                fail(
                    "resource",
                    "Playlist contains MBIDs not verified in this session: "
                    + ", ".join(missing),
                ),
            )

        # Searching all owned playlists before POST makes non-idempotent create
        # safe across a retry even if an earlier response was lost.
        list_error, owned = self._list_owned_playlists()
        if list_error:
            return self._close_if_terminal(permit, list_error)
        # Exact title equality avoids mutating a merely similar human playlist.
        existing = next(
            (item for item in owned if (item.get("title") or "") == normalized_title),
            None,
        )
        if existing:
            existing_mbid = _identifier_mbid(existing.get("identifier"))
            result = ok(
                {
                    "playlist_mbid": existing_mbid,
                    "title": normalized_title,
                    "created": False,
                    "reused": True,
                }
            )
            return self._finish(
                fingerprint, [*targets, f"playlist:{existing_mbid}"], permit, result
            )

        # Create private by default with only identity-bearing JSPF fields.
        body = {
            "playlist": {
                "title": normalized_title,
                "track": [_jspf_track(mbid) for mbid in recording_mbids],
                "extension": {JSPF_EXTENSION: {"public": False}},
            }
        }
        # The client adds the token only because auth=True is explicit here.
        status, response, _ = self.client.request(
            "POST",
            f"{LB_API}/playlist/create",
            body=body,
            auth=True,
            timeout=WRITE_TIMEOUT_S,
        )
        if status != 200:
            # The playlist may exist even if the response was lost.
            return self._uncertain(
                targets,
                permit,
                fail(
                    classify_status(status),
                    f"Playlist creation failed (HTTP {status}).",
                ),
            )
        created_mbid = response.get("playlist_mbid")
        result = ok(
            {
                "playlist_mbid": created_mbid,
                "title": normalized_title,
                "created": True,
                "reused": False,
            }
        )
        return self._finish(
            fingerprint, [*targets, f"playlist:{created_mbid}"], permit, result
        )

    def add_playlist_items(
        self,
        playlist_mbid: str,
        recording_mbids: list[str],
        *,
        permit: WritePermit | None = None,
    ) -> ToolResult:
        """Append a fully verified logical track set in sequential batches."""
        # Permit binds both destination playlist and complete ordered track list.
        arguments = {
            "playlist_mbid": playlist_mbid,
            "recording_mbids": recording_mbids,
        }
        targets = [f"playlist:{playlist_mbid}"]
        fingerprint, denied = self._authorize(
            "add_playlist_items", arguments, permit, targets
        )
        if denied:
            return denied
        assert permit is not None
        # An empty write is not meaningful; duplicates complicate repair/idempotency.
        if not recording_mbids or len(set(recording_mbids)) != len(recording_mbids):
            return self._close_if_terminal(
                permit, fail("input", "recording_mbids must be non-empty and unique.")
            )
        if any(not _valid_mbid(mbid) for mbid in recording_mbids):
            return self._close_if_terminal(
                permit, fail("input", "every recording_mbid must be a UUID.")
            )
        missing = self.evidence.unverified(recording_mbids)
        if missing:
            return self._close_if_terminal(
                permit,
                fail("resource", "Unverified recording MBIDs: " + ", ".join(missing)),
            )
        # Prove ownership and existence before the first external mutation.
        state_error, _, title = self._fetch_owned_playlist(playlist_mbid)
        if state_error:
            return self._close_if_terminal(permit, state_error)

        # Count only service-confirmed tracks for exact partial-progress reporting.
        completed = 0
        for start in range(0, len(recording_mbids), ADD_BATCH_SIZE):
            # This ordinary loop is deliberately sequential—no parallel POSTs.
            batch = recording_mbids[start : start + ADD_BATCH_SIZE]
            status, _, _ = self.client.request(
                "POST",
                f"{LB_API}/playlist/{playlist_mbid}/item/add",
                body={"playlist": {"track": [_jspf_track(mbid) for mbid in batch]}},
                auth=True,
                timeout=WRITE_TIMEOUT_S,
            )
            if status != 200:
                # A partial batch sequence must not be blindly retried from the
                # beginning. Close the permit and require a newly previewed repair.
                return self._uncertain(
                    targets,
                    permit,
                    fail(
                        classify_status(status),
                        f"Add stopped after {completed} of {len(recording_mbids)} tracks "
                        f"(HTTP {status}); current state must be re-fetched before repair.",
                    ),
                )
            # Advance only after this batch returns success.
            completed += len(batch)
        return self._finish(
            fingerprint,
            targets,
            permit,
            ok({"playlist_mbid": playlist_mbid, "title": title, "added": completed}),
        )

    def remove_playlist_items(
        self,
        playlist_mbid: str,
        index: int,
        count: int,
        *,
        permit: WritePermit | None = None,
    ) -> ToolResult:
        """Remove one exact positional range after current-state preflight."""
        # Index/count are approval-bound because MBID alone cannot identify repeats.
        arguments = {"playlist_mbid": playlist_mbid, "index": index, "count": count}
        targets = [f"playlist:{playlist_mbid}"]
        fingerprint, denied = self._authorize(
            "remove_playlist_items", arguments, permit, targets
        )
        if denied:
            return denied
        assert permit is not None
        if index < 0 or count < 1:
            return self._close_if_terminal(
                permit, fail("input", "index must be >= 0 and count must be >= 1.")
            )
        state_error, playlist, title = self._fetch_owned_playlist(playlist_mbid)
        if state_error:
            return self._close_if_terminal(permit, state_error)
        # Validate against freshly fetched tracks, never the model's remembered list.
        tracks = (playlist or {}).get("track", [])
        if index + count > len(tracks):
            return self._close_if_terminal(
                permit,
                fail(
                    "resource",
                    f"Removal range exceeds current playlist length {len(tracks)}.",
                ),
            )
        # ListenBrainz removal accepts positional index/count rather than track UUID.
        status, _, _ = self.client.request(
            "POST",
            f"{LB_API}/playlist/{playlist_mbid}/item/delete",
            body={"index": index, "count": count},
            auth=True,
            timeout=WRITE_TIMEOUT_S,
        )
        if status != 200:
            # Once POST was attempted, a timeout/5xx is ambiguous: removing the
            # same range again could delete different tracks. Require a fresh
            # state check and approval even when the category is transient.
            return self._uncertain(
                targets,
                permit,
                fail(
                    classify_status(status),
                    f"Playlist removal may be partial (HTTP {status}); re-fetch state before repair.",
                ),
            )
        return self._finish(
            fingerprint,
            targets,
            permit,
            ok({"playlist_mbid": playlist_mbid, "title": title, "removed": count}),
        )

    def set_feedback(
        self,
        recording_mbid: str,
        rating: str,
        *,
        permit: WritePermit | None = None,
    ) -> ToolResult:
        """Set explicit feedback after checking the recording's current score."""
        # Human-readable rating remains in the exact approval fingerprint.
        arguments = {"recording_mbid": recording_mbid, "rating": rating}
        targets = [f"recording:{recording_mbid}"]
        fingerprint, denied = self._authorize(
            "set_feedback", arguments, permit, targets
        )
        if denied:
            return denied
        assert permit is not None
        # Translate schema enum values into ListenBrainz numeric scores.
        scores = {"love": 1, "clear": 0, "hate": -1}
        if not _valid_mbid(recording_mbid) or rating not in scores:
            return self._close_if_terminal(
                permit, fail("input", "recording_mbid or feedback rating is invalid.")
            )
        if not self.listenbrainz_user:
            return self._close_if_terminal(
                permit, fail("input", "LISTENBRAINZ_USER is not configured.")
            )
        # Reading current state makes naturally idempotent feedback a visible no-op.
        status, body, _ = self.client.request(
            "GET",
            f"{LB_API}/feedback/user/{self.listenbrainz_user}/get-feedback-for-recordings",
            params={"recording_mbids": recording_mbid},
            timeout=SLOW_READ_TIMEOUT_S,
        )
        if status != 200:
            return self._close_if_terminal(
                permit,
                fail(classify_status(status), f"Feedback preflight failed (HTTP {status})."),
            )
        # No existing row is equivalent to neutral/clear score zero.
        feedback = body.get("feedback", [])
        current = feedback[0].get("score") if feedback else 0
        # Matching current state completes safely without issuing POST.
        if current == scores[rating]:
            return self._finish(
                fingerprint,
                targets,
                permit,
                ok({"recording_mbid": recording_mbid, "rating": rating, "changed": False}),
            )
        status, _, _ = self.client.request(
            "POST",
            f"{LB_API}/feedback/recording-feedback",
            body={"recording_mbid": recording_mbid, "score": scores[rating]},
            auth=True,
            timeout=WRITE_TIMEOUT_S,
        )
        if status != 200:
            return self._uncertain(
                targets,
                permit,
                fail(classify_status(status), f"Feedback write failed (HTTP {status})."),
            )
        return self._finish(
            fingerprint,
            targets,
            permit,
            ok({"recording_mbid": recording_mbid, "rating": rating, "changed": True}),
        )

    def export_playlist(
        self,
        playlist_mbid: str,
        service: str,
        *,
        permit: WritePermit | None = None,
    ) -> ToolResult:
        """Export one proven-owned playlist exactly once per approved request."""
        # Destination service materially changes the effect and is approval-bound.
        arguments = {"playlist_mbid": playlist_mbid, "service": service}
        targets = [f"playlist:{playlist_mbid}"]
        fingerprint, denied = self._authorize(
            "export_playlist", arguments, permit, targets
        )
        if denied:
            return denied
        assert permit is not None
        if service not in {"spotify", "apple_music"}:
            return self._close_if_terminal(
                permit, fail("input", "service must be spotify or apple_music.")
            )
        # Validate current ownership/existence even though LB also authenticates.
        state_error, _, title = self._fetch_owned_playlist(playlist_mbid)
        if state_error:
            return self._close_if_terminal(permit, state_error)
        status, body, _ = self.client.request(
            "POST",
            f"{LB_API}/playlist/{playlist_mbid}/export/{service}",
            params={"is_public": "false"},
            auth=True,
            timeout=WRITE_TIMEOUT_S,
        )
        if status != 200:
            detail = _safe_service_error(body)
            detail_suffix = f" ListenBrainz reported: {detail}" if detail else ""
            # ListenBrainz reports missing links/scopes as 400 even though the
            # model cannot repair them; make those configuration failures fatal.
            lowered_detail = (detail or "").casefold()
            category = (
                "fatal"
                if status == 400
                and ("not linked" in lowered_detail or "missing scope" in lowered_detail)
                else classify_status(status)
            )
            # External services may create the playlist even when ListenBrainz
            # loses the response. Never retry export under the same approval.
            return self._uncertain(
                targets,
                permit,
                fail(
                    category,
                    f"Playlist export outcome is uncertain (HTTP {status}); "
                    f"verify the service before retrying.{detail_suffix}",
                ),
            )
        # The folder note prevents overstating what the API accomplished.
        result = ok(
            {
                "playlist_mbid": playlist_mbid,
                "title": title,
                "service": service,
                "service_result": body,
                "folder_note": (
                    "Move the exported playlist manually into the Spotify folder "
                    "named 'listenbrainz'; service folders are not API-accessible."
                    if service == "spotify"
                    else None
                ),
            }
        )
        return self._finish(fingerprint, targets, permit, result)

    def delete_playlist(
        self,
        playlist_mbid: str,
        *,
        permit: WritePermit | None = None,
    ) -> ToolResult:
        """Delete only one current, owned, explicitly marked LB playlist."""
        # The playlist UUID is the complete destructive target shown in preview.
        arguments = {"playlist_mbid": playlist_mbid}
        targets = [f"playlist:{playlist_mbid}"]
        fingerprint, denied = self._authorize(
            "delete_playlist", arguments, permit, targets
        )
        if denied:
            return denied
        assert permit is not None
        state_error, _, title = self._fetch_owned_playlist(playlist_mbid)
        if state_error:
            return self._close_if_terminal(permit, state_error)
        # Marker enforcement is deterministic containment, not prompt guidance.
        if not title.startswith(PLAYLIST_PREFIX):
            return self._close_if_terminal(
                permit,
                fail(
                    "fatal",
                    f"{POLICY_HALT_PREFIX} deletion refused: playlist lacks the "
                    "Encore [ListenBrainz] marker.",
                ),
            )
        # This fixed endpoint cannot target Spotify or Apple Music playlists.
        status, _, _ = self.client.request(
            "POST",
            f"{LB_API}/playlist/{playlist_mbid}/delete",
            auth=True,
            timeout=WRITE_TIMEOUT_S,
        )
        # Deleting also ends any remembered create for this title, so a later
        # create of the same name makes a real new playlist.
        deleted_targets = [*targets, f"title:{title}"]
        if status != 200:
            return self._uncertain(
                deleted_targets,
                permit,
                fail(classify_status(status), f"Playlist deletion failed (HTTP {status})."),
            )
        return self._finish(
            fingerprint,
            deleted_targets,
            permit,
            ok({"playlist_mbid": playlist_mbid, "title": title, "deleted": True}),
        )

    def registry(self) -> dict[str, Callable[..., ToolResult]]:
        """Bind exactly the six model-visible write schemas to guarded handlers."""
        # getattr fails loudly at startup if schema/implementation ever drift.
        return {spec.name: getattr(self, spec.name) for spec in WRITE_TOOL_SPECS}
