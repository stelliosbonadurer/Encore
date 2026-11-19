"""Read-only ListenBrainz and MusicBrainz tools.

Every method validates before I/O, calls a fixed endpoint, trims its response,
and returns ToolResult. Expected API failures never escape into the agent loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from encore.api_client import LB_API, MB_API, SLOW_READ_TIMEOUT_S, ApiClient
from encore.contracts import ToolResult, classify_status, fail, ok
from encore.playlist_policy import PLAYLIST_PREFIX
from encore.session_state import SessionEvidence
from encore.tool_catalog import READ_TOOL_SPECS


def _valid_mbid(value: str) -> bool:
    """Validate UUID syntax locally so malformed IDs never consume API quota."""
    try:
        # UUID accepts every valid canonical MBID UUID representation.
        UUID(value)
    except (ValueError, AttributeError, TypeError):
        # Model-produced nulls/non-strings are input errors, not crashes.
        return False
    # Parsing without exception is sufficient for MusicBrainz identifier syntax.
    return True


def _recording_mbid(identifier: Any) -> str | None:
    """Extract an MBID from the string-or-list JSPF identifier field.

    ListenBrainz JSPF responses are not perfectly uniform: ``identifier`` may
    be one URL or a list of URLs. Normalizing that variation here keeps the
    individual tools small and makes malformed identifiers non-fatal.
    """
    if isinstance(identifier, list):
        identifier = identifier[0] if identifier else None
    if not isinstance(identifier, str):
        return None
    # Strip a trailing slash, then keep the last path segment of the URL.
    candidate = identifier.rstrip("/").rsplit("/", 1)[-1]
    # Never let an arbitrary URL tail enter later verification/write state.
    return candidate if _valid_mbid(candidate) else None


def _api_error(service: str, status: int) -> ToolResult:
    """Translate HTTP status into the shared error taxonomy without response data."""
    return fail(classify_status(status), f"{service} request failed (HTTP {status}).")


class ReadTools:
    """Read-only tool handlers bound to one API client and one LB username.

    Dependency injection is important here: tests provide a fake client, while
    production receives the polite, allowlisted ``ApiClient``. The model never
    receives the client, token, endpoint URL, or authorization headers.
    """

    def __init__(
        self,
        client: ApiClient,
        listenbrainz_user: str | None,
        evidence: SessionEvidence | None = None,
    ) -> None:
        # The allowlisted client owns all transport and credential behavior.
        self.client = client
        # Username is a constructor dependency so models cannot choose another user.
        self.listenbrainz_user = listenbrainz_user
        # Share evidence with writes; create a local set for isolated read tests.
        self.evidence = evidence or SessionEvidence()

    def get_recent_listens(self, count: int = 25) -> ToolResult:
        """Return compact recent-listen evidence for the configured user."""
        # Validate bounds before consuming network quota.
        if not 1 <= count <= 100:
            return fail("input", f"count must be between 1 and 100; got {count}.")
        if not self.listenbrainz_user:
            return fail("input", "LISTENBRAINZ_USER is not configured.")

        # Public recent listens do not need or receive the private token.
        status, body, _ = self.client.request(
            "GET",
            f"{LB_API}/user/{self.listenbrainz_user}/listens",
            params={"count": count},
            timeout=SLOW_READ_TIMEOUT_S,
        )
        # Every non-success is translated into the four-category contract.
        if status != 200:
            return fail(
                classify_status(status),
                f"ListenBrainz listens request failed (HTTP {status}).",
            )

        # Track metadata contains many provider-specific fields. Keeping only
        # what the next reasoning step needs reduces context and privacy cost.
        listens = []
        for item in body["payload"]["listens"]:
            # ListenBrainz nests provider fields under track_metadata.
            metadata = item["track_metadata"]
            listens.append(
                {
                    "listened_at": item["listened_at"],
                    "artist": metadata.get("artist_name"),
                    "track": metadata.get("track_name"),
                    "recording_mbid": (metadata.get("mbid_mapping") or {}).get(
                        "recording_mbid"
                    ),
                }
            )
        return ok({"listens": listens})

    def get_feedback(self, rating: str, count: int = 25) -> ToolResult:
        """Return explicit love/hate signals; ListenBrainz represents them as +/-1."""
        # The model sees readable enum values; ListenBrainz uses numeric scores.
        scores = {"love": 1, "hate": -1}
        # Runtime validation mirrors the schema in case a non-model caller errs.
        if rating not in scores:
            return fail("input", "rating must be 'love' or 'hate'.")
        if not 1 <= count <= 100:
            return fail("input", f"count must be between 1 and 100; got {count}.")
        if not self.listenbrainz_user:
            return fail("input", "LISTENBRAINZ_USER is not configured.")

        # Feedback reads are public; omitting auth minimizes credential exposure.
        status, body, _ = self.client.request(
            "GET",
            f"{LB_API}/feedback/user/{self.listenbrainz_user}/get-feedback",
            params={"score": scores[rating], "count": count},
            timeout=SLOW_READ_TIMEOUT_S,
        )
        if status != 200:
            return _api_error("ListenBrainz feedback", status)
        # Translate numeric API representation back to the readable requested enum.
        feedback = [
            {
                "recording_mbid": item.get("recording_mbid"),
                "recording_msid": item.get("recording_msid"),
                "rating": rating,
                "created": item.get("created"),
            }
            for item in body.get("feedback", [])
        ]
        if not feedback:
            return fail("resource", f"No {rating} feedback was found for this user.")
        return ok({"feedback": feedback, "total_count": body.get("total_count")})

    def search_recordings(
        self, title: str, artist: str = "", limit: int = 5
    ) -> ToolResult:
        """Search recording identity without treating model text as an MBID."""
        # Empty/whitespace searches are costly and cannot produce intentional hits.
        if not title.strip():
            return fail("input", "title must not be empty.")
        if not 1 <= limit <= 25:
            return fail("input", f"limit must be between 1 and 25; got {limit}.")

        # Fielded MusicBrainz search is less ambiguous than concatenating title
        # and artist into free text, especially when both share common words.
        query = f'recording:"{title}"' + (
            f' AND artist:"{artist}"' if artist else ""
        )
        # ApiClient adds fmt=json and enforces MusicBrainz pacing.
        status, body, _ = self.client.request(
            "GET", f"{MB_API}/recording", params={"query": query, "limit": limit}
        )
        if status != 200:
            return fail(
                classify_status(status),
                f"MusicBrainz search failed (HTTP {status}).",
            )

        # Flatten only identity, credit, duration, and release evidence needed next.
        hits = [
            {
                "recording_mbid": recording["id"],
                "title": recording["title"],
                "artist": "".join(
                    credit["name"] + credit.get("joinphrase", "")
                    for credit in recording.get("artist-credit", [])
                ),
                "length_s": (recording.get("length") or 0) // 1000,
                "first_release": recording.get("first-release-date"),
            }
            for recording in body.get("recordings", [])
        ]
        # Empty successful search is a resource miss, never ambiguous success.
        if not hits:
            suffix = f" by {artist}" if artist else ""
            return fail("resource", f"No recording matches '{title}'{suffix}.")
        return ok({"recordings": hits})

    def lookup_artist(self, artist: str) -> ToolResult:
        """Resolve an artist name when needed, then fetch artist relationships."""
        if not artist.strip():
            return fail("input", "artist must be a non-empty name or MBID.")
        # Preserve a direct UUID fast path while supporting ordinary artist names.
        artist_mbid = artist.strip()
        # A direct MBID needs one request. A human-readable name needs a search
        # first because MusicBrainz relationship lookup is MBID-addressed.
        if not _valid_mbid(artist_mbid):
            status, search_body, _ = self.client.request(
                "GET",
                f"{MB_API}/artist",
                params={"query": f'artist:"{artist_mbid}"', "limit": 5},
            )
            if status != 200:
                return _api_error("MusicBrainz artist search", status)
            # Search results are relevance ordered by MusicBrainz.
            matches = search_body.get("artists", [])
            if not matches:
                return fail("resource", f"No MusicBrainz artist matches '{artist}'.")
            # MusicBrainz sorts searches by relevance. Returning the selected
            # identity and disambiguation lets the agent surface uncertainty.
            artist_mbid = matches[0]["id"]
        status, body, _ = self.client.request(
            "GET", f"{MB_API}/artist/{artist_mbid}", params={"inc": "artist-rels"}
        )
        if status != 200:
            return _api_error("MusicBrainz artist lookup", status)

        # Relationship payloads are verbose; keep human-useful identity and dates.
        relationships = []
        for relation in body.get("relations", []):
            # Defensive fallback tolerates a malformed relation without crashing.
            related = relation.get("artist") or {}
            relationships.append(
                {
                    "type": relation.get("type"),
                    "direction": relation.get("direction"),
                    "artist_mbid": related.get("id"),
                    "artist": related.get("name"),
                    "begin": relation.get("begin"),
                    "end": relation.get("end"),
                    "ended": relation.get("ended"),
                }
            )
        return ok(
            {
                "artist_mbid": body.get("id"),
                "name": body.get("name"),
                "type": body.get("type"),
                "disambiguation": body.get("disambiguation", ""),
                # Bound model context while retaining enough for band membership.
                "relationships": relationships[:50],
            }
        )

    def get_top_recordings_for_artist(
        self, artist_mbid: str, limit: int = 10
    ) -> ToolResult:
        """Return popularity-ranked recordings for one known artist identity."""
        # Popularity lookup is MBID-addressed and cannot resolve names itself.
        if not _valid_mbid(artist_mbid):
            return fail("input", "artist_mbid must be a UUID.")
        if not 1 <= limit <= 25:
            return fail("input", f"limit must be between 1 and 25; got {limit}.")
        # This anti-scraping endpoint currently requires the user's token.
        status, body, _ = self.client.request(
            "GET",
            f"{LB_API}/popularity/top-recordings-for-artist/{artist_mbid}",
            auth=True,
            timeout=SLOW_READ_TIMEOUT_S,
        )
        if status != 200:
            return _api_error("ListenBrainz popularity", status)
        # This endpoint uniquely returns a top-level list rather than a payload
        # object. Treat an unexpected shape as no usable resource, not a crash.
        source = body if isinstance(body, list) else []
        recordings = [
            {
                "recording_mbid": item.get("recording_mbid"),
                "title": item.get("recording_name"),
                "artist": item.get("artist_name"),
                "length_s": (item.get("length") or 0) // 1000,
                "listen_count": item.get("total_listen_count"),
                "listener_count": item.get("total_user_count"),
            }
            for item in source[:limit]
        ]
        if not recordings:
            return fail("resource", "No popular recordings were found for that artist.")
        return ok({"recordings": recordings})

    def verify_recordings(self, recording_mbids: list[str]) -> ToolResult:
        """Succeed only if every supplied MBID exists, making writes all-or-nothing.

        A successful result adds MBIDs to current-session verification evidence.
        Returning a resource error when even one ID is missing ensures a partial
        result can never be mistaken for authorization to write the valid IDs.
        """
        # Limit one verification call to a predictable amount of polite MB traffic.
        if not 1 <= len(recording_mbids) <= 25:
            return fail("input", "recording_mbids must contain 1 to 25 items.")
        if len(set(recording_mbids)) != len(recording_mbids):
            return fail("input", "recording_mbids must not contain duplicates.")
        # Reject the entire batch locally before verifying any individual entity.
        malformed = [mbid for mbid in recording_mbids if not _valid_mbid(mbid)]
        if malformed:
            return fail("input", f"Malformed recording MBID: {malformed[0]}.")

        # We inspect every syntactically valid ID so the model gets one complete
        # missing-ID report instead of discovering bad inputs one retry at a time.
        verified = []
        missing = []
        # MusicBrainz has no batch lookup for arbitrary recording UUIDs.
        for mbid in recording_mbids:
            status, body, _ = self.client.request(
                "GET", f"{MB_API}/recording/{mbid}", params={"inc": "artist-credits"}
            )
            # Continue through 404s to report every missing ID in this batch.
            if status == 404:
                missing.append(mbid)
                continue
            if status != 200:
                return _api_error("MusicBrainz recording verification", status)
            # Store compact evidence the next reasoning step can show to the user.
            verified.append(
                {
                    "recording_mbid": body.get("id"),
                    "title": body.get("title"),
                    "artist": "".join(
                        credit.get("name", "") + credit.get("joinphrase", "")
                        for credit in body.get("artist-credit", [])
                    ),
                }
            )
        if missing:
            return fail("resource", "Recording MBIDs not found: " + ", ".join(missing))
        # Record only after the complete batch succeeds. A partially verified
        # request must not grant write eligibility to a subset by accident.
        self.evidence.record_verified(recording_mbids)
        return ok({"verified_recordings": verified})

    def get_fresh_releases(self, days: int = 14) -> ToolResult:
        """Return a bounded window of releases linked to the user's listening."""
        # ListenBrainz documents 90 days as this endpoint's maximum window.
        if not 1 <= days <= 90:
            return fail("input", f"days must be between 1 and 90; got {days}.")
        if not self.listenbrainz_user:
            return fail("input", "LISTENBRAINZ_USER is not configured.")
        status, body, _ = self.client.request(
            "GET",
            f"{LB_API}/user/{self.listenbrainz_user}/fresh_releases",
            params={"days": days},
            timeout=SLOW_READ_TIMEOUT_S,
        )
        if status != 200:
            return _api_error("ListenBrainz fresh releases", status)
        # Normalize API-specific release field names into concise agent vocabulary.
        releases = [
            {
                "release": item.get("release_name"),
                "artist": item.get("artist_credit_name"),
                "release_date": item.get("release_date"),
                "release_mbid": item.get("release_mbid"),
                "release_group_mbid": item.get("release_group_mbid"),
                "type": item.get("release_group_primary_type"),
            }
            for item in (body.get("payload") or {}).get("releases", [])
        ]
        if not releases:
            return fail("resource", f"No fresh releases were found in the {days}-day window.")
        return ok({"releases": releases})

    def lb_radio(self, prompt: str, mode: str = "easy") -> ToolResult:
        """Generate read-only candidates and normalize their JSPF identifiers."""
        # LB Radio returns 400 for an empty prompt; catch it before a slow request.
        if not prompt.strip():
            return fail("input", "prompt must not be empty.")
        if mode not in {"easy", "medium", "hard"}:
            return fail("input", "mode must be easy, medium, or hard.")
        status, body, _ = self.client.request(
            "GET",
            f"{LB_API}/explore/lb-radio",
            params={"prompt": prompt, "mode": mode},
            auth=True,
            timeout=SLOW_READ_TIMEOUT_S,
        )
        if status != 200:
            # Invalid prompt grammar is actionable, so preserve only a bounded
            # service-owned error string rather than the complete response body.
            detail = None
            if isinstance(body, dict):
                candidate = body.get("error") or body.get("message")
                if isinstance(candidate, str) and candidate.strip():
                    detail = " ".join(candidate.split())[:300]
            suffix = f" {detail}" if detail else ""
            return fail(
                classify_status(status),
                f"ListenBrainz radio request failed (HTTP {status}).{suffix}",
            )
        if not isinstance(body, dict):
            return fail(
                "resource",
                "ListenBrainz radio returned an unexpected response shape.",
            )
        # Safely walk nested payload/JSPF objects because service errors can vary.
        payload = body.get("payload")
        if not isinstance(payload, dict):
            return fail(
                "resource",
                "ListenBrainz radio returned an unexpected response shape.",
            )
        jspf = payload.get("jspf")
        if not isinstance(jspf, dict):
            return fail(
                "resource",
                "ListenBrainz radio returned an unexpected response shape.",
            )
        playlist = jspf.get("playlist")
        if not isinstance(playlist, dict):
            return fail(
                "resource",
                "ListenBrainz radio returned an unexpected response shape.",
            )
        raw_tracks = playlist.get("track", [])
        if not isinstance(raw_tracks, list):
            return fail(
                "resource",
                "ListenBrainz radio returned an unexpected response shape.",
            )
        # A candidate without a valid recording MBID cannot pass the later
        # verification gate, so remove it before it reaches model context.
        tracks = [
            {
                "recording_mbid": _recording_mbid(item.get("identifier")),
                "title": item.get("title"),
                "artist": item.get("creator"),
                "duration_ms": item.get("duration"),
            }
            for item in raw_tracks
            if isinstance(item, dict)
        ]
        # Discard candidates that cannot be proved as real MusicBrainz recordings.
        tracks = [track for track in tracks if track["recording_mbid"]]
        if not tracks:
            return fail("resource", "LB Radio returned no usable recording candidates.")
        # Twenty candidates is ample for the bounded playlist tools and avoids a
        # radio response dominating working memory and later model requests.
        return ok(
            {
                "title": playlist.get("title"),
                "tracks": tracks[:20],
                # Code-owned workflow guidance helps local models advance from
                # discovery instead of repeatedly calling the same radio query.
                "next_action": (
                    "Select the requested number of recording_mbid values, call "
                    "verify_recordings once, and do not repeat this lb_radio call."
                ),
            }
        )

    def find_playlists(self, query: str = "", count: int = 25) -> ToolResult:
        """List the user's own playlists, including private ones, with local filtering."""
        if not 1 <= count <= 100:
            return fail("input", f"count must be between 1 and 100; got {count}.")
        if not self.listenbrainz_user:
            return fail("input", "LISTENBRAINZ_USER is not configured.")
        # Auth includes private playlists; public-only results are unsafe preflight.
        status, body, _ = self.client.request(
            "GET",
            f"{LB_API}/user/{self.listenbrainz_user}/playlists",
            params={"count": count},
            auth=True,
            timeout=SLOW_READ_TIMEOUT_S,
        )
        if status != 200:
            return _api_error("ListenBrainz playlists", status)
        # The user's playlist endpoint returns the authoritative owned set.
        # Filtering locally avoids the public fuzzy-search endpoint, which could
        # mix other users' playlists into a later mutation decision.
        needle = query.strip().casefold()
        # Produce a new trimmed list rather than exposing raw JSPF extensions.
        playlists = []
        for item in body.get("playlists", []):
            playlist = item.get("playlist") or {}
            title = playlist.get("title") or ""
            # Casefold gives Unicode-aware case-insensitive title matching.
            if needle and needle not in title.casefold():
                continue
            playlists.append(
                {
                    "playlist_mbid": _recording_mbid(playlist.get("identifier")),
                    "title": title,
                    "description": playlist.get("annotation") or "",
                    "creator": playlist.get("creator"),
                    # ListenBrainz records the owning account as creator even when
                    # Encore performed the approved creation. The code-owned title
                    # marker identifies playlists managed through this application.
                    "encore_managed": title.startswith(PLAYLIST_PREFIX),
                    "date": playlist.get("date"),
                }
            )
        if not playlists:
            detail = f" matching '{query}'" if query.strip() else ""
            return fail("resource", f"No playlists{detail} were found.")
        return ok(
            {
                "playlists": playlists,
                "total_count": body.get("playlist_count"),
                "source_service": "listenbrainz",
                "external_service_library_searched": False,
            }
        )

    def registry(self) -> dict[str, Callable[..., ToolResult]]:
        """Bind catalog names to methods and fail loudly if either side drifts.

        ``getattr`` intentionally raises during composition if a catalog entry
        lacks a handler. That is safer than letting the error appear halfway
        through an agent session.
        """
        # getattr intentionally fails during startup if a schema has no method.
        return {spec.name: getattr(self, spec.name) for spec in READ_TOOL_SPECS}
