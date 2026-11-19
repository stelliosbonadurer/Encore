"""Tool metadata shared by the router and ReAct loop.

Keeping one catalog prevents three copies of the tool set (schemas, router
domains, and handlers) from quietly drifting apart. Only tools with real
handlers are placed in the catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Domain = Literal["listening", "catalog", "discovery", "library", "memory", "control"]


@dataclass(frozen=True)
class ToolSpec:
    """The model-visible contract for one implemented tool.

    ``name`` must match a method on the tool container. ``domain`` controls
    routing. ``description`` and ``parameters`` are sent to the main model, so
    neither may contain credentials or runtime-only safety state.
    """

    # Stable evaluation name and Python handler name.
    name: str
    # Coarse router bucket; control is deliberately not a routable domain.
    domain: Domain
    # Model instructions include selection and non-selection guidance.
    description: str
    # Closed JSON Schema validates all model-produced arguments.
    parameters: dict[str, Any]
    # Approval is execution metadata. It is intentionally absent from
    # ``openai_schema`` so a model cannot authorize its own side effect.
    requires_approval: bool = False

    def openai_schema(self) -> dict[str, Any]:
        """Wrap the neutral ToolSpec in OpenAI-compatible function format."""
        # This method contains model-visible data only; approval metadata stays out.
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _object(properties: dict[str, Any], *required: str) -> dict[str, Any]:
    """Build the common closed-object schema used by every Encore tool.

    ``additionalProperties=False`` is deliberate: a model may invent plausible
    arguments, and silently accepting them makes traces misleading and future
    compatibility harder to reason about.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


# This tuple is intentionally limited to tools with working handlers. Adding a
# schema here automatically makes its name available to the router, so placeholder
# schemas would allow the model to request actions the executor cannot perform.
READ_TOOL_SPECS = (
    # Listening: observations about this user's behavior.
    # Recent listens is broad behavioral evidence. Count is bounded to control
    # response/context size and protect the shared nonprofit service.
    ToolSpec(
        "get_recent_listens",
        "listening",
        "Read the configured user's latest ListenBrainz listens. Pass a count "
        "from 1 to 100. Prefer this for what the user played recently. Do not "
        "use this when asking what the user explicitly loved or hated; use "
        "get_feedback instead.",
        _object({"count": {"type": "integer", "minimum": 1, "maximum": 100}}),
    ),
    # Feedback is a stronger explicit taste signal than mere playback. The enum
    # prevents the model from inventing unsupported sentiment categories.
    ToolSpec(
        "get_feedback",
        "listening",
        "Read recordings the configured user loved or hated. Pass rating and a "
        "count from 1 to 100. Prefer this for explicit taste signals. Do not use "
        "this to infer taste from recent plays; use get_recent_listens instead.",
        _object(
            {
                "rating": {"type": "string", "enum": ["love", "hate"]},
                "count": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "rating",
        ),
    ),
    # Catalog: identity and factual metadata from MusicBrainz/ListenBrainz.
    # Search is the recovery path when a recording identity is unknown. Artist
    # remains optional because title-only discovery is a valid user request.
    ToolSpec(
        "search_recordings",
        "catalog",
        "Search MusicBrainz for real recordings by title and optional artist. "
        "Pass a non-empty title, optional artist, and limit from 1 to 25. Prefer "
        "this when no recording MBID is known. Do not use this to verify a known "
        "list of MBIDs before a write; use verify_recordings instead. Do not pass "
        "a genre, mood, or activity as the title; use lb_radio for those themes.",
        _object(
            {
                "title": {"type": "string", "minLength": 1},
                "artist": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25},
            },
            "title",
        ),
    ),
    # Artist lookup accepts a name for usability but ultimately resolves an MBID
    # before requesting relationship data from MusicBrainz.
    ToolSpec(
        "lookup_artist",
        "catalog",
        "Look up one MusicBrainz artist and its artist relationships. Pass an "
        "artist name or exact artist MBID. Prefer this for members and "
        "collaborators. Do not use this to find recordings by title; use "
        "search_recordings instead.",
        _object({"artist": {"type": "string", "minLength": 1}}, "artist"),
    ),
    # Popularity requires an existing artist identity. Limit bounds the bare-list
    # endpoint before results enter model context.
    ToolSpec(
        "get_top_recordings_for_artist",
        "catalog",
        "Read an artist's most-listened recordings from ListenBrainz. Pass an "
        "artist MBID and limit from 1 to 25. Prefer this for popularity-ranked "
        "tracks. Do not use this for relationship questions; use lookup_artist.",
        _object(
            {
                "artist_mbid": {"type": "string", "format": "uuid"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25},
            },
            "artist_mbid",
        ),
    ),
    # Verification deliberately accepts a batch but caps it at 25 polite serial
    # MusicBrainz lookups. Unique items simplify session-evidence semantics.
    ToolSpec(
        "verify_recordings",
        "catalog",
        "Confirm that one to 25 recording MBIDs exist in MusicBrainz. Pass the "
        "complete unique MBID list intended for a playlist write. Prefer this "
        "immediately before playlist mutation. Do not use this to discover songs "
        "when their MBIDs are unknown; use search_recordings.",
        _object(
            {
                "recording_mbids": {
                    "type": "array",
                    "items": {"type": "string", "format": "uuid"},
                    "minItems": 1,
                    "maxItems": 25,
                    "uniqueItems": True,
                }
            },
            "recording_mbids",
        ),
    ),
    # Discovery: candidate generation only; neither tool mutates an account.
    # Fresh-release days follows the service's documented 90-day maximum.
    ToolSpec(
        "get_fresh_releases",
        "discovery",
        "Read recent or upcoming releases associated with the configured user's "
        "listening, for 1 to 90 days. Prefer this for 'what is new?' requests. Do "
        "not use this to generate a mood or similarity playlist; use lb_radio.",
        _object({"days": {"type": "integer", "minimum": 1, "maximum": 90}}),
    ),
    # LB Radio mode is a service-defined enum. Its output is still only candidate
    # data and cannot bypass verify_recordings before playlist writes.
    ToolSpec(
        "lb_radio",
        "discovery",
        "Generate ListenBrainz radio candidates from LB Radio expression syntax, "
        "such as tag:(bluegrass) or artist:(MusicBrainz artist MBID). Pass a "
        "non-empty expression and mode: easy, medium, or hard. Prefer this for guided "
        "discovery. Do not use this to create or alter an account playlist; it is "
        "read-only and its candidates must be verified before any write.",
        _object(
            {
                "prompt": {"type": "string", "minLength": 1},
                "mode": {"type": "string", "enum": ["easy", "medium", "hard"]},
            },
            "prompt",
            "mode",
        ),
    ),
    # Library read: this discovers mutation targets but cannot change them.
    # Query is applied only within the configured user's authoritative playlist
    # list so similarly named public playlists never become accidental targets.
    ToolSpec(
        "find_playlists",
        "library",
        "Read the configured user's own ListenBrainz playlist metadata, including "
        "private playlists, and optionally filter titles by a query. Results mark "
        "titles beginning [ListenBrainz] as encore_managed; creator is the owning "
        "account, not evidence of which app created it. This tool cannot inspect "
        "the user's Spotify library or prove an earlier export. Pass count "
        "from 1 to 100. Prefer this before creating, editing, exporting, or "
        "deleting. Do not use this to search other users' public playlists.",
        _object(
            {
                "query": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 100},
            }
        ),
    ),
)


# Account writes are model-visible but their handlers reject calls unless the
# executor supplies an opaque, argument-bound permit after human confirmation.
WRITE_TOOL_SPECS = (
    # Creation defaults private and applies the logical-folder prefix in code.
    # No maxItems is imposed because volume is warned/confirmed, not hard-capped.
    ToolSpec(
        "create_playlist",
        "library",
        "Create one private ListenBrainz playlist whose title begins with "
        "[ListenBrainz]. Pass the requested title and verified recording MBIDs. "
        "This changes the user's account and requires approval; creation is not "
        "naturally idempotent, so Encore reuses an exact marked-title match. "
        "Prefer this after find_playlists and verify_recordings. Do not use this "
        "when an existing target playlist should receive tracks.",
        _object(
            {
                "title": {"type": "string", "minLength": 1, "maxLength": 200},
                "recording_mbids": {
                    "type": "array",
                    "items": {"type": "string", "format": "uuid"},
                    "uniqueItems": True,
                },
            },
            "title",
            "recording_mbids",
        ),
        True,
    ),
    # Addition is a logical operation over a complete list; the handler performs
    # service-sized sequential batches only after one approval of the total.
    ToolSpec(
        "add_playlist_items",
        "library",
        "Append verified recordings to one owned ListenBrainz playlist. Pass the "
        "playlist MBID and complete recording-MBID list. This changes the user's "
        "account and requires approval; Encore records completed operations to "
        "make retries idempotent. Prefer this for an existing playlist. Do not "
        "use this before find_playlists and verify_recordings.",
        _object(
            {
                "playlist_mbid": {"type": "string", "format": "uuid"},
                "recording_mbids": {
                    "type": "array",
                    "items": {"type": "string", "format": "uuid"},
                    "minItems": 1,
                    "uniqueItems": True,
                },
            },
            "playlist_mbid",
            "recording_mbids",
        ),
        True,
    ),
    # Removal is positional because ListenBrainz requires index/count. Both values
    # are approval-bound and checked against freshly fetched playlist state.
    ToolSpec(
        "remove_playlist_items",
        "library",
        "Remove a contiguous range from one owned ListenBrainz playlist by "
        "zero-based index and count. This changes the user's account and "
        "requires approval; Encore checks current playlist state and records a "
        "completed operation for idempotent retry. Prefer this after locating "
        "the exact positions. Do not use this to delete the whole playlist.",
        _object(
            {
                "playlist_mbid": {"type": "string", "format": "uuid"},
                "index": {"type": "integer", "minimum": 0},
                "count": {"type": "integer", "minimum": 1},
            },
            "playlist_mbid",
            "index",
            "count",
        ),
        True,
    ),
    # Feedback maps three readable enum values to API scores 1, 0, and -1. It is
    # naturally idempotent but still changes account state and needs approval.
    ToolSpec(
        "set_feedback",
        "library",
        "Set one recording's ListenBrainz feedback to love, clear, or hate. Pass "
        "a recording MBID and rating. This changes the user's account and "
        "requires approval, but setting the same value is naturally idempotent. "
        "Prefer this for explicit user feedback. Do not use this merely because "
        "a recommendation was played or skipped.",
        _object(
            {
                "recording_mbid": {"type": "string", "format": "uuid"},
                "rating": {"type": "string", "enum": ["love", "clear", "hate"]},
            },
            "recording_mbid",
            "rating",
        ),
        True,
    ),
    # Export's enum intentionally excludes unsupported services. The operation can
    # create an external playlist and therefore always crosses the approval gate.
    ToolSpec(
        "export_playlist",
        "library",
        "Export one owned ListenBrainz playlist to Spotify or Apple Music. Pass "
        "the playlist MBID and service. This creates an external-service playlist "
        "and always requires approval; Encore prevents duplicate retries. Prefer "
        "this only after the ListenBrainz playlist is final. Do not use this to "
        "claim the playlist was placed in a service folder; folder movement is "
        "manual.",
        _object(
            {
                "playlist_mbid": {"type": "string", "format": "uuid"},
                "service": {"type": "string", "enum": ["spotify", "apple_music"]},
            },
            "playlist_mbid",
            "service",
        ),
        True,
    ),
    # Deletion exposes only a playlist UUID. Ownership and the marker cannot be
    # asserted by the model; the handler re-fetches and proves both itself.
    ToolSpec(
        "delete_playlist",
        "library",
        "Delete one owned, Encore-marked ListenBrainz playlist. Pass its playlist "
        "MBID. This permanently changes the user's account and always requires "
        "approval; retries are recorded. Prefer this only after find_playlists "
        "identifies the exact target. Do not use this for an unmarked playlist "
        "or a Spotify/Apple Music playlist.",
        _object(
            {"playlist_mbid": {"type": "string", "format": "uuid"}},
            "playlist_mbid",
        ),
        True,
    ),
)


# Memory tools operate on Encore's injected stores, never on the ListenBrainz
# account. Their backends can evolve without changing the model-facing schema.
MEMORY_TOOL_SPECS = (
    # Episodic search answers “what happened before?” and is always filtered by
    # ENCORE_USER_ID inside the backend rather than by model-provided identity.
    ToolSpec(
        "search_past_sessions",
        "memory",
        "Search up to 10 of this user's episodic session summaries. Pass only the "
        "search query; the application owns the bounded result count. Prefer this "
        "for what was requested, decided, accepted, or "
        "rejected in earlier sessions. Do not use this for service documentation "
        "or general music terminology; use search_music_knowledge.",
        _object({"query": {"type": "string", "minLength": 1}}, "query"),
    ),
    # Semantic search answers documentation/terminology questions and returns
    # source-bearing chunks instead of personal history.
    ToolSpec(
        "search_music_knowledge",
        "memory",
        "Search the indexed ListenBrainz/MusicBrainz documentation and music "
        "glossary. Pass a query and limit from 1 to 5. Prefer this for how a "
        "service works or what a music term means. Do not use this to recall the "
        "user's prior requests; use search_past_sessions.",
        _object(
            {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "query",
        ),
    ),
    # Preference notes are bounded local-memory writes. Evidence is optional but
    # stored beside the claim so later memory quality checks can inspect origin.
    ToolSpec(
        "note_preference",
        "memory",
        "Record one explicit user preference in this session's hot state. Pass a "
        "concise preference and optional evidence. Prefer this when the user "
        "directly states or corrects a durable taste. Do not use this to infer a "
        "preference from one listen, skip, or model-generated suggestion.",
        _object(
            {
                "preference": {"type": "string", "minLength": 1, "maxLength": 500},
                "evidence": {"type": "string", "maxLength": 500},
            },
            "preference",
        ),
    ),
)


IMPLEMENTED_TOOL_SPECS = (
    # Tuple expansion preserves domain order for stable router enums and traces.
    *READ_TOOL_SPECS,
    *WRITE_TOOL_SPECS,
    *MEMORY_TOOL_SPECS,
)


# final_answer is a control action, not a routed information domain. The main
# model always receives it so even a `none` route can terminate normally.
FINAL_ANSWER_SPEC = ToolSpec(
    # The loop will verify/override claimed status when execution history conflicts.
    "final_answer",
    "control",  # Control tools are always available and never router-selected.
    "Finish the current request with a status, answer, confidence, and sources. "
    "Prefer this only when no more tool evidence is needed. Do not use this to "
    "hide an unattempted required action or unresolved transient failure.",
    _object(
        {
            "status": {
                "type": "string",
                "enum": [
                    "answered",
                    "answered_low_confidence",
                    "stuck",
                    "budget_exceeded",
                    "policy_halt",
                ],
            },
            "answer": {"type": "string", "minLength": 1},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
        "status",
        "answer",
        "confidence",
        "sources",
    ),
)


def specs_by_name() -> dict[str, ToolSpec]:
    """Return a fresh lookup map so callers cannot mutate shared catalog state."""
    # Rebuilding this tiny map avoids sharing mutable dictionary state.
    return {spec.name: spec for spec in IMPLEMENTED_TOOL_SPECS}
