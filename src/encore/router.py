"""Cheap-model router with enum-constrained domains.

The router narrows the main model's choices; it never executes a tool or chooses
the next action. Invalid output safely exposes every implemented tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from encore.model_client import ModelClient
from encore.structured_output import parse_forced_arguments
from encore.tool_catalog import FINAL_ANSWER_SPEC, IMPLEMENTED_TOOL_SPECS, ToolSpec

# This order is stable so traces and fallback tool ordering remain comparable.
ROUTABLE_DOMAINS = ("listening", "catalog", "discovery", "library", "memory")
# `none` is a routing outcome rather than a tool domain. It tells the main model
# that it can answer conversationally with only the always-present final_answer.
ROUTER_DOMAINS = (*ROUTABLE_DOMAINS, "none")


def _domain_tools(specs: tuple[ToolSpec, ...]) -> dict[str, tuple[str, ...]]:
    """Derive domain membership from the catalog instead of duplicating it."""
    # Comprehension derives each tuple from the same ToolSpec domain field used
    # everywhere else; no second hand-maintained mapping exists.
    return {
        domain: tuple(spec.name for spec in specs if spec.domain == domain)
        for domain in ROUTABLE_DOMAINS
    }


# Materialize once because the catalog is immutable for the process lifetime.
DOMAIN_TOOLS = _domain_tools(IMPLEMENTED_TOOL_SPECS)
# Stable catalog order is used by route properties and trace output.
IMPLEMENTED_NAMES = tuple(spec.name for spec in IMPLEMENTED_TOOL_SPECS)

ROUTER_PROMPT = """You are Encore's request router. Call classify_intent once.
Select every relevant domain. Python will expose all tools in those domains; the
main agent, not you, decides which tool to call and in what order.

The user input may contain an explicitly delimited recent conversation followed
by CURRENT_REQUEST. Treat recent conversation only as context. Classify the
complete current intent, including an unfinished workflow or a reply to a prior
clarifying question. CURRENT_REQUEST is always authoritative.

- listening: the user's listens and explicit feedback
- catalog: recordings, artists, relationships, popularity, and MBID verification
- discovery: fresh releases and ListenBrainz radio
- library: finding or changing playlists, feedback writes, and export
- memory: prior sessions or music-service documentation
- none: conversation needing no tool; choose no tools and no other domain

Examples that are easy to misroute:
- "What is a deep cut?" or an exact API endpoint question: memory.
- "What preferences have I told you before?": memory.
- "Recommend something upbeat for later" without enough musical context: none;
  the main agent should ask one focused clarification before discovery.

Playlist creation and editing normally require catalog, discovery, and library.
Select all three so the agent can discover recordings, verify MBIDs, locate the
playlist, perform the requested mutation, and export it.

To locate earlier Spotify or Apple Music exports made through Encore, select
both library and memory: playlist lookup identifies marked ListenBrainz staging
records, while session memory may contain successful export URLs. Neither domain
directly searches the external service's playlist library.

Do not answer the request.
"""


def build_classifier_tool() -> dict[str, Any]:
    """Build the forced enum-constrained domain classifier."""
    # OpenAI function-tool format wraps our ordinary JSON Schema object.
    return {
        "type": "function",
        "function": {
            "name": "classify_intent",
            "description": "Select every domain relevant to the current request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domains": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(ROUTER_DOMAINS)},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                },
                "required": ["domains"],
                "additionalProperties": False,
            },
        },
    }


# The production classifier covers every currently implemented action handler.
CLASSIFIER_TOOL = build_classifier_tool()
# Forcing one named call prevents a prose answer from masquerading as routing.
FORCE_CLASSIFIER = {"type": "function", "function": {"name": "classify_intent"}}


@dataclass(frozen=True)
class RouteDecision:
    """Validated routing output consumed by the ReAct loop."""

    # Domains explain coarse intent in traces and support rubric evaluation.
    domains: tuple[str, ...]
    # Traces can separate model routing quality from fallback recovery.
    used_fallback: bool = False

    @property
    def tools(self) -> tuple[str, ...]:
        """Derive visible tools from selected domains and the one catalog."""
        if self.domains == ("none",):
            return ()
        selected = set(self.domains)
        return tuple(
            spec.name for spec in IMPLEMENTED_TOOL_SPECS if spec.domain in selected
        )

    def model_tool_schemas(self) -> list[dict[str, Any]]:
        """Return exactly the routed schemas plus final_answer.

        Reducing the model's visible choices improves selection reliability.
        ``final_answer`` is appended in code so the router cannot accidentally
        prevent the main model from terminating.
        """
        # Build a local immutable-by-convention lookup from authoritative specs.
        by_name = {spec.name: spec for spec in IMPLEMENTED_TOOL_SPECS}
        # Validation already proved every routed name exists in this mapping.
        selected = [by_name[name].openai_schema() for name in self.tools]
        # Control termination remains available even when selected is empty.
        return [*selected, FINAL_ANSWER_SPEC.openai_schema()]


def _validate_semantics(domains: tuple[str, ...]) -> None:
    """Enforce the meaning of the special tool-free domain."""
    # `none` means ordinary conversation and is mutually exclusive by meaning.
    if "none" in domains:
        if domains != ("none",):
            raise ValueError("none cannot be combined with another domain")


def parse_router_message(message: dict[str, Any]) -> RouteDecision:
    """Decode, schema-validate, and semantically validate one forced tool call.

    Validation happens before any selected name is used for handler lookup. A
    malformed model response is therefore data for fallback, never executable
    control flow.
    """
    arguments = parse_forced_arguments(message, CLASSIFIER_TOOL)
    # Tuples make the validated decision immutable and hash/comparison friendly.
    domains = tuple(arguments["domains"])
    _validate_semantics(domains)
    return RouteDecision(domains)


def _is_ambiguous_continuation(request: str) -> bool:
    """Recognize short follow-ups whose intent lives in conversation history."""
    # Context-aware routing input ends with this wrapper; inspect only its current
    # request rather than comparing the entire bounded conversation.
    if "<CURRENT_REQUEST>" in request:
        request = request.split("<CURRENT_REQUEST>", 1)[1].split(
            "</CURRENT_REQUEST>", 1
        )[0]
    normalized = " ".join(request.casefold().split()).strip(" .!?")
    return normalized in {
        "again",
        "do it",
        "go ahead",
        "retry",
        "retry it",
        "try again",
        "try it again",
        "yes",
    }


def _current_request_text(request: str) -> str:
    """Extract only the authoritative request from context-aware router input."""
    if "<CURRENT_REQUEST>" in request:
        return request.split("<CURRENT_REQUEST>", 1)[1].split(
            "</CURRENT_REQUEST>", 1
        )[0]
    return request


def _requires_memory_domain(request: str) -> bool:
    """Recover explicit documentation, terminology, and prior-memory intents."""
    normalized = " ".join(_current_request_text(request).casefold().split())
    return any(
        marker in normalized
        for marker in (
            "music terminology",
            "what is a deep cut",
            "what does deep cut mean",
            "listenbrainz endpoint",
            "musicbrainz endpoint",
            "/1/",
            "preferences have i told you",
            "told you about before",
        )
    )


def _needs_recommendation_clarification(request: str) -> bool:
    """Keep highly underspecified recommendations conversational and read-only."""
    normalized = " ".join(_current_request_text(request).casefold().split())
    starts_recommendation = normalized.startswith(("recommend ", "suggest "))
    indefinite = "something" in normalized
    grounding = (
        "artist",
        "song",
        "track",
        "album",
        "genre",
        "similar to",
        "based on",
        "from my",
    )
    return starts_recommendation and indefinite and not any(
        marker in normalized for marker in grounding
    )


def _with_memory_domain(decision: RouteDecision) -> RouteDecision:
    """Add memory in stable catalog order without broadening unrelated domains."""
    if "memory" in decision.domains:
        return decision
    selected = set(() if decision.domains == ("none",) else decision.domains)
    selected.add("memory")
    domains = tuple(domain for domain in ROUTABLE_DOMAINS if domain in selected)
    return RouteDecision(domains, used_fallback=decision.used_fallback)


def fallback_decision() -> RouteDecision:
    """Expose all implemented tools so a router failure cannot block recovery.

    This broadens model choice but does not execute anything. Write tools still
    cross the executor's approval boundary regardless of routing.
    """
    # Exclude empty domains so fallback traces describe actual capability.
    active_domains = tuple(domain for domain in ROUTABLE_DOMAINS if DOMAIN_TOOLS[domain])
    return RouteDecision(active_domains, used_fallback=True)


def route_request(request: str, client: ModelClient) -> RouteDecision:
    """Ask the cheap tier for one classification and fall back on any bad shape."""
    # Routing always uses the configured cheap tier to control cost and latency.
    response = client.chat(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": request},
        ],
        tier="cheap",
        tools=[CLASSIFIER_TOOL],
        tool_choice=FORCE_CLASSIFIER,
    )
    # Model output is untrusted even though tool_choice forces the function name:
    # providers can omit calls, emit invalid JSON, or violate the schema.
    try:
        # The happy path returns only fully validated immutable data.
        decision = parse_router_message(response.message)
        # The router sees only the current objective, while the main agent retains
        # bounded conversation. A short continuation cannot safely be classified
        # as tool-free, so expose the complete approved catalog for recovery.
        if decision.domains == ("none",) and _is_ambiguous_continuation(request):
            return fallback_decision()
        # These narrow deterministic repairs address measured local-model router
        # failures without replacing the enum-constrained model router generally.
        if _needs_recommendation_clarification(request):
            return RouteDecision(("none",), used_fallback=decision.used_fallback)
        if _requires_memory_domain(request):
            return _with_memory_domain(decision)
        return decision
    except Exception:
        # Broadening candidates preserves availability; it executes nothing.
        return fallback_decision()
