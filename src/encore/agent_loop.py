"""Bounded ReAct-style orchestration for one Encore user request.

The model may propose one catalogued action per step, but ordinary code owns
routing, schema validation, execution, approvals, the step budget, and final
termination status.  A cheap-tier ``final_answer`` call is only a stop signal;
the expensive tier always produces the user-facing structured answer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from encore.contracts import POLICY_HALT_PREFIX, ToolResult, fail
from encore.model_client import ModelClient
from encore.router import RouteDecision, fallback_decision, route_request
from encore.structured_output import parse_forced_arguments
from encore.tool_catalog import FINAL_ANSWER_SPEC
from encore.tool_executor import ExecutionReport, ToolExecutor
from encore.tracing import TraceWriteError, TraceWriter, TracingModelClient
from encore.working_memory import WorkingMemoryService

# Twelve model action steps is the assignment's deterministic runaway guard.
MAX_STEPS = 12
# Local models occasionally need several schema-correction turns, especially in
# multi-track workflows. Six still stops a genuine format loop well before the
# twelve-step request bound without terminating after a brief provider wobble.
MAX_CONSECUTIVE_INVALID_OUTPUTS = 6
# Different arguments may be legitimate recovery, but a fourth failed execution
# of the same tool in one request is a no-progress loop in Encore's workflows.
MAX_FAILED_EXECUTIONS_PER_TOOL = 3
# LB Radio returns a candidate batch in one call. Changing its mode after a
# successful response does not add required evidence and was a common loop.
SINGLE_SUCCESS_TOOLS = frozenset({"lb_radio"})
# These are the only externally reportable ways one request may terminate.
TerminationStatus = Literal[
    "answered",
    "answered_low_confidence",
    "stuck",
    "budget_exceeded",
    "policy_halt",
]
# The confidence vocabulary matches the closed final_answer JSON Schema.
Confidence = Literal["high", "medium", "low"]

# Force the expensive tier to return the one control function, not free text.
FORCE_FINAL_ANSWER = {
    "type": "function",
    "function": {"name": "final_answer"},
}

# LB Radio returns candidate identifiers rather than write-ready evidence. After
# one successful batch, constrain the next model turn to the required verifier.
# The model still chooses the requested subset; Python does not select tracks.
FORCE_VERIFY_RECORDINGS = {
    "type": "function",
    "function": {"name": "verify_recordings"},
}

FORCE_SEARCH_PAST_SESSIONS = {
    "type": "function",
    "function": {"name": "search_past_sessions"},
}


@dataclass(frozen=True)
class ToolObservation:
    """One executed proposal and the deterministic metadata around it."""

    # Name makes the observation understandable without replaying model history.
    tool_name: str
    # Result is the exact success/error object shown to the reasoning model.
    result: ToolResult
    # Attempts and delays preserve retry evidence for the future JSONL trace.
    attempts: int
    retry_delays_s: tuple[float, ...]
    # Approval flags prove whether an account-write boundary was crossed.
    approval_requested: bool
    approved: bool


@dataclass(frozen=True)
class AgentResult:
    """Complete, trace-ready outcome of one bounded agent run."""

    # Code reconciles this status with observed control flow before returning it.
    status: TerminationStatus
    # Answer is always non-empty, including deterministic fallback outcomes.
    answer: str
    # Confidence and sources come from the validated final control call.
    confidence: Confidence
    sources: tuple[str, ...]
    # Route explains which action schemas were made visible to the main model.
    route: RouteDecision
    # Steps count cheap-tier action decisions, including malformed decisions.
    steps: int
    # Model calls include routing, action decisions, and expensive finalization.
    model_calls: int
    # Tool calls count only proposals that reached the deterministic executor.
    tool_calls: int
    # Immutable observations support later tracing and evaluation without logs.
    observations: tuple[ToolObservation, ...]
    # Bounded conversation state can be carried into the next session request.
    conversation: tuple[dict[str, Any], ...]
    # Safe operator-facing detail explains fallback outcomes without exposing
    # prompts, provider response bodies, URLs, or credentials.
    diagnostic: str | None = None


# Injection keeps router behavior independently testable without network calls.
RouteFunction = Callable[[str, ModelClient], RouteDecision]


class AgentLoop:
    """Route once, execute at most one proposed action per step, then finalize."""

    def __init__(
        self,
        client: ModelClient,
        executor: ToolExecutor,
        system_prompt: str,
        *,
        route: RouteFunction = route_request,
        max_steps: int = MAX_STEPS,
        trace: TraceWriter | None = None,
        working_memory: WorkingMemoryService | None = None,
        verbosity: int = 0,
        progress_output: Callable[[str], None] = print,
    ) -> None:
        # Model access remains behind one injected provider/tier boundary.
        # When tracing is enabled, every router/main/final model call is covered.
        self._client = TracingModelClient(client, trace) if trace is not None else client
        # Executor is the only path from untrusted model output to handlers.
        self._executor = executor
        # Strip accidental edge whitespace while preserving prompt formatting.
        self._system_prompt = system_prompt.strip()
        # Router injection makes routing failure and narrowing deterministic in tests.
        self._route = route
        # The same writer receives route, tool, approval, and final control events.
        self._trace = trace
        # Optional working memory owns compression, archival, and context ordering.
        self._working_memory = working_memory
        # Console visibility is separate from durable, privacy-minimized traces.
        if verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be 0, 1, or 2")
        self._verbosity = verbosity
        self._progress_output = progress_output
        # A positive bound guarantees the loop can never spin indefinitely.
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._max_steps = max_steps

    @property
    def max_steps(self) -> int:
        """Expose the immutable segment bound for user-facing continuation text."""
        return self._max_steps

    def _show(self, minimum_level: int, label: str, payload: Any = None) -> None:
        """Print one observable loop event without exposing prompts or credentials."""
        if self._verbosity < minimum_level:
            return
        if payload is None:
            self._progress_output(label)
            return
        # JSON encoding escapes terminal control characters carried by untrusted data.
        try:
            rendered = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            rendered = json.dumps(
                {"unprintable_type": type(payload).__name__},
                indent=2,
                sort_keys=True,
            )
        self._progress_output(f"{label}\n{rendered}")

    def _write_trace(self, event: str, **fields: Any) -> None:
        """Write one event when tracing is active; failures intentionally propagate."""
        # A missing writer is the explicit development/test opt-out.
        if self._trace is not None:
            self._trace.write(event, **fields)

    @staticmethod
    def _trace_arguments(arguments: Any) -> dict[str, Any]:
        """Return parsed object arguments or a non-sensitive invalid marker."""
        try:
            # Valid provider calls normally carry a JSON string.
            decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
            # Round-trip detaches dictionaries before the trace sanitizer sees them.
            if isinstance(decoded, dict):
                return json.loads(json.dumps(decoded, allow_nan=False))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        # Never copy malformed raw text because it could contain arbitrary secrets.
        return {"_invalid_arguments": True}

    def _messages(
        self,
        history: list[dict[str, Any]],
        *,
        objective: str | None = None,
        retrieved_memories: tuple[dict[str, Any], ...] = (),
        final_instruction: str | None = None,
    ) -> list[dict[str, Any]]:
        """Prepend one fresh safety prompt without storing duplicates in history."""
        # Final requirements join the same single system message when applicable.
        content = self._system_prompt
        if final_instruction:
            content = f"{content}\n\n{final_instruction}"
        # Working-memory context follows the accepted front-loading order.
        if self._working_memory is not None and objective is not None:
            conversation = self._working_memory.frontload(
                objective,
                history,
                retrieved_memories=retrieved_memories,
            )
        else:
            # Legacy/test composition keeps the original single-request history.
            conversation = history
        # A new list prevents the provider or a test double mutating loop history.
        return [{"role": "system", "content": content}, *conversation]

    @staticmethod
    def _single_call(message: Any) -> dict[str, Any] | None:
        """Return one structurally usable tool call, otherwise no executable call."""
        # The HTTP client type is advisory; an untrusted provider may return a scalar.
        if not isinstance(message, dict):
            return None
        # Providers may omit tool_calls or return a non-list despite tool metadata.
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        # A scalar call cannot contain the required OpenAI function structure.
        call = calls[0]
        if not isinstance(call, dict):
            return None
        # Tool observations must reference a stable, non-empty provider call ID.
        if not isinstance(call.get("id"), str) or not call["id"]:
            return None
        # Function name and arguments are both required before executor dispatch.
        function = call.get("function")
        if not isinstance(function, dict):
            return None
        if not isinstance(function.get("name"), str) or not function["name"]:
            return None
        if not isinstance(function.get("arguments"), (str, dict)):
            return None
        return call

    @staticmethod
    def _correction(reason: str) -> dict[str, str]:
        """Create a plain history turn that requests one clean replacement call."""
        # It contains no model-supplied tool-call structure, so API history remains valid.
        return {
            "role": "user",
            "content": (
                f"Your previous action was not executable: {reason}. "
                "Call exactly one offered tool now. Use final_answer if finished."
            ),
        }

    @staticmethod
    def _proposal_key(tool_name: str, arguments: dict[str, Any]) -> str | None:
        """Return a stable identity for one valid-looking model proposal."""
        # Malformed arguments need executor/schema feedback and may hide distinct
        # provider text behind the same safe trace marker, so never deduplicate them.
        if arguments.get("_invalid_arguments") is True:
            return None
        canonical_arguments = dict(arguments)
        # Result size does not change the meaning of a named-playlist lookup.
        # Treat count-only variations as the same proposal.
        if tool_name == "find_playlists":
            canonical_arguments.pop("count", None)
        return json.dumps(
            {"tool": tool_name, "arguments": canonical_arguments},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def _request_requires_account_write(request: str) -> bool:
        """Conservatively recognize objectives that remain incomplete without a write."""
        normalized = " ".join(request.casefold().split())
        write_phrases = (
            "create a playlist",
            "create playlist",
            "add ",
            "remove ",
            "delete ",
            "export ",
            "save it to spotify",
            "save this playlist",
            "mark ",
            "set feedback",
        )
        return any(phrase in normalized for phrase in write_phrases)

    @staticmethod
    def _request_requires_semantic_evidence(request: str) -> bool:
        """Require indexed evidence for explicit terminology/documentation requests."""
        normalized = " ".join(request.casefold().split())
        return any(
            marker in normalized
            for marker in (
                "music terminology",
                "what is a deep cut",
                "what does deep cut mean",
                "listenbrainz endpoint",
                "musicbrainz endpoint",
                "/1/",
            )
        )

    @staticmethod
    def _bulk_delete_forbidden(request: str) -> bool:
        """Reject broad destructive playlist requests before exposing a write path."""
        normalized = " ".join(request.casefold().split())
        return (
            "playlist" in normalized
            and "delete" in normalized
            and any(
                phrase in normalized
                for phrase in ("every playlist", "all playlists", "all my playlists")
            )
        )

    @staticmethod
    def _unmarked_delete_forbidden(
        request: str,
        tool_name: str,
        result: ToolResult,
    ) -> bool:
        """Recognize a requested unmarked target from authoritative playlist data."""
        if tool_name != "find_playlists" or not result.success:
            return False
        normalized = request.casefold()
        if "delete" not in normalized or not isinstance(result.data, dict):
            return False
        playlists = result.data.get("playlists")
        if not isinstance(playlists, list):
            return False
        for playlist in playlists:
            if not isinstance(playlist, dict) or playlist.get("encore_managed") is not False:
                continue
            title = playlist.get("title")
            if isinstance(title, str) and title.strip() and title.casefold() in normalized:
                return True
        return False

    @staticmethod
    def _observation(tool_name: str, report: ExecutionReport) -> ToolObservation:
        """Convert executor output to the loop's immutable trace-facing shape."""
        # Copy every existing report field needed to audit retries and approval.
        return ToolObservation(
            tool_name=tool_name,
            result=report.result,
            attempts=report.attempts,
            retry_delays_s=report.retry_delays_s,
            approval_requested=report.approval_requested,
            approved=report.approved,
        )

    @staticmethod
    def _routing_request(
        request: str,
        prior_conversation: tuple[dict[str, Any], ...],
    ) -> str:
        """Add bounded dialogue context so the router can resolve follow-ups."""
        # Tool payloads and assistant function calls are deliberately excluded:
        # routing needs the conversational task, not bulky or untrusted API data.
        recent: list[str] = []
        for message in reversed(prior_conversation):
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            compact = " ".join(content.split())
            if not compact:
                continue
            recent.append(f"{role}: {compact[:500]}")
            if len(recent) == 4:
                break
        if not recent:
            return request
        recent.reverse()
        return (
            "<UNTRUSTED_RECENT_CONVERSATION>\n"
            + "\n".join(recent)
            + "\n</UNTRUSTED_RECENT_CONVERSATION>\n"
            + "<CURRENT_REQUEST>\n"
            + request
            + "\n</CURRENT_REQUEST>"
        )

    @staticmethod
    def _requires_past_session_search(request: str) -> bool:
        """Recognize explicit history requests that startup retrieval cannot answer."""
        normalized = " ".join(request.casefold().split())
        return any(
            phrase in normalized
            for phrase in (
                "prior session",
                "past session",
                "previous session",
                "last session",
                "earlier session",
                "session history",
                "what did we do last",
            )
        )

    @staticmethod
    def _fallback_final(
        reason: TerminationStatus | None,
        observations: tuple[ToolObservation, ...] = (),
    ) -> dict[str, Any]:
        """Return a truthful local answer when expensive finalization is malformed."""
        # Known deterministic stops receive specific wording rather than false success.
        if reason == "budget_exceeded":
            answer = "Encore reached its action-step limit before completing the request."
            status: TerminationStatus = "budget_exceeded"
        elif reason == "policy_halt":
            answer = "Encore stopped because a deterministic safety policy refused the action."
            status = "policy_halt"
        elif observations:
            # Preserve bounded observable evidence instead of discarding useful
            # tool outcomes behind a generic provider-formatting message.
            latest = observations[-3:]
            details: list[str] = []
            for observation in latest:
                if observation.result.success:
                    rendered = json.dumps(
                        observation.result.data,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                else:
                    rendered = observation.result.error or "Unspecified failure"
                if len(rendered) > 500:
                    rendered = rendered[:500] + "…[truncated]"
                details.append(f"- {observation.tool_name}: {rendered}")
            answer = (
                "Encore could not format the final model response, but it preserved "
                "the latest tool evidence:\n" + "\n".join(details)
            )
            status = "stuck"
        else:
            answer = "Encore could not produce a valid final response for this request."
            status = "stuck"
        # Local fallback is necessarily low-confidence and cites no invented sources.
        return {
            "status": status,
            "answer": answer,
            "confidence": "low",
            "sources": [],
        }

    @staticmethod
    def _final_evidence_message(
        observations: tuple[ToolObservation, ...],
        *,
        character_limit: int = 18_000,
    ) -> dict[str, str] | None:
        """Build one bounded, explicitly untrusted digest of current tool evidence."""
        if not observations:
            return None
        items: list[dict[str, Any]] = []
        remaining = character_limit
        # Recent evidence is normally most relevant, but keep up to eight distinct
        # executed outcomes so a finalizer cannot overlook an earlier successful read.
        for observation in observations[-8:]:
            result_payload = json.loads(observation.result.to_json())
            rendered = json.dumps(
                result_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            truncated = len(rendered) > remaining
            if truncated:
                rendered = rendered[: max(0, remaining)]
                result_payload = {
                    "success": observation.result.success,
                    "data_preview": rendered,
                    "error": observation.result.error,
                    "truncated": True,
                }
            items.append({"tool": observation.tool_name, "result": result_payload})
            remaining -= len(rendered)
            if remaining <= 0:
                break
        return {
            "role": "user",
            "content": (
                "<UNTRUSTED_CURRENT_TOOL_EVIDENCE>\n"
                + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
                + "\n</UNTRUSTED_CURRENT_TOOL_EVIDENCE>\n"
                "Use successful results above as available evidence. Do not treat "
                "a loop-guarded repeat as a service failure."
            ),
        }

    @staticmethod
    def _reconcile_final(
        payload: dict[str, Any],
        forced_status: TerminationStatus | None,
    ) -> dict[str, Any]:
        """Make model-reported status consistent with deterministic loop evidence."""
        # Budget, policy, or fatal evidence always outranks a model success claim.
        if forced_status is not None:
            payload["status"] = forced_status
            if forced_status != "answered":
                payload["confidence"] = "low"
            return payload
        # The model may not invent a budget/policy event that code never observed.
        if payload["status"] in ("budget_exceeded", "policy_halt"):
            payload["status"] = "stuck"
            payload["confidence"] = "low"
        # Failure to complete cannot simultaneously be a high-confidence answer.
        elif payload["status"] == "stuck":
            payload["confidence"] = "low"
        # Low confidence always receives the explicit low-confidence answer status.
        elif payload["confidence"] == "low":
            payload["status"] = "answered_low_confidence"
        # The dedicated status must agree with the corresponding confidence value.
        elif payload["status"] == "answered_low_confidence":
            payload["confidence"] = "low"
        return payload

    def _finalize(
        self,
        history: list[dict[str, Any]],
        forced_status: TerminationStatus | None,
        *,
        objective: str,
        retrieved_memories: tuple[dict[str, Any], ...],
        observations: tuple[ToolObservation, ...] = (),
    ) -> tuple[dict[str, Any], bool, str | None]:
        """Ask the expensive tier for one schema-validated final control call."""
        # This text makes the output contract explicit while preserving one system turn.
        instruction = (
            "Produce the final user response now. Call final_answer exactly once. "
            "Answer only the CURRENT_OBJECTIVE for this request. Use only tool "
            "evidence produced during this current request; previous assistant "
            "answers are context, not evidence. Cite only actual current sources, "
            "and do not propose another action. The action phase is already over, "
            "so action tools are intentionally not offered during this final call. "
            "Never claim a tool was unavailable when CURRENT_TOOL_EVIDENCE contains "
            "a successful result for it. If the objective asks for a summary of "
            "prior sessions, summarize every session returned in current tool "
            "evidence; do not substitute the three startup memories."
        )
        # Do not replay historical assistant tool calls while offering only the
        # final_answer schema. Some local providers interpret those now-unoffered
        # functions as missing capabilities and ignore their successful results.
        # The objective, protected state, memories, and this bounded evidence digest
        # are sufficient for final response generation.
        final_history: list[dict[str, Any]] = [
            message
            for message in history
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"].startswith("<UNTRUSTED_WORKING_MEMORY_SUMMARY>")
        ]
        evidence_message = self._final_evidence_message(observations)
        if evidence_message is not None:
            final_history.append(evidence_message)
        try:
            # Only final_answer is visible and tool_choice forces that named function.
            response = self._client.chat(
                self._messages(
                    final_history,
                    objective=objective,
                    retrieved_memories=retrieved_memories,
                    final_instruction=instruction,
                ),
                tier="expensive",
                tools=[FINAL_ANSWER_SPEC.openai_schema()],
                tool_choice=FORCE_FINAL_ANSWER,
                max_tokens=1_600,
            )
            # Structural checks reject missing, parallel, or incomplete final calls.
            call = self._single_call(response.message)
            if call is None or call["function"]["name"] != "final_answer":
                raise ValueError("expensive model did not call final_answer once")
            payload = parse_forced_arguments(
                response.message, FINAL_ANSWER_SPEC.openai_schema()
            )
            # Copy before reconciliation so provider/test-owned dictionaries stay intact.
            return self._reconcile_final(dict(payload), forced_status), True, None
        except TraceWriteError:
            # Missing audit evidence is not a recoverable model-format problem.
            raise
        except Exception as exc:
            # Provider, decoding, and validation failures terminate instead of recursing.
            # Only code-owned metadata enters the diagnostic. Provider response bodies,
            # exception messages, request URLs, prompts, and headers remain private.
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if isinstance(status_code, int):
                diagnostic = (
                    "Expensive finalization model request failed "
                    f"(HTTP {status_code}). Check the configured expensive model "
                    "and provider availability."
                )
            elif isinstance(exc, (json.JSONDecodeError, ValueError)):
                diagnostic = (
                    "The expensive model returned an invalid final_answer payload. "
                    "Try a model with reliable tool-calling support."
                )
            else:
                diagnostic = (
                    "Expensive finalization failed "
                    f"({type(exc).__name__}). Check the expensive-model configuration."
                )
            return self._fallback_final(forced_status, observations), False, diagnostic

    def run(
        self,
        request: str,
        *,
        retrieved_memories: tuple[dict[str, Any], ...] = (),
        prior_conversation: tuple[dict[str, Any], ...] = (),
        request_id: str | None = None,
    ) -> AgentResult:
        """Execute one user request within the route, tool, and step boundaries."""
        # Correlation starts before routing so its LLM call shares the request ID.
        if self._trace is not None:
            self._trace.start_request(request_id)
        # Strict detachment prevents a caller from mutating active loop history.
        history: list[dict[str, Any]] = json.loads(
            json.dumps(prior_conversation, allow_nan=False)
        )
        # The exact user turn remains in bounded history for future requests and
        # end-of-request ordering. Front-loading removes its duplicate at call time.
        history.append({"role": "user", "content": request})
        # Counters are explicit so later traces do not infer behavior from messages.
        model_calls = 1  # Routing is one cheap-model decision attempt.
        tool_calls = 0
        steps = 0
        observations: list[ToolObservation] = []
        # Code-owned loop guards prevent repeated successful reads from consuming
        # the assignment's entire 12-step budget.
        executed_proposals: dict[str, ToolObservation] = {}
        successful_by_tool: dict[str, ToolObservation] = {}
        duplicate_rejections: dict[str, int] = {}
        failed_executions_by_tool: dict[str, int] = {}
        consecutive_invalid_outputs = 0
        # The structural compression trigger resets after each successful summary.
        tool_results_since_compression = 0
        # A failed optional compaction must not invalidate completed account work.
        # Disable further attempts for this request and continue with raw evidence.
        compression_disabled = False

        try:
            # The router runs once; tools are never broadened mid-request by the model.
            route = self._route(
                self._routing_request(request, prior_conversation),
                self._client,
            )
        except TraceWriteError:
            # Never replace an observability failure with an unrecorded fallback route.
            raise
        except Exception:
            # Router transport/implementation failures safely broaden choices only.
            route = fallback_decision()
        # Record the validated/fallback selection without storing the raw request.
        self._write_trace(
            "route",
            domains=route.domains,
            tools=route.tools,
            used_fallback=route.used_fallback,
        )
        route_summary = (
            f"[route] domains={','.join(route.domains)}; "
            f"tools={len(route.tools)}; fallback={str(route.used_fallback).lower()}"
        )
        self._show(1, route_summary)
        self._show(
            2,
            "[route details]",
            {
                "domains": route.domains,
                "tools": route.tools,
                "used_fallback": route.used_fallback,
            },
        )
        # This fixed schema subset is reused for every cheap action decision.
        schemas = route.model_tool_schemas()
        # A forced status is set only by deterministic budget, fatal, or policy evidence.
        forced_status: TerminationStatus | None = (
            "policy_halt" if self._bulk_delete_forbidden(request) else None
        )
        if forced_status == "policy_halt":
            self._show(1, "[policy] broad playlist deletion refused before action")
        # A cheap final call exits normally; exhausting the range is a budget stop.
        signaled_final = False

        step_numbers = () if forced_status is not None else range(1, self._max_steps + 1)
        for step_number in step_numbers:
            # Each iteration consumes exactly one action-decision step.
            steps = step_number
            model_calls += 1
            try:
                # The cheap tier selects one routed action or signals finalization.
                # Some providers ignore a forced function choice, so expose one
                # required function for explicit history lookup and radio-batch
                # verification edges. All other turns retain the routed schema.
                must_search_past_sessions = (
                    self._requires_past_session_search(request)
                    and "search_past_sessions" not in successful_by_tool
                    and failed_executions_by_tool.get("search_past_sessions", 0) == 0
                    and any(
                        schema["function"]["name"] == "search_past_sessions"
                        for schema in schemas
                    )
                )
                must_verify_radio_batch = (
                    "lb_radio" in successful_by_tool
                    and "verify_recordings" not in successful_by_tool
                    and failed_executions_by_tool.get("verify_recordings", 0) == 0
                    and any(
                        schema["function"]["name"] == "verify_recordings"
                        for schema in schemas
                    )
                )
                if must_search_past_sessions:
                    step_schemas = [
                        schema
                        for schema in schemas
                        if schema["function"]["name"] == "search_past_sessions"
                    ]
                elif must_verify_radio_batch:
                    step_schemas = [
                        schema
                        for schema in schemas
                        if schema["function"]["name"] == "verify_recordings"
                    ]
                else:
                    step_schemas = schemas
                action_choice: Any = (
                    FORCE_SEARCH_PAST_SESSIONS
                    if must_search_past_sessions
                    else FORCE_VERIFY_RECORDINGS
                    if must_verify_radio_batch
                    else "required"
                )
                response = self._client.chat(
                    self._messages(
                        history,
                        objective=request,
                        retrieved_memories=retrieved_memories,
                    ),
                    tier="cheap",
                    tools=step_schemas,
                    # Every valid action, including termination, is represented by
                    # an offered function. Requiring one prevents verbose prose from
                    # consuming the bounded loop without doing anything.
                    tool_choice=action_choice,
                    max_tokens=1_200,
                )
            except TraceWriteError:
                # Trace durability failure outranks normal provider recovery.
                raise
            except Exception:
                # Model transport/provider failure cannot be repaired by executing a tool.
                self._show(1, f"[step {step_number}] model call failed; stopping")
                forced_status = "stuck"
                break

            # Level 2 exposes the complete provider message actually received. In
            # normal operation it is a function call, not hidden chain-of-thought.
            self._show(2, f"[step {step_number} model output]", response.message)

            # Never execute partial or parallel proposals from malformed output.
            call = self._single_call(response.message)
            if call is None:
                consecutive_invalid_outputs += 1
                self._show(
                    1,
                    f"[step {step_number}] invalid model output; requesting correction",
                )
                history.append(
                    self._correction(
                        "it must contain exactly one complete tool call; do not "
                        "repeat a tool whose result is already in the conversation"
                    )
                )
                if consecutive_invalid_outputs >= MAX_CONSECUTIVE_INVALID_OUTPUTS:
                    # The expensive finalizer gets one chance to turn any gathered
                    # evidence into an answer; no more action steps are spent.
                    self._show(1, "[loop guard] repeated invalid model output; finalizing")
                    signaled_final = True
                    break
                continue
            consecutive_invalid_outputs = 0
            # One valid control call is only a termination signal at the cheap tier.
            tool_name = call["function"]["name"]
            if tool_name == "final_answer":
                self._show(1, f"[step {step_number}] final_answer signal")
                # A nonempty validated route means the router determined that live
                # evidence is relevant. Do not let stale dialogue short-circuit the
                # current factual/recommendation request before any tool executes.
                if route.tools and tool_calls == 0 and (
                    prior_conversation
                    or self._request_requires_semantic_evidence(request)
                ):
                    history.append(
                        self._correction(
                            "no current-request tool evidence has been gathered yet"
                        )
                    )
                    continue
                signaled_final = True
                break

            # Providers can emit a remembered or invented function name even when
            # it was not included in this request's routed schemas. Routing narrows
            # executable capability as well as presentation, so reject it here.
            if tool_name not in route.tools:
                self._show(
                    1,
                    f"[step {step_number}] rejected unavailable tool {tool_name}",
                )
                history.append(
                    self._correction(
                        f"tool '{tool_name}' was not offered for this request"
                    )
                )
                continue

            # Keep the exact executable assistant call so the following tool turn is valid.
            history.append(response.message)
            # A pre-execution intent line proves the trace was writable before dispatch.
            traced_arguments = self._trace_arguments(call["function"]["arguments"])
            self._show(1, f"[step {step_number}] call {tool_name}")
            self._show(2, f"[tool arguments: {tool_name}]", traced_arguments)
            self._write_trace(
                "tool_call",
                phase="proposed",
                tool=tool_name,
                args=traced_arguments,
            )
            proposal_key = self._proposal_key(tool_name, traced_arguments)
            if proposal_key is not None and proposal_key in executed_proposals:
                previous = executed_proposals[proposal_key]
                duplicate_rejections[proposal_key] = (
                    duplicate_rejections.get(proposal_key, 0) + 1
                )
                message = (
                    "This exact tool call already completed. Use its existing result; "
                    "choose a different necessary action or call final_answer."
                )
                self._show(1, f"[loop guard] rejected duplicate {tool_name}")
                self._write_trace(
                    "tool_call",
                    phase="rejected_duplicate",
                    tool=tool_name,
                    args=traced_arguments,
                    message=message,
                )
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": tool_name,
                        # Replay the real evidence. A synthetic failure here makes
                        # the finalizer falsely report that a successful read failed.
                        "content": previous.result.to_json(),
                    }
                )
                history.append(self._correction(message))
                if duplicate_rejections[proposal_key] >= 2:
                    if previous.result.success and not self._request_requires_account_write(request):
                        signaled_final = True
                    else:
                        forced_status = "stuck"
                    break
                continue
            if tool_name in SINGLE_SUCCESS_TOOLS and tool_name in successful_by_tool:
                previous = successful_by_tool[tool_name]
                guard_key = f"single-success:{tool_name}"
                duplicate_rejections[guard_key] = (
                    duplicate_rejections.get(guard_key, 0) + 1
                )
                message = (
                    f"{tool_name} already returned a successful candidate batch. "
                    "Use its recording_mbid values for the next required action; "
                    "do not run discovery again."
                )
                self._show(1, f"[loop guard] rejected redundant {tool_name}")
                self._write_trace(
                    "tool_call",
                    phase="rejected_redundant",
                    tool=tool_name,
                    args=traced_arguments,
                    message=message,
                )
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": tool_name,
                        "content": previous.result.to_json(),
                    }
                )
                history.append(self._correction(message))
                if duplicate_rejections[guard_key] >= 2:
                    forced_status = "stuck"
                    break
                continue
            if (
                failed_executions_by_tool.get(tool_name, 0)
                >= MAX_FAILED_EXECUTIONS_PER_TOOL
            ):
                message = (
                    f"{tool_name} already failed {MAX_FAILED_EXECUTIONS_PER_TOOL} "
                    "times without "
                    "finishing the request. Stop repeating it."
                )
                self._show(1, f"[loop guard] {message}")
                self._write_trace(
                    "tool_call",
                    phase="rejected_no_progress",
                    tool=tool_name,
                    args=traced_arguments,
                    message=message,
                )
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": tool_name,
                        "content": fail("resource", message).to_json(),
                    }
                )
                if not self._request_requires_account_write(request) and any(
                    item.result.success for item in observations
                ):
                    signaled_final = True
                else:
                    forced_status = "stuck"
                break
            # Executor validates the routed name, strict JSON, schema, and approval.
            report = self._executor.execute(
                tool_name,
                call["function"]["arguments"],
            )
            tool_calls += 1
            observations.append(self._observation(tool_name, report))
            if report.result.success:
                failed_executions_by_tool[tool_name] = 0
                successful_by_tool[tool_name] = observations[-1]
            else:
                failed_executions_by_tool[tool_name] = (
                    failed_executions_by_tool.get(tool_name, 0) + 1
                )
            if proposal_key is not None:
                executed_proposals[proposal_key] = observations[-1]
            result_label = "success" if report.result.success else (
                report.result.error_category or "failure"
            )
            self._show(
                1,
                f"[result] {tool_name}: {result_label}; attempts={report.attempts}",
            )
            self._show(
                2,
                f"[tool result: {tool_name}]",
                {
                    "result": json.loads(report.result.to_json()),
                    "attempts": report.attempts,
                    "retry_delays_s": report.retry_delays_s,
                    "approval_requested": report.approval_requested,
                    "approved": report.approved,
                },
            )
            if report.approval_requested:
                self._show(
                    1,
                    "[approval] " + ("approved" if report.approved else "declined"),
                )
            # Completion metadata audits retries without storing private result payloads.
            self._write_trace(
                "tool_call",
                phase="completed",
                tool=tool_name,
                success=report.result.success,
                error_category=report.result.error_category,
                message=report.result.error,
                attempts=report.attempts,
                retry_delays_s=report.retry_delays_s,
            )
            # Approval text and opaque permits are never included in this event.
            if report.approval_requested:
                self._write_trace(
                    "approval",
                    tool=tool_name,
                    approved=report.approved,
                    high_volume_confirmation_required=(
                        report.high_volume_confirmation_required
                    ),
                )
            # Tool output is serialized data; prompt rules forbid treating it as instructions.
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": tool_name,
                    "content": report.result.to_json(),
                }
            )

            # Persist deterministic observation state before evaluating terminal
            # outcomes. In particular, a declined write must remain available for
            # a later "go ahead" turn even though it ends the current request.
            if self._working_memory is not None and not compression_disabled:
                try:
                    changes = self._working_memory.apply_observation(
                        tool_name,
                        report.result,
                        approval_requested=report.approval_requested,
                        approved=report.approved,
                        approval_summary=report.approval_summary,
                    )
                    if changes:
                        self._write_trace(
                            "memory_write",
                            store="hot_state",
                            operation="update:" + ",".join(sorted(changes)),
                            success=True,
                            record_count=len(changes),
                        )
                    self._working_memory.record_tool_result()
                except TraceWriteError:
                    raise
                except Exception:
                    forced_status = "stuck"
                    break

            # A deterministic safety prefix distinguishes policy stops from ordinary fatality.
            if (
                not report.result.success
                and report.result.error
                and report.result.error.startswith(POLICY_HALT_PREFIX)
            ):
                forced_status = "policy_halt"
                break
            # Fatal observations stop immediately; retryable categories remain model evidence.
            if report.result.error_category == "fatal":
                forced_status = "stuck"
                break
            # Executor-owned retries have already been exhausted. More model
            # proposals cannot repair the same transient transport outcome.
            if report.result.error_category == "transient":
                forced_status = "stuck"
                break
            # A declined exact write is terminal for this request by assignment.
            if report.approval_requested and not report.approved:
                forced_status = "stuck"
                break
            if self._unmarked_delete_forbidden(request, tool_name, report.result):
                forced_status = "policy_halt"
                break

            if self._working_memory is not None and not compression_disabled:
                # Hot state is code-owned and therefore survives generated summaries.
                tool_results_since_compression += 1
                try:
                    compression = self._working_memory.maybe_compress(
                        history,
                        tool_results_since_compression=(
                            tool_results_since_compression
                        ),
                    )
                except TraceWriteError:
                    # A missing compression trace is a fatal observability failure.
                    raise
                except Exception:
                    # Local state/storage failures stop further actions safely.
                    forced_status = "stuck"
                    break
                # Compression uses the cheap model but does not consume an action step.
                model_calls += compression.model_calls
                if compression.error is not None:
                    self._show(1, "[compression] failed; retaining active history")
                    # Compression improves availability but is not part of the
                    # user's requested action. Preserve the original history and
                    # let the action loop finish instead of reporting a successful
                    # write as stuck.
                    compression_disabled = True
                    tool_results_since_compression = 0
                    continue
                if compression.compressed:
                    self._show(
                        1,
                        "[compression] completed; "
                        f"tokens={compression.tokens_before}->{compression.tokens_after}",
                    )
                    self._show(
                        2,
                        "[compression details]",
                        {
                            "reasons": compression.reasons,
                            "tokens_before": compression.tokens_before,
                            "tokens_after": compression.tokens_after,
                            "archived_messages": compression.archived_messages,
                        },
                    )
                    history = list(compression.history)
                    tool_results_since_compression = 0

        # Reaching the bound without a final signal or earlier stop is budget exhaustion.
        if not signaled_final and forced_status is None and steps >= self._max_steps:
            forced_status = "budget_exceeded"

        # The expensive final call occurs once regardless of the preceding stop reason.
        model_calls += 1
        final_payload, _, final_diagnostic = self._finalize(
            history,
            forced_status,
            objective=request,
            retrieved_memories=retrieved_memories,
            observations=tuple(observations),
        )
        display_phase = (
            "segment limit"
            if final_payload["status"] == "budget_exceeded"
            else "final"
        )
        self._show(
            1,
            f"[{display_phase}] {final_payload['status']}; "
            f"confidence={final_payload['confidence']}",
        )
        self._show(2, f"[{display_phase} payload]", final_payload)
        # Schema validation/reconciliation guarantees these casts match public aliases.
        status: TerminationStatus = final_payload["status"]
        confidence: Confidence = final_payload["confidence"]
        # Freeze sequence fields so callers cannot rewrite trace evidence after return.
        result = AgentResult(
            status=status,
            answer=final_payload["answer"],
            confidence=confidence,
            sources=tuple(final_payload["sources"]),
            route=route,
            steps=steps,
            model_calls=model_calls,
            tool_calls=tool_calls,
            observations=tuple(observations),
            # Cross-request context carries only user-visible dialogue. Tool calls
            # remain in the trace/transcript, while hot state preserves exact active
            # playlist and recording references. This prevents one large radio
            # result from poisoning every later request in the chat session.
            conversation=tuple(
                [
                    *[
                        message
                        for message in prior_conversation
                        if message.get("role") in {"user", "assistant"}
                        and isinstance(message.get("content"), str)
                    ][-6:],
                    {"role": "user", "content": request},
                    {"role": "assistant", "content": final_payload["answer"]},
                ]
            ),
            diagnostic=final_diagnostic,
        )
        # Final is the compact terminal record graders can locate without replay.
        self._write_trace(
            "final",
            status=result.status,
            confidence=result.confidence,
            sources=result.sources,
            steps=result.steps,
            model_calls=result.model_calls,
            tool_calls=result.tool_calls,
        )
        return result
