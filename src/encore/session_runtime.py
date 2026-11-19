"""Interactive session lifecycle connecting retrieval, ReAct, and memory writes.

One controller owns the state spanning multiple user requests.  It retrieves
episodic context once from the first request, carries the loop's bounded working
conversation forward, separately retains a raw session transcript, and invokes
the mandatory summary/quality/store path exactly once when the session closes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from encore.agent_loop import AgentLoop, AgentResult, TerminationStatus
from encore.episodic_memory import EpisodicMemoryStore
from encore.session_memory import SessionMemoryResult, SessionMemoryService
from encore.tracing import TraceWriter

# Three episodes provide useful continuity without overwhelming the live request.
SESSION_START_MEMORY_LIMIT = 3
# These commands are process controls and never become conversation evidence.
EXIT_COMMANDS = frozenset({"/quit", "/exit"})
CONTINUE_RESPONSE = "yes"
# These terse replies depend entirely on the immediately preceding dialogue.
# Expand them for the reasoning loop while preserving the user's literal turn in
# the auditable session transcript.
CONFIRMATION_RESPONSES = frozenset({"yes", "yes please", "go ahead", "do it"})


@dataclass(frozen=True)
class SessionCloseReport:
    """Compact result of closing one interactive session."""

    # Requests counts completed user turns, excluding commands and blank input.
    requests: int
    # Last status gives the session summary a truthful terminal condition.
    final_status: str
    # Retrieval count demonstrates whether session-start memory was available.
    retrieved_memories: int
    # Empty sessions perform no artificial episodic write.
    memory_result: SessionMemoryResult | None


class SessionController:
    """Coordinate first-message retrieval and exactly-once session closure."""

    def __init__(
        self,
        loop: AgentLoop,
        episodic: EpisodicMemoryStore,
        session_memory: SessionMemoryService,
        user_id: str,
        session_id: str,
        *,
        trace: TraceWriter | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not user_id.strip():
            raise ValueError("user_id must not be empty")
        # Canonical validation prevents session identity from becoming a path value.
        self.session_id = str(UUID(session_id))
        self.user_id = user_id
        self._loop = loop
        # Test doubles predating this feature retain the assignment's normal bound.
        self.step_limit = getattr(loop, "max_steps", 12)
        self._episodic = episodic
        self._session_memory = session_memory
        self._trace = trace
        self._monotonic = monotonic
        # Retrieval occurs lazily because the first request supplies its query.
        self._retrieval_attempted = False
        self._retrieved_memories: tuple[dict[str, Any], ...] = ()
        # Bounded conversation is model-facing and may contain compression summaries.
        self._conversation: tuple[dict[str, Any], ...] = ()
        # Raw transcript remains separate for the end-of-session evidence check.
        self._transcript: list[dict[str, Any]] = []
        self._requests = 0
        self._last_status: TerminationStatus | None = None
        # A user-approved continuation repeats the original objective explicitly
        # instead of relying on a vague synthetic "continue" turn.
        self._continuation_objective: str | None = None
        self._closed = False
        self._close_report: SessionCloseReport | None = None

    def _retrieve_once(self, first_request: str) -> None:
        """Retrieve top-three same-user episodes once, tolerating store unavailability."""
        if self._retrieval_attempted:
            return
        # Mark before I/O so a failed backend is not hammered on every user turn.
        self._retrieval_attempted = True
        started = self._monotonic()
        try:
            # Store-side code filters user/90-day window and preserves annotations.
            matches = self._episodic.search_sessions(
                self.user_id,
                first_request,
                SESSION_START_MEMORY_LIMIT,
            )
            # Strict JSON-compatible copies prevent backend mutation after retrieval.
            import json

            detached = json.loads(json.dumps(matches, allow_nan=False))
            self._retrieved_memories = tuple(detached[:SESSION_START_MEMORY_LIMIT])
        except Exception:
            # Prior memory improves an answer but is not required to answer safely.
            self._retrieved_memories = ()
        latency_ms = max(0, round((self._monotonic() - started) * 1_000))
        if self._trace is not None:
            # Counts/timing are auditable; retrieved private text never enters traces.
            self._trace.write(
                "memory_read",
                store="episodic",
                query_kind="session_start",
                result_count=len(self._retrieved_memories),
                latency_ms=latency_ms,
            )

    def handle(
        self,
        request: str,
        *,
        continuation_of: str | None = None,
    ) -> AgentResult:
        """Run one non-empty request and add its evidence to both memory lanes."""
        if self._closed:
            raise RuntimeError("session is already closed")
        normalized = request.strip()
        if not normalized:
            raise ValueError("request must not be empty")
        effective_request = normalized
        if normalized.casefold() in CONFIRMATION_RESPONSES and self._conversation:
            effective_request = (
                "The user confirms the immediately preceding assistant question. "
                "Continue the pending user request using the bounded conversation. "
                "Do not ask for another conversational confirmation; any account "
                "write will use the application's exact approval prompt."
            )
        if continuation_of is None:
            self._continuation_objective = effective_request
        # One request UUID covers retrieval, routing, actions, and finalization.
        request_id = str(uuid4())
        if self._trace is not None:
            self._trace.start_request(request_id)
        self._retrieve_once(effective_request)
        # Record user evidence before execution so failures are summarized truthfully.
        self._transcript.append({"role": "user", "content": normalized})
        try:
            result = self._loop.run(
                effective_request,
                retrieved_memories=self._retrieved_memories,
                prior_conversation=self._conversation,
                request_id=request_id,
            )
        except Exception:
            # Session closure will record that this request did not finish.
            self._last_status = "stuck"
            raise

        # Tool results are evidence data; intermediate model reasoning is omitted.
        for observation in result.observations:
            self._transcript.append(
                {
                    "role": "tool",
                    "name": observation.tool_name,
                    "content": observation.result.to_json(),
                }
            )
        # The code-reconciled final answer is the assistant evidence users observed.
        self._transcript.append({"role": "assistant", "content": result.answer})
        self._conversation = result.conversation
        self._requests += 1
        self._last_status = result.status
        return result

    def continue_after_budget(self) -> AgentResult:
        """Run another bounded segment only after the preceding segment exhausted."""
        if self._closed:
            raise RuntimeError("session is already closed")
        if self._last_status != "budget_exceeded":
            raise RuntimeError("there is no budget-exceeded request to continue")
        if not self._continuation_objective:
            raise RuntimeError("the original request is unavailable")
        objective = self._continuation_objective
        continuation = (
            "The user approved another bounded reasoning segment. Continue the "
            "same objective using progress already present in this session:\n"
            + objective
        )
        return self.handle(continuation, continuation_of=objective)

    def close(self) -> SessionCloseReport:
        """Summarize/store a non-empty transcript exactly once and return its result."""
        if self._close_report is not None:
            return self._close_report
        self._closed = True
        # No user request means there is no episode to manufacture or quality-check.
        if not self._transcript:
            report = SessionCloseReport(0, "stuck", 0, None)
            self._close_report = report
            return report
        # A request that raised before returning is correctly summarized as stuck.
        final_status = self._last_status or "stuck"
        memory_result = self._session_memory.summarize_and_store(
            self.user_id,
            self.session_id,
            list(self._transcript),
            final_status=final_status,
        )
        report = SessionCloseReport(
            requests=self._requests,
            final_status=final_status,
            retrieved_memories=len(self._retrieved_memories),
            memory_result=memory_result,
        )
        self._close_report = report
        return report


def format_budget_progress(results: list[AgentResult]) -> str:
    """Summarize observable progress across user-approved bounded segments."""
    steps = sum(result.steps for result in results)
    observations = [
        observation
        for result in results
        for observation in result.observations
    ]
    successes = [
        observation.tool_name
        for observation in observations
        if observation.result.success
    ]
    failures = [
        f"{observation.tool_name} ({observation.result.error_category or 'failure'})"
        for observation in observations
        if not observation.result.success
    ]
    lines = [
        f"Encore used {steps} bounded reasoning steps without finishing.",
        "Progress so far:",
    ]
    if observations:
        lines.append(
            f"- Tool calls completed: {len(observations)} "
            f"({len(successes)} successful, {len(failures)} unsuccessful)."
        )
        if successes:
            lines.append("- Successful tools: " + ", ".join(successes) + ".")
        if failures:
            lines.append("- Unresolved results: " + ", ".join(failures) + ".")
        lines.append("- Latest tool evidence:")
        # Five bounded observations keep repeated continuations readable while
        # still showing the user concrete progress rather than only tool names.
        for observation in observations[-5:]:
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
            if len(rendered) > 300:
                rendered = rendered[:300] + "…[truncated]"
            lines.append(f"  - {observation.tool_name}: {rendered}")
    else:
        lines.append("- No executable tool calls completed in these segments.")
    # Sources are identifiers/chunk labels already approved for the final contract.
    sources = list(
        dict.fromkeys(source for result in results for source in result.sources)
    )
    if sources:
        lines.append("- Sources gathered: " + ", ".join(sources) + ".")
    return "\n".join(lines)


def _show_terminal_result(
    result: AgentResult,
    output_fn: Callable[[str], None],
) -> None:
    """Render one completed or declined terminal result consistently."""
    output_fn(result.answer)
    if result.status not in ("answered", "answered_low_confidence"):
        output_fn(
            f"Encore status: {result.status} "
            f"(confidence: {result.confidence})."
        )
    diagnostic = getattr(result, "diagnostic", None)
    if diagnostic:
        output_fn(f"Troubleshooting: {diagnostic}")


def run_console_session(
    controller: SessionController,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run a small multi-request console and always close non-empty sessions."""
    output_fn("Encore session started. Type /quit or /exit to finish.")
    exit_code = 0
    try:
        while True:
            try:
                request = input_fn("Encore> ")
            except EOFError:
                # End-of-file is a normal explicit session boundary in piped demos.
                break
            except KeyboardInterrupt:
                output_fn("Session interrupted; closing memory safely.")
                break
            normalized = request.strip()
            if not normalized:
                continue
            if normalized.casefold() in EXIT_COMMANDS:
                break
            try:
                result = controller.handle(normalized)
                budget_segments: list[AgentResult] = []
                while result.status == "budget_exceeded":
                    budget_segments.append(result)
                    output_fn(format_budget_progress(budget_segments))
                    try:
                        response = input_fn(
                            f"Continue thinking for up to {controller.step_limit} "
                            "more steps? "
                            "Type yes to continue: "
                        )
                    except (EOFError, KeyboardInterrupt):
                        response = ""
                    if response.strip().casefold() != CONTINUE_RESPONSE:
                        _show_terminal_result(result, output_fn)
                        break
                    output_fn("Continuing the same request for another bounded segment.")
                    result = controller.continue_after_budget()
                else:
                    _show_terminal_result(result, output_fn)
            except Exception as exc:
                # Class-only diagnostics avoid leaking provider URLs or private values.
                output_fn(f"Request stopped safely ({type(exc).__name__}).")
                exit_code = 1
                break
    finally:
        try:
            report = controller.close()
            if report.memory_result is not None:
                if report.memory_result.stored:
                    output_fn("Session memory saved.")
                else:
                    detail = report.memory_result.error
                    if detail:
                        output_fn(f"Session memory was not saved safely: {detail}")
                    else:
                        output_fn("Session memory was not saved safely.")
        except KeyboardInterrupt:
            # A second Ctrl-C during the bounded shutdown call exits cleanly
            # instead of dumping a requests/SSL traceback into the console.
            output_fn("Session closure interrupted; session memory was not saved.")
            exit_code = 130
        except Exception as exc:
            # A closure failure must be visible rather than silently losing memory.
            output_fn(f"Session closure failed ({type(exc).__name__}).")
            exit_code = 2
    return exit_code
