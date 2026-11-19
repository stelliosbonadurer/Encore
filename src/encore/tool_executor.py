"""Deterministic validation, approval, execution, and retry boundary.

The model proposes a name and JSON arguments; it never invokes Python handlers
directly. This module turns that untrusted proposal into a validated tool
observation while keeping approval, retry count, timing, and write policy under
ordinary code control.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jsonschema

from encore.approval import ApprovalGate, ApprovalRequest
from encore.contracts import ToolResult, fail
from encore.structured_output import strict_json_object
from encore.tool_catalog import IMPLEMENTED_TOOL_SPECS, ToolSpec

# Approval UI returns raw user text; only ApprovalGate interprets it.
@dataclass(frozen=True)
class ApprovalDecision:
    """The two user responses collected for one exact write preview."""

    # Every account write needs an unambiguous ordinary confirmation.
    response: str
    # High-volume operations additionally need the generated exact phrase.
    high_volume_response: str = ""


@dataclass(frozen=True)
class ExecutionReport:
    """Tool observation plus trace-ready deterministic execution metadata."""

    # ToolResult is the only payload returned to the reasoning model.
    result: ToolResult
    # Attempts counts real handler invocations, excluding validation/declines.
    attempts: int
    # Delays records chosen full-jitter waits for later observability.
    retry_delays_s: tuple[float, ...]
    # Approval fields let traces prove writes crossed the human boundary.
    approval_requested: bool = False
    approved: bool = False
    high_volume_confirmation_required: bool = False
    # Exact deterministic preview shown to the user, when one was shown.
    approval_summary: str | None = None


ApprovalHandler = Callable[[ApprovalRequest], ApprovalDecision]


class ConsoleApprovalHandler:
    """Small injectable terminal UI for the interactive CLI.

    Tests provide fake input/output callables, so neither tests nor model code
    can accidentally block waiting for a real terminal response.
    """

    def __init__(
        self,
        *,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
    ) -> None:
        # Keep I/O behind callables so approval behavior is deterministic in tests.
        self._read = read
        self._write = write

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        # Preview prose comes from deterministic code, never the reasoning model.
        self._write(f"Proposed account change: {request.summary}")
        # A plain yes is required for every write; any other text declines.
        response = self._read("Approve this exact change? Type yes to approve: ")
        high_volume_response = ""
        # The second prompt appears only after the normal response is collected.
        if request.high_volume and response.strip().casefold() == "yes":
            self._write(
                "Large operation: type this exact phrase to continue: "
                f"{request.high_volume_phrase}"
            )
            high_volume_response = self._read("High-volume confirmation: ")
        # ApprovalGate, not this UI, decides whether the text is valid.
        return ApprovalDecision(response, high_volume_response)


class ToolExecutor:
    """Safely execute validated action tools and report every retry decision."""

    def __init__(
        self,
        registry: Mapping[str, Callable[..., ToolResult]],
        *,
        specs: tuple[ToolSpec, ...] = IMPLEMENTED_TOOL_SPECS,
        approval_gate: ApprovalGate | None = None,
        approval_handler: ApprovalHandler | None = None,
        max_retries: int = 2,
        retry_base_s: float = 1.0,
        retry_cap_s: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        # Copy the handler map so a caller cannot swap functions after validation.
        self._registry = dict(registry)
        # Name lookup connects validated proposals to model-visible specifications.
        self._specs = {spec.name: spec for spec in specs}
        # One gate must be shared with WriteTools so issued permits validate there.
        self._approval_gate = approval_gate or ApprovalGate()
        # Missing handler means writes fail closed rather than auto-approving.
        self._approval_handler = approval_handler
        # Retry settings are ordinary code-owned policy, never model arguments.
        self._max_retries = max_retries
        self._retry_base_s = retry_base_s
        self._retry_cap_s = retry_cap_s
        # Time and randomness are injected for deterministic, instant tests.
        self._sleep = sleep
        self._uniform = uniform

        # Negative retry counts or delays would create nonsensical loop behavior.
        if max_retries < 0 or retry_base_s < 0 or retry_cap_s < 0:
            raise ValueError("retry settings must be non-negative")
        # Every model-visible action needs exactly one executable handler.
        if self._registry.keys() != self._specs.keys():
            missing = self._specs.keys() - self._registry.keys()
            extra = self._registry.keys() - self._specs.keys()
            raise ValueError(
                f"executor registry mismatch; missing={missing}, extra={extra}"
            )

    @staticmethod
    def _decode_arguments(arguments: dict[str, Any] | str) -> dict[str, Any]:
        """Return a detached, strict-JSON dictionary from model output."""
        return strict_json_object(arguments)

    def _validated_arguments(
        self, tool_name: str, arguments: dict[str, Any] | str
    ) -> tuple[ToolSpec, dict[str, Any]]:
        """Resolve a known tool and validate its detached JSON arguments."""
        # Checking the spec first prevents arbitrary registry/key access.
        spec = self._specs.get(tool_name)
        if spec is None:
            raise KeyError(tool_name)
        # Decode before schema validation so strings and dictionaries behave alike.
        decoded = self._decode_arguments(arguments)
        # FormatChecker makes ``format: uuid`` enforceable rather than annotation.
        jsonschema.validate(
            decoded,
            spec.parameters,
            format_checker=jsonschema.FormatChecker(),
        )
        return spec, decoded

    def _delay_for_retry(self, retry_number: int) -> float:
        """Choose full jitter from zero to the capped exponential ceiling."""
        # retry_number starts at zero after the first failed handler attempt.
        ceiling = min(self._retry_cap_s, self._retry_base_s * (2**retry_number))
        # Full jitter spreads clients throughout the entire safe interval.
        return self._uniform(0.0, ceiling)

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | str,
        *,
        logical_count: int = 1,
    ) -> ExecutionReport:
        """Validate one proposal, obtain approval if needed, and run its handler."""
        # Aggregate logical counts are loop metadata and must never be zero/negative.
        if logical_count < 1:
            return ExecutionReport(
                fail("input", "logical_count must be positive."), 0, ()
            )
        try:
            # Nothing is executed until both name and complete arguments validate.
            spec, decoded = self._validated_arguments(tool_name, arguments)
        except KeyError:
            # Unknown names are model input failures, not Python dispatch errors.
            return ExecutionReport(
                fail("input", f"Unknown tool '{tool_name}'."), 0, ()
            )
        except (TypeError, ValueError, jsonschema.ValidationError) as exc:
            # Use only the validation message; never include credentials or headers.
            return ExecutionReport(
                fail("input", f"Invalid arguments for {tool_name}: {exc}"), 0, ()
            )

        # Write permits remain out-of-band and never enter ``decoded``.
        permit = None
        approval_requested = spec.requires_approval
        high_volume = False
        if spec.requires_approval:
            # No UI/callback means deterministic decline, not implicit authorization.
            if self._approval_handler is None:
                return ExecutionReport(
                    fail("resource", "Account write declined: no approval handler."),
                    0,
                    (),
                    approval_requested=True,
                )
            # Preview binds exactly the already validated detached arguments.
            request = self._approval_gate.prepare(
                tool_name, decoded, logical_count=logical_count
            )
            high_volume = request.high_volume
            # Only this ordinary callback may interact with the human user.
            try:
                decision = self._approval_handler(request)
            except Exception as exc:
                # Terminal/UI failures are sanitized just like handler exceptions.
                return ExecutionReport(
                    fail(
                        "fatal",
                        f"Approval interaction failed ({type(exc).__name__}).",
                    ),
                    0,
                    (),
                    approval_requested=True,
                    high_volume_confirmation_required=high_volume,
                )
            # A callback must return the explicit two-field decision contract.
            if not isinstance(decision, ApprovalDecision):
                return ExecutionReport(
                    fail("fatal", "Approval handler returned an invalid decision."),
                    0,
                    (),
                    approval_requested=True,
                    high_volume_confirmation_required=high_volume,
                )
            # Gate applies exact yes/phrase rules and returns an opaque capability.
            permit = self._approval_gate.confirm(
                request,
                decision.response,
                high_volume_response=decision.high_volume_response,
            )
            if permit is None:
                # Declines are resource observations and must never be retried.
                return ExecutionReport(
                    fail("resource", "User declined the proposed account write."),
                    0,
                    (),
                    approval_requested=True,
                    approved=False,
                    high_volume_confirmation_required=high_volume,
                    approval_summary=request.summary,
                )

        # Retry metadata begins empty and grows only after transient read failures.
        delays: list[float] = []
        attempts = 0
        # Initial attempt plus max_retries gives three total attempts by default.
        for attempt_index in range(self._max_retries + 1):
            attempts += 1
            try:
                if spec.requires_approval:
                    # Permit is a Python-only keyword absent from the JSON schema.
                    result = self._registry[tool_name](**decoded, permit=permit)
                else:
                    # Read/memory handlers receive only validated model arguments.
                    result = self._registry[tool_name](**decoded)
            except Exception as exc:  # Unexpected bugs become sanitized fatal data.
                # A crashing write must not leave its permit active for reuse.
                if permit is not None:
                    self._approval_gate.complete(permit)
                return ExecutionReport(
                    fail(
                        "fatal",
                        f"{tool_name} failed internally ({type(exc).__name__}).",
                    ),
                    attempts,
                    tuple(delays),
                    approval_requested,
                    permit is not None,
                    high_volume,
                )

            # Handlers must honor the shared contract; arbitrary returns are fatal.
            if not isinstance(result, ToolResult):
                if permit is not None:
                    self._approval_gate.complete(permit)
                result = fail("fatal", f"{tool_name} returned an invalid result type.")

            # Success and every non-transient failure terminate immediately.
            if result.success or result.error_category != "transient":
                break
            # Account writes never retry automatically after any attempted handler.
            if spec.requires_approval:
                break
            # Stop when this was the final permitted attempt.
            if attempt_index >= self._max_retries:
                break
            # Choose, record, and apply full-jitter delay before the next attempt.
            delay = self._delay_for_retry(attempt_index)
            delays.append(delay)
            self._sleep(delay)

        # Executor owns permit lifetime because it owns the no-write-retry policy.
        if permit is not None:
            self._approval_gate.complete(permit)
        # Return both the observation and trace-ready control-flow evidence.
        return ExecutionReport(
            result,
            attempts,
            tuple(delays),
            approval_requested,
            permit is not None,
            high_volume,
        )
