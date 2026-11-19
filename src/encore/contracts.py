"""Small data contracts shared across Encore modules.

This module comes from the starter's most important boundary: tools never return
an ambiguous empty value or raise an expected API failure into the agent loop.
They return one explicit, JSON-serializable result shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

# The four values come directly from the assignment. Keeping them as a Literal
# lets type checkers catch misspellings before a trace contains an unknown class.
ErrorCategory = Literal["transient", "input", "resource", "fatal"]
# Fatal results beginning with this marker are deterministic safety refusals.
# The loop recognizes only this code-owned prefix; model prose cannot create it.
POLICY_HALT_PREFIX = "POLICY_HALT:"


@dataclass(frozen=True)
class ToolResult:
    # True means ``data`` is usable; False means the error fields explain why.
    success: bool
    # Data is intentionally JSON-compatible because it enters prompts and traces.
    data: Any | None
    # Human-readable recovery guidance for the model and the user.
    error: str | None
    # Machine-readable retry/termination guidance for the deterministic loop.
    error_category: ErrorCategory | None

    def to_json(self) -> str:
        """Serialize strictly so NaN/Infinity cannot corrupt a trace later."""
        # ``asdict`` removes the dataclass wrapper without custom encoders.
        payload = asdict(self)
        # Disallow NaN/Infinity because neither is valid portable JSON.
        return json.dumps(payload, allow_nan=False)


def ok(data: Any) -> ToolResult:
    """Construct the only valid success shape."""
    # Central construction prevents contradictory success/error combinations.
    return ToolResult(True, data, None, None)


def fail(category: ErrorCategory, message: str) -> ToolResult:
    """Construct the only valid expected-failure shape."""
    # Expected service/input failures are observations, not Python exceptions.
    return ToolResult(False, None, message, category)


def classify_status(status: int) -> ErrorCategory:
    """Map HTTP/no-response statuses to the assignment's four categories."""
    # Status 0 is our sentinel for timeout/connection failure: retry may help.
    if status == 0 or status == 429 or status >= 500:
        return "transient"
    # Authentication/authorization failures should stop rather than hammer APIs.
    if status in (401, 403):
        return "fatal"
    # A valid request for a missing entity should prompt an alternate resource.
    if status == 404:
        return "resource"
    # Remaining 4xx-like outcomes normally indicate malformed model arguments.
    return "input"
