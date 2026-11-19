"""Secret-minimized JSONL observability for one Encore session.

Each writer owns exactly one session file.  Event fields are allowlisted by
type, nested values are sanitized, and model prompts/results are deliberately
absent.  The writer raises on I/O failure so the application never silently
continues an untraceable session.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from encore.model_client import ModelClient, ModelResponse

# The assignment names these event categories; future memory/compression work can
# use the reserved categories without changing the file format.
TraceEvent = Literal[
    "llm_call",
    "tool_call",
    "route",
    "approval",
    "memory_read",
    "memory_write",
    "compress",
    "final",
]

# Each category accepts only fields needed to audit the corresponding behavior.
# Unknown top-level fields raise instead of quietly leaking new private values.
EVENT_FIELDS: dict[str, frozenset[str]] = {
    "llm_call": frozenset(
        {
            "model",
            "tier",
            "tokens_in",
            "tokens_out",
            "cost_usd",
            "latency_ms",
            "success",
            "error",
        }
    ),
    "tool_call": frozenset(
        {
            "phase",
            "tool",
            "args",
            "success",
            "error_category",
            "message",
            "attempts",
            "retry_delays_s",
        }
    ),
    "route": frozenset({"domains", "tools", "used_fallback"}),
    "approval": frozenset(
        {
            "tool",
            "approved",
            "high_volume_confirmation_required",
        }
    ),
    "memory_read": frozenset(
        {"store", "query_kind", "result_count", "latency_ms"}
    ),
    "memory_write": frozenset(
        {"store", "operation", "success", "record_count"}
    ),
    "compress": frozenset(
        {"model", "tokens_before", "tokens_after", "quality_passed"}
    ),
    "final": frozenset(
        {
            "status",
            "confidence",
            "sources",
            "steps",
            "model_calls",
            "tool_calls",
        }
    ),
}

# Nested argument keys containing these markers are always replaced by REDACTED.
SENSITIVE_KEY_MARKERS = (
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "cookie",
    "headers",
)
# Bound arbitrary external strings so one API error cannot overwhelm a trace.
MAX_TRACE_STRING_LENGTH = 2_000


class TraceWriteError(RuntimeError):
    """Signal that required session evidence could not be durably recorded."""


def _utc_now() -> datetime:
    """Return a timezone-aware timestamp for the production writer."""
    # UTC keeps traces comparable across machines and daylight-saving changes.
    return datetime.now(timezone.utc)


def _sanitize(value: Any, *, key: str = "") -> Any:
    """Recursively produce a bounded JSON-compatible, secret-minimized value."""
    # Field-name checks catch credentials even when nested inside tool arguments.
    normalized_key = key.casefold().replace("-", "_")
    if any(marker in normalized_key for marker in SENSITIVE_KEY_MARKERS):
        return "[REDACTED]"
    # Mappings are copied so callers cannot mutate a record after serialization.
    if isinstance(value, Mapping):
        return {str(item_key): _sanitize(item, key=str(item_key)) for item_key, item in value.items()}
    # Tuples and lists share one portable JSON array representation.
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    # Strings are useful audit data but receive a hard size ceiling.
    if isinstance(value, str):
        if len(value) > MAX_TRACE_STRING_LENGTH:
            return value[:MAX_TRACE_STRING_LENGTH] + "…[truncated]"
        return value
    # Only native JSON scalars pass through unchanged.
    if value is None or isinstance(value, (bool, int, float)):
        return value
    # Unknown objects expose only their type, never a potentially secret repr().
    return f"[UNSERIALIZABLE:{type(value).__name__}]"


class TraceWriter:
    """Append strict JSON events to one user-keyed, permission-limited file."""

    def __init__(
        self,
        root: Path,
        user_id: str,
        *,
        session_id: str | None = None,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        # Empty identity would make retention and user-scoped deletion ambiguous.
        if not user_id.strip():
            raise ValueError("user_id must not be empty")
        # Caller-supplied IDs are UUID-validated so they cannot alter file paths.
        parsed_session = UUID(session_id) if session_id is not None else uuid4()
        self.session_id = str(parsed_session)
        # Hashing keys the directory by user without placing the raw ID in paths.
        self.user_key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
        # Root is application configuration, never a model-produced path.
        self.root = Path(root)
        self.directory = self.root / self.user_key
        self.path = self.directory / f"{self.session_id}.jsonl"
        # Timestamp injection makes exact JSONL assertions deterministic in tests.
        self._now = now
        # Request ID remains absent until AgentLoop begins one user request.
        self._request_id: str | None = None
        # One lock preserves complete lines if UI and callbacks later share a writer.
        self._lock = threading.Lock()
        self._closed = False

        # Private-by-default directories keep local listening activity user-scoped.
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # O_NOFOLLOW prevents an existing session filename from redirecting writes.
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        # A newly created trace is readable/writable only by the local account.
        descriptor = os.open(self.path, flags, 0o600)
        # Line buffering flushes every newline without requiring callers to remember.
        self._file = os.fdopen(descriptor, "a", encoding="utf-8", buffering=1)

    def start_request(self, request_id: str | None = None) -> str:
        """Set the correlation UUID attached to subsequent session events."""
        # UUID validation prevents arbitrary values from entering the trace envelope.
        parsed = UUID(request_id) if request_id is not None else uuid4()
        self._request_id = str(parsed)
        return self._request_id

    def write(self, event: TraceEvent, **fields: Any) -> None:
        """Validate, sanitize, and durably append one compact JSON object."""
        # Event validation protects the stable rubric-facing trace vocabulary.
        allowed = EVENT_FIELDS.get(event)
        if allowed is None:
            raise ValueError(f"unknown trace event '{event}'")
        # Reject accidental schema expansion rather than silently discarding data.
        unknown = fields.keys() - allowed
        if unknown:
            raise ValueError(f"unsupported fields for {event}: {sorted(unknown)}")
        # The envelope contains correlation metadata but no raw user identity.
        record: dict[str, Any] = {
            "ts": self._now().isoformat(),
            "session_id": self.session_id,
            "event": event,
        }
        if self._request_id is not None:
            record["request_id"] = self._request_id
        # Sanitization occurs before strict encoding and before acquiring the lock.
        record.update({name: _sanitize(value, key=name) for name, value in fields.items()})
        # allow_nan=False keeps every physical line valid portable JSON.
        try:
            line = json.dumps(record, allow_nan=False, separators=(",", ":"))
            with self._lock:
                if self._closed:
                    raise TraceWriteError("trace writer is closed")
                self._file.write(line + "\n")
                self._file.flush()
        except TraceWriteError:
            # Preserve the dedicated type so the agent loop will never recover past it.
            raise
        except (OSError, TypeError, ValueError) as exc:
            # Sanitize the public message; filesystem paths and values stay private.
            raise TraceWriteError(
                f"trace event could not be written ({type(exc).__name__})"
            ) from exc

    def close(self) -> None:
        """Flush and close the session file; repeated closes are harmless."""
        with self._lock:
            if not self._closed:
                self._file.flush()
                self._file.close()
                self._closed = True

    def __enter__(self) -> TraceWriter:
        """Return this writer for explicit session-lifetime management."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close the session trace regardless of request success or failure."""
        self.close()


class TracingModelClient:
    """Record safe timing/model metadata around any ModelClient-compatible object."""

    def __init__(
        self,
        client: ModelClient,
        trace: TraceWriter,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        # The wrapped client continues to own endpoints, headers, and credentials.
        self._client = client
        # Trace receives only returned public metadata and exception type names.
        self._trace = trace
        # Clock injection avoids sleeping in latency tests.
        self._monotonic = monotonic

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
        """Delegate one call while tracing tier, exact model, usage, and latency."""
        # Tier is application-owned metadata and safe to record.
        tier = kwargs.get("tier")
        started = self._monotonic()
        try:
            # Prompts, tool schemas, and choices are deliberately not traced.
            response = self._client.chat(messages, **kwargs)
        except Exception as exc:
            # Failure traces contain only the exception class, never its message.
            latency_ms = round((self._monotonic() - started) * 1000, 3)
            self._trace.write(
                "llm_call",
                tier=tier,
                latency_ms=latency_ms,
                success=False,
                error=type(exc).__name__,
            )
            raise
        # Exact returned model string supports version pinning and evaluation.
        latency_ms = round((self._monotonic() - started) * 1000, 3)
        self._trace.write(
            "llm_call",
            model=response.model,
            tier=tier,
            tokens_in=response.prompt_tokens,
            tokens_out=response.completion_tokens,
            cost_usd=response.cost_usd,
            latency_ms=latency_ms,
            success=True,
        )
        return response
