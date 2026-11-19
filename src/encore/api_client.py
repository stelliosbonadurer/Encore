"""Polite HTTP access for the two MetaBrainz services.

Adapted from the provided starter. Model code never receives this client's token
or headers. Tools choose from fixed endpoint templates and only request auth when
an endpoint requires it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import requests

from encore.config import Settings
from encore.fault_injection import TransientFaultInjector

# Fixed roots are a security allowlist: tools may fill paths, never choose hosts.
LB_API = "https://api.listenbrainz.org/1"
MB_API = "https://musicbrainz.org/ws/2"
# Normal reads fail promptly enough for loop-owned retry to be useful.
READ_TIMEOUT_S = 5
# Writes get longer because an ambiguous timeout needs careful reconciliation.
WRITE_TIMEOUT_S = 15
# Measured slow endpoints (listens/radio/playlists) opt into this override.
SLOW_READ_TIMEOUT_S = 15
# MusicBrainz asks clients sharing an IP to average about one request per second.
MB_MIN_INTERVAL_S = 1.5


class ApiClient:
    """Own rate-limit state for one process instead of using module globals."""

    def __init__(
        self,
        settings: Settings,
        *,
        request: Callable[..., Any] = requests.request,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        fault_injector: TransientFaultInjector | None = None,
    ) -> None:
        # Settings supplies only configuration; it is never serialized to models.
        self.settings = settings
        # Injecting requests/time functions makes pacing tests instant and exact.
        self._request = request
        self._monotonic = monotonic
        self._sleep = sleep
        # Evaluation failures are decided locally before any network dispatch.
        self._fault_injector = fault_injector
        # Per-client timestamps avoid hidden module-global state between sessions.
        self._last_mb_call = 0.0
        self._lb_wait_until = 0.0

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
        auth: bool = False,
        timeout: int = READ_TIMEOUT_S,
    ) -> tuple[int, Any, dict[str, str]]:
        """Make one polite request; status 0 means no response was received."""
        # Resolve the allowlisted service before auth, injection, or dispatch.
        if url.startswith(MB_API):
            service = "musicbrainz"
        elif url.startswith(LB_API):
            service = "listenbrainz"
        else:
            # A fixed-host allowlist prevents tools/models from turning this
            # helper into an arbitrary authenticated HTTP client.
            raise ValueError("ApiClient only permits ListenBrainz and MusicBrainz URLs")

        # Both nonprofit services need an identifiable application User-Agent.
        headers = {
            "User-Agent": f"Encore-CSCI391/0.1 ({self.settings.contact_email})"
        }
        if auth:
            # Refuse locally rather than emitting an empty/malformed auth header.
            if not self.settings.listenbrainz_token:
                return 401, "ListenBrainz token is not configured", {}
            # The token exists only inside this request boundary.
            headers["Authorization"] = f"Token {self.settings.listenbrainz_token}"

        # Seeded evaluation faults occur after local validation but before pacing
        # or transport. The ordinary tool/executor path classifies and retries them.
        if self._fault_injector is not None:
            injected_status = self._fault_injector.status_for(service)
            if injected_status is not None:
                return (
                    injected_status,
                    {"error": f"Injected transient failure (HTTP {injected_status})."},
                    {"X-Encore-Fault-Injected": "true"},
                )

        if service == "musicbrainz":
            # Calculate only the remaining portion of the minimum interval.
            wait = self._last_mb_call + MB_MIN_INTERVAL_S - self._monotonic()
            if wait > 0:
                # Sleeping here protects every MusicBrainz tool uniformly.
                self._sleep(wait)
            # Record dispatch time after sleeping, not the earlier attempted time.
            self._last_mb_call = self._monotonic()
            # MusicBrainz defaults to XML; all tools expect JSON dictionaries.
            params = {**(params or {}), "fmt": "json"}
        else:
            # ListenBrainz tells us exactly when exhausted quota becomes available.
            wait = self._lb_wait_until - self._monotonic()
            if wait > 0:
                self._sleep(wait)

        try:
            # requests handles URL encoding and JSON serialization; tools never
            # construct raw HTTP bytes or authorization headers themselves.
            response = self._request(
                method,
                url,
                params=params,
                json=body,
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            # Status 0 distinguishes no response from any real HTTP status.
            return 0, type(exc).__name__, {}

        # Persist server rate-limit state for the next ListenBrainz request.
        if service == "listenbrainz" and response.headers.get("X-RateLimit-Remaining") == "0":
            # Defaulting to one second is safer than immediately retrying a
            # response that omitted its reset duration.
            reset = float(response.headers.get("X-RateLimit-Reset-In", "1"))
            self._lb_wait_until = self._monotonic() + reset

        try:
            # Most endpoints return JSON on both success and expected failure.
            payload = response.json()
        except ValueError:
            # Preserve HTML/text error pages as observations without parsing them.
            payload = response.text
        # Headers are copied to a plain dict so fake and real responses align.
        return response.status_code, payload, dict(response.headers)
