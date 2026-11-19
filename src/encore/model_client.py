"""Minimal OpenAI-compatible model client adapted from Lab 3.

This module chooses providers and model tiers and makes one request. It does not
retry, route, execute tools, or decide termination; those belong to the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import requests

from encore.config import Settings

# Mode selects provider; tier selects a configured model within that provider.
Mode = Literal["online", "offline"]
Tier = Literal["cheap", "expensive"]


@dataclass(frozen=True)
class ModelResponse:
    # The assistant message may contain text or OpenAI-style tool calls.
    message: dict[str, Any]
    # Exact returned configuration identifier is needed in traces/evaluation.
    model: str
    # Providers may omit usage, so token counts are explicitly optional.
    prompt_tokens: int | None
    completion_tokens: int | None
    # OpenRouter includes the billed request cost in usage; local providers may omit it.
    cost_usd: float | None = None


class ModelClient:
    def __init__(self, settings: Settings, mode: Mode) -> None:
        # Keep provider credentials encapsulated in this boundary.
        self.settings = settings
        # Mode is fixed for the client lifetime to avoid accidental provider swaps.
        self.mode = mode

    def _connection(self, tier: Tier) -> tuple[str, dict[str, str], str]:
        """Resolve endpoint, private headers, and exact model for one tier."""
        if self.mode == "online":
            # Online calls cannot proceed without an explicit OpenRouter key.
            if not self.settings.openrouter_api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not configured")
            # Routing/tool work uses cheap; final answers will use expensive.
            model = (
                self.settings.openrouter_model_cheap
                if tier == "cheap"
                else self.settings.openrouter_model_expensive
            )
            # Authorization stays here and is never returned in ModelResponse.
            headers = {
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
            }
            url = self.settings.openrouter_url
        else:
            # Local servers generally need no bearer credential.
            model = (
                self.settings.local_llm_model_cheap
                if tier == "cheap"
                else self.settings.local_llm_model_expensive
            )
            headers = {"Content-Type": "application/json"}
            url = self.settings.local_llm_url

        # Fail at composition/call time with a targeted configuration message.
        if not model:
            raise RuntimeError(f"The {self.mode} {tier} model is not configured")
        # Returning a tuple keeps request construction below provider-agnostic.
        return url, headers, model

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tier: Tier,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
        max_tokens: int = 800,
        timeout_s: float = 120,
    ) -> ModelResponse:
        """Make one model call; retries and agent policy belong to the loop."""
        # Resolve secrets and exact model immediately before the HTTP boundary.
        url, headers, model = self._connection(tier)
        # Temperature zero reduces router/tool-selection variance.
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            # Tool metadata is model-visible, but execution capabilities are not.
            payload.update(
                {
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "parallel_tool_calls": False,
                }
            )

        # Callers may shorten the outer bound for nonessential lifecycle work.
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        # Transport/provider failures become exceptions for the loop layer.
        response.raise_for_status()
        # Decode only after a successful HTTP status.
        data = response.json()
        try:
            # OpenAI-compatible providers place the assistant turn at choices[0].
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            # A malformed success response is not a valid empty assistant turn.
            raise RuntimeError("Model response has no assistant message") from exc

        # Usage is valuable but not universally implemented by local providers.
        usage = data.get("usage") or {}
        # Return sanitized metadata only—never endpoint headers or API keys.
        return ModelResponse(
            message=message,
            model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost_usd=usage.get("cost"),
        )
