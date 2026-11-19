"""Shared validation for model-produced function calls."""

from __future__ import annotations

import json
from typing import Any

import jsonschema


def strict_json_object(value: Any) -> dict[str, Any]:
    """Decode a detached portable-JSON object, rejecting NaN and scalars."""
    if isinstance(value, str):
        decoded = json.loads(
            value,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {constant}")
            ),
        )
    else:
        decoded = json.loads(json.dumps(value, allow_nan=False))
    if not isinstance(decoded, dict):
        raise ValueError("structured arguments must be a JSON object")
    return decoded


def parse_forced_arguments(
    message: Any,
    tool: dict[str, Any],
) -> dict[str, Any]:
    """Validate exactly one call to the supplied function tool."""
    if not isinstance(message, dict):
        raise ValueError("model response is not an assistant message object")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("model must return exactly one structured call")
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    expected_name = tool["function"]["name"]
    if not isinstance(function, dict) or function.get("name") != expected_name:
        raise ValueError(f"model did not call {expected_name}")
    arguments = strict_json_object(function.get("arguments"))
    jsonschema.validate(
        arguments,
        tool["function"]["parameters"],
        format_checker=jsonschema.FormatChecker(),
    )
    return arguments
