"""Console visibility levels for the observable ReAct action loop."""

from __future__ import annotations

import json

import pytest

from encore.agent_loop import AgentLoop
from encore.contracts import ok
from encore.cli import parse_args
from encore.model_client import ModelResponse
from encore.router import RouteDecision
from encore.tool_executor import ExecutionReport


def tool_message(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def final_message(call_id):
    return tool_message(
        "final_answer",
        {
            "status": "answered",
            "answer": "Found it.",
            "confidence": "high",
            "sources": ["MusicBrainz"],
        },
        call_id,
    )


class QueueModel:
    def __init__(self):
        self.messages = [
            tool_message("search_recordings", {"title": "Black"}, "action-1"),
            final_message("cheap-final"),
            final_message("expensive-final"),
        ]

    def chat(self, messages, **kwargs):
        return ModelResponse(self.messages.pop(0), "test-model", 10, 3)


class OneExecutor:
    def execute(self, name, arguments):
        return ExecutionReport(
            ok({"recordings": [{"title": "Black", "artist": "Pearl Jam"}]}),
            1,
            (),
        )


def catalog_route(request, client):
    return RouteDecision(("catalog",))


def run_with_verbosity(level):
    output = []
    result = AgentLoop(
        QueueModel(),
        OneExecutor(),
        "RULES",
        route=catalog_route,
        verbosity=level,
        progress_output=output.append,
    ).run("Find Black")
    return result, "\n".join(output)


def test_verbosity_zero_preserves_silent_loop():
    result, output = run_with_verbosity(0)
    assert result.status == "answered"
    assert output == ""


def test_verbosity_one_shows_descriptions_without_full_payloads():
    _, output = run_with_verbosity(1)
    assert "[route] domains=catalog" in output
    assert "[step 1] call search_recordings" in output
    assert "[result] search_recordings: success; attempts=1" in output
    assert "[step 2] final_answer signal" in output
    assert "[final] answered; confidence=high" in output
    assert "Pearl Jam" not in output


def test_verbosity_two_shows_complete_observable_action_and_result():
    _, output = run_with_verbosity(2)
    assert "[step 1 model output]" in output
    assert '"title": "Black"' in output
    assert "[tool result: search_recordings]" in output
    assert '"artist": "Pearl Jam"' in output
    assert "[final payload]" in output
    assert '"answer": "Found it."' in output


def test_cli_accepts_only_three_verbosity_levels():
    assert parse_args(["offline"]).verbosity == 0
    assert parse_args(["offline", "--verbosity", "2"]).verbosity == 2
    with pytest.raises(SystemExit):
        parse_args(["offline", "--verbosity", "3"])


def test_loop_rejects_invalid_programmatic_verbosity():
    with pytest.raises(ValueError, match="verbosity"):
        AgentLoop(
            QueueModel(),
            OneExecutor(),
            "RULES",
            route=catalog_route,
            verbosity=-1,
        )
