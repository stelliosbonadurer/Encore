"""Offline tests for first-message retrieval and exactly-once session closure."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from encore.contracts import ok
from encore.session_memory import SessionMemoryResult, SummaryQuality
from encore.session_runtime import SessionController, run_console_session
from encore.tracing import TraceWriter


SESSION = "11111111-1111-4111-8111-111111111111"


class RecordingEpisodic:
    """Return configured session-start memories or a controlled failure."""

    def __init__(self, matches=(), *, fail=False):
        self.matches = list(matches)
        self.fail = fail
        self.calls = []

    def search_sessions(self, user_id, query, limit):
        self.calls.append((user_id, query, limit))
        if self.fail:
            raise RuntimeError("private vector failure")
        return list(self.matches)


class RecordingLoop:
    """Return small AgentResult-shaped objects while recording context inputs."""

    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def run(self, request, **kwargs):
        self.calls.append((request, kwargs))
        if self.fail:
            raise RuntimeError("private model failure")
        prior = list(kwargs["prior_conversation"])
        conversation = (
            *prior,
            {"role": "user", "content": request},
            {"role": "assistant", "content": f"Answered: {request}"},
        )
        observation = SimpleNamespace(
            tool_name="search_recordings",
            result=ok({"recordings": [{"title": "Black"}]}),
        )
        return SimpleNamespace(
            status="answered",
            answer=f"Answered: {request}",
            confidence="high",
            diagnostic=None,
            observations=(observation,),
            conversation=conversation,
        )


class RecordingSessionMemory:
    """Capture the raw session transcript passed to mandatory closure."""

    def __init__(self, *, stored=True):
        self.calls = []
        self.result = SessionMemoryResult(
            stored=stored,
            memory_id="memory-1" if stored else None,
            merged=False,
            quality=SummaryQuality(True, 1.0, ()) if stored else None,
            error=None if stored else "safe failure",
        )

    def summarize_and_store(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def controller(loop, episodic, memory, *, trace=None):
    """Build one controller with the shared deterministic session identity."""
    return SessionController(
        loop,
        episodic,
        memory,
        "alice",
        SESSION,
        trace=trace,
        monotonic=lambda: 10.0,
    )


def test_first_request_retrieves_top_three_once_and_later_turns_reuse_context():
    matches = [
        {
            "memory_id": f"memory-{index}",
            "quality_passed": index != 2,
            "annotation": None if index != 2 else "VERIFY BEFORE USING",
        }
        for index in range(4)
    ]
    episodic = RecordingEpisodic(matches)
    loop = RecordingLoop()
    memory = RecordingSessionMemory()
    session = controller(loop, episodic, memory)

    first = session.handle("Make the focus mix more upbeat")
    second = session.handle("Add two more tracks")
    report = session.close()

    assert first.status == second.status == "answered"
    assert episodic.calls == [("alice", "Make the focus mix more upbeat", 3)]
    assert len(loop.calls[0][1]["retrieved_memories"]) == 3
    # The flagged memory keeps its warning instead of becoming an asserted fact.
    assert loop.calls[0][1]["retrieved_memories"][2]["annotation"] == (
        "VERIFY BEFORE USING"
    )
    assert loop.calls[1][1]["retrieved_memories"] == (
        loop.calls[0][1]["retrieved_memories"]
    )
    assert loop.calls[1][1]["prior_conversation"] == first.conversation
    assert report.requests == 2 and report.retrieved_memories == 3
    # Raw closure evidence includes both user turns, tool observations, and answers.
    transcript = memory.calls[0][0][2]
    assert [message["role"] for message in transcript] == [
        "user",
        "tool",
        "assistant",
        "user",
        "tool",
        "assistant",
    ]


def test_bare_yes_resumes_pending_request_without_losing_literal_transcript():
    loop = RecordingLoop()
    memory = RecordingSessionMemory()
    session = controller(loop, RecordingEpisodic(), memory)

    session.handle("Create the jazz playlist")
    session.handle("yes")
    session.close()

    resumed = loop.calls[1][0]
    assert "confirms the immediately preceding assistant question" in resumed
    assert "exact approval prompt" in resumed
    assert loop.calls[1][1]["prior_conversation"]
    transcript = memory.calls[0][0][2]
    assert [
        message["content"]
        for message in transcript
        if message["role"] == "user"
    ] == ["Create the jazz playlist", "yes"]


def test_retrieval_failure_is_nonfatal_and_not_retried_every_turn():
    episodic = RecordingEpisodic(fail=True)
    loop = RecordingLoop()
    memory = RecordingSessionMemory()
    session = controller(loop, episodic, memory)

    result = session.handle("Find something new")
    session.handle("Try another search")
    report = session.close()

    assert result.status == "answered"
    assert len(episodic.calls) == 1
    assert loop.calls[0][1]["retrieved_memories"] == ()
    assert report.retrieved_memories == 0


def test_failed_request_is_still_summarized_as_stuck_on_close():
    loop = RecordingLoop(fail=True)
    memory = RecordingSessionMemory()
    session = controller(loop, RecordingEpisodic(), memory)

    with pytest.raises(RuntimeError, match="private model failure"):
        session.handle("A request that fails")
    report = session.close()

    assert report.final_status == "stuck"
    assert memory.calls[0][1]["final_status"] == "stuck"
    assert memory.calls[0][0][2] == [
        {"role": "user", "content": "A request that fails"}
    ]


def test_close_is_idempotent_and_empty_session_creates_no_episode():
    memory = RecordingSessionMemory()
    empty = controller(RecordingLoop(), RecordingEpisodic(), memory)

    first = empty.close()
    second = empty.close()

    assert first is second and first.memory_result is None
    assert memory.calls == []
    with pytest.raises(RuntimeError, match="closed"):
        empty.handle("too late")


def test_console_quit_always_runs_session_end_memory_path():
    loop = RecordingLoop()
    memory = RecordingSessionMemory()
    session = controller(loop, RecordingEpisodic(), memory)
    inputs = iter(["Find Black", "/quit"])
    output = []

    exit_code = run_console_session(
        session,
        input_fn=lambda prompt: next(inputs),
        output_fn=output.append,
    )

    assert exit_code == 0 and len(memory.calls) == 1
    assert "Answered: Find Black" in output
    assert output[-1] == "Session memory saved."


def test_budget_limit_summarizes_progress_and_yes_continues_same_objective():
    class BudgetThenAnswerLoop(RecordingLoop):
        def run(self, request, **kwargs):
            self.calls.append((request, kwargs))
            prior = list(kwargs["prior_conversation"])
            if len(self.calls) == 1:
                observation = SimpleNamespace(
                    tool_name="search_recordings",
                    result=ok({"recordings": [{"title": "Black"}]}),
                )
                return SimpleNamespace(
                    status="budget_exceeded",
                    answer="Generic budget fallback.",
                    confidence="low",
                    sources=(),
                    steps=12,
                    diagnostic="invalid final payload",
                    observations=(observation,),
                    conversation=(
                        *prior,
                        {"role": "user", "content": request},
                        {"role": "assistant", "content": "Segment incomplete."},
                    ),
                )
            return SimpleNamespace(
                status="answered",
                answer="Completed after continuing.",
                confidence="high",
                sources=("MusicBrainz",),
                steps=2,
                diagnostic=None,
                observations=(),
                conversation=(
                    *prior,
                    {"role": "user", "content": request},
                    {"role": "assistant", "content": "Completed after continuing."},
                ),
            )

    loop = BudgetThenAnswerLoop()
    memory = RecordingSessionMemory()
    session = controller(loop, RecordingEpisodic(), memory)
    prompts = []
    inputs = iter(["Build a careful playlist", "yes", "/quit"])
    output = []

    exit_code = run_console_session(
        session,
        input_fn=lambda prompt: prompts.append(prompt) or next(inputs),
        output_fn=output.append,
    )

    assert exit_code == 0 and len(loop.calls) == 2
    assert loop.calls[0][0] == "Build a careful playlist"
    assert "Build a careful playlist" in loop.calls[1][0]
    assert "another bounded reasoning segment" in loop.calls[1][0]
    assert any("Progress so far" in line for line in output)
    assert any("search_recordings" in line for line in output)
    assert any('"title":"Black"' in line for line in output)
    assert "Generic budget fallback." not in output
    assert "Completed after continuing." in output
    assert any("Continue thinking" in prompt for prompt in prompts)
    assert memory.calls[0][1]["final_status"] == "answered"


def test_budget_limit_decline_keeps_budget_status_and_does_not_continue():
    class AlwaysBudgetLoop(RecordingLoop):
        def run(self, request, **kwargs):
            self.calls.append((request, kwargs))
            return SimpleNamespace(
                status="budget_exceeded",
                answer="Stopped at the bounded limit.",
                confidence="low",
                sources=(),
                steps=12,
                diagnostic=None,
                observations=(),
                conversation=(
                    {"role": "user", "content": request},
                    {"role": "assistant", "content": "Stopped."},
                ),
            )

    loop = AlwaysBudgetLoop()
    session = controller(loop, RecordingEpisodic(), RecordingSessionMemory())
    inputs = iter(["Hard request", "no", "/quit"])
    output = []

    run_console_session(
        session,
        input_fn=lambda prompt: next(inputs),
        output_fn=output.append,
    )

    assert len(loop.calls) == 1
    assert any("No executable tool calls" in line for line in output)
    assert "Stopped at the bounded limit." in output
    assert any("Encore status: budget_exceeded" in line for line in output)


def test_second_interrupt_during_session_close_exits_without_traceback():
    class InterruptingController:
        def close(self):
            raise KeyboardInterrupt

    output = []
    inputs = iter(["/exit"])

    exit_code = run_console_session(
        InterruptingController(),
        input_fn=lambda prompt: next(inputs),
        output_fn=output.append,
    )

    assert exit_code == 130
    assert output[-1] == (
        "Session closure interrupted; session memory was not saved."
    )


def test_retrieval_trace_contains_count_not_private_memory_text(tmp_path):
    private_text = "The user discussed an extremely private playlist."
    episodic = RecordingEpisodic(
        [{"memory_id": "memory-1", "summary": private_text}]
    )
    with TraceWriter(tmp_path / "traces", "alice", session_id=SESSION) as trace:
        session = controller(
            RecordingLoop(),
            episodic,
            RecordingSessionMemory(),
            trace=trace,
        )
        session.handle("Use the earlier playlist")

    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert events[0]["event"] == "memory_read"
    assert events[0]["result_count"] == 1
    assert private_text not in trace.path.read_text()
