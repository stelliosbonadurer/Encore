"""Offline tests for code-owned hot state and session-scoped preferences."""

from __future__ import annotations

import json

from encore.agent_loop import AgentLoop
from encore.contracts import fail, ok
from encore.hot_state import MAX_LAST_RECORDINGS, updates_for_result
from encore.memory_store import ArchiveStore, PersonalMemoryStore
from encore.router import RouteDecision
from encore.tool_executor import ExecutionReport
from encore.tracing import TraceWriter
from encore.working_memory import WorkingMemoryService

from test_agent_loop import (
    SESSION,
    FakeExecutor,
    FakeModel,
    compression_message,
    final_message,
    report,
    tool_message,
)

PLAYLIST = "aaaaaaaa-0000-4000-8000-000000000001"
OTHER_PLAYLIST = "bbbbbbbb-0000-4000-8000-000000000002"


def recording(index):
    return {
        "recording_mbid": f"5b11f4ce-0000-4000-8000-{index:012d}",
        "title": f"Song {index}",
        "artist": "Band",
    }


def service(personal, tmp_path, **kwargs):
    return WorkingMemoryService(
        FakeModel(),
        personal,
        ArchiveStore(tmp_path / "archives"),
        "alice",
        SESSION,
        **kwargs,
    )


# --- pure mapping -------------------------------------------------------------


def test_recording_lists_become_ordered_positions_for_the_second_song():
    result = ok({"recordings": [recording(1), recording(2), recording(3)]})
    updates = updates_for_result("search_recordings", result)
    second = updates["last_recordings"][1]
    assert second["position"] == 2 and second["title"] == "Song 2"
    assert second["recording_mbid"].endswith("000000000002")


def test_listens_use_track_field_and_skip_unmapped_and_cap_length():
    listens = [{"track": "Unmapped", "artist": "X", "recording_mbid": None}]
    listens += [
        {"track": f"Song {i}", "artist": "Band", "recording_mbid": recording(i)["recording_mbid"]}
        for i in range(1, 15)
    ]
    updates = updates_for_result("get_recent_listens", ok({"listens": listens}))
    kept = updates["last_recordings"]
    assert len(kept) == MAX_LAST_RECORDINGS
    assert kept[0] == {
        "position": 1,
        "title": "Song 1",
        "artist": "Band",
        "recording_mbid": recording(1)["recording_mbid"],
    }


def test_artist_and_playlist_writes_set_their_keys():
    artist = updates_for_result(
        "lookup_artist", ok({"artist_mbid": "a-1", "name": "Khruangbin"})
    )
    assert artist == {"last_artist": {"name": "Khruangbin", "artist_mbid": "a-1"}}
    for tool in ("create_playlist", "add_playlist_items", "export_playlist"):
        updates = updates_for_result(
            tool, ok({"playlist_mbid": PLAYLIST, "title": "[ListenBrainz] Run"})
        )
        assert updates["active_playlist"]["playlist_mbid"] == PLAYLIST


def test_find_playlists_sets_active_only_for_one_unambiguous_match():
    one = ok({"playlists": [{"playlist_mbid": PLAYLIST, "title": "Running"}]})
    many = ok(
        {
            "playlists": [
                {"playlist_mbid": PLAYLIST, "title": "Running"},
                {"playlist_mbid": OTHER_PLAYLIST, "title": "Running 2"},
            ]
        }
    )
    assert updates_for_result("find_playlists", one)["active_playlist"]["title"] == "Running"
    assert updates_for_result("find_playlists", many) == {}


def test_delete_clears_active_playlist_only_when_it_was_the_target():
    active = {"playlist_mbid": PLAYLIST, "title": "Run"}
    same = ok({"playlist_mbid": PLAYLIST, "deleted": True})
    other = ok({"playlist_mbid": OTHER_PLAYLIST, "deleted": True})
    assert updates_for_result("delete_playlist", same, active) == {"active_playlist": None}
    assert updates_for_result("delete_playlist", other, active) == {}


def test_failed_results_change_nothing():
    assert updates_for_result("search_recordings", fail("resource", "none")) == {}


# --- persistence through the working-memory service ---------------------------


def test_declined_write_is_remembered_then_cleared_by_an_approved_write(tmp_path):
    with PersonalMemoryStore(tmp_path / "p.sqlite3") as personal:
        working = service(personal, tmp_path)
        working.apply_observation(
            "delete_playlist",
            fail("resource", "User declined the proposed account write."),
            approval_requested=True,
            approved=False,
            approval_summary=f"Permanently delete playlist {PLAYLIST}.",
        )
        pending = personal.get_hot("alice", "pending_approval")
        assert pending == {
            "tool": "delete_playlist",
            "preview": f"Permanently delete playlist {PLAYLIST}.",
            "status": "declined",
        }
        working.apply_observation(
            "set_feedback",
            ok({"recording_mbid": "r", "rating": "love", "changed": True}),
            approval_requested=True,
            approved=True,
        )
        assert personal.get_hot("alice", "pending_approval") is None


def test_reset_session_state_clears_hot_keys_but_keeps_preferences(tmp_path):
    with PersonalMemoryStore(tmp_path / "p.sqlite3") as personal:
        personal.set_hot("alice", "active_playlist", {"title": "Old session"})
        personal.set_hot("alice", "last_recordings", [{"title": "Old"}])
        personal.set_hot("bob", "active_playlist", {"title": "Bob's"})
        personal.note_preference("alice", "No Christmas music", "user said it")
        removed = service(personal, tmp_path).reset_session_state()
        assert removed == 2
        assert personal.get_hot("alice", "active_playlist") is None
        assert personal.get_hot("bob", "active_playlist")["title"] == "Bob's"
        assert len(personal.list_preferences("alice")) == 1


def test_only_long_term_and_current_session_preferences_are_front_loaded(tmp_path):
    clock = [1_000.0]
    with PersonalMemoryStore(tmp_path / "p.sqlite3", now=lambda: clock[0]) as personal:
        # Session 1 (t=1000): one passing note, one later promoted to long-term.
        personal.note_preference("alice", "I only want metal", "session 1")
        standing = personal.note_preference("alice", "No Christmas music", "session 1")
        personal.promote_preference("alice", standing["id"])
        # Session 2 starts at t=5000 and notes a new preference.
        clock[0] = 5_000.0
        working = service(personal, tmp_path, session_started_at=5_000.0)
        personal.note_preference("alice", "Folk today", "session 2")
        texts = {p["preference"] for p in working.protected_state()["preferences"]}

    assert texts == {"No Christmas music", "Folk today"}
    # The old note is retained for retention/episodic use, just not front-loaded.


# --- loop integration ---------------------------------------------------------


def test_active_playlist_survives_compression_and_is_traced(tmp_path):
    created = ok(
        {"playlist_mbid": PLAYLIST, "title": "[ListenBrainz] Focus", "created": True}
    )
    actions = [tool_message("create_playlist", {"title": "Focus"}, "call-0")]
    actions += [
        tool_message("search_recordings", {"title": f"T{i}"}, f"call-{i}")
        for i in range(1, 8)
    ]
    model = FakeModel(
        *actions,
        compression_message(),
        final_message(call_id="cheap-final"),
        final_message(answer="Built it."),
    )
    executor = FakeExecutor(
        ExecutionReport(created, 1, (), approval_requested=True, approved=True),
        *[report(ok({"recordings": [{"title": "x" * 300}]})) for _ in range(7)],
    )
    with PersonalMemoryStore(tmp_path / "p.sqlite3") as personal:
        with TraceWriter(tmp_path / "traces", "alice", session_id=SESSION) as trace:
            working = WorkingMemoryService(
                model,
                personal,
                ArchiveStore(tmp_path / "archives"),
                "alice",
                SESSION,
                trace=trace,
            )
            AgentLoop(
                model,
                executor,
                "RULES",
                route=lambda request, client: RouteDecision(("catalog", "library")),
                working_memory=working,
                trace=trace,
            ).run("Build a focus playlist")
        events = [json.loads(line) for line in trace.path.read_text().splitlines()]

    final_context = model.calls[-1][0]
    key_facts = final_context[2]["content"]
    assert key_facts.startswith("<UNTRUSTED_KEY_FACTS>")
    # The playlist being built is still exact after older turns were compressed.
    assert PLAYLIST in key_facts and "[ListenBrainz] Focus" in key_facts
    assert any(event["event"] == "compress" for event in events)
    writes = [
        event
        for event in events
        if event["event"] == "memory_write" and event.get("store") == "hot_state"
    ]
    assert writes[0]["operation"] == "update:active_playlist"
    # Values are not traced, only key names.
    assert all(PLAYLIST not in json.dumps(event) for event in writes)
