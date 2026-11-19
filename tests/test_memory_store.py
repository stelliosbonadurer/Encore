"""Offline tests for deletion-first SQLite and archived working memory."""

from __future__ import annotations

import json
import stat

import pytest

from encore.cli import main, parse_args
from encore.memory_store import (
    ArchiveStore,
    MemoryPrivacyManager,
    PersonalMemoryStore,
    user_storage_key,
)
from encore.memory_tools import MemoryTools


SESSION = "11111111-1111-4111-8111-111111111111"


def test_hot_state_is_user_scoped_persistent_and_ttl_aware(tmp_path):
    now = [100.0]
    path = tmp_path / "memory" / "personal.sqlite3"
    store = PersonalMemoryStore(path, now=lambda: now[0])
    store.set_hot("alice", "active_playlist", {"title": "Focus"})
    store.set_hot("bob", "active_playlist", {"title": "Running"})
    store.set_hot("alice", "pending_approval", "export", ttl_s=10)

    assert store.get_hot("alice", "active_playlist")["title"] == "Focus"
    assert store.get_hot("bob", "active_playlist")["title"] == "Running"
    assert store.get_hot("alice", "pending_approval") == "export"
    store.close()

    # A new connection proves exact hot state survives process restart.
    store = PersonalMemoryStore(path, now=lambda: now[0])
    assert store.get_hot("alice", "active_playlist")["title"] == "Focus"
    now[0] = 111.0
    assert store.get_hot("alice", "pending_approval", "missing") == "missing"
    assert store.count_user("alice").hot_state == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    store.close()


def test_hot_state_rejects_invalid_keys_ttls_and_nonportable_json(tmp_path):
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as store:
        with pytest.raises(ValueError, match="key"):
            store.set_hot("alice", " ", {})
        with pytest.raises(ValueError, match="positive"):
            store.set_hot("alice", "pending", {}, ttl_s=0)
        with pytest.raises(ValueError):
            store.set_hot("alice", "bad_number", {"value": float("nan")})


def test_preferences_deduplicate_without_silent_long_term_promotion(tmp_path):
    with PersonalMemoryStore(tmp_path / "personal.sqlite3", now=lambda: 10.0) as store:
        first = store.note_preference(
            "alice",
            "Prefers   instrumental focus music",
            "said directly",
        )
        second = store.note_preference(
            "alice",
            "prefers instrumental focus music",
            "repeated today",
        )

        assert first["id"] == second["id"]
        assert second["evidence"] == "repeated today"
        assert second["scope"] == "session"
        assert len(store.list_preferences("alice")) == 1
        # Promotion is a separate explicit retention operation.
        assert store.promote_preference("alice", first["id"]) is True
        assert store.list_preferences("alice")[0]["scope"] == "long_term"


def test_exact_preference_lookup_is_normalized_and_user_scoped(tmp_path):
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as store:
        stored = store.note_preference(
            "alice",
            "Prefers   instrumental focus music",
            "user stated it",
        )
        store.note_preference("bob", "Prefers instrumental focus music", "bob said it")

        found = store.find_preference("alice", "prefers instrumental focus music")

        assert found is not None and found["id"] == stored["id"]
        assert found["evidence"] == "user stated it"
        assert store.find_preference("alice", "Prefers vocal music") is None


def test_memory_tool_writes_session_preference_to_sqlite(tmp_path):
    with PersonalMemoryStore(tmp_path / "personal.sqlite3") as store:
        tools = MemoryTools(store, object(), object(), "alice")
        result = tools.note_preference("No Christmas music", "user stated it")

        assert result.success
        rows = store.list_preferences("alice")
        assert rows[0]["preference"] == "No Christmas music"
        assert rows[0]["scope"] == "session"


def test_archive_is_hashed_private_separate_and_append_only(tmp_path):
    archive = ArchiveStore(tmp_path / "archives", now=lambda: 123.0)
    count = archive.archive_messages(
        "alice@example.edu",
        SESSION,
        [
            {"role": "user", "content": "make focus music"},
            {"role": "assistant", "content": "I can help."},
        ],
    )
    archive.archive_messages(
        "alice@example.edu",
        SESSION,
        [{"role": "tool", "content": '{"success":true}'}],
    )

    path = (
        tmp_path
        / "archives"
        / user_storage_key("alice@example.edu")
        / f"{SESSION}.jsonl"
    )
    lines = path.read_text().splitlines()
    assert count == 2 and len(lines) == 3
    assert json.loads(lines[0])["message"]["role"] == "user"
    assert "alice@example.edu" not in str(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_archive_rejects_system_messages_and_invalid_session_names(tmp_path):
    archive = ArchiveStore(tmp_path / "archives")
    with pytest.raises(ValueError, match="may not be archived"):
        archive.archive_messages(
            "alice",
            SESSION,
            [{"role": "system", "content": "private procedural prompt"}],
        )
    with pytest.raises(ValueError):
        archive.archive_messages(
            "alice",
            "../../escape",
            [{"role": "user", "content": "hello"}],
        )


def test_forget_removes_only_target_user_from_all_current_stores(tmp_path):
    class EmptyEpisodic:
        def forget_user(self, user_id):
            return 0

    store = PersonalMemoryStore(tmp_path / "personal.sqlite3")
    archives = ArchiveStore(tmp_path / "archives")
    for user in ("alice", "bob"):
        store.set_hot(user, "active_playlist", {"title": user})
        store.note_preference(user, f"preference for {user}", "direct")
        archives.archive_messages(
            user,
            SESSION,
            [{"role": "user", "content": f"private {user} turn"}],
        )

    report = MemoryPrivacyManager(store, archives, EmptyEpisodic()).forget_user("alice")

    assert report.hot_state == 1 and report.preferences == 1
    assert report.archived_messages == 1 and report.archive_files == 1
    assert report.episodic_summaries == 0 and report.total_items == 3
    assert store.count_user("alice").hot_state == 0
    assert store.list_preferences("alice") == []
    assert archives.count_user("alice").messages == 0
    # Bob's similarly shaped records prove deletion is properly scoped.
    assert store.get_hot("bob", "active_playlist")["title"] == "bob"
    assert len(store.list_preferences("bob")) == 1
    assert archives.count_user("bob").messages == 1
    store.close()


def test_cli_forget_is_local_reports_counts_and_needs_no_model_or_api(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ENCORE_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("ENCORE_USER_ID", "cli-user")
    store = PersonalMemoryStore(tmp_path / "memory" / "personal.sqlite3")
    store.set_hot("cli-user", "last_recording", {"title": "Black"})
    store.note_preference("cli-user", "likes Pearl Jam", "direct")
    store.close()
    ArchiveStore(tmp_path / "memory" / "archives").archive_messages(
        "cli-user",
        SESSION,
        [{"role": "user", "content": "hello"}],
    )

    assert main(["online", "--forget"]) == 0
    output = capsys.readouterr().out
    assert "hot state: 1" in output
    assert "preferences: 1" in output
    assert "archived messages: 1" in output
    assert "ListenBrainz account data was not changed" in output
    with PersonalMemoryStore(tmp_path / "memory" / "personal.sqlite3") as verify:
        assert verify.count_user("cli-user").hot_state == 0
        assert verify.list_preferences("cli-user") == []


def test_cli_parser_exposes_forget_as_explicit_boolean():
    assert parse_args(["online", "--forget"]).forget is True
    assert parse_args(["offline"]).forget is False


def test_cli_parser_supports_timezone_aware_retention_simulation():
    args = parse_args(
        ["offline", "--retention-plan", "--now", "2027-02-01T12:30:00-07:00"]
    )

    assert args.retention_plan is True
    assert args.apply_retention is False
    assert args.now.isoformat() == "2027-02-01T19:30:00+00:00"


def test_cli_parser_rejects_naive_now_and_conflicting_memory_actions():
    with pytest.raises(SystemExit):
        parse_args(["offline", "--now", "2027-02-01T12:30:00"])
    with pytest.raises(SystemExit):
        parse_args(["offline", "--forget", "--retention-plan"])
    with pytest.raises(SystemExit):
        parse_args(["offline", "--chat", "--forget"])


def test_cli_parser_requires_explicit_chat_flag_for_interactive_session():
    assert parse_args(["offline", "--chat"]).chat is True
    assert parse_args(["offline"]).chat is False
