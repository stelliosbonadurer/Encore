"""Safety and composition tests for the complete Part 3 tool set."""

from __future__ import annotations

from encore.approval import ApprovalGate, IdempotencyLedger
from encore.config import RuntimePolicy
from encore.memory_tools import MemoryTools
from encore.session_state import SessionEvidence
from encore.tool_catalog import FINAL_ANSWER_SPEC, IMPLEMENTED_TOOL_SPECS, WRITE_TOOL_SPECS
from encore.tool_suite import ToolSuite
from encore.tools import ReadTools
from encore.write_tools import PLAYLIST_PREFIX, WriteTools

RECORDING = "5b11f4ce-0000-4000-8000-000000000002"
PLAYLIST = "aaaaaaaa-0000-4000-8000-000000000001"


class QueueClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError(f"unexpected network call: {method} {url}")
        return self.responses.pop(0)


def make_writes(client, *, enabled=True, evidence=None, gate=None, ledger=None):
    return WriteTools(
        client,
        "student",
        # Use the offline provider mode throughout these tool tests to prove
        # model selection does not control account-write authorization.
        RuntimePolicy("offline", enabled),
        evidence or SessionEvidence(),
        gate or ApprovalGate(),
        ledger or IdempotencyLedger(),
    )


def approve(gate, tool_name, arguments, *, high_volume_response=""):
    request = gate.prepare(tool_name, arguments)
    return gate.confirm(
        request,
        "yes",
        high_volume_response=high_volume_response,
    )


def test_all_part3_tools_have_schemas_and_all_actions_have_handlers():
    client = QueueClient()
    evidence = SessionEvidence()
    suite = ToolSuite(
        ReadTools(client, "student", evidence),
        make_writes(client, evidence=evidence),
        MemoryTools(object(), object(), object(), "student"),
    )
    required_actions = {
        "get_recent_listens",
        "get_feedback",
        "search_recordings",
        "lookup_artist",
        "get_top_recordings_for_artist",
        "verify_recordings",
        "get_fresh_releases",
        "lb_radio",
        "find_playlists",
        "create_playlist",
        "add_playlist_items",
        "remove_playlist_items",
        "set_feedback",
        "export_playlist",
        "delete_playlist",
        "search_past_sessions",
        "search_music_knowledge",
        "note_preference",
    }
    assert set(suite.registry()) == required_actions
    assert {schema["function"]["name"] for schema in suite.schemas()} == {
        *required_actions,
        "final_answer",
    }
    assert len(IMPLEMENTED_TOOL_SPECS) == 18
    assert FINAL_ANSWER_SPEC.name == "final_answer"


def test_every_account_write_is_marked_as_requiring_approval():
    assert len(WRITE_TOOL_SPECS) == 6
    assert all(spec.requires_approval for spec in WRITE_TOOL_SPECS)
    assert not any(
        spec.requires_approval
        for spec in IMPLEMENTED_TOOL_SPECS
        if spec not in WRITE_TOOL_SPECS
    )


def test_permit_is_bound_to_exact_tool_and_arguments():
    gate = ApprovalGate()
    arguments = {"recording_mbid": RECORDING, "rating": "love"}
    permit = approve(gate, "set_feedback", arguments)
    assert gate.validate(permit, "set_feedback", arguments)
    assert not gate.validate(permit, "set_feedback", {**arguments, "rating": "hate"})
    assert not gate.validate(permit, "delete_playlist", {"playlist_mbid": PLAYLIST})


def test_ambiguous_or_missing_confirmation_never_issues_permit():
    gate = ApprovalGate()
    request = gate.prepare("delete_playlist", {"playlist_mbid": PLAYLIST})
    assert gate.confirm(request, "") is None
    assert gate.confirm(request, "sure") is None
    assert gate.confirm(request, "no") is None


def test_create_preview_shows_exact_prefixed_title():
    request = ApprovalGate().prepare(
        "create_playlist", {"title": "Focus", "recording_mbids": [RECORDING]}
    )
    assert "'[ListenBrainz] Focus'" in request.summary


def test_high_volume_has_no_hard_cap_but_requires_exact_second_phrase():
    gate = ApprovalGate()
    arguments = {"playlist_mbid": PLAYLIST, "recording_mbids": [str(i) for i in range(51)]}
    request = gate.prepare("add_playlist_items", arguments)
    assert request.high_volume
    assert request.high_volume_phrase == "CONFIRM ADD_PLAYLIST_ITEMS 51 TRACKS"
    assert gate.confirm(request, "yes") is None
    permit = gate.confirm(
        request, "yes", high_volume_response="CONFIRM ADD_PLAYLIST_ITEMS 51 TRACKS"
    )
    assert permit is not None


def test_disabled_write_flag_rejects_write_even_with_real_permit_before_network():
    client = QueueClient()
    gate = ApprovalGate()
    arguments = {"recording_mbid": RECORDING, "rating": "love"}
    permit = approve(gate, "set_feedback", arguments)
    result = make_writes(client, enabled=False, gate=gate).set_feedback(
        RECORDING, "love", permit=permit
    )
    assert not result.success and result.error_category == "fatal"
    assert client.calls == []


def test_create_requires_session_verification_before_network():
    client = QueueClient()
    gate = ApprovalGate()
    arguments = {"title": "Focus", "recording_mbids": [RECORDING]}
    permit = approve(gate, "create_playlist", arguments)
    result = make_writes(client, gate=gate).create_playlist(
        "Focus", [RECORDING], permit=permit
    )
    assert not result.success and result.error_category == "resource"
    assert "not verified" in result.error
    assert client.calls == []


def test_create_prefixes_title_posts_private_jspf_and_is_idempotent():
    client = QueueClient(
        (200, {"playlists": [], "playlist_count": 0}, {}),
        (200, {"status": "ok", "playlist_mbid": PLAYLIST}, {}),
    )
    evidence = SessionEvidence({RECORDING})
    gate = ApprovalGate()
    ledger = IdempotencyLedger()
    writes = make_writes(client, evidence=evidence, gate=gate, ledger=ledger)
    arguments = {"title": "Focus", "recording_mbids": [RECORDING]}
    permit = approve(gate, "create_playlist", arguments)

    first = writes.create_playlist("Focus", [RECORDING], permit=permit)
    assert first.success and first.data["title"] == f"{PLAYLIST_PREFIX} Focus"
    post = client.calls[1]
    assert post[0] == "POST" and post[1].endswith("/playlist/create")
    assert post[2]["body"]["playlist"]["extension"]

    # Without a fresh approval, the remembered success is not returned.
    unapproved = writes.create_playlist("Focus", [RECORDING])
    assert not unapproved.success and unapproved.error_category == "resource"
    assert len(client.calls) == 2

    # An approved exact retry is reported as a replay without a second POST.
    retry = writes.create_playlist(
        "Focus", [RECORDING], permit=approve(gate, "create_playlist", arguments)
    )
    assert retry.success and retry.data["replayed"] is True
    assert retry.data["playlist_mbid"] == PLAYLIST
    assert len(client.calls) == 2


def feedback_client(score):
    """Fake feedback service that tracks score so tests can observe real POSTs."""

    class Client:
        def __init__(self):
            self.score = score
            self.posts = 0

        def request(self, method, url, **kwargs):
            if method == "GET":
                return 200, {"feedback": [{"score": self.score}]}, {}
            self.posts += 1
            self.score = kwargs["body"]["score"]
            return 200, {"status": "ok"}, {}

    return Client()


def test_feedback_love_clear_love_sends_every_real_change():
    client = feedback_client(0)
    gate = ApprovalGate()
    writes = make_writes(client, gate=gate)
    for rating in ("love", "clear", "love"):
        arguments = {"recording_mbid": RECORDING, "rating": rating}
        result = writes.set_feedback(
            RECORDING, rating, permit=approve(gate, "set_feedback", arguments)
        )
        assert result.success and "replayed" not in result.data
    assert client.posts == 3 and client.score == 1


def test_disabled_write_flag_never_returns_a_remembered_write_as_success():
    client = feedback_client(0)
    gate = ApprovalGate()
    ledger = IdempotencyLedger()
    arguments = {"recording_mbid": RECORDING, "rating": "love"}
    online = make_writes(client, gate=gate, ledger=ledger)
    assert online.set_feedback(
        RECORDING, "love", permit=approve(gate, "set_feedback", arguments)
    ).success

    read_only = make_writes(client, enabled=False, gate=gate, ledger=ledger)
    result = read_only.set_feedback(
        RECORDING, "love", permit=approve(gate, "set_feedback", arguments)
    )
    assert not result.success and result.error_category == "fatal"


def test_failed_post_invalidates_replay_so_repair_runs_for_real():
    gate = ApprovalGate()
    ledger = IdempotencyLedger()
    arguments = {"recording_mbid": RECORDING, "rating": "love"}
    # First love succeeds; latest write on this recording is now "love".
    first = feedback_client(0)
    assert make_writes(first, gate=gate, ledger=ledger).set_feedback(
        RECORDING, "love", permit=approve(gate, "set_feedback", arguments)
    ).success
    # A later "clear" POST fails with 503: the outcome is unknown.
    failing = QueueClient((200, {"feedback": [{"score": 1}]}, {}), (503, {}, {}))
    clear_args = {"recording_mbid": RECORDING, "rating": "clear"}
    failed = make_writes(failing, gate=gate, ledger=ledger).set_feedback(
        RECORDING, "clear", permit=approve(gate, "set_feedback", clear_args)
    )
    assert not failed.success
    # Repeating "love" must re-check live state rather than replaying old success.
    recheck = QueueClient((200, {"feedback": [{"score": 0}]}, {}), (200, {}, {}))
    result = make_writes(recheck, gate=gate, ledger=ledger).set_feedback(
        RECORDING, "love", permit=approve(gate, "set_feedback", arguments)
    )
    assert result.success and "replayed" not in result.data
    assert recheck.calls[-1][0] == "POST"


def test_delete_then_create_same_title_makes_a_real_new_playlist():
    new_playlist = "bbbbbbbb-0000-4000-8000-000000000002"
    title = f"{PLAYLIST_PREFIX} Focus"
    owned = {
        "playlists": [
            {
                "playlist": {
                    "title": title,
                    "identifier": f"https://listenbrainz.org/playlist/{PLAYLIST}",
                }
            }
        ],
        "playlist_count": 1,
    }
    client = QueueClient(
        (200, {"playlists": [], "playlist_count": 0}, {}),  # create preflight
        (200, {"playlist_mbid": PLAYLIST}, {}),  # create POST
        (200, owned, {}),  # delete ownership proof
        (200, {"playlist": {"track": []}}, {}),  # delete state fetch
        (200, {}, {}),  # delete POST
        (200, {"playlists": [], "playlist_count": 0}, {}),  # create preflight
        (200, {"playlist_mbid": new_playlist}, {}),  # second create POST
    )
    gate = ApprovalGate()
    writes = make_writes(
        client, evidence=SessionEvidence({RECORDING}), gate=gate
    )
    create_args = {"title": "Focus", "recording_mbids": [RECORDING]}
    writes.create_playlist(
        "Focus", [RECORDING], permit=approve(gate, "create_playlist", create_args)
    )
    delete_args = {"playlist_mbid": PLAYLIST}
    assert writes.delete_playlist(
        PLAYLIST, permit=approve(gate, "delete_playlist", delete_args)
    ).success
    again = writes.create_playlist(
        "Focus", [RECORDING], permit=approve(gate, "create_playlist", create_args)
    )
    assert again.success and again.data["playlist_mbid"] == new_playlist
    assert "replayed" not in again.data


def test_delete_refuses_owned_but_unmarked_playlist_without_delete_post():
    owned = {
        "title": "Personal playlist",
        "identifier": f"https://listenbrainz.org/playlist/{PLAYLIST}",
    }
    client = QueueClient(
        (200, {"playlists": [{"playlist": owned}], "playlist_count": 1}, {}),
        (200, {"playlist": {"track": []}}, {}),
    )
    gate = ApprovalGate()
    arguments = {"playlist_mbid": PLAYLIST}
    permit = approve(gate, "delete_playlist", arguments)
    result = make_writes(client, gate=gate).delete_playlist(PLAYLIST, permit=permit)
    assert not result.success and result.error_category == "fatal"
    assert "lacks" in result.error
    assert all(not call[1].endswith("/delete") for call in client.calls)


def owned_page(title=f"{PLAYLIST_PREFIX} Test"):
    """One-page ownership proof used by write preflight tests."""
    return {
        "playlists": [
            {
                "playlist": {
                    "title": title,
                    "identifier": f"https://listenbrainz.org/playlist/{PLAYLIST}",
                }
            }
        ],
        "playlist_count": 1,
    }


def test_add_playlist_items_fetches_state_then_posts_verified_tracks():
    client = QueueClient(
        (200, owned_page(), {}),
        (200, {"playlist": {"track": []}}, {}),
        (200, {"status": "ok"}, {}),
    )
    gate = ApprovalGate()
    evidence = SessionEvidence({RECORDING})
    arguments = {"playlist_mbid": PLAYLIST, "recording_mbids": [RECORDING]}
    permit = approve(gate, "add_playlist_items", arguments)
    result = make_writes(client, gate=gate, evidence=evidence).add_playlist_items(
        PLAYLIST, [RECORDING], permit=permit
    )
    assert result.success and result.data["added"] == 1
    assert client.calls[-1][1].endswith(f"/playlist/{PLAYLIST}/item/add")


def test_remove_playlist_items_validates_range_then_posts_position():
    tracks = [{"identifier": f"https://musicbrainz.org/recording/{RECORDING}"}] * 2
    client = QueueClient(
        (200, owned_page(), {}),
        (200, {"playlist": {"track": tracks}}, {}),
        (200, {"status": "ok"}, {}),
    )
    gate = ApprovalGate()
    arguments = {"playlist_mbid": PLAYLIST, "index": 1, "count": 1}
    permit = approve(gate, "remove_playlist_items", arguments)
    result = make_writes(client, gate=gate).remove_playlist_items(
        PLAYLIST, 1, 1, permit=permit
    )
    assert result.success and result.data["removed"] == 1
    assert client.calls[-1][2]["body"] == {"index": 1, "count": 1}


def test_set_feedback_reads_current_value_before_posting_change():
    client = QueueClient(
        (200, {"feedback": [{"recording_mbid": RECORDING, "score": 0}]}, {}),
        (200, {"status": "ok"}, {}),
    )
    gate = ApprovalGate()
    arguments = {"recording_mbid": RECORDING, "rating": "love"}
    permit = approve(gate, "set_feedback", arguments)
    result = make_writes(client, gate=gate).set_feedback(
        RECORDING, "love", permit=permit
    )
    assert result.success and result.data["changed"] is True
    assert client.calls[-1][2]["body"]["score"] == 1


def test_export_playlist_checks_ownership_and_returns_manual_folder_note():
    client = QueueClient(
        (200, owned_page(), {}),
        (200, {"playlist": {"track": []}}, {}),
        (200, {"status": "ok"}, {}),
    )
    gate = ApprovalGate()
    arguments = {"playlist_mbid": PLAYLIST, "service": "spotify"}
    permit = approve(gate, "export_playlist", arguments)
    result = make_writes(client, gate=gate).export_playlist(
        PLAYLIST, "spotify", permit=permit
    )
    assert result.success and "manually" in result.data["folder_note"]
    assert client.calls[-1][1].endswith(f"/playlist/{PLAYLIST}/export/spotify")


def test_export_failure_preserves_bounded_service_detail_for_diagnosis():
    client = QueueClient(
        (200, owned_page(), {}),
        (200, {"playlist": {"track": []}}, {}),
        (500, {"error": "Spotify mapping failed for every recording."}, {}),
    )
    gate = ApprovalGate()
    arguments = {"playlist_mbid": PLAYLIST, "service": "spotify"}
    permit = approve(gate, "export_playlist", arguments)

    result = make_writes(client, gate=gate).export_playlist(
        PLAYLIST, "spotify", permit=permit
    )

    assert not result.success and result.error_category == "transient"
    assert "Spotify mapping failed" in result.error


def test_delete_marked_owned_playlist_calls_only_listenbrainz_delete():
    client = QueueClient(
        (200, owned_page(), {}),
        (200, {"playlist": {"track": []}}, {}),
        (200, {"status": "ok"}, {}),
    )
    gate = ApprovalGate()
    arguments = {"playlist_mbid": PLAYLIST}
    permit = approve(gate, "delete_playlist", arguments)
    result = make_writes(client, gate=gate).delete_playlist(PLAYLIST, permit=permit)
    assert result.success and result.data["deleted"] is True
    assert client.calls[-1][1].endswith(f"/playlist/{PLAYLIST}/delete")


def test_memory_tools_keep_users_separate_and_memory_types_distinct():
    class Episodic:
        sessions = [
            {"user_id": "student", "summary": "Made a quiet focus playlist", "topics": ["focus"]},
            {"user_id": "other", "summary": "Made a loud focus playlist", "topics": ["focus"]},
        ]

        def search_sessions(self, user_id, query, limit):
            return [item for item in self.sessions if item["user_id"] == user_id][:limit]

    class Semantic:
        knowledge = [
            {"source": "ListenBrainz docs", "title": "Spotify", "text": "Connect Spotify in settings."}
        ]

        def search_knowledge(self, query, limit):
            return self.knowledge[:limit]

    class Personal:
        def __init__(self):
            self.preferences = {}

        def note_preference(self, user_id, preference, evidence, *, scope):
            record = {"preference": preference, "evidence": evidence, "scope": scope}
            self.preferences.setdefault(user_id, []).append(record)
            return record

    personal = Personal()
    tools = MemoryTools(personal, Episodic(), Semantic(), "student")
    past = tools.search_past_sessions("focus")
    docs = tools.search_music_knowledge("connect Spotify")
    preference = tools.note_preference("prefers instrumental focus music", "said directly")
    assert past.success and len(past.data["sessions"]) == 1
    assert docs.success and docs.data["chunks"][0]["source"] == "ListenBrainz docs"
    assert preference.success
    assert personal.preferences["student"][0]["preference"].startswith("prefers")
