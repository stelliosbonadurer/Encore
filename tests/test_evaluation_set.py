"""Structural checks for the frozen 30-request agent evaluation set."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from scripts.run_agent_evaluation import _assess_playlist_requirements


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SET = ROOT / "evaluation" / "agent_requests.json"
VALID_STATUSES = {
    "answered",
    "answered_low_confidence",
    "stuck",
    "budget_exceeded",
    "policy_halt",
}


def test_agent_evaluation_set_is_frozen_balanced_and_predefined():
    payload = json.loads(EVALUATION_SET.read_text(encoding="utf-8"))
    requests = payload["requests"]

    assert payload["frozen_at"] == "2026-11-02"
    assert len(requests) == 30
    assert Counter(item["category"] for item in requests) == {
        "happy": 10,
        "boundary": 10,
        "failure": 10,
    }
    assert len({item["request_id"] for item in requests}) == 30

    for item in requests:
        assert item["request"].strip()
        assert item["expected"]["status"] in VALID_STATUSES
        assert item["expected"]["outcome"].strip()
        assert isinstance(item["expected"]["required_tools"], list)
        assert isinstance(item["expected"]["forbidden_tools"], list)


def test_required_fault_injection_fixtures_are_valid_and_referenced():
    payload = json.loads(EVALUATION_SET.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    fixture_paths = sorted((ROOT / "evaluation" / "injections").glob("*.json"))

    assert len(fixture_paths) == 3
    for path in fixture_paths:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        assert path.relative_to(ROOT).as_posix() in serialized
        assert isinstance(fixture["seed"], int)
        if "transient_rate" in fixture:
            assert 0.0 <= fixture["transient_rate"] <= 1.0
        if "max_steps" in fixture:
            assert fixture["max_steps"] >= 1


def test_live_playlist_setup_assessment_detects_missing_frozen_state():
    playlists = [
        {"title": "[ListenBrainz] Test Playlist", "encore_managed": True},
        {"title": "Fall2026", "encore_managed": False},
    ]

    assert _assess_playlist_requirements("F03", playlists)[0] is True
    assert _assess_playlist_requirements("H09", playlists)[0] is False
    assert _assess_playlist_requirements("B06", playlists)[0] is False
    assert _assess_playlist_requirements("F10", playlists)[0] is False
