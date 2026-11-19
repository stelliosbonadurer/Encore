"""Fast offline checks for core contracts and API behavior."""

from __future__ import annotations

import json

import pytest

from encore.api_client import LB_API, MB_API, ApiClient
from encore.config import RuntimePolicy, Settings
from encore.contracts import classify_status, fail, ok
from encore.router import parse_router_message
from encore.tools import ReadTools


def settings(**overrides):
    values = {
        "listenbrainz_token": "secret",
        "listenbrainz_user": "student",
        "encore_user_id": "student",
        "contact_email": "student@example.edu",
        "openrouter_api_key": None,
        "openrouter_url": "https://openrouter.example/chat",
        "openrouter_model_cheap": None,
        "openrouter_model_expensive": None,
        "local_llm_url": "http://localhost/chat",
        "local_llm_model_cheap": None,
        "local_llm_model_expensive": None,
    }
    values.update(overrides)
    return Settings(**values)


class FakeResponse:
    def __init__(self, body, *, status=200, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = ""

    def json(self):
        return self._body


def test_tool_result_contract():
    assert json.loads(ok({"x": 1}).to_json())["success"] is True
    result = json.loads(fail("input", "bad value").to_json())
    assert result["error_category"] == "input" and result["data"] is None


@pytest.mark.parametrize(
    "status, expected",
    [(0, "transient"), (429, "transient"), (503, "transient"),
     (401, "fatal"), (403, "fatal"), (404, "resource"), (400, "input")],
)
def test_status_classification(status, expected):
    assert classify_status(status) == expected


def test_model_provider_mode_is_independent_from_write_authorization():
    assert RuntimePolicy("offline", True).writes_enabled is True
    assert RuntimePolicy("offline", False).writes_enabled is False
    assert RuntimePolicy("online", False).writes_enabled is False
    assert RuntimePolicy("online", True).writes_enabled is True


def test_api_client_rejects_arbitrary_hosts():
    client = ApiClient(settings())
    with pytest.raises(ValueError, match="only permits"):
        client.request("GET", "https://example.com/private")


def test_api_client_adds_musicbrainz_json_and_never_sends_token_by_default():
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return FakeResponse({})

    client = ApiClient(settings(), request=request, monotonic=lambda: 100.0)
    client.request("GET", f"{MB_API}/recording", params={"query": "Black"})
    kwargs = calls[0][2]
    assert kwargs["params"]["fmt"] == "json"
    assert "Authorization" not in kwargs["headers"]


def test_listenbrainz_rate_limit_is_honored_on_next_request():
    responses = [
        FakeResponse({}, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset-In": "7"}),
        FakeResponse({}),
    ]
    now = [100.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    client = ApiClient(
        settings(),
        request=lambda *args, **kwargs: responses.pop(0),
        monotonic=lambda: now[0],
        sleep=sleep,
    )
    client.request("GET", f"{LB_API}/user/student/listens")
    client.request("GET", f"{LB_API}/user/student/listens")
    assert sleeps == [7.0]


def test_read_tool_validates_before_network():
    class NeverCall:
        def request(self, *args, **kwargs):
            raise AssertionError("network should not be called")

    tools = ReadTools(NeverCall(), "student")
    assert tools.get_recent_listens(0).error_category == "input"
    assert tools.search_recordings("  ").error_category == "input"


def test_read_tool_trims_recent_listens():
    class FakeClient:
        def request(self, *args, **kwargs):
            return 200, {"payload": {"listens": [{
                "listened_at": 123,
                "track_metadata": {"artist_name": "Artist", "track_name": "Song"},
            }]}}, {}

    result = ReadTools(FakeClient(), "student").get_recent_listens(1)
    assert result.success
    assert result.data["listens"] == [{
        "listened_at": 123,
        "artist": "Artist",
        "track": "Song",
        "recording_mbid": None,
    }]


def test_router_parses_forced_validated_tool_call():
    message = {
        "tool_calls": [{
            "function": {
                "name": "classify_intent",
                "arguments": json.dumps({
                    "domains": ["listening", "catalog"],
                }),
            }
        }]
    }
    decision = parse_router_message(message)
    assert decision.domains == ("listening", "catalog")
    assert "get_recent_listens" in decision.tools
    assert "search_recordings" in decision.tools


def test_router_rejects_unknown_domain():
    message = {
        "tool_calls": [{
            "function": {
                "name": "classify_intent",
                "arguments": json.dumps({"domains": ["filesystem"]}),
            }
        }]
    }
    with pytest.raises(Exception):
        parse_router_message(message)
