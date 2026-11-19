"""Offline contract tests for the read tools and enum router."""

from __future__ import annotations

import json

import pytest

from encore.api_client import LB_API, MB_API, SLOW_READ_TIMEOUT_S
from encore.model_client import ModelResponse
from encore.router import (
    CLASSIFIER_TOOL,
    IMPLEMENTED_NAMES,
    fallback_decision,
    parse_router_message,
    route_request,
)
from encore.tool_catalog import FINAL_ANSWER_SPEC, IMPLEMENTED_TOOL_SPECS, READ_TOOL_SPECS
from encore.tools import ReadTools

ARTIST_MBID = "83b9cbe7-9857-49e2-ab8e-b57b01038103"
RECORDING_MBID = "5b11f4ce-0000-4000-8000-000000000002"
PLAYLIST_MBID = "aaaaaaaa-0000-4000-8000-000000000001"


class QueueClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def router_message(domains):
    return {
        "tool_calls": [
            {
                "function": {
                    "name": "classify_intent",
                    "arguments": json.dumps({"domains": domains}),
                }
            }
        ]
    }


class RouterModel:
    def __init__(self, domains):
        self.domains = domains

    def chat(self, messages, **kwargs):
        return ModelResponse(router_message(self.domains), "router-model", None, None)


def test_router_repairs_measured_memory_and_clarification_misroutes():
    terminology = route_request(
        "In music terminology, what is a deep cut?",
        RouterModel(["none"]),
    )
    vague = route_request(
        "Recommend something upbeat for later.",
        RouterModel(["discovery"]),
    )

    assert terminology.domains == ("memory",)
    assert "search_music_knowledge" in terminology.tools
    assert vague.domains == ("none",) and vague.tools == ()


def test_catalog_and_registry_have_exactly_the_same_implemented_names():
    registry = ReadTools(QueueClient(), "student").registry()
    assert tuple(registry) == tuple(spec.name for spec in READ_TOOL_SPECS)
    assert len(registry) == 9
    assert all(callable(handler) for handler in registry.values())


def test_every_description_contains_required_guidance_and_closed_schema():
    for spec in (*IMPLEMENTED_TOOL_SPECS, FINAL_ANSWER_SPEC):
        assert "Prefer" in spec.description
        assert "Do not use" in spec.description
        assert spec.parameters["additionalProperties"] is False


def test_bounded_values_are_enums_in_tool_schemas():
    by_name = {spec.name: spec for spec in IMPLEMENTED_TOOL_SPECS}
    assert by_name["get_feedback"].parameters["properties"]["rating"]["enum"] == [
        "love",
        "hate",
    ]
    assert by_name["lb_radio"].parameters["properties"]["mode"]["enum"] == [
        "easy",
        "medium",
        "hard",
    ]


def test_get_feedback_maps_rating_and_trims_result():
    client = QueueClient(
        (200, {"feedback": [{"recording_mbid": RECORDING_MBID, "score": 1}], "total_count": 1}, {})
    )
    result = ReadTools(client, "student").get_feedback("love", 3)
    assert result.success and result.data["feedback"][0]["rating"] == "love"
    assert client.calls == [
        (
            "GET",
            f"{LB_API}/feedback/user/student/get-feedback",
            {"params": {"score": 1, "count": 3}, "timeout": SLOW_READ_TIMEOUT_S},
        )
    ]


def test_lookup_artist_name_searches_then_fetches_relationships():
    client = QueueClient(
        (200, {"artists": [{"id": ARTIST_MBID}]}, {}),
        (
            200,
            {
                "id": ARTIST_MBID,
                "name": "Pearl Jam",
                "type": "Group",
                "relations": [
                    {
                        "type": "member of band",
                        "direction": "backward",
                        "artist": {"id": "b95ce3ff-3d05-4e87-9e01-c97b66af13d4", "name": "Eddie Vedder"},
                    }
                ],
            },
            {},
        ),
    )
    result = ReadTools(client, "student").lookup_artist("Pearl Jam")
    assert result.success and result.data["relationships"][0]["artist"] == "Eddie Vedder"
    assert client.calls[0][1] == f"{MB_API}/artist"
    assert client.calls[1][1] == f"{MB_API}/artist/{ARTIST_MBID}"


def test_popularity_uses_auth_and_trims_to_limit():
    item = {
        "recording_mbid": RECORDING_MBID,
        "recording_name": "Black",
        "artist_name": "Pearl Jam",
        "length": 343000,
        "total_listen_count": 100,
        "total_user_count": 20,
    }
    client = QueueClient((200, [item, item], {}))
    result = ReadTools(client, "student").get_top_recordings_for_artist(ARTIST_MBID, 1)
    assert result.success and len(result.data["recordings"]) == 1
    assert client.calls[0][2]["auth"] is True


def test_verify_recordings_is_all_or_nothing():
    missing = "00000000-0000-4000-8000-000000000000"
    client = QueueClient(
        (200, {"id": RECORDING_MBID, "title": "Black", "artist-credit": [{"name": "Pearl Jam"}]}, {}),
        (404, {"error": "not found"}, {}),
    )
    result = ReadTools(client, "student").verify_recordings([RECORDING_MBID, missing])
    assert not result.success and result.error_category == "resource"
    assert missing in result.error


def test_fresh_releases_and_radio_trim_service_payloads():
    fresh = {
        "payload": {
            "releases": [
                {
                    "release_name": "New Album",
                    "artist_credit_name": "Artist",
                    "release_date": "2026-10-01",
                    "release_mbid": "bbbbbbbb-0000-4000-8000-000000000001",
                    "release_group_mbid": "cccccccc-0000-4000-8000-000000000001",
                    "release_group_primary_type": "Album",
                }
            ]
        }
    }
    radio = {
        "payload": {
            "jspf": {
                "playlist": {
                    "title": "Radio",
                    "track": [
                        {
                            "identifier": f"https://musicbrainz.org/recording/{RECORDING_MBID}",
                            "title": "Black",
                            "creator": "Pearl Jam",
                        }
                    ],
                }
            }
        }
    }
    client = QueueClient((200, fresh, {}), (200, radio, {}))
    tools = ReadTools(client, "student")
    assert tools.get_fresh_releases(30).data["releases"][0]["release"] == "New Album"
    radio_result = tools.lb_radio("artist:(Pearl Jam)", "easy")
    assert radio_result.data["tracks"][0]["recording_mbid"] == RECORDING_MBID
    assert "verify_recordings once" in radio_result.data["next_action"]
    assert client.calls[1][2]["auth"] is True


def test_radio_rejects_unexpected_success_shape_without_crashing():
    client = QueueClient((200, ["not", "a", "payload"], {}))

    result = ReadTools(client, "student").lb_radio("tag:(bluegrass)", "easy")

    assert not result.success and result.error_category == "resource"
    assert result.error == "ListenBrainz radio returned an unexpected response shape."


def test_radio_preserves_only_bounded_actionable_input_error():
    detail = "Invalid prompt expression " + ("x" * 500)
    client = QueueClient((400, {"error": detail, "private": "do not copy"}, {}))

    result = ReadTools(client, "student").lb_radio("bluegrass song", "easy")

    assert not result.success and result.error_category == "input"
    assert "Invalid prompt expression" in result.error
    assert "private" not in result.error and len(result.error) < 360


def test_find_playlists_authenticates_for_private_and_filters_locally():
    body = {
        "playlist_count": 2,
        "playlists": [
            {"playlist": {"title": "[ListenBrainz] Focus", "identifier": f"https://listenbrainz.org/playlist/{PLAYLIST_MBID}"}},
            {"playlist": {"title": "Road Trip", "identifier": "https://listenbrainz.org/playlist/dddddddd-0000-4000-8000-000000000001"}},
        ],
    }
    client = QueueClient((200, body, {}))
    result = ReadTools(client, "student").find_playlists("focus", 25)
    assert result.success and [p["title"] for p in result.data["playlists"]] == ["[ListenBrainz] Focus"]
    assert result.data["playlists"][0]["encore_managed"] is True
    assert result.data["source_service"] == "listenbrainz"
    assert result.data["external_service_library_searched"] is False
    assert client.calls[0][2]["auth"] is True


def test_find_playlists_distinguishes_account_owner_from_encore_marker():
    body = {
        "playlist_count": 2,
        "playlists": [
            {
                "playlist": {
                    "title": "[ListenBrainz] Encore Mix",
                    "creator": "alice",
                    "identifier": f"https://listenbrainz.org/playlist/{PLAYLIST_MBID}",
                }
            },
            {
                "playlist": {
                    "title": "Fall2026",
                    "creator": "alice",
                    "identifier": "https://listenbrainz.org/playlist/dddddddd-0000-4000-8000-000000000001",
                }
            },
        ],
    }

    result = ReadTools(QueueClient((200, body, {})), "student").find_playlists()

    by_title = {item["title"]: item for item in result.data["playlists"]}
    assert by_title["[ListenBrainz] Encore Mix"]["creator"] == "alice"
    assert by_title["[ListenBrainz] Encore Mix"]["encore_managed"] is True
    assert by_title["Fall2026"]["encore_managed"] is False


def test_router_schema_is_domain_only():
    properties = CLASSIFIER_TOOL["function"]["parameters"]["properties"]
    assert "none" in properties["domains"]["items"]["enum"]
    assert set(properties) == {"domains"}


def test_router_accepts_enum_subset_and_always_adds_final_answer_schema():
    decision = parse_router_message(
        router_message("catalog".split())
    )
    names = [schema["function"]["name"] for schema in decision.model_tool_schemas()]
    expected = tuple(
        spec.name for spec in IMPLEMENTED_TOOL_SPECS if spec.domain == "catalog"
    )
    assert decision.tools == expected
    assert names == [*expected, "final_answer"]


def test_playlist_route_exposes_all_tools_in_selected_domains():
    decision = parse_router_message(
        router_message(["catalog", "discovery", "library"])
    )
    assert decision.domains == ("catalog", "discovery", "library")
    assert "search_recordings" in decision.tools
    assert "verify_recordings" in decision.tools
    assert "lb_radio" in decision.tools
    assert "find_playlists" in decision.tools
    assert "create_playlist" in decision.tools
    assert "export_playlist" in decision.tools


def test_route_request_derives_playlist_tools_from_model_domains():
    class PlaylistRouterClient:
        def chat(self, *args, **kwargs):
            return ModelResponse(
                router_message(["catalog", "discovery", "library"]),
                "test",
                None,
                None,
            )

    decision = route_request("create a three-song playlist", PlaylistRouterClient())

    assert "lb_radio" in decision.tools
    assert "verify_recordings" in decision.tools
    assert "find_playlists" in decision.tools
    assert "export_playlist" in decision.tools


def test_router_rejects_none_combined_with_another_domain():
    with pytest.raises(ValueError):
        parse_router_message(router_message(["none", "catalog"]))


def test_router_rejects_duplicate_domains():
    with pytest.raises(Exception):
        parse_router_message(router_message(["catalog", "catalog"]))


def test_router_rejects_legacy_or_invented_fields():
    message = router_message(["catalog"])
    message["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"domains": ["catalog"], "tools": ["search_recordings"]}
    )
    with pytest.raises(Exception):
        parse_router_message(message)


def test_invalid_model_route_falls_back_to_every_implemented_tool():
    class BadRouterClient:
        def chat(self, *args, **kwargs):
            return ModelResponse({"content": "not a tool call"}, "test", None, None)

    decision = route_request("make a playlist", BadRouterClient())
    assert decision == fallback_decision()
    assert decision.used_fallback and decision.tools == IMPLEMENTED_NAMES


def test_ambiguous_retry_broadens_none_route_for_conversation_aware_main_agent():
    class NoneRouterClient:
        def chat(self, *args, **kwargs):
            return ModelResponse(
                router_message(["none"]),
                "test",
                None,
                None,
            )

    decision = route_request("try again", NoneRouterClient())

    assert decision == fallback_decision()
    assert "export_playlist" in decision.tools
