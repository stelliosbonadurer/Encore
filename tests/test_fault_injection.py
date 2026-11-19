"""Offline tests for reproducible assignment-required fault injection."""

from __future__ import annotations

import json

import pytest

from encore.api_client import LB_API, MB_API, ApiClient
from encore.config import Settings
from encore.cli import parse_args
from encore.fault_injection import InjectionConfig, TransientFaultInjector


def settings() -> Settings:
    return Settings(
        listenbrainz_token="secret",
        listenbrainz_user="student",
        encore_user_id="student",
        contact_email="student@example.edu",
        openrouter_api_key=None,
        openrouter_url="https://openrouter.example/chat",
        openrouter_model_cheap=None,
        openrouter_model_expensive=None,
        local_llm_url="http://localhost/chat",
        local_llm_model_cheap=None,
        local_llm_model_expensive=None,
    )


class FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}
    text = ""

    def json(self):
        return {"ok": True}


def write_fixture(tmp_path, payload) -> str:
    path = tmp_path / "inject.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_cli_loads_strict_injection_config(tmp_path):
    path = write_fixture(
        tmp_path, {"seed": 17, "transient_rate": 0.25, "max_steps": 3}
    )

    args = parse_args(["online", "--chat", "--inject", path])

    assert args.inject == InjectionConfig(17, 0.25, 3)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"seed": True},
        {"seed": 1, "transient_rate": -0.1},
        {"seed": 1, "transient_rate": 1.1},
        {"seed": 1, "max_steps": 0},
        {"seed": 1, "max_steps": 13},
        {"seed": 1, "surprise": "network off"},
    ],
)
def test_cli_rejects_invalid_or_expansive_injection_config(tmp_path, payload):
    path = write_fixture(tmp_path, payload)

    with pytest.raises(SystemExit):
        parse_args(["online", "--inject", path])


def test_seeded_listenbrainz_faults_are_reproducible_and_precede_network():
    network_calls = []
    client = ApiClient(
        settings(),
        request=lambda *args, **kwargs: network_calls.append((args, kwargs))
        or FakeResponse(),
        fault_injector=TransientFaultInjector(InjectionConfig(39102, 0.5)),
    )

    statuses = [
        client.request("GET", f"{LB_API}/user/student/listens")[0]
        for _ in range(3)
    ]

    assert statuses == [429, 429, 200]
    assert len(network_calls) == 1


def test_injected_status_matches_service_and_never_dispatches_network():
    def no_network(*args, **kwargs):
        raise AssertionError("injected requests must not reach the network")

    lb = ApiClient(
        settings(),
        request=no_network,
        fault_injector=TransientFaultInjector(InjectionConfig(1, 1.0)),
    )
    mb = ApiClient(
        settings(),
        request=no_network,
        fault_injector=TransientFaultInjector(InjectionConfig(1, 1.0)),
    )

    assert lb.request("GET", f"{LB_API}/user/student/listens")[0] == 429
    assert mb.request("GET", f"{MB_API}/recording")[0] == 503
