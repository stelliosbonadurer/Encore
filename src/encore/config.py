"""Environment configuration and runtime safety policy.

Loading configuration is kept out of tool modules. This makes secrets easier to
contain and lets tests construct settings without touching the real environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # ListenBrainz credentials and identity are kept out of tool/model schemas.
    listenbrainz_token: str | None
    listenbrainz_user: str | None
    encore_user_id: str | None
    # MusicBrainz requires an identifying contact in the User-Agent header.
    contact_email: str
    # Online provider settings inherited from the hardened Lab 3 client.
    openrouter_api_key: str | None
    openrouter_url: str
    openrouter_model_cheap: str | None
    openrouter_model_expensive: str | None
    # Offline provider settings use the same OpenAI-compatible request format.
    local_llm_url: str
    local_llm_model_cheap: str | None
    local_llm_model_expensive: str | None
    # Personal-memory location is configurable so tests never touch real stores.
    memory_dir: str = "memory"

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load .env once, then return an immutable snapshot for this process."""
        # python-dotenv fills missing environment values but does not override
        # variables explicitly supplied by the shell.
        load_dotenv()
        # Read the account name once so the memory deletion identity can default
        # to exactly the same value.
        lb_user = os.getenv("LISTENBRAINZ_USER")
        # Constructing a dataclass makes dependencies explicit and easy to fake.
        return cls(
            listenbrainz_token=os.getenv("LISTENBRAINZ_TOKEN"),
            listenbrainz_user=lb_user,
            encore_user_id=os.getenv("ENCORE_USER_ID") or lb_user,
            contact_email=os.getenv("CONTACT_EMAIL", "you@umontana.edu"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_url=os.getenv(
                "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
            ),
            openrouter_model_cheap=os.getenv("OPENROUTER_MODEL_CHEAP"),
            openrouter_model_expensive=os.getenv("OPENROUTER_MODEL_EXPENSIVE"),
            local_llm_url=os.getenv(
                "LOCAL_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions"
            ),
            local_llm_model_cheap=os.getenv("LOCAL_LLM_MODEL_CHEAP"),
            local_llm_model_expensive=os.getenv("LOCAL_LLM_MODEL_EXPENSIVE"),
            memory_dir=os.getenv("ENCORE_MEMORY_DIR", "memory"),
        )


@dataclass(frozen=True)
class RuntimePolicy:
    """The deterministic gate checked before any write tool runs."""

    # ``mode`` selects the model provider and remains useful diagnostic context.
    # It does not grant or revoke music-account capabilities.
    mode: str
    # This independent switch prevents accidental writes in either provider mode.
    allow_writes_flag: bool = False

    @property
    def writes_enabled(self) -> bool:
        # Provider location and account authorization are separate concerns.
        # Individual writes still require an exact, deterministic approval permit.
        return self.allow_writes_flag
