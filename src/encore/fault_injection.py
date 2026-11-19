"""Reproducible evaluation-only transport and step-budget fault injection."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MAX_ALLOWED_STEPS = 12
_ALLOWED_FIELDS = {"seed", "transient_rate", "max_steps"}


@dataclass(frozen=True)
class InjectionConfig:
    """Strict, portable controls loaded from one evaluator-owned JSON file."""

    seed: int
    transient_rate: float = 0.0
    max_steps: int | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "InjectionConfig":
        """Load and validate a fault configuration without accepting coercions."""
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(
                f"could not read injection file ({type(exc).__name__})"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError("injection file must contain valid JSON") from exc

        if not isinstance(payload, dict):
            raise ValueError("injection file must contain one JSON object")
        unknown = payload.keys() - _ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"unknown injection fields: {sorted(unknown)}")
        if "seed" not in payload:
            raise ValueError("injection file requires an integer seed")

        seed = payload["seed"]
        rate = payload.get("transient_rate", 0.0)
        max_steps = payload.get("max_steps")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("injection seed must be an integer")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError("transient_rate must be a number from 0 to 1")
        rate = float(rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("transient_rate must be a number from 0 to 1")
        if max_steps is not None:
            if isinstance(max_steps, bool) or not isinstance(max_steps, int):
                raise ValueError("max_steps must be an integer from 1 to 12")
            if not 1 <= max_steps <= MAX_ALLOWED_STEPS:
                raise ValueError("max_steps must be an integer from 1 to 12")
        return cls(seed=seed, transient_rate=rate, max_steps=max_steps)


class TransientFaultInjector:
    """Return seeded synthetic HTTP statuses before a real API dispatch."""

    def __init__(self, config: InjectionConfig) -> None:
        self.config = config
        self._random = random.Random(config.seed)

    def status_for(self, service: Literal["listenbrainz", "musicbrainz"]) -> int | None:
        """Inject LB 429 or MB 503 according to the configured Bernoulli rate."""
        if self._random.random() >= self.config.transient_rate:
            return None
        return 429 if service == "listenbrainz" else 503
