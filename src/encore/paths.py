"""Filesystem locations shared by the CLI, scripts, and tests.

Versioned resources (prompts, knowledge base) ship inside the package.
Runtime state (personal memory, traces, evaluation runs) lives under the
git-ignored ``data/`` directory at the repository root, so generated output
never lands beside source code.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

# Versioned resources.
PROMPTS_DIR = PACKAGE_DIR / "prompts"
KNOWLEDGE_DIR = PACKAGE_DIR / "knowledge_base"
SYSTEM_PROMPT = PROMPTS_DIR / "system_v4.md"

# Versioned evaluation inputs (datasets and fault-injection fixtures).
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
INJECTIONS_DIR = EVALUATION_DIR / "injections"

# Git-ignored runtime state.
DATA_DIR = PROJECT_ROOT / "data"
TRACES_DIR = DATA_DIR / "traces"
RUNS_DIR = DATA_DIR / "runs"


def resolve_memory_root(configured: str | Path) -> Path:
    """Absolute ENCORE_MEMORY_DIR values win; relative ones resolve under data/."""
    path = Path(configured)
    return path if path.is_absolute() else DATA_DIR / path
