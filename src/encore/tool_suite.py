"""Composition object for the complete Part 3 tool registry."""

from __future__ import annotations

from collections.abc import Callable

from encore.contracts import ToolResult
from encore.memory_tools import MemoryTools
from encore.tool_catalog import FINAL_ANSWER_SPEC, IMPLEMENTED_TOOL_SPECS
from encore.tools import ReadTools
from encore.write_tools import WriteTools


class ToolSuite:
    """Combine focused tool groups while rejecting duplicate or missing names."""

    def __init__(
        self, reads: ReadTools, writes: WriteTools, memory: MemoryTools
    ) -> None:
        # Each focused class owns one concern; composition happens only here.
        groups = (reads.registry(), writes.registry(), memory.registry())
        # Begin empty so overlaps can be detected before any name is overwritten.
        registry: dict[str, Callable[..., ToolResult]] = {}
        for group in groups:
            # A duplicate would make handler selection depend on merge order.
            overlap = registry.keys() & group.keys()
            if overlap:
                raise ValueError(f"duplicate tool handlers: {sorted(overlap)}")
            # Safe only after the overlap check succeeds.
            registry.update(group)

        # The catalog is authoritative for what models may request.
        expected = {spec.name for spec in IMPLEMENTED_TOOL_SPECS}
        # Dict-key views compare as sets, so order does not hide a mismatch.
        if registry.keys() != expected:
            # Report both directions to make a future integration error obvious.
            missing = expected - registry.keys()
            extra = registry.keys() - expected
            raise ValueError(f"tool registry mismatch; missing={missing}, extra={extra}")
        # Keep the validated mapping private after construction.
        self._registry = registry

    def registry(self) -> dict[str, Callable[..., ToolResult]]:
        """Return a copy so callers cannot mutate the suite's authoritative map."""
        # A shallow copy prevents callers from removing/replacing suite handlers.
        return self._registry.copy()

    def schemas(self) -> list[dict]:
        """Return all action schemas plus the always-available control schema."""
        # Action schemas are routed; final_answer is a separate control action.
        return [
            *(spec.openai_schema() for spec in IMPLEMENTED_TOOL_SPECS),
            FINAL_ANSWER_SPEC.openai_schema(),
        ]
