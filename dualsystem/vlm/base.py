"""VLM client interface."""

from __future__ import annotations

from typing import Protocol

from dualsystem.core.types import PlannerInput, PlannerOutput


class VLMClient(Protocol):
    """Protocol implemented by VLM planner adapters."""

    def generate(self, planner_input: PlannerInput) -> PlannerOutput:
        """Generate one or more raw planner outputs."""
