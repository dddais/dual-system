"""Executor client interface."""

from __future__ import annotations

from typing import Protocol

from dualsystem.core.types import ExecutorInput, ExecutorOutput


class ExecutorClient(Protocol):
    """Protocol implemented by VLA or robot executor adapters."""

    def execute(self, executor_input: ExecutorInput) -> ExecutorOutput:
        """Execute a subtask and return actions or status."""
