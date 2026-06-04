"""Standalone Dual-System VLM -> executor coordination loop."""

from __future__ import annotations

import copy

from dualsystem.core.parsers import (
    parse_subtask_plan,
    parse_subtask_selection,
    parse_vlm_output,
)
from dualsystem.core.prompts import (
    build_prompt,
    build_subtask_plan_prompt,
    build_subtask_selection_prompt,
)
from dualsystem.core.types import (
    ExecutorInput,
    ExecutorOutput,
    Observation,
    PlannerInput,
    RunOptions,
    SessionState,
    StepResult,
)
from dualsystem.executor.base import ExecutorClient
from dualsystem.vlm.base import VLMClient

DIRECT_PLANNING_MODE = "direct"
SUBTASK_SELECTION_PLANNING_MODE = "subtask_selection"


class DualSystemAgentLoop:
    """Coordinate one VLM planning turn and one executor turn."""

    def __init__(
        self,
        vlm_client: VLMClient,
        executor_client: ExecutorClient,
        options: RunOptions | None = None,
    ) -> None:
        self.vlm_client = vlm_client
        self.executor_client = executor_client
        self.options = options or RunOptions()
        self.options.frequency = max(1, int(self.options.frequency))
        self.options.planning_mode = _normalize_planning_mode(
            self.options.planning_mode
        )

    def step(
        self,
        observation: Observation,
        session_state: SessionState | None = None,
    ) -> tuple[StepResult, SessionState]:
        """Run one standalone Dual-System step."""
        state = copy.deepcopy(session_state or SessionState())
        step_index = state.step_index
        if self.options.planning_mode == SUBTASK_SELECTION_PLANNING_MODE:
            return self._step_with_subtask_selection(
                observation=observation,
                state=state,
                step_index=step_index,
            )

        prompt = build_prompt(
            observation.task,
            memory=state.memory,
            enable_memory=self.options.enable_memory,
        )

        should_call_vlm = (
            state.cached_subtask is None
            or step_index % self.options.frequency == 0
        )
        parse_ok = True
        parse_error = None

        if should_call_vlm:
            planner_output = self.vlm_client.generate(
                PlannerInput(
                    observation=observation,
                    prompt=prompt,
                    sampling_params=dict(self.options.sampling_params),
                )
            )
            raw_output = (
                planner_output.raw_outputs[0] if planner_output.raw_outputs else ""
            )
            parsed = parse_vlm_output(raw_output, self.options.enable_memory)
            subtask = parsed.subtask
            parse_ok = parsed.parse_ok
            parse_error = parsed.parse_error
            if self.options.enable_memory and parsed.memory:
                state.memory = parsed.memory
            state.cached_subtask = subtask
            state.cached_raw_output = raw_output
            skipped_vlm = False
        else:
            subtask = state.cached_subtask or "continue current action"
            raw_output = state.cached_raw_output or subtask
            skipped_vlm = True

        executor_output = self.executor_client.execute(
            ExecutorInput(
                observation=observation,
                subtask=subtask,
                memory=state.memory if self.options.enable_memory else None,
                metadata={
                    **observation.metadata,
                    "step_index": step_index,
                    "trajectory_index": state.trajectory_index,
                    "skipped_vlm": skipped_vlm,
                },
            )
        )
        if executor_output is None:
            executor_output = ExecutorOutput(status="ok")

        result = StepResult(
            session_id=observation.session_id,
            step_index=step_index,
            trajectory_index=state.trajectory_index,
            task=observation.task,
            subtask=subtask,
            memory=state.memory if self.options.enable_memory else None,
            vlm_raw_output=raw_output,
            executor_output=executor_output,
            skipped_vlm=skipped_vlm,
            prompt=prompt,
            parse_ok=parse_ok,
            parse_error=parse_error,
            planning_mode=self.options.planning_mode,
        )

        state.step_index += 1
        return result, state

    def _step_with_subtask_selection(
        self,
        observation: Observation,
        state: SessionState,
        step_index: int,
    ) -> tuple[StepResult, SessionState]:
        plan_prompt = None
        plan_raw_output = state.cached_plan_raw_output
        plan_parse_ok = True
        plan_parse_error = None
        created_subtask_plan = False

        if not state.subtasks:
            plan_prompt = build_subtask_plan_prompt(observation.task)
            planner_output = self.vlm_client.generate(
                PlannerInput(
                    observation=observation,
                    prompt=plan_prompt,
                    sampling_params=dict(self.options.sampling_params),
                )
            )
            plan_raw_output = (
                planner_output.raw_outputs[0] if planner_output.raw_outputs else ""
            )
            parsed_plan = parse_subtask_plan(plan_raw_output)
            state.subtasks = parsed_plan.subtasks
            state.cached_plan_raw_output = plan_raw_output
            plan_parse_ok = parsed_plan.parse_ok
            plan_parse_error = parsed_plan.parse_error
            created_subtask_plan = True

        prompt = build_subtask_selection_prompt(
            observation.task,
            state.subtasks,
            memory=state.memory,
            enable_memory=self.options.enable_memory,
        )
        should_call_vlm = (
            state.cached_subtask is None
            or step_index % self.options.frequency == 0
        )
        parse_ok = plan_parse_ok
        parse_error = plan_parse_error

        if should_call_vlm:
            planner_output = self.vlm_client.generate(
                PlannerInput(
                    observation=observation,
                    prompt=prompt,
                    sampling_params=dict(self.options.sampling_params),
                )
            )
            raw_output = (
                planner_output.raw_outputs[0] if planner_output.raw_outputs else ""
            )
            parsed = parse_subtask_selection(
                raw_output,
                state.subtasks,
                enable_memory=self.options.enable_memory,
            )
            subtask = parsed.subtask
            parse_ok, parse_error = _merge_parse_status(
                parse_ok,
                parse_error,
                parsed.parse_ok,
                parsed.parse_error,
            )
            subtask_index = parsed.subtask_index
            if self.options.enable_memory and parsed.memory:
                state.memory = parsed.memory
            state.cached_subtask = subtask
            state.cached_subtask_index = subtask_index
            state.cached_raw_output = raw_output
            skipped_vlm = False
        else:
            fallback_subtask = (
                state.subtasks[0] if state.subtasks else "continue current action"
            )
            subtask = state.cached_subtask or fallback_subtask
            raw_output = state.cached_raw_output or subtask
            subtask_index = state.cached_subtask_index
            skipped_vlm = True

        executor_output = self.executor_client.execute(
            ExecutorInput(
                observation=observation,
                subtask=subtask,
                memory=state.memory if self.options.enable_memory else None,
                metadata={
                    **observation.metadata,
                    "step_index": step_index,
                    "trajectory_index": state.trajectory_index,
                    "skipped_vlm": skipped_vlm,
                    "planning_mode": self.options.planning_mode,
                    "subtasks": list(state.subtasks),
                    "subtask_index": subtask_index,
                },
            )
        )
        if executor_output is None:
            executor_output = ExecutorOutput(status="ok")

        result = StepResult(
            session_id=observation.session_id,
            step_index=step_index,
            trajectory_index=state.trajectory_index,
            task=observation.task,
            subtask=subtask,
            memory=state.memory if self.options.enable_memory else None,
            vlm_raw_output=raw_output,
            executor_output=executor_output,
            skipped_vlm=skipped_vlm,
            prompt=prompt,
            parse_ok=parse_ok,
            parse_error=parse_error,
            planning_mode=self.options.planning_mode,
            subtasks=list(state.subtasks),
            subtask_index=subtask_index,
            created_subtask_plan=created_subtask_plan,
            plan_prompt=plan_prompt,
            plan_raw_output=plan_raw_output,
        )

        state.step_index += 1
        return result, state


def _normalize_planning_mode(planning_mode: str | None) -> str:
    normalized = (planning_mode or DIRECT_PLANNING_MODE).strip().lower()
    aliases = {
        "default": DIRECT_PLANNING_MODE,
        "memoryless": DIRECT_PLANNING_MODE,
        "subtask": SUBTASK_SELECTION_PLANNING_MODE,
        "subtasks": SUBTASK_SELECTION_PLANNING_MODE,
        "subtask_select": SUBTASK_SELECTION_PLANNING_MODE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {DIRECT_PLANNING_MODE, SUBTASK_SELECTION_PLANNING_MODE}:
        raise ValueError(
            f"Unsupported planning_mode: {planning_mode}. "
            f"Expected '{DIRECT_PLANNING_MODE}' or "
            f"'{SUBTASK_SELECTION_PLANNING_MODE}'."
        )
    return normalized


def _merge_parse_status(
    left_ok: bool,
    left_error: str | None,
    right_ok: bool,
    right_error: str | None,
) -> tuple[bool, str | None]:
    errors = [error for error in (left_error, right_error) if error]
    return left_ok and right_ok, "; ".join(errors) if errors else None
