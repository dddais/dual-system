"""Shared dataclasses for the standalone Dual-System package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

JsonDict = dict[str, Any]


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses and array-like values to JSON-safe data."""
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return str(value)
    return value


def _to_jsonable(value: Any) -> Any:
    return to_jsonable(value)


@dataclass
class ImageInput:
    """Serializable image payload."""

    type: str
    data: str
    mime_type: str | None = None
    path: str | None = None

    def to_dict(self) -> JsonDict:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ImageInput":
        return cls(
            type=str(data["type"]),
            data=str(data["data"]),
            mime_type=data.get("mime_type"),
            path=data.get("path"),
        )


@dataclass
class Observation:
    """Observation supplied to one Dual-System step."""

    session_id: str
    task: str
    images: dict[str, ImageInput] = field(default_factory=dict)
    state: JsonDict | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Observation":
        images = {
            key: value if isinstance(value, ImageInput) else ImageInput.from_dict(value)
            for key, value in data.get("images", {}).items()
        }
        return cls(
            session_id=str(data["session_id"]),
            task=str(data["task"]),
            images=images,
            state=data.get("state"),
            metadata=data.get("metadata") or {},
        )


@dataclass
class SessionState:
    """Persisted state for one session."""

    memory: str = ""
    cached_subtask: str | None = None
    cached_raw_output: str | None = None
    cached_subtask_index: int | None = None
    cached_plan_raw_output: str | None = None
    subtasks: list[str] = field(default_factory=list)
    step_index: int = 0
    trajectory_index: int = 0

    def to_dict(self) -> JsonDict:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, data: JsonDict | None) -> "SessionState":
        if not data:
            return cls()
        return cls(
            memory=str(data.get("memory") or ""),
            cached_subtask=data.get("cached_subtask"),
            cached_raw_output=data.get("cached_raw_output"),
            cached_subtask_index=_optional_int(data.get("cached_subtask_index")),
            cached_plan_raw_output=data.get("cached_plan_raw_output"),
            subtasks=[str(item) for item in data.get("subtasks", [])],
            step_index=int(data.get("step_index", 0)),
            trajectory_index=int(data.get("trajectory_index", 0)),
        )


@dataclass
class RunOptions:
    """Runtime options for the Dual-System loop."""

    enable_memory: bool = False
    frequency: int = 1
    planning_mode: str = "direct"
    sampling_params: JsonDict = field(default_factory=dict)


@dataclass
class PlannerInput:
    """Input passed to a VLM planner."""

    observation: Observation
    prompt: str
    sampling_params: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return _to_jsonable(self)


@dataclass
class PlannerOutput:
    """Output returned by a VLM planner."""

    raw_outputs: list[str]
    raw_response: JsonDict | None = None

    def to_dict(self) -> JsonDict:
        return _to_jsonable(self)


@dataclass
class ExecutorInput:
    """Input passed to a VLA or robot executor."""

    observation: Observation
    subtask: str
    memory: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return _to_jsonable(self)


@dataclass
class ExecutorOutput:
    """Output returned by a VLA or robot executor."""

    actions: list[Any] | dict[str, Any] | None = None
    status: str = "ok"
    raw_response: JsonDict | None = None

    def to_dict(self) -> JsonDict:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, data: JsonDict | None) -> "ExecutorOutput":
        if data is None:
            return cls()
        return cls(
            actions=data.get("actions"),
            status=str(data.get("status", "ok")),
            raw_response=data.get("raw_response") or data,
        )


@dataclass
class StepResult:
    """Structured result for one Dual-System step."""

    session_id: str
    step_index: int
    subtask: str
    memory: str | None
    vlm_raw_output: str
    executor_output: ExecutorOutput
    skipped_vlm: bool
    task: str = ""
    trajectory_index: int = 0
    prompt: str | None = None
    parse_ok: bool = True
    parse_error: str | None = None
    planning_mode: str = "direct"
    subtasks: list[str] = field(default_factory=list)
    subtask_index: int | None = None
    created_subtask_plan: bool = False
    plan_prompt: str | None = None
    plan_raw_output: str | None = None

    def to_dict(self) -> JsonDict:
        return _to_jsonable(self)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
