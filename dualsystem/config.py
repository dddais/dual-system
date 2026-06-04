"""Configuration loading for standalone Dual-System."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VLMConfig:
    """VLM adapter configuration."""

    provider: str = "openai_compatible"
    model: str | None = None
    model_path: str | None = None
    model_family: str = "auto"
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    dtype: str = "bf16"
    device: str = "auto"
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    sampling_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutorConfig:
    """Executor adapter configuration."""

    provider: str = "http"
    endpoint: str | None = None
    timeout: float = 30.0


@dataclass
class LoggingConfig:
    """Trajectory logging configuration."""

    enabled: bool = True
    run_dir: str = "./runs"
    copy_images: bool = False


@dataclass
class RealRobotConfig:
    """Realtime robot deployment configuration."""

    task: str | None = None
    session_id: str | None = None
    image_dir: str = "/tmp/img"
    camera_files: dict[str, str] = field(default_factory=dict)
    control_hz: float = 1.0
    max_steps: int | None = None
    wait_timeout: float = 5.0
    poll_interval: float = 0.05
    stop_status: str = "done"
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.camera_files = dict(self.camera_files or {})
        self.control_hz = float(self.control_hz)
        self.wait_timeout = float(self.wait_timeout)
        self.poll_interval = float(self.poll_interval)
        self.max_steps = None if self.max_steps is None else int(self.max_steps)
        self.dry_run = bool(self.dry_run)


@dataclass
class DualSystemConfig:
    """Top-level standalone Dual-System configuration."""

    enable_memory: bool = False
    frequency: int = 1
    planning_mode: str = "direct"
    session_dir: str = "~/.dualsystem/sessions"
    vlm: VLMConfig = field(default_factory=VLMConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    real_robot: RealRobotConfig = field(default_factory=RealRobotConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DualSystemConfig":
        data = _expand_env_vars(data or {})
        vlm_data = data.get("vlm") or {}
        executor_data = data.get("executor") or {}
        logging_data = data.get("logging") or {}
        real_robot_data = data.get("real_robot") or {}
        config = cls(
            enable_memory=bool(data.get("enable_memory", False)),
            frequency=int(data.get("frequency", 1)),
            planning_mode=str(data.get("planning_mode", "direct")),
            session_dir=str(data.get("session_dir", "~/.dualsystem/sessions")),
            vlm=VLMConfig(**vlm_data),
            executor=ExecutorConfig(**executor_data),
            logging=LoggingConfig(**logging_data),
            real_robot=RealRobotConfig(**real_robot_data),
        )
        config.apply_env_defaults()
        return config

    def apply_env_defaults(self) -> None:
        """Fill unset fields from environment variables."""
        if self.vlm.api_key is None:
            self.vlm.api_key = os.getenv("DUALSYSTEM_VLM_API_KEY") or os.getenv(
                "OPENAI_API_KEY"
            )
        if self.vlm.base_url is None:
            self.vlm.base_url = os.getenv("DUALSYSTEM_VLM_BASE_URL") or os.getenv(
                "OPENAI_BASE_URL"
            )
        if self.executor.endpoint is None:
            self.executor.endpoint = os.getenv("DUALSYSTEM_EXECUTOR_URL")


def load_config(path: str | Path | None) -> DualSystemConfig:
    """Load JSON/YAML config from disk, or return defaults when path is None."""
    if path is None:
        return DualSystemConfig.from_dict({})
    config_path = Path(path).expanduser()
    with open(config_path) as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to load YAML configs") from exc
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)
    return DualSystemConfig.from_dict(data)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value
