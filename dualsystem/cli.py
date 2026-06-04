"""Command line interface for standalone Dual-System."""

from __future__ import annotations

import argparse
import json
from typing import Any

from dualsystem.config import DualSystemConfig, load_config
from dualsystem.core.loop import DualSystemAgentLoop
from dualsystem.core.types import Observation, RunOptions
from dualsystem.executor.http_executor import HTTPExecutorClient
from dualsystem.io.image import parse_image_spec
from dualsystem.io.trajectory_logger import TrajectoryLogger
from dualsystem.state.session_store import SessionStore
from dualsystem.vlm.local_qwen import LocalQwenVLMClient
from dualsystem.vlm.openai_compatible import OpenAICompatibleVLMClient


def main(argv: list[str] | None = None) -> int:
    """Run the Dual-System CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "step":
        return _run_step(args)
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dualsystem")
    subparsers = parser.add_subparsers(dest="command")

    step = subparsers.add_parser("step", help="run one Dual-System step")
    step.add_argument("--config", help="Path to JSON/YAML config")
    step.add_argument("--session-id", required=True)
    step.add_argument("--task", required=True)
    step.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image input in key=path, key=url, or key=base64 syntax",
    )
    step.add_argument("--state", default=None, help="JSON object for robot state")
    step.add_argument("--metadata", default=None, help="JSON object for metadata")
    step.add_argument("--reset-session", action="store_true")
    step.add_argument("--session-dir", default=None)
    step.add_argument("--run-dir", default=None)
    step.add_argument("--no-log", action="store_true")
    step.add_argument("--pretty", action="store_true")
    return parser


def _run_step(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.session_dir:
        config.session_dir = args.session_dir
    if args.run_dir:
        config.logging.run_dir = args.run_dir
    if args.no_log:
        config.logging.enabled = False

    images = dict(parse_image_spec(spec) for spec in args.image)
    observation = Observation(
        session_id=args.session_id,
        task=args.task,
        images=images,
        state=_json_arg(args.state, default={}),
        metadata=_json_arg(args.metadata, default={}),
    )

    store = SessionStore(config.session_dir)
    state = store.reset(args.session_id) if args.reset_session else store.load(args.session_id)

    loop = DualSystemAgentLoop(
        vlm_client=_build_vlm(config),
        executor_client=_build_executor(config),
        options=RunOptions(
            enable_memory=config.enable_memory,
            frequency=config.frequency,
            planning_mode=config.planning_mode,
            sampling_params=config.vlm.sampling_params,
        ),
    )
    result, new_state = loop.step(observation, state)
    store.save(args.session_id, new_state)

    if config.logging.enabled:
        TrajectoryLogger(
            run_dir=config.logging.run_dir,
            copy_images=config.logging.copy_images,
        ).log_step(result, observation, new_state)

    print(json.dumps(result.to_dict(), indent=2 if args.pretty else None))
    return 0


def _build_vlm(config: DualSystemConfig):
    provider = config.vlm.provider
    if provider == "openai_compatible":
        if not config.vlm.model:
            raise ValueError("vlm.model is required for openai_compatible provider")
        return OpenAICompatibleVLMClient(
            model=config.vlm.model,
            base_url=config.vlm.base_url,
            api_key=config.vlm.api_key,
            timeout=config.vlm.timeout,
            default_sampling_params=config.vlm.sampling_params,
        )
    if provider == "local_qwen":
        if not config.vlm.model_path:
            raise ValueError("vlm.model_path is required for local_qwen provider")
        return LocalQwenVLMClient(
            model_path=config.vlm.model_path,
            model_family=config.vlm.model_family,
            dtype=config.vlm.dtype,
            device=config.vlm.device,
            min_pixels=config.vlm.min_pixels,
            max_pixels=config.vlm.max_pixels,
            default_sampling_params=config.vlm.sampling_params,
        )
    raise ValueError(f"Unsupported VLM provider: {provider}")


def _build_executor(config: DualSystemConfig):
    provider = config.executor.provider
    if provider == "http":
        if not config.executor.endpoint:
            raise ValueError("executor.endpoint is required for http provider")
        return HTTPExecutorClient(
            endpoint=config.executor.endpoint,
            timeout=config.executor.timeout,
        )
    raise ValueError(f"Unsupported executor provider: {provider}")


def _json_arg(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


if __name__ == "__main__":
    raise SystemExit(main())
