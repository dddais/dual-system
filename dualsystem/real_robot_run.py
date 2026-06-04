"""Realtime Dual-System runner for robot camera snapshots.

Reads the latest camera images from a directory, plans the next instruction
with the configured VLM, and sends that instruction to the configured executor.

Usage:
    python -m dualsystem.real_robot_run \
        --config examples/config_real_robot.yaml \
        --reset-session
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dualsystem import DualSystemAgentLoop, Observation, RunOptions
from dualsystem.cli import _build_executor, _build_vlm
from dualsystem.config import load_config
from dualsystem.core.types import ExecutorInput, ExecutorOutput, SessionState
from dualsystem.executor.base import ExecutorClient
from dualsystem.io import TrajectoryLogger, image_from_path
from dualsystem.state import SessionStore

DEFAULT_CONFIG_PATH = "/home/dais/workspace/dualsystem/examples/config_real_robot.yaml"
DEFAULT_IMAGE_DIR = "/tmp/img"
DEFAULT_SESSION_ID = "real-robot-001"
DEFAULT_TASK = "Organize the desk and ensure no items are left scattered."
DEFAULT_CONTROL_HZ = 1.0
DEFAULT_STOP_STATUS = "done"

DEFAULT_CAMERA_FILES = {
    "cam_high": "base_0_rgb.jpg",
    "cam_left_wrist": "left_wrist_0_rgb.jpg",
    "cam_right_wrist": "right_wrist_0_rgb.jpg",
}


class DryRunExecutor:
    """Executor that records the instruction without commanding the robot."""

    def __init__(self) -> None:
        self.inputs: list[ExecutorInput] = []

    def execute(self, executor_input: ExecutorInput) -> ExecutorOutput:
        self.inputs.append(executor_input)
        return ExecutorOutput(
            status="dry_run",
            raw_response={
                "subtask": executor_input.subtask,
                "metadata": executor_input.metadata,
            },
        )


def build_observation_from_image_dir(
    image_dir: str | Path,
    session_id: str,
    task: str,
    camera_files: dict[str, str] | None = None,
    metadata: dict | None = None,
    wait_timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> Observation:
    """Build one observation from the latest robot camera image files."""
    image_root = Path(image_dir).expanduser()
    selected_camera_files = camera_files or DEFAULT_CAMERA_FILES
    paths = {
        camera_name: image_root / filename
        for camera_name, filename in selected_camera_files.items()
    }
    _wait_for_stable_files(paths, wait_timeout, poll_interval)
    images = {
        camera_name: image_from_path(path)
        for camera_name, path in paths.items()
    }
    return Observation(
        session_id=session_id,
        task=task,
        images=images,
        state={},
        metadata={
            **(metadata or {}),
            "image_dir": str(image_root),
            "camera_files": dict(selected_camera_files),
        },
    )


def parse_camera_file_specs(specs: list[str]) -> dict[str, str]:
    """Parse repeated ``camera=filename`` CLI overrides."""
    if not specs:
        return dict(DEFAULT_CAMERA_FILES)

    camera_files: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Camera spec must use camera=filename syntax: {spec}")
        camera_name, filename = spec.split("=", 1)
        camera_name = camera_name.strip()
        filename = filename.strip()
        if not camera_name or not filename:
            raise ValueError(f"Camera spec must not be empty: {spec}")
        camera_files[camera_name] = filename
    return camera_files


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    real_robot = config.real_robot
    if args.planning_mode:
        config.planning_mode = args.planning_mode
    if args.session_dir:
        config.session_dir = args.session_dir
    if args.run_dir:
        config.logging.run_dir = args.run_dir
    if args.no_log:
        config.logging.enabled = False

    task = args.task or real_robot.task or DEFAULT_TASK
    session_id = args.session_id or real_robot.session_id or DEFAULT_SESSION_ID
    image_dir = args.image_dir or real_robot.image_dir or DEFAULT_IMAGE_DIR
    camera_files = (
        parse_camera_file_specs(args.camera_file)
        if args.camera_file
        else dict(real_robot.camera_files or DEFAULT_CAMERA_FILES)
    )
    control_hz = (
        args.control_hz
        if args.control_hz is not None
        else real_robot.control_hz
    )
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else real_robot.max_steps
    )
    wait_timeout = (
        args.wait_timeout
        if args.wait_timeout is not None
        else real_robot.wait_timeout
    )
    poll_interval = (
        args.poll_interval
        if args.poll_interval is not None
        else real_robot.poll_interval
    )
    stop_status = args.stop_status or real_robot.stop_status
    dry_run = args.dry_run or real_robot.dry_run

    vlm = _build_vlm(config)
    executor: ExecutorClient
    if dry_run:
        executor = DryRunExecutor()
    else:
        executor = _build_executor(config)

    logger = TrajectoryLogger(
        run_dir=config.logging.run_dir,
        copy_images=config.logging.copy_images,
    )
    loop = DualSystemAgentLoop(
        vlm_client=vlm,
        executor_client=executor,
        options=RunOptions(
            enable_memory=config.enable_memory,
            frequency=config.frequency,
            planning_mode=config.planning_mode,
            sampling_params=config.vlm.sampling_params,
        ),
    )

    state = SessionState()
    store = None
    if not args.no_session:
        store = SessionStore(config.session_dir)
        state = (
            store.reset(session_id)
            if args.reset_session
            else store.load(session_id)
        )

    print(
        "Starting realtime robot loop: "
        f"image_dir={Path(image_dir).expanduser()} "
        f"cameras={camera_files} control_hz={control_hz}"
    )

    step_count = 0
    period = 1.0 / control_hz if control_hz > 0 else 0.0
    try:
        while max_steps is None or step_count < max_steps:
            started_at = time.monotonic()
            observation = build_observation_from_image_dir(
                image_dir=image_dir,
                session_id=session_id,
                task=task,
                camera_files=camera_files,
                metadata={"realtime_step": step_count},
                wait_timeout=wait_timeout,
                poll_interval=poll_interval,
            )

            result, state = loop.step(observation, state)

            if store is not None:
                store.save(session_id, state)

            if config.logging.enabled:
                logger.log_step(result, observation, state)

            if result.created_subtask_plan and result.subtasks:
                print(
                    "initial_subtasks="
                    + json.dumps(result.subtasks, ensure_ascii=False)
                )

            print(
                f"step={result.step_index} "
                f"instruction={result.subtask!r} "
                f"subtask_index={result.subtask_index} "
                f"skipped_vlm={result.skipped_vlm} "
                f"parse_ok={result.parse_ok} "
                f"executor_status={result.executor_output.status}"
            )

            step_count += 1
            if result.executor_output.status == stop_status:
                print(f"Stopping: executor returned status={stop_status!r}.")
                break

            elapsed = time.monotonic() - started_at
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Stopping realtime robot loop.")

    if config.logging.enabled:
        print(
            "Trajectory saved to "
            f"{config.logging.run_dir}/{session_id}/trajectory.jsonl"
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dualsystem.real_robot_run")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="YAML/JSON config path",
    )
    parser.add_argument("--task", default=None, help="Task description")
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session identifier",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory containing realtime camera images",
    )
    parser.add_argument(
        "--camera-file",
        action="append",
        default=[],
        help=(
            "Camera image mapping in camera=filename syntax. Repeat to override "
            "the default cam_high/base_0_rgb.jpg, cam_left_wrist/"
            "left_wrist_0_rgb.jpg, cam_right_wrist/right_wrist_0_rgb.jpg."
        ),
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help="Realtime loop frequency; config frequency still controls VLM cadence",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum realtime iterations before stopping",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=None,
        help="Seconds to wait for all camera files to appear and become stable",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Seconds between camera file readiness checks",
    )
    parser.add_argument(
        "--planning-mode",
        choices=("direct", "subtask_selection"),
        default=None,
        help="Override config planning mode",
    )
    parser.add_argument(
        "--stop-status",
        default=None,
        help="Stop when executor returns this status",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip real executor calls",
    )
    parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Clear session state first",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="Skip session persistence",
    )
    parser.add_argument(
        "--session-dir",
        default=None,
        help="Override session storage directory",
    )
    parser.add_argument("--run-dir", default=None, help="Override log output directory")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable trajectory logging",
    )
    return parser


def _wait_for_stable_files(
    paths: dict[str, Path],
    timeout: float,
    poll_interval: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_sizes: dict[str, int] | None = None

    while True:
        missing = {
            camera_name: path
            for camera_name, path in paths.items()
            if not path.is_file()
        }
        if not missing:
            sizes = {
                camera_name: path.stat().st_size
                for camera_name, path in paths.items()
            }
            if all(size > 0 for size in sizes.values()) and sizes == last_sizes:
                return
            last_sizes = sizes

        if time.monotonic() >= deadline:
            missing_desc = ", ".join(
                f"{camera_name}={path}" for camera_name, path in missing.items()
            )
            if missing_desc:
                raise FileNotFoundError(
                    f"Timed out waiting for camera image files: {missing_desc}"
                )
            raise TimeoutError(
                "Timed out waiting for camera image files to become stable: "
                + ", ".join(
                    f"{camera_name}={path}" for camera_name, path in paths.items()
                )
            )

        time.sleep(poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
