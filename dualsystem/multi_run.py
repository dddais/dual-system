"""Full-loop example: run Dual-System steps with video-based observations.

Reads frames from video files in a data directory at a fixed sampling interval,
feeds them through the VLM → executor pipeline.

Usage:
    python -m dualsystem.multi_run \\
        --config examples/config_test.yaml \\
        --task "turn on the radio" \\
        --session-id task-radio-001 \\
        --data-dir data/turn_on_radio_1 \\
        --sample-interval 100
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from dualsystem import DualSystemAgentLoop, Observation, RunOptions
from dualsystem.cli import _build_executor, _build_vlm
from dualsystem.config import DualSystemConfig, load_config
from dualsystem.core.types import ExecutorOutput, SessionState
from dualsystem.executor.base import ExecutorClient
from dualsystem.io import TrajectoryLogger, image_from_video
from dualsystem.state import SessionStore

# Video key → filename mapping for Behavior multi-camera setup.
# The key (e.g. "head", "left", "right") becomes the image name in Observation.
_DEFAULT_VIDEO_PATTERN = {
    "head": "head.mp4",
    "left": "left.mp4",
    "right": "right.mp4",
}

DEFAULT_CONFIG_PATH =  "/home/dais/workspace/dualsystem/examples/config_test_local.yaml"
DEFAULT_DATA_DIR = "/home/dais/workspace/dualsystem/data/pick3suc_1_carrot"
DEFAULT_SESSION_ID = "task-carrot-001_select_1"
DEFAULT_TASK = "pick up the carrot and put it on the yellow plate"
DEFAULT_SAMPLE_INTERVAL = 30

class MockExecutor:
    def __init__(self):
        self.inputs: list = []

    def execute(self, executor_input) -> ExecutorOutput:
        self.inputs.append(executor_input)
        return ExecutorOutput(actions=[[1, 2, 3]], status="ok")


def _discover_videos(data_dir: Path) -> dict[str, Path]:
    """Find video files in data_dir and map them to image keys.

    Matches by filename suffix: files ending in ``head.mp4``, ``left.mp4``,
    ``right.mp4`` are mapped to the corresponding keys.  If no standard
    names are found, falls back to using all video files with sequential keys.
    """
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
    videos: dict[str, Path] = {}

    all_videos = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in video_exts
    )

    for vp in all_videos:
        name_lower = vp.name.lower()
        for key, suffix in _DEFAULT_VIDEO_PATTERN.items():
            if name_lower.endswith(suffix):
                videos[key] = vp
                break

    # Fallback: assign sequential keys if no standard names matched.
    if not videos and all_videos:
        for i, vp in enumerate(all_videos):
            videos[f"cam{i}"] = vp

    return videos


def _get_frame_count(video_path: Path) -> int:
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for video input")
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dualsystem.multi_run")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML/JSON config path")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task description")
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID, help="Session identifier")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory with video files")
    parser.add_argument("--sample-interval", type=int, default=DEFAULT_SAMPLE_INTERVAL, help="Frame sampling interval")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps (default: auto from video)",
    )
    parser.add_argument(
        "--planning-mode",
        choices=("direct", "subtask_selection"),
        default=None,
        help="Planning route: direct next-subtask planning or fixed-plan subtask selection",
    )
    parser.add_argument("--reset-session", default=True, action="store_true")
    parser.add_argument(
        "--no-session",
        default=True,
        action="store_true",
        help="Skip session state persistence",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.planning_mode:
        config.planning_mode = args.planning_mode
    data_dir = Path(args.data_dir).resolve()

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 1

    # Discover video files and map to camera keys.
    video_map = _discover_videos(data_dir)
    if not video_map:
        print(f"No video files found in {data_dir}")
        return 1
    print(f"Found videos: { {k: v.name for k, v in video_map.items()} }")

    # Determine number of steps from the first video.
    first_video = next(iter(video_map.values()))
    total_frames = _get_frame_count(first_video)
    max_steps = args.max_steps or (total_frames // args.sample_interval)
    print(
        f"Video: {total_frames} frames, sample interval: "
        f"{args.sample_interval}, steps: {max_steps}"
    )

    # Build components.
    vlm = _build_vlm(config)
    executor: ExecutorClient = MockExecutor()
    # executor: ExecutorClient = _build_executor(config)
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
    if not args.no_session:
        store = SessionStore(config.session_dir)
        state = store.reset(args.session_id) if args.reset_session else store.load(args.session_id)

    for step in tqdm(range(max_steps), desc="Dual-System steps"):
        frame_index = step * args.sample_interval

        # Extract the same frame_index from all camera videos.
        images = {}
        for key, video_path in video_map.items():
            images[key] = image_from_video(video_path, frame_index=frame_index)

        observation = Observation(
            session_id=args.session_id,
            task=args.task,
            images=images,
        )

        result, state = loop.step(observation, state)

        if not args.no_session:
            store.save(args.session_id, state)

        if config.logging.enabled:
            logger.log_step(result, observation, state)

        if result.created_subtask_plan and result.subtasks:
            tqdm.write(
                "initial_subtasks="
                + json.dumps(result.subtasks, ensure_ascii=False)
            )

        tqdm.write(
            f"step={result.step_index} frame={frame_index} "
            f"subtask={result.subtask!r} "
            f"subtask_index={result.subtask_index} "
            f"skipped_vlm={result.skipped_vlm} "
            f"parse_ok={result.parse_ok}"
        )

        if result.executor_output.status == "done":
            tqdm.write("Task completed.")
            break

    print(f"Trajectory saved to {config.logging.run_dir}/{args.session_id}/trajectory.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
