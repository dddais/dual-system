"""JSONL trajectory logger for standalone Dual-System."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from dualsystem.core.types import ImageInput, Observation, SessionState, StepResult


class TrajectoryLogger:
    """Append one JSON record per Dual-System step."""

    def __init__(self, run_dir: str | Path = "./runs", copy_images: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.copy_images = copy_images

    def log_step(
        self,
        result: StepResult,
        observation: Observation,
        state: SessionState,
    ) -> Path:
        session_dir = self.run_dir / observation.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "result": result.to_dict(),
            "observation": {
                "session_id": observation.session_id,
                "task": observation.task,
                "images": {
                    key: self._image_summary(
                        image,
                        session_dir=session_dir,
                        key=key,
                        step_index=result.step_index,
                    )
                    for key, image in observation.images.items()
                },
                "state": observation.state or {},
                "metadata": observation.metadata,
            },
            "session_state": state.to_dict(),
        }
        jsonl_path = session_dir / "trajectory.jsonl"
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return jsonl_path

    def _image_summary(
        self,
        image: ImageInput,
        session_dir: Path,
        key: str,
        step_index: int,
    ) -> dict[str, Any]:
        summary = {
            "type": image.type,
            "mime_type": image.mime_type,
            "path": image.path,
            "data_length": len(image.data) if image.data else 0,
        }
        if self.copy_images:
            image_dir = session_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            dst = image_dir / f"step_{step_index:06d}_{key}.jpg"

            if image.type == "base64" and image.data:
                import base64
                raw = base64.b64decode(image.data)
                dst.write_bytes(raw)
                summary["copied_path"] = str(dst)
            elif image.type == "path":
                src = Path(image.data)
                if src.exists():
                    shutil.copy2(src, dst)
                    summary["copied_path"] = str(dst)
        return summary
