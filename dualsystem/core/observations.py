"""Observation conversion helpers for realtime and standalone entry points."""

from __future__ import annotations

from typing import Any

from dualsystem.core.types import Observation, to_jsonable


def observation_from_raw_obs(
    raw_obs: dict[str, Any],
    session_id: str,
    task: str,
    camera_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Observation:
    """Build an ``Observation`` from a raw ``get_obs()`` style dictionary."""
    frames = raw_obs.get("frames", {})
    images = frames_to_image_inputs(frames, camera_names or [])
    if not images:
        raise RuntimeError("get_obs() returned no camera frames")
    return Observation(
        session_id=session_id,
        task=task,
        images=images,
        state=to_jsonable(raw_obs.get("state", {})),
        metadata=metadata or {},
    )


def frames_to_image_inputs(
    frames: dict[str, Any],
    camera_names: list[str] | None = None,
):
    """Convert selected raw camera frames into serializable image inputs."""
    from dualsystem.io.image import normalize_image

    selected_names = camera_names or list(frames.keys())
    images = {}
    for name in selected_names:
        if name not in frames:
            available = ", ".join(sorted(frames.keys())) or "(none)"
            raise KeyError(f"Camera '{name}' not found. Available cameras: {available}")
        images[name] = normalize_image(frames[name])
    return images
