"""Image normalization helpers."""

from __future__ import annotations

import base64
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any

from dualsystem.core.types import ImageInput


def parse_image_spec(spec: str) -> tuple[str, ImageInput]:
    """Parse ``key=path`` CLI image syntax.

    Supports:
        - ``main=./obs.jpg`` — image file
        - ``main=./video.mp4`` — video file (extracts frame 0)
        - ``main=./video.mp4@5`` — video file, extract frame 5
        - ``main=https://...`` — URL
    """
    if "=" not in spec:
        raise ValueError(f"Image spec must use key=path syntax: {spec}")
    key, value = spec.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Image key must not be empty: {spec}")
    value = value.strip()

    # Check for video@frame syntax: "video.mp4@5" → path="video.mp4", frame=5
    if "@" in value:
        path_part, frame_part = value.rsplit("@", 1)
        path = Path(path_part).expanduser()
        if path.exists() and _is_video_extension(path):
            try:
                frame_index = int(frame_part)
            except ValueError:
                raise ValueError(
                    f"Frame index must be an integer: got '{frame_part}' in {spec}"
                )
            return key, image_from_video(path, frame_index)

    return key, normalize_image(value)


def normalize_image(value: Any, mime_type: str | None = None) -> ImageInput:
    """Normalize path/base64/bytes/PIL/NumPy/Torch inputs into ``ImageInput``."""
    if isinstance(value, ImageInput):
        return value
    if isinstance(value, bytes):
        return image_from_bytes(value, mime_type=mime_type)
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return ImageInput(type="url", data=value)
        if value.startswith("data:image/"):
            header, data = value.split(",", 1)
            return ImageInput(
                type="base64",
                data=data,
                mime_type=header.removeprefix("data:").split(";")[0],
            )
        path = Path(value).expanduser()
        if path.exists():
            if _is_video_extension(path):
                return image_from_video(path, frame_index=0)
            return image_from_path(path)
        return ImageInput(type="base64", data=value, mime_type=mime_type)
    if _looks_like_pil(value):
        return image_from_pil(value)
    if _looks_like_numpy(value):
        return image_from_numpy(value)
    if _looks_like_torch(value):
        return image_from_torch(value)
    raise TypeError(f"Unsupported image input type: {type(value)}")


def image_from_path(path: str | Path) -> ImageInput:
    """Read a local image path as a base64 payload."""
    image_path = Path(path).expanduser()
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return ImageInput(
        type="base64",
        data=data,
        mime_type=mime_type,
        path=str(image_path),
    )


def image_from_bytes(data: bytes, mime_type: str | None = None) -> ImageInput:
    """Create a base64 image payload from bytes."""
    return ImageInput(
        type="base64",
        data=base64.b64encode(data).decode("ascii"),
        mime_type=mime_type or "image/jpeg",
    )


def image_from_pil(image, image_format: str = "JPEG") -> ImageInput:
    """Create an image payload from a PIL image."""
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return image_from_bytes(
        buffer.getvalue(),
        mime_type=f"image/{image_format.lower()}",
    )


def image_from_numpy(array) -> ImageInput:
    """Create an image payload from a NumPy array."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for NumPy image conversion") from exc
    return image_from_pil(Image.fromarray(array))


def image_from_torch(tensor) -> ImageInput:
    """Create an image payload from a Torch tensor."""
    array = tensor.detach().cpu().numpy()
    return image_from_numpy(array)


def image_from_video(
    video_path: str | Path,
    frame_index: int = 0,
) -> ImageInput:
    """Extract a single frame from a video file and return as ``ImageInput``.

    Args:
        video_path: Path to a video file (mp4, avi, mkv, etc.).
        frame_index: Zero-based frame index to extract.

    Returns:
        ``ImageInput`` with the extracted frame encoded as JPEG base64.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for video frame extraction. "
            "Install with: pip install opencv-python"
        ) from exc

    path = Path(video_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0 or (total_frames > 0 and frame_index >= total_frames):
        cap.release()
        raise IndexError(
            f"Frame index {frame_index} out of range "
            f"(video has {total_frames} frames)"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_index} from {path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for video frame conversion") from exc
    result = image_from_pil(Image.fromarray(frame_rgb))
    result.path = str(path)
    return result


def image_to_openai_content(image: ImageInput) -> dict[str, Any]:
    """Convert ``ImageInput`` to OpenAI-compatible message content."""
    if image.type == "url":
        url = image.data
    elif image.type == "base64":
        mime_type = image.mime_type or "image/jpeg"
        url = f"data:{mime_type};base64,{image.data}"
    elif image.type == "path":
        converted = image_from_path(image.data)
        mime_type = converted.mime_type or "image/jpeg"
        url = f"data:{mime_type};base64,{converted.data}"
    else:
        raise ValueError(f"Unsupported image type for OpenAI content: {image.type}")
    return {"type": "image_url", "image_url": {"url": url}}


def image_input_to_pil(image: ImageInput):
    """Convert a base64 or path image input to PIL."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for PIL conversion") from exc

    if image.type == "base64":
        raw = base64.b64decode(image.data)
        return Image.open(BytesIO(raw)).convert("RGB")
    if image.type == "path":
        return Image.open(Path(image.data).expanduser()).convert("RGB")
    raise ValueError(f"Cannot convert image type to PIL: {image.type}")


def _looks_like_pil(value: Any) -> bool:
    return hasattr(value, "save") and hasattr(value, "mode")


def _looks_like_numpy(value: Any) -> bool:
    return value.__class__.__module__.startswith("numpy")


def _looks_like_torch(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch")


_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}


def _is_video_extension(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_EXTENSIONS
