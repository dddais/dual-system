"""Local Qwen-VL adapter for standalone Dual-System."""

from __future__ import annotations

import re
from typing import Any

from dualsystem.core.types import ImageInput, PlannerInput, PlannerOutput
from dualsystem.io.image import image_input_to_pil

_THINK_END_TOKEN_ID = 151668


class LocalQwenVLMClient:
    """Run Qwen2.5-VL or Qwen3-VL locally through HuggingFace Transformers."""

    def __init__(
        self,
        model_path: str,
        model_family: str = "auto",
        dtype: str = "bf16",
        device: str = "auto",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        default_sampling_params: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = model_path
        self.model_family = _infer_model_family(model_path, model_family)
        self.dtype = dtype
        self.device = device
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.default_sampling_params = default_sampling_params or {}
        self._model = None
        self._processor = None
        self._process_vision_info = None

    def generate(self, planner_input: PlannerInput) -> PlannerOutput:
        import torch

        self._ensure_loaded()
        with torch.no_grad():
            return self._generate_impl(planner_input)

    def _generate_impl(self, planner_input: PlannerInput) -> PlannerOutput:
        messages = self._build_messages(planner_input)
        texts = [
            self._processor.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        image_inputs, video_inputs = self._process_vision_info([[m] for m in messages])
        inputs = self._processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        sampling_params = {
            **self.default_sampling_params,
            **planner_input.sampling_params,
        }
        generated_ids = self._model.generate(**inputs, **sampling_params)
        outputs = []
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids):
            trimmed_ids = output_ids[len(input_ids) :].tolist()
            if self.model_family == "qwen3":
                outputs.append(self._strip_thinking(trimmed_ids))
            else:
                outputs.append(
                    self._processor.decode(
                        trimmed_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
        return PlannerOutput(raw_outputs=outputs)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        torch_dtype = _torch_dtype(torch, self.dtype)
        if self.model_family == "qwen3":
            from transformers import Qwen3VLForConditionalGeneration

            model_cls = Qwen3VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration

        self._model = model_cls.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self._processor.tokenizer.padding_side = "left"
        if self.device != "auto":
            self._model.to(self.device)
        self._model.eval()
        self._process_vision_info = process_vision_info

    def _build_messages(self, planner_input: PlannerInput) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for image in planner_input.observation.images.values():
            content.append({"type": "image", "image": _to_pil(image)})
        content.append({"type": "text", "text": planner_input.prompt})
        return [{"role": "user", "content": content}]

    def _strip_thinking(self, output_ids: list[int]) -> str:
        try:
            idx = len(output_ids) - 1 - output_ids[::-1].index(_THINK_END_TOKEN_ID)
            content_ids = output_ids[idx + 1 :]
        except ValueError:
            content_ids = output_ids
        text = self._processor.decode(
            content_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _to_pil(image: ImageInput):
    try:
        return image_input_to_pil(image)
    except ImportError as exc:
        raise RuntimeError("Pillow is required to use LocalQwenVLMClient") from exc


def _infer_model_family(model_path: str, model_family: str) -> str:
    if model_family != "auto":
        return model_family.lower()
    lowered = model_path.lower()
    if "qwen3" in lowered:
        return "qwen3"
    return "qwen2.5"


def _torch_dtype(torch_module, dtype: str):
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {dtype}")
