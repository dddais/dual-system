"""VLM adapters for standalone Dual-System."""

from dualsystem.vlm.base import VLMClient
from dualsystem.vlm.openai_compatible import OpenAICompatibleVLMClient

__all__ = ["OpenAICompatibleVLMClient", "VLMClient"]
