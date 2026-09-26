from __future__ import annotations

from .base import DecoderOnlyAdapter


class Qwen2Adapter(DecoderOnlyAdapter):
    architecture_name = "qwen2"
    model_types = frozenset({"qwen2", "qwen2_moe"})
    class_prefixes = ("Qwen2",)
