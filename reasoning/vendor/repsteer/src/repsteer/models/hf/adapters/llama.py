from __future__ import annotations

from .base import DecoderOnlyAdapter


class LlamaAdapter(DecoderOnlyAdapter):
    architecture_name = "llama"
    model_types = frozenset({"llama"})
    class_prefixes = ("Llama",)
