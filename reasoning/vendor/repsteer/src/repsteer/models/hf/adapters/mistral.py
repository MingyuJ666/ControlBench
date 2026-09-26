from __future__ import annotations

from .base import DecoderOnlyAdapter


class MistralAdapter(DecoderOnlyAdapter):
    architecture_name = "mistral"
    model_types = frozenset({"mistral"})
    class_prefixes = ("Mistral",)
