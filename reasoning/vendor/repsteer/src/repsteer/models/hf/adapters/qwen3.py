from __future__ import annotations

from .base import DecoderOnlyAdapter


class Qwen3Adapter(DecoderOnlyAdapter):
    """Semantic-site adapter for Hugging Face Qwen3 decoder-only models.

    Qwen3-8B uses the standard ``model.layers`` decoder layout with
    ``self_attn``, ``post_attention_layernorm``, and ``mlp`` modules, so its
    residual-site contract is the shared decoder-only contract.
    """

    architecture_name = "qwen3"
    model_types = frozenset({"qwen3", "qwen3_moe"})
    # Do not claim Qwen3-Next or Qwen3-VL layouts through the class-name
    # fallback: their decoder structures have separate adapter contracts.
    class_prefixes = ("Qwen3For", "Qwen3Model", "Qwen3Moe")
