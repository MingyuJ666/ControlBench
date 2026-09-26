from __future__ import annotations

from typing import Any

from torch import nn

from .base import DecoderOnlyAdapter, PathTensorAccessor, ResolvedSite, _site_error


class GemmaAdapter(DecoderOnlyAdapter):
    architecture_name = "gemma"
    model_types = frozenset({"gemma", "gemma2"})
    class_prefixes = ("GemmaFor", "GemmaModel", "Gemma2")

    def resolve(self, model: nn.Module, site: Any) -> ResolvedSite:
        resolved = super().resolve(model, site)
        model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
        if model_type != "gemma2" or getattr(site, "component", None) != "resid_mid":
            return resolved

        # Gemma2 normalizes the raw attention output before adding the residual:
        # post_attention_layernorm(attn_out) + resid_pre.  The semantic
        # attention-residual midpoint is therefore the input to
        # pre_feedforward_layernorm, not post_attention_layernorm (which is the
        # correct midpoint hook for Gemma1/Llama/Mistral/Qwen2).
        layer_index = int(site.layer)
        layers, layers_path = self._layers(model)
        layer = layers[layer_index]
        module = getattr(layer, "pre_feedforward_layernorm", None)
        if not isinstance(module, nn.Module):
            raise _site_error(
                f"Gemma2 decoder layer {layer_index} has no "
                "pre_feedforward_layernorm module"
            )
        explicit_path = tuple(getattr(site, "tensor_path", ()) or ())
        return ResolvedSite(
            site=site,
            module=module,
            module_path=f"{layers_path}.{layer_index}.pre_feedforward_layernorm",
            hook_kind="forward_pre",
            tensor_accessor=PathTensorAccessor(explicit_path or (0,)),
            hidden_dim=resolved.hidden_dim,
            architecture_name=self.architecture_name,
        )
