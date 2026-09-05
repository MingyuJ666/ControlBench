"""The ITI steering method: sparse per-head pushes at the last position.

Mechanics from the official intervener (honest_llama interveners.py): for
each selected head, dose * sigma_head * unit_direction is added to that
head's slice of the attention output -- the o_proj input -- at the LAST
position of every forward pass. Under a KV cache that is every generated
token; prompt positions are untouched except the final one, exactly as the
official code behaves.

Dose here is ITI's alpha (their experiments use ~15): dimensionless, scaled
per head by that head's own projected std. It is NOT the raw-norm-units
dose of the residual methods -- dose semantics are method-defined, and the
per-head sigma scaling is the method's identity.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict

import torch

from ..base import Method


class ITI(Method):
    name = "iti"

    def prepare(self, model, tokenizer) -> None:
        data = torch.load(self.config["vector"])
        by_layer = defaultdict(list)
        for (layer, head), direction, sigma in zip(
                data["heads"], data["directions"], data["proj_std"]):
            by_layer[layer].append((head, direction, float(sigma)))
        self.by_layer = dict(by_layer)
        self.head_dim = data["directions"].shape[1]

    @contextlib.contextmanager
    def hooks(self, model, dose: float):
        if dose == 0:
            yield
            return
        device, dtype = model.device, model.dtype
        adds = {layer: [(head, (direction * sigma * dose).to(device=device,
                                                            dtype=dtype))
                        for head, direction, sigma in entries]
                for layer, entries in self.by_layer.items()}

        def make(layer):
            def hook(module, args, kwargs=None):
                hidden = args[0]                       # [batch, seq, H*D]
                shaped = hidden.view(*hidden.shape[:-1], -1, self.head_dim)
                for head, delta in adds[layer]:
                    shaped[:, -1, head, :] += delta    # last position only
                return None                            # modified in place
            return hook

        handles = [model.model.layers[layer].self_attn.o_proj
                   .register_forward_pre_hook(make(layer))
                   for layer in adds]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
