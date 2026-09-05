"""The SADI steering method: amplify the input's own signal, once.

Mechanics from the official code (steer_heads): at the last prompt token
only -- the prefill pass; generation steps are untouched -- the selected
heads' activations are SCALED, not shifted: the push direction is whatever
the input itself activated, which is what makes the intervention
semantics-adaptive. No vector is added anywhere.

Dose mapping, because multiplication breaks the dose-0-is-a-no-op contract
if taken raw: the paper's update A' = A + delta * (A ⊙ M) is multiplication
by (1 + delta), so our dose IS delta -- dose 0 multiplies by 1 (no-op), and
the official strength sweep 5..80 corresponds to doses 4..79.

Config: {"vector": <sadi artifact>, "heads": int (how many, official best
1..10, default 8), "side": "pos" to push the axis behaviour, "neg" for the
opposite}.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict

import torch

from ..base import Method


class SADI(Method):
    name = "sadi"

    def prepare(self, model, tokenizer) -> None:
        data = torch.load(self.config["vector"])
        count = int(self.config.get("heads", 8))
        side = self.config.get("side", "pos")
        chosen = data[f"heads_{side}"][:count]
        by_layer = defaultdict(list)
        for layer, head in chosen:
            by_layer[layer].append(head)
        self.by_layer = dict(by_layer)
        # head_dim is declared on GQA configs (Gemma 2: 256, not 3584/16);
        # only fall back to the even-split assumption when it is absent.
        self.head_dim = getattr(model.config, "head_dim", None) or (
            model.config.hidden_size // model.config.num_attention_heads)

    @contextlib.contextmanager
    def hooks(self, model, dose: float):
        if dose == 0:
            yield
            return
        factor = 1.0 + dose

        def make(layer):
            def hook(module, args, kwargs=None):
                hidden = args[0]                        # [batch, seq, H*D]
                if hidden.shape[1] <= 1:
                    return None                         # prefill only
                shaped = hidden.view(*hidden.shape[:-1], -1, self.head_dim)
                for head in self.by_layer[layer]:
                    shaped[:, -1, head, :] *= factor    # last prompt token
                return None
            return hook

        handles = [model.model.layers[layer].self_attn.o_proj
                   .register_forward_pre_hook(make(layer))
                   for layer in self.by_layer]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
