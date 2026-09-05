"""The CAST steering method: gate the push on a learned condition.

Mechanics verified against IBM's official code (leash_layer.py @52be602):

- On the PROMPT pass only (detected as seq_len > 1; generation steps under a
  KV cache have seq_len 1), the hidden state entering the condition layer is
  aggregated over prompt tokens (mean, their default), projected through the
  rank-1 projector P = vv^T/(v.v) of the condition direction, squashed with
  tanh, and compared to the raw state by cosine. Similarity past the
  threshold => the condition latches ON for the whole generation.
- While ON, the behavior direction (unit norm) times dose is added to the
  INPUT of every layer in the behavior band -- official default band
  [10..15]; band and condition layer are method hyperparameters calibrated
  on TRAIN pairs, never on dev items.
- Their "OOI preventive normalization" is applied: if the addition GREW the
  hidden-state norm, scale back to the original norm (shrinkage is left
  alone).

Departures from the official code, by necessity: theirs assumes batch size 1
(`hidden_states[0]`); here the condition is evaluated per batch row and the
push is masked to the rows whose condition fired. Batched generation pads on
the left, so the condition pooling uses the 2D attention mask (captured at
the model's own forward) to exclude padding -- otherwise a sample's gate
would depend on its batch neighbours, and calibration ran unpadded.

Config: {"vector": <cast artifact>, "condition_layer": int (official id),
"threshold": float, "behavior_layers": [ints], "comparator": "larger"}.
"""

from __future__ import annotations

import contextlib

import torch

from ..base import Method


class CAST(Method):
    name = "cast"

    def prepare(self, model, tokenizer) -> None:
        data = torch.load(self.config["vector"])
        vecs = data["vectors"]
        calibration = data.get("calibration", {})
        cond_id = self.config.get("condition_layer",
                                  calibration.get("condition_layer"))
        self.config["condition_layer"] = cond_id        # for hooks()
        self.behavior_ids = list(self.config.get("behavior_layers",
                                                 range(10, 16)))
        # brick row R = official layer id R-1  =>  official L -> row L+1
        self.cond_dir = vecs["condition"][cond_id + 1]
        self.behav = {L: vecs["behavior"][L + 1] for L in self.behavior_ids}
        self.threshold = float(self.config.get("threshold",
                                               calibration.get("threshold")))
        self.comparator = self.config.get(
            "comparator", calibration.get("comparator", "larger"))

    @contextlib.contextmanager
    def hooks(self, model, dose: float):
        if dose == 0:
            yield
            return
        cond_id = self.config["condition_layer"]
        device, dtype = model.device, model.dtype
        v = self.cond_dir.to(device=device, dtype=torch.float32)
        projector = torch.outer(v, v) / torch.dot(v, v)
        behav = {L: (d.to(device=device, dtype=dtype) * dose)
                 for L, d in self.behav.items()}
        state = {"met": None, "mask": None}              # per-row latch

        def mask_hook(module, args, kwargs):
            state["mask"] = kwargs.get("attention_mask")
            return None

        def condition_hook(module, args, kwargs=None):
            hidden = args[0] if args else kwargs["hidden_states"]
            if hidden.shape[1] > 1:                      # prompt pass only
                mask = state["mask"]
                if mask is not None and mask.dim() == 2 \
                        and mask.shape == hidden.shape[:2]:
                    weights = mask.to(hidden.dtype).unsqueeze(-1)
                    pooled = ((hidden * weights).sum(dim=1)
                              / weights.sum(dim=1).clamp_min(1)).float()
                else:                                    # unpadded fallback
                    pooled = hidden.float().mean(dim=1)  # [batch, hidden]
                proj = torch.tanh(pooled @ projector.T)
                sim = torch.nn.functional.cosine_similarity(pooled, proj, dim=-1)
                met = sim < self.threshold if self.comparator == "larger" \
                    else sim > self.threshold
                state["met"] = met                       # [batch] bool
            return None

        def behavior_hook(layer_id):
            def hook(module, args, kwargs=None):
                hidden = args[0] if args else kwargs["hidden_states"]
                met = state["met"]
                if met is None or not met.any():
                    return None
                original = hidden.norm(dim=-1, keepdim=True)
                pushed = hidden + behav[layer_id] * met.view(-1, 1, 1).to(hidden.dtype)
                new = pushed.norm(dim=-1, keepdim=True)
                if (new / original).max() > 1:           # OOI: shrink back growth
                    pushed = pushed * (original / new)
                if args:
                    return (pushed,) + args[1:]
                kwargs["hidden_states"] = pushed
                return None
            return hook

        layers = model.model.layers
        handles = [model.model.register_forward_pre_hook(mask_hook,
                                                         with_kwargs=True),
                   layers[cond_id].register_forward_pre_hook(condition_hook)]
        handles += [layers[L].register_forward_pre_hook(behavior_hook(L))
                    for L in self.behavior_ids]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
