"""The SAE steering method: push along one dictionary feature's direction.

Injection mechanics are identical to dim/caa/probe -- unit direction,
residual addition at the SAE's own layer, dose in raw norm units -- so any
difference in the grid is attributable to WHERE the direction came from: a
sparse dictionary feature rather than a contrast statistic.

Config: {"vector": <sae artifact>, "feature_rank": which stored feature to
use (default 0, the top-scoring one)}. The injection layer is the SAE's
layer, carried inside the artifact; passing "layer" is not supported, since
a decoder direction only means anything at the layer its SAE was trained on.
"""

from __future__ import annotations

import contextlib

import torch

from ..base import Method
from ..shared import residual_add


class SAE(Method):
    name = "sae"

    def prepare(self, model, tokenizer) -> None:
        data = torch.load(self.config["vector"])
        rank = int(self.config.get("feature_rank", 0))
        self.vector = data["directions"][rank]
        self.hidden_idx = data["sae_layer"] + 1   # brick-row convention
        self.feature = data["features"][rank]

    def hooks(self, model, dose: float):
        if dose == 0:
            return contextlib.nullcontext()
        return residual_add.steering(model, self.hidden_idx, self.vector,
                                     dose)   # unit direction: dose = push

    def describe(self) -> dict:
        return {"method": self.name, **self.config,
                "feature": getattr(self, "feature", None),
                "hidden_idx": getattr(self, "hidden_idx", None)}
