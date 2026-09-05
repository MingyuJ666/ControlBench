"""The DiM steering method: one direction, one layer, residual addition.

Config (via --method-config or code):
    vector     path to an extraction archive (.pt)
    layer      hidden_idx to inject at (see shared/residual_add.py)
    position   which read-point's direction to use (default prompt_last)

Dose is the push in raw norm units: alpha = dose / ||v||, so dose 48 means
"push the stream 48 units along v" regardless of how large the extracted
vector happens to be. A negative dose is a real intervention -- it pushes
toward tool-calling -- and only dose 0 is a no-op. DiM's two piles make -v a
meaningful opposite direction, which is what lets one axis serve both
"more clarify" and "more call".
"""

from __future__ import annotations

import contextlib

from ..base import Method
from ..shared import residual_add


class DiM(Method):
    name = "dim"

    def prepare(self, model, tokenizer) -> None:
        self.vector = residual_add.load_vector(
            self.config["vector"],
            self.config.get("position", "prompt_last"),
            self.config["layer"])
        self.norm = self.vector.norm().item()

    def hooks(self, model, dose: float):
        if dose == 0:
            return contextlib.nullcontext()
        return residual_add.steering(model, self.config["layer"], self.vector,
                                     dose / self.norm)
