"""The probe steering method: inject the classifier's weight direction.

Identical mechanics to DiM -- one direction, one layer, residual addition,
dose = push in norm units -- so any performance difference between the two
is attributable to the estimator, not the plumbing.
"""

from __future__ import annotations

import contextlib

from ..base import Method
from ..shared import residual_add


class Probe(Method):
    name = "probe"

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
