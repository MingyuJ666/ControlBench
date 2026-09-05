"""The CAA steering method (Rimsky et al. 2024): the response-contrast
direction, added from the end of the instruction onward.

Two things the paper fixes, so they are not configurable here:

- The direction is the first-response-token contrast (`resp_first`; see
  caa/extract.py for the mapping from the paper's answer-letter token).
- The official code adds the vector at every position from the last token
  of the instruction onward (`position_ids >= from_pos`, utils/helpers.py):
  the final prompt token and every generated token, never the prompt before
  it. Under a KV cache that is the last position of every forward pass,
  which is positions="last" in shared/residual_add.py. Same hook, same
  layer convention as dim/probe/sae -- only the position mask differs, and
  it differs because the paper says so.

Config: vector (artifact path), layer. Dose is the push in raw norm units
(alpha = dose / ||v||); the paper's multiplier m on the raw vector is
dose = m * ||v||.

One caution the extraction inherits: CAA's two piles are NOT symmetric --
the direction is "this behaviour minus tool call" estimated only on prompts
where the behaviour was warranted. A negative dose is still a real push
along -v, but it is not a certified "opposite behaviour" the way DiM's
symmetric construction makes it.
"""

from __future__ import annotations

import contextlib

from ..base import Method
from ..shared import residual_add


class CAA(Method):
    name = "caa"

    def prepare(self, model, tokenizer) -> None:
        if "position" in self.config:
            raise ValueError("caa reads resp_first by definition; "
                             "'position' is not a knob")
        self.vector = residual_add.load_vector(
            self.config["vector"], "resp_first", self.config["layer"])
        self.norm = self.vector.norm().item()

    def hooks(self, model, dose: float):
        if dose == 0:
            return contextlib.nullcontext()
        return residual_add.steering(model, self.config["layer"], self.vector,
                                     dose / self.norm, positions="last")
