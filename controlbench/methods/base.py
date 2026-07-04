"""Pluggable steering-method interface for ControlBench.

A method decides, at eval time, how to intervene on the model. To keep the
eval harness method-agnostic, a method returns a `SteerConfig` describing a
single-layer additive intervention (layer, vector, coeff). Baseline returns
None (no intervention). Future methods can subclass and override.
"""

from dataclasses import dataclass
import torch


@dataclass
class SteerConfig:
    layer: int
    vector: torch.Tensor  # shape [d_model]
    coeff: float


class SteeringMethod:
    name = "abstract"

    def steer_config(self):
        """Return a SteerConfig to apply during generation, or None for no-op."""
        raise NotImplementedError
