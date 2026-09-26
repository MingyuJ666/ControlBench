"""Head-result operator used by Inference-Time Intervention (ITI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from repsteer.artifacts import ITIArtifact
from repsteer.core import StepContext

from .base import Operator, strength_tensor


@dataclass(frozen=True, slots=True)
class ITIAdd(Operator):
    """Add calibrated ITI directions to selected query-head results.

    The controlled activation is the flattened input of an attention output
    projection, with layout ``[..., query_heads * head_dim]``. This operator
    reshapes it only long enough to modify the profile's selected heads, then
    restores the exact original layout. It never approximates ITI by editing
    the residual stream after the output projection.
    """

    layer: int
    is_additive = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
        ):
            raise ValueError("ITIAdd.layer must be a non-negative integer")

    def apply(
        self,
        activation: Tensor,
        artifact: Any,
        strength: Tensor | float,
        context: StepContext,
    ) -> Tensor:
        del context
        if not isinstance(artifact, ITIArtifact):
            raise TypeError("ITIAdd requires an ITIArtifact")
        if activation.ndim < 1:
            raise ValueError("ITI head results need a final feature dimension")
        selected = artifact.entries_for_layer(self.layer)
        if not selected:
            raise ValueError(f"ITI profile has no selected heads at layer {self.layer}")
        width = int(activation.shape[-1])
        head_dim = artifact.head_dim
        if width == 0 or width % head_dim:
            raise ValueError(
                "ITI head-result width must be a positive multiple of the profile "
                f"head dimension ({width} vs {head_dim})"
            )
        head_count = width // head_dim
        invalid = [entry.head for _, entry in selected if entry.head >= head_count]
        if invalid:
            raise ValueError(
                f"ITI profile references query head(s) {invalid}, but this layer has "
                f"{head_count} query heads"
            )

        # Build the complete layer delta once. The common alpha schedule is
        # applied afterwards, exactly matching alpha * sigma_lh * theta_lh.
        delta = torch.zeros_like(activation).reshape(
            *activation.shape[:-1], head_count, head_dim
        )
        for index, entry in selected:
            direction = artifact.directions[index].to(
                device=activation.device, dtype=activation.dtype
            )
            delta[..., entry.head, :] = direction * entry.projected_std
        alpha = strength_tensor(strength, activation)
        return activation + alpha * delta.reshape_as(activation)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "iti_add", "layer": self.layer}


__all__ = ["ITIAdd"]
