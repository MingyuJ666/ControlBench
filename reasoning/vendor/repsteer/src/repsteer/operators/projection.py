"""Direction/subspace projection removal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from repsteer.artifacts import SubspaceArtifact
from repsteer.core import StepContext

from .base import Operator, direction_tensor, strength_tensor


@dataclass(frozen=True, slots=True)
class RemoveProjection(Operator):
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("RemoveProjection.eps must be positive")
        object.__setattr__(self, "eps", float(self.eps))

    def _subspace_projection(
        self, activation: Tensor, artifact: SubspaceArtifact
    ) -> Tensor:
        basis = artifact.basis.to(device=activation.device, dtype=activation.dtype)
        if basis.shape[-1] != activation.shape[-1]:
            raise ValueError(
                f"Subspace hidden size {basis.shape[-1]} does not match activation "
                f"hidden size {activation.shape[-1]}"
            )
        centered = activation
        if artifact.mean is not None:
            centered = centered - artifact.mean.to(
                device=activation.device, dtype=activation.dtype
            )
        compute_dtype = (
            torch.float32
            if activation.dtype in (torch.float16, torch.bfloat16)
            else activation.dtype
        )
        x = centered.to(compute_dtype)
        rows = basis.to(compute_dtype)
        gram = rows @ rows.transpose(-1, -2)
        identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        # pinv handles non-orthonormal and mildly rank-deficient learned bases.
        inverse = torch.linalg.pinv(gram + self.eps * identity)
        coefficients = (x @ rows.transpose(-1, -2)) @ inverse
        return cast(Tensor, coefficients @ rows).to(dtype=activation.dtype)

    def apply(
        self,
        activation: Tensor,
        artifact: Any,
        strength: Tensor | float,
        context: StepContext,
    ) -> Tensor:
        if isinstance(artifact, SubspaceArtifact):
            projection = self._subspace_projection(activation, artifact)
        else:
            direction = direction_tensor(artifact, activation)
            compute_dtype = (
                torch.float32
                if activation.dtype in (torch.float16, torch.bfloat16)
                else activation.dtype
            )
            x = activation.to(compute_dtype)
            vector = direction.to(compute_dtype)
            denominator = vector.square().sum() + self.eps
            coefficient = torch.einsum("...d,d->...", x, vector) / denominator
            projection = (coefficient[..., None] * vector).to(dtype=activation.dtype)
        alpha = strength_tensor(strength, activation)
        return activation - alpha * projection

    def to_dict(self) -> dict[str, Any]:
        return {"type": "remove_projection", "eps": self.eps}


__all__ = ["RemoveProjection"]
