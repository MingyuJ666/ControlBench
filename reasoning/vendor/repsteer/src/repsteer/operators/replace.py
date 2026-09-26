"""Linear activation replacement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from repsteer.core import StepContext

from .base import Operator, replacement_tensor, strength_tensor


@dataclass(frozen=True, slots=True)
class Replace(Operator):
    """Interpolate toward an artifact vector: ``(1-alpha)h + alpha*v``."""

    def apply(
        self,
        activation: Tensor,
        artifact: Any,
        strength: Tensor | float,
        context: StepContext,
    ) -> Tensor:
        target = replacement_tensor(artifact, activation)
        alpha = strength_tensor(strength, activation)
        return activation + alpha * (target - activation)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "replace"}


__all__ = ["Replace"]
