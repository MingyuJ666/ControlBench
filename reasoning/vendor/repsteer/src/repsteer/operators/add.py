"""Additive direction operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from repsteer.core import StepContext

from .base import Operator, direction_tensor, strength_tensor


@dataclass(frozen=True, slots=True)
class Add(Operator):
    is_additive = True

    def apply(
        self,
        activation: Tensor,
        artifact: Any,
        strength: Tensor | float,
        context: StepContext,
    ) -> Tensor:
        direction = direction_tensor(artifact, activation)
        alpha = strength_tensor(strength, activation)
        return activation + alpha * direction

    def to_dict(self) -> dict[str, Any]:
        return {"type": "add"}


@dataclass(frozen=True, slots=True)
class Subtract(Operator):
    is_additive = True

    def apply(
        self,
        activation: Tensor,
        artifact: Any,
        strength: Tensor | float,
        context: StepContext,
    ) -> Tensor:
        direction = direction_tensor(artifact, activation)
        alpha = strength_tensor(strength, activation)
        return activation - alpha * direction

    def to_dict(self) -> dict[str, Any]:
        return {"type": "subtract"}


__all__ = ["Add", "Subtract"]
