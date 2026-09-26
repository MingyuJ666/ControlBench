"""Constant intervention strength."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from repsteer.core import StepContext

from .base import StrengthSchedule


@dataclass(frozen=True, slots=True)
class Constant(StrengthSchedule):
    alpha: float = 1.0

    def __post_init__(self) -> None:
        value = torch.as_tensor(self.alpha)
        if value.numel() != 1:
            raise ValueError("Constant alpha must be scalar")
        object.__setattr__(self, "alpha", float(value.item()))

    def value(self, activation: Tensor, context: StepContext) -> Tensor:
        return torch.as_tensor(
            self.alpha, device=activation.device, dtype=activation.dtype
        )

    @property
    def is_always_zero(self) -> bool:
        return self.alpha == 0.0

    def with_alpha(self, alpha: float) -> Constant:
        return replace(self, alpha=alpha)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "constant", "alpha": self.alpha}


__all__ = ["Constant"]
