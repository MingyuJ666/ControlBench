"""Strength schedule protocol and broadcast helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from repsteer.core import StepContext


class StrengthSchedule(ABC):
    """Compute an intervention coefficient for the current activation."""

    @property
    def is_always_zero(self) -> bool:
        """Whether the schedule is provably zero without evaluating an activation."""

        return False

    @abstractmethod
    def value(self, activation: Tensor, context: StepContext) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def __call__(self, activation: Tensor, context: StepContext) -> Tensor:
        return self.value(activation, context)


def as_strength_tensor(value: Any, activation: Tensor) -> Tensor:
    tensor = torch.as_tensor(value, device=activation.device, dtype=activation.dtype)
    # A [B, S] strength naturally controls token activations [B, S, D].
    while tensor.ndim < activation.ndim:
        tensor = tensor.unsqueeze(-1)
    try:
        torch.broadcast_to(tensor, activation.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Strength shape {tuple(tensor.shape)} cannot broadcast to activation "
            f"shape {tuple(activation.shape)}"
        ) from exc
    return tensor


def strength_schedule_from_dict(value: Mapping[str, Any]) -> StrengthSchedule:
    from .constant import Constant
    from .norm import NormRelative

    kind = str(value.get("type", "")).lower()
    if kind == "constant":
        return Constant(value.get("alpha", value.get("value", 1.0)))
    if kind in ("norm_relative", "normrelative"):
        return NormRelative(
            value.get("ratio", 1.0),
            p=value.get("p", 2.0),
            eps=value.get("eps", 0.0),
        )
    raise ValueError(f"Unknown strength schedule type {kind!r}")


__all__ = [
    "StrengthSchedule",
    "as_strength_tensor",
    "strength_schedule_from_dict",
]
