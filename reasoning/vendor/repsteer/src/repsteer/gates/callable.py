"""User-defined gate wrapper."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from repsteer.core import GateContext

from .base import Gate


@dataclass(frozen=True, slots=True)
class CallableGate(Gate):
    fn: Callable[[GateContext], Any]
    name: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError("CallableGate.fn must be callable")

    def evaluate(self, context: GateContext) -> Tensor:
        return torch.as_tensor(self.fn(context), device=context.runtime_device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "callable",
            "name": self.name
            or getattr(self.fn, "__qualname__", type(self.fn).__qualname__),
        }


__all__ = ["CallableGate"]
