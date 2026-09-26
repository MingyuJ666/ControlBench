"""Unconditional gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from repsteer.core import GateContext

from .base import Gate


@dataclass(frozen=True, slots=True)
class Always(Gate):
    def evaluate(self, context: GateContext) -> Tensor:
        batch = context.resolved_batch_size()
        return torch.ones(batch, dtype=torch.bool, device=context.runtime_device)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "always"}


__all__ = ["Always"]
