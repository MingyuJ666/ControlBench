"""Cosine-similarity activation gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from repsteer.core import GateContext

from .activation import context_tensor, reduce_token_scores
from .base import Gate


@dataclass(frozen=True, slots=True)
class CosineGate(Gate):
    direction: Any
    threshold: float
    evaluate_at: Any | None = None
    positions: Any | None = None
    reduction: str = "mean"
    activation_key: str | None = None
    eps: float = 1e-8

    def __post_init__(self) -> None:
        threshold = float(self.threshold)
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("CosineGate.threshold must lie in [-1, 1]")
        if self.eps <= 0:
            raise ValueError("CosineGate.eps must be positive")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "eps", float(self.eps))

    def evaluate(self, context: GateContext) -> Tensor:
        activation = context_tensor(context, key=self.activation_key)
        value = getattr(self.direction, "direction", self.direction)
        direction = torch.as_tensor(
            value, device=activation.device, dtype=activation.dtype
        )
        if direction.ndim != 1 or direction.shape[0] != activation.shape[-1]:
            raise ValueError(
                "CosineGate direction must be one-dimensional and match the "
                "activation hidden size"
            )
        scores = F.cosine_similarity(
            activation,
            direction.expand_as(activation),
            dim=-1,
            eps=self.eps,
        )
        pooled, valid = reduce_token_scores(
            scores,
            activation,
            context,
            positions=self.positions,
            reduction=self.reduction,
        )
        return (pooled >= self.threshold) & valid

    def to_dict(self) -> dict[str, Any]:
        fingerprint = (
            self.direction.fingerprint()
            if callable(getattr(self.direction, "fingerprint", None))
            else None
        )
        evaluate_at = self.evaluate_at
        positions = self.positions
        return {
            "type": "cosine",
            "threshold": self.threshold,
            "direction_fingerprint": fingerprint,
            "evaluate_at": (
                evaluate_at.to_dict()
                if evaluate_at is not None and hasattr(evaluate_at, "to_dict")
                else None
            ),
            "positions": (
                positions.to_dict()
                if positions is not None and hasattr(positions, "to_dict")
                else None
            ),
            "reduction": self.reduction,
            "activation_key": self.activation_key,
            "eps": self.eps,
        }


__all__ = ["CosineGate"]
