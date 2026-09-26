"""Activation-norm-relative intervention strength."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch import Tensor

from repsteer.core import StepContext

from .base import StrengthSchedule


@dataclass(frozen=True, slots=True)
class NormRelative(StrengthSchedule):
    """Return ``ratio * ||activation||_p`` for every token.

    This schedule assumes a normalized direction artifact, so the magnitude of
    the added vector is a fixed fraction of the current residual norm.
    """

    ratio: float = 1.0
    p: float = 2.0
    eps: float = 0.0

    def __post_init__(self) -> None:
        if self.p <= 0:
            raise ValueError("NormRelative.p must be positive")
        if self.eps < 0:
            raise ValueError("NormRelative.eps cannot be negative")
        object.__setattr__(self, "ratio", float(self.ratio))
        object.__setattr__(self, "p", float(self.p))
        object.__setattr__(self, "eps", float(self.eps))

    def value(self, activation: Tensor, context: StepContext) -> Tensor:
        if activation.ndim < 1:
            raise ValueError("NormRelative requires a hidden dimension")
        # CPU linalg does not implement all half dtypes.  Calculate in float32
        # when necessary and return the activation dtype for downstream math.
        work = (
            activation.float()
            if activation.dtype in (torch.float16, torch.bfloat16)
            else activation
        )
        norm = cast(Tensor, torch.linalg.vector_norm(work, ord=self.p, dim=-1))
        if self.eps:
            norm = norm.clamp_min(self.eps)
        return (norm * self.ratio).to(dtype=activation.dtype)

    @property
    def is_always_zero(self) -> bool:
        return self.ratio == 0.0

    def with_ratio(self, ratio: float) -> NormRelative:
        return replace(self, ratio=ratio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "norm_relative",
            "ratio": self.ratio,
            "p": self.p,
            "eps": self.eps,
        }


__all__ = ["NormRelative"]
