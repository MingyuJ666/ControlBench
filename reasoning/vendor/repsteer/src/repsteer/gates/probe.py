"""Probe-conditioned sequence gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from repsteer.artifacts import ProbeArtifact
from repsteer.core import GateContext

from .activation import context_tensor, reduce_token_scores
from .base import Gate


@dataclass(frozen=True, slots=True)
class ProbeGate(Gate):
    """Trigger when a probe probability crosses ``threshold``."""

    probe: ProbeArtifact
    threshold: float = 0.5
    evaluate_at: Any | None = None
    positions: Any | None = None
    reduction: str = "mean"
    class_index: int | None = None
    activation_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.probe, ProbeArtifact):
            raise TypeError("ProbeGate.probe must be a ProbeArtifact")
        threshold = float(self.threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("ProbeGate.threshold must lie in [0, 1]")
        if self.class_index is not None and (
            isinstance(self.class_index, bool)
            or not isinstance(self.class_index, int)
            or self.class_index < 0
        ):
            raise ValueError("ProbeGate.class_index must be a non-negative int")
        object.__setattr__(self, "threshold", threshold)

    def evaluate(self, context: GateContext) -> Tensor:
        activation = context_tensor(context, key=self.activation_key)
        probabilities = self.probe.probabilities(activation)
        if self.probe.weight.ndim == 2:
            outputs = self.probe.weight.shape[0]
            class_index = self.class_index
            if class_index is None:
                if outputs != 2:
                    raise ValueError(
                        "A multiclass ProbeGate needs class_index unless the probe "
                        "has exactly two outputs"
                    )
                class_index = 1
            if class_index >= outputs:
                raise IndexError(
                    f"class_index {class_index} is outside {outputs} probe outputs"
                )
            probabilities = probabilities[..., class_index]
        scores, valid = reduce_token_scores(
            probabilities,
            activation,
            context,
            positions=self.positions,
            reduction=self.reduction,
        )
        return (scores >= self.threshold) & valid

    def to_dict(self) -> dict[str, Any]:
        evaluate_at = self.evaluate_at
        positions = self.positions
        return {
            "type": "probe",
            "threshold": self.threshold,
            "probe_fingerprint": self.probe.fingerprint(),
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
            "class_index": self.class_index,
            "activation_key": self.activation_key,
        }


__all__ = ["ProbeGate"]
