"""Paired activation addition / contrastive activation addition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import Tensor

from repsteer.capture import ActivationStore
from repsteer.data import ContrastivePairs

from .base import (
    ContrastiveLearner,
    activation_matrix,
    artifact_metadata,
    normalize_vector,
    weighted_mean,
)


def paired_difference_mean(
    positive: Tensor,
    negative: Tensor,
    *,
    weights: Tensor | None = None,
    normalize: str | None = "l2",
) -> Tensor:
    """Mean per-example ``positive - negative`` contrastive activation."""

    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("positive and negative activations must be matrices")
    if positive.shape != negative.shape:
        raise ValueError(
            "paired positive and negative activations must have equal shape"
        )
    return normalize_vector(weighted_mean(positive - negative, weights), normalize)


compute_actadd = paired_difference_mean


@dataclass
class ActAdd(ContrastiveLearner):
    """Learn activation addition from aligned positive/negative examples.

    ``normalize`` defaults to ``None``, so :meth:`fit` returns the raw
    ``mean(positive - negative)`` vector rather than a unit-length direction.
    This matches the official CAA implementation (Rimsky et al., 2023,
    ``nrimsky/CAA``) and the official ActAdd/algebraic-value-editing
    implementation (Turner et al., 2023, ``montemac/activation_additions``),
    neither of which normalizes the mean-difference vector: both save/use the
    raw activation difference and apply their reported ``multiplier``/
    ``coeff`` directly against that scale. Pass ``normalize="l2"`` for a
    unit-length direction whose magnitude is controlled entirely by
    ``strength`` instead.
    """

    normalize: str | None = None
    method: ClassVar[str] = "actadd"

    def fit(
        self,
        model: Any,
        data: ContrastivePairs,
        *,
        store: ActivationStore | None = None,
    ) -> Any:
        from repsteer.artifacts import DirectionArtifact

        if not data.paired:
            raise ValueError("ActAdd requires paired positive/negative records")
        positive_batch, negative_batch = self._capture_pair(model, data, store)
        positive = activation_matrix(positive_batch, name="positive activations")
        negative = activation_matrix(negative_batch, name="negative activations")
        weights = torch.as_tensor(
            data.pair_weights, dtype=positive.dtype, device=positive.device
        )
        direction = (
            paired_difference_mean(
                positive,
                negative,
                weights=weights,
                normalize=self.normalize,
            )
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        metadata = artifact_metadata(
            learner=self,
            model=model,
            data=data,
            artifact_type="direction",
            hidden_size=direction.numel(),
            extra_config={"pairs": positive.shape[0], "paired_differences": True},
        )
        return DirectionArtifact(metadata=metadata, direction=direction)


@dataclass
class CAA(ActAdd):
    """Contrastive Activation Addition (paired ActAdd estimator).

    Inherits :class:`ActAdd`'s ``normalize=None`` default, so the artifact
    holds the raw paired-difference-of-means vector, as in the official
    ``nrimsky/CAA`` implementation.
    """

    method: ClassVar[str] = "caa"


ContrastiveActivationAddition = CAA


__all__ = [
    "ActAdd",
    "CAA",
    "ContrastiveActivationAddition",
    "compute_actadd",
    "paired_difference_mean",
]
