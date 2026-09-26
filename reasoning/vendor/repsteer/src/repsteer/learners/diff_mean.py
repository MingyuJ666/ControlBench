"""Difference-in-means learner. See :mod:`repsteer.learners.actadd` for CAA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from torch import Tensor

from repsteer.capture import ActivationStore
from repsteer.data import ContrastivePairs

from .base import (
    ContrastiveLearner,
    activation_matrix,
    artifact_metadata,
    normalize_vector,
    sample_weights,
    weighted_mean,
)


def compute_diff_mean(
    positive: Tensor,
    negative: Tensor,
    *,
    positive_weights: Tensor | None = None,
    negative_weights: Tensor | None = None,
    normalize: str | None = "l2",
) -> Tensor:
    """Compute positive weighted mean minus negative weighted mean."""

    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("positive and negative activations must be matrices")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative hidden sizes differ")
    direction = weighted_mean(positive, positive_weights) - weighted_mean(
        negative, negative_weights
    )
    return normalize_vector(direction, normalize)


diff_mean = compute_diff_mean


@dataclass
class DiffMean(ContrastiveLearner):
    """Learn a direction from independently weighted positive/negative groups.

    ``normalize`` defaults to ``"l2"``: repsteer's own convention, so the
    returned direction has unit norm and callers control magnitude entirely
    through ``strength``. This is a design choice, not a specific paper's
    reproduction — mass-mean-difference readers such as RepE's
    ``ClusterMeanRepReader`` return the raw, unnormalized ``mean(positive) -
    mean(negative)`` vector instead. Pass ``normalize=None`` to match that
    raw-vector convention.
    """

    method: ClassVar[str] = "diff_mean"

    def fit(
        self,
        model: Any,
        data: ContrastivePairs,
        *,
        store: ActivationStore | None = None,
    ) -> Any:
        from repsteer.artifacts import DirectionArtifact

        positive_batch, negative_batch = self._capture_pair(model, data, store)
        positive = activation_matrix(positive_batch, name="positive activations")
        negative = activation_matrix(negative_batch, name="negative activations")
        direction = (
            compute_diff_mean(
                positive,
                negative,
                positive_weights=sample_weights(positive_batch, positive),
                negative_weights=sample_weights(negative_batch, negative),
                normalize=self.normalize,
            )
            .detach()
            .to(device="cpu", dtype=positive.dtype)
        )
        metadata = artifact_metadata(
            learner=self,
            model=model,
            data=data,
            artifact_type="direction",
            hidden_size=direction.numel(),
            extra_config={
                "positive_samples": positive.shape[0],
                "negative_samples": negative.shape[0],
                "weighted": True,
            },
        )
        return DirectionArtifact(metadata=metadata, direction=direction)


# CAA is mathematically the paired-difference form and is implemented in actadd.


__all__ = ["DiffMean", "compute_diff_mean", "diff_mean"]
