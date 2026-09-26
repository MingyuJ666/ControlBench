"""Centered SVD principal-component learner."""

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
    canonicalize_component_signs,
    sample_weights,
    weighted_mean,
)


def principal_components(
    activations: Tensor,
    *,
    n_components: int = 1,
    weights: Tensor | None = None,
    center: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(basis, mean, explained_variance)`` using deterministic SVD signs."""

    if activations.ndim != 2:
        raise ValueError("PCA activations must be [samples, hidden]")
    samples, hidden = activations.shape
    if not 1 <= n_components <= min(samples, hidden):
        raise ValueError(
            f"n_components must be in [1, {min(samples, hidden)}], got {n_components}"
        )
    if weights is None:
        weights = torch.ones(
            samples, dtype=activations.dtype, device=activations.device
        )
    else:
        weights = torch.as_tensor(
            weights, dtype=activations.dtype, device=activations.device
        ).flatten()
    mean = (
        weighted_mean(activations, weights)
        if center
        else torch.zeros(hidden, dtype=activations.dtype, device=activations.device)
    )
    centered = activations - mean if center else activations
    weighted = centered * weights.clamp_min(0).sqrt()[:, None]
    # full_matrices=False avoids constructing a hidden_size squared matrix.
    _, singular_values, vh = torch.linalg.svd(weighted, full_matrices=False)
    basis = canonicalize_component_signs(vh[:n_components])
    denominator = max(float(weights.sum()) - 1.0, 1.0)
    explained_variance = singular_values[:n_components].square() / denominator
    return basis, mean, explained_variance


compute_pca = principal_components


def _orient_by_group_difference(
    basis: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    positive_weights: Tensor,
    negative_weights: Tensor,
) -> Tensor:
    """Flip each component to align with ``mean(positive) - mean(negative)``.

    ``principal_components()`` only resolves the arbitrary SVD sign
    deterministically (largest-magnitude coordinate positive), which carries
    no relation to the positive/negative labels. This reorients each
    component toward the group that actually carries the "positive" label,
    so a rank-1 result is safe to hand to an additive steering operator
    without an external sign check. A near-zero alignment (e.g. the group
    means coincide) leaves the deterministic magnitude-based sign as-is
    rather than flipping on noise.
    """

    result = basis.clone()
    group_difference = weighted_mean(positive, positive_weights) - weighted_mean(
        negative, negative_weights
    )
    alignment = result @ group_difference.to(dtype=result.dtype)
    result[alignment < 0] *= -1
    return result


@dataclass
class PCA(ContrastiveLearner):
    """Centered PCA/SVD over pooled positive and negative activations.

    This pools ``positive`` and ``negative`` activations into one matrix
    (``data.paired`` is not required) and centers by their combined mean, so
    the returned components capture whatever axes carry the most variance in
    that pooled cloud — not necessarily the positive/negative contrast. Each
    component's sign is then oriented toward the positive group via
    :func:`_orient_by_group_difference` (aligned with
    ``mean(positive) - mean(negative)``), so it is safe to use directly with
    an additive steering operator.

    This is still **not** RepE's PCA reading vector (Zou et al., 2023): the
    official implementation always PCAs the *paired difference*
    ``positive - negative`` (see ``repe/rep_reading_pipeline.py``'s
    unconditional adjacent-pair differencing before every
    ``direction_method``, including ``'pca'``), which cancels any variance
    shared by both groups before extracting components — this class does
    not, so a component may still track a confound rather than the intended
    contrast, even though its sign is now well-defined.
    :class:`~repsteer.learners.LAT` reproduces the official method
    faithfully, including its pairwise sign vote; prefer
    ``LAT(n_components=1)`` (which requires ``data.paired``) when you want
    that stronger guarantee, and reserve ``PCA`` for independent/unpaired
    groups or exploratory use.
    """

    n_components: int = 1
    center: bool = True
    normalize: str | None = "orthonormal"
    method: ClassVar[str] = "pca"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_components = int(self.n_components)
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")

    def fit(
        self,
        model: Any,
        data: ContrastivePairs,
        *,
        store: ActivationStore | None = None,
    ) -> Any:
        from repsteer.artifacts import SubspaceArtifact

        positive_batch, negative_batch = self._capture_pair(model, data, store)
        positive = activation_matrix(positive_batch, name="positive activations")
        negative = activation_matrix(negative_batch, name="negative activations")
        positive_weights = sample_weights(positive_batch, positive)
        negative_weights = sample_weights(negative_batch, negative)
        activations = torch.cat((positive, negative), dim=0)
        weights = torch.cat((positive_weights, negative_weights), dim=0)
        basis, mean, explained_variance = principal_components(
            activations,
            n_components=self.n_components,
            weights=weights,
            center=self.center,
        )
        basis = _orient_by_group_difference(
            basis,
            positive,
            negative,
            positive_weights=positive_weights,
            negative_weights=negative_weights,
        )
        basis = basis.detach().to(device="cpu", dtype=torch.float32)
        mean = mean.detach().to(device="cpu", dtype=torch.float32)
        explained_variance = explained_variance.detach().to(
            device="cpu", dtype=torch.float32
        )
        metadata = artifact_metadata(
            learner=self,
            model=model,
            data=data,
            artifact_type="subspace",
            hidden_size=basis.shape[1],
            extra_config={
                "n_components": self.n_components,
                "center": self.center,
                "samples": activations.shape[0],
                "svd": "torch.linalg.svd",
                "sign_alignment": "positive_negative_mean_difference",
            },
        )
        return SubspaceArtifact(
            metadata=metadata,
            basis=basis,
            mean=mean,
            explained_variance=explained_variance,
        )


__all__ = ["PCA", "compute_pca", "principal_components"]
