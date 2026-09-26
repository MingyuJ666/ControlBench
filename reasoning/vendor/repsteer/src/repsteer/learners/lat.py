"""Linear Artificial Tomography over paired contrastive differences."""

from __future__ import annotations

import random
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
    normalize_vector,
    weighted_mean,
)


def lat_components(
    positive: Tensor,
    negative: Tensor,
    *,
    weights: Tensor | None = None,
    n_components: int = 1,
    center: bool = True,
    normalize: str | None = "l2",
    seed: int = 42,
    pair_signs: Tensor | None = None,
    shuffle_pair_order: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Learn LAT components from paired activations.

    The official RepE example builders shuffle the two members of each pair
    before their adjacent subtraction.  With explicit positive/negative pairs,
    this is equivalent to applying an independent ``+1`` or ``-1`` sign to each
    ``positive - negative`` difference before centered PCA.  The learned axes
    are then oriented back toward the positive side using a pairwise vote.

    ``pair_signs`` exposes a fixed shuffled order for reproducible comparisons.
    When omitted, ``seed`` controls a local, deterministic emulation of the
    official pair shuffling.  Set ``shuffle_pair_order=False`` to retain the
    supplied pair order, as supported by the lower-level official pipeline.
    """

    if positive.ndim != 2 or negative.ndim != 2 or positive.shape != negative.shape:
        raise ValueError("LAT requires equally shaped positive/negative matrices")
    differences = positive - negative
    samples, hidden = differences.shape
    if not 1 <= n_components <= min(samples, hidden):
        raise ValueError(
            f"n_components must be in [1, {min(samples, hidden)}], got {n_components}"
        )
    if weights is None:
        weights = torch.ones(
            samples, dtype=differences.dtype, device=differences.device
        )
    else:
        weights = torch.as_tensor(
            weights, dtype=differences.dtype, device=differences.device
        ).flatten()
    signs = _lat_pair_signs(
        samples,
        dtype=differences.dtype,
        device=differences.device,
        seed=seed,
        pair_signs=pair_signs,
        shuffle_pair_order=shuffle_pair_order,
    )
    pca_inputs = differences * signs[:, None]
    mean_difference = weighted_mean(differences, weights)
    pca_mean = weighted_mean(pca_inputs, weights)
    matrix = pca_inputs - pca_mean if center else pca_inputs
    weighted = matrix * weights.clamp_min(0).sqrt()[:, None]
    _, singular_values, vh = torch.linalg.svd(weighted, full_matrices=False)
    basis = vh[:n_components].clone()

    # Centered identical differences contain no variance.  Their only meaningful
    # steering direction is the contrastive mean itself.
    if float(torch.linalg.vector_norm(weighted)) <= torch.finfo(weighted.dtype).eps:
        first = normalize_vector(mean_difference.clone(), normalize)
        basis[0] = first
        if n_components > 1:
            basis[1:] = canonicalize_component_signs(basis[1:])
    else:
        basis = _orient_lat_components(
            basis,
            differences,
            weights=weights,
        )
        for index in range(n_components):
            basis[index] = normalize_vector(basis[index], normalize)
    denominator = max(float(weights.sum()) - 1.0, 1.0)
    explained_variance = singular_values[:n_components].square() / denominator
    return basis, mean_difference, explained_variance


def _lat_pair_signs(
    samples: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    pair_signs: Tensor | None,
    shuffle_pair_order: bool,
) -> Tensor:
    """Return the signs induced by independently shuffling each pair."""

    if pair_signs is not None:
        signs = torch.as_tensor(pair_signs, dtype=dtype, device=device).flatten()
        if signs.numel() != samples:
            raise ValueError("pair_signs must have one value per paired difference")
        if not bool(torch.all((signs == -1) | (signs == 1))):
            raise ValueError("pair_signs values must be either -1 or 1")
        return signs
    if not isinstance(shuffle_pair_order, bool):
        raise TypeError("shuffle_pair_order must be a bool")
    if not shuffle_pair_order:
        return torch.ones(samples, dtype=dtype, device=device)

    # The official example datasets use ``random.shuffle`` on every pair.  Keep
    # that exact pair-order operation local so fitting neither consumes nor
    # depends on global RNG state.
    generator = random.Random(int(seed))
    values: list[int] = []
    for _ in range(samples):
        pair = [1, -1]
        generator.shuffle(pair)
        values.append(pair[0])
    return torch.tensor(values, dtype=dtype, device=device)


def _orient_lat_components(
    basis: Tensor,
    differences: Tensor,
    *,
    weights: Tensor,
) -> Tensor:
    """Orient PCA axes with the official pairwise label-sign decision."""

    result = basis.clone()
    projections = differences @ result.T
    positive_votes = (projections > 0).to(weights.dtype).T @ weights
    negative_votes = (projections < 0).to(weights.dtype).T @ weights
    result[positive_votes < negative_votes] *= -1
    return result


def compute_lat(
    positive: Tensor,
    negative: Tensor,
    *,
    weights: Tensor | None = None,
    center: bool = True,
    normalize: str | None = "l2",
    seed: int = 42,
    pair_signs: Tensor | None = None,
    shuffle_pair_order: bool = True,
) -> Tensor:
    return lat_components(
        positive,
        negative,
        weights=weights,
        n_components=1,
        center=center,
        normalize=normalize,
        seed=seed,
        pair_signs=pair_signs,
        shuffle_pair_order=shuffle_pair_order,
    )[0][0]


@dataclass
class LAT(ContrastiveLearner):
    n_components: int = 1
    center: bool = True
    shuffle_pair_order: bool = True
    method: ClassVar[str] = "lat"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_components = int(self.n_components)
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if not isinstance(self.shuffle_pair_order, bool):
            raise TypeError("shuffle_pair_order must be a bool")

    def fit(
        self,
        model: Any,
        data: ContrastivePairs,
        *,
        store: ActivationStore | None = None,
    ) -> Any:
        from repsteer.artifacts import DirectionArtifact, SubspaceArtifact

        if not data.paired:
            raise ValueError("LAT requires paired positive/negative records")
        positive_batch, negative_batch = self._capture_pair(model, data, store)
        positive = activation_matrix(positive_batch, name="positive activations")
        negative = activation_matrix(negative_batch, name="negative activations")
        pair_weights = torch.as_tensor(
            data.pair_weights, dtype=positive.dtype, device=positive.device
        )
        basis, mean, explained_variance = lat_components(
            positive,
            negative,
            weights=pair_weights,
            n_components=self.n_components,
            center=self.center,
            normalize=self.normalize,
            seed=self.seed,
            shuffle_pair_order=self.shuffle_pair_order,
        )
        basis = basis.detach().to(device="cpu", dtype=torch.float32)
        mean = mean.detach().to(device="cpu", dtype=torch.float32)
        explained_variance = explained_variance.detach().to(
            device="cpu", dtype=torch.float32
        )
        artifact_type = "direction" if self.n_components == 1 else "subspace"
        metadata = artifact_metadata(
            learner=self,
            model=model,
            data=data,
            artifact_type=artifact_type,
            hidden_size=basis.shape[1],
            extra_config={
                "n_components": self.n_components,
                "center": self.center,
                "pairs": positive.shape[0],
                "pair_signing": (
                    "seeded_pair_shuffle"
                    if self.shuffle_pair_order
                    else "preserved_pair_order"
                ),
                "sign_alignment": "positive_pairwise_vote",
            },
        )
        if self.n_components == 1:
            return DirectionArtifact(metadata=metadata, direction=basis[0])
        return SubspaceArtifact(
            metadata=metadata,
            basis=basis,
            mean=mean,
            explained_variance=explained_variance,
        )


__all__ = ["LAT", "compute_lat", "lat_components"]
