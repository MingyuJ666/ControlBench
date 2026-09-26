"""Deterministic full-batch logistic linear probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from repsteer.capture import ActivationStore
from repsteer.data import ContrastivePairs

from .base import (
    ContrastiveLearner,
    activation_matrix,
    artifact_metadata,
    normalize_vector,
    sample_weights,
)


def fit_logistic_probe(
    features: Tensor,
    labels: Tensor,
    *,
    sample_weight: Tensor | None = None,
    l2: float = 1e-4,
    max_iter: int = 100,
    tolerance: float = 1e-7,
    learning_rate: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Fit binary logistic regression with deterministic damped Newton steps."""

    weight, bias, _ = _fit_logistic_probe(
        features,
        labels,
        sample_weight=sample_weight,
        l2=l2,
        max_iter=max_iter,
        tolerance=tolerance,
        learning_rate=learning_rate,
    )
    return weight, bias


def _fit_logistic_probe(
    features: Tensor,
    labels: Tensor,
    *,
    sample_weight: Tensor | None,
    l2: float,
    max_iter: int,
    tolerance: float,
    learning_rate: float,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    if features.ndim != 2:
        raise ValueError("probe features must be [samples, hidden]")
    labels = torch.as_tensor(labels).flatten()
    if labels.numel() != features.shape[0]:
        raise ValueError("probe labels must have one value per sample")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("linear probe supports binary labels 0/1")
    if not bool((labels == 0).any()) or not bool((labels == 1).any()):
        raise ValueError("linear probe requires both positive and negative samples")
    if l2 < 0 or max_iter <= 0 or tolerance <= 0 or learning_rate <= 0:
        raise ValueError("invalid logistic regression optimization configuration")

    # Float64 CPU solves give stable, backend-independent small-matrix Newton steps.
    x = features.detach().to(device="cpu", dtype=torch.float64)
    y = labels.detach().to(device="cpu", dtype=torch.float64)
    if sample_weight is None:
        weights = torch.ones_like(y)
    else:
        weights = torch.as_tensor(
            sample_weight, dtype=torch.float64, device="cpu"
        ).flatten()
    if (
        weights.numel() != y.numel()
        or bool((weights < 0).any())
        or float(weights.sum()) <= 0
    ):
        raise ValueError("probe sample weights must be non-negative with positive sum")

    ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
    design = torch.cat((x, ones), dim=1)
    beta = torch.zeros(design.shape[1], dtype=x.dtype)
    regularizer = torch.ones_like(beta)
    regularizer[-1] = 0.0  # The intercept is not regularized.
    total_weight = weights.sum()

    def objective(candidate: Tensor) -> Tensor:
        logits = design @ candidate
        data_loss = (weights * (F.softplus(logits) - y * logits)).sum() / total_weight
        penalty = 0.5 * l2 * (candidate[:-1].square().sum())
        return data_loss + penalty

    converged = False
    iterations = 0
    for _iterations in range(1, max_iter + 1):
        iterations = _iterations
        logits = design @ beta
        probability = torch.sigmoid(logits)
        gradient = design.T @ (weights * (probability - y)) / total_weight
        gradient = gradient + l2 * regularizer * beta
        curvature = weights * probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design / total_weight
        hessian = hessian + torch.diag(l2 * regularizer + 1e-10)
        try:
            step = torch.linalg.solve(hessian, gradient)
        except RuntimeError:
            step = torch.linalg.lstsq(hessian, gradient[:, None]).solution[:, 0]
        if float(torch.linalg.vector_norm(step, ord=float("inf"))) <= tolerance:
            converged = True
            break

        old_objective = objective(beta)
        scale = learning_rate
        accepted = False
        for _ in range(24):
            candidate = beta - scale * step
            if bool(torch.isfinite(candidate).all()) and float(
                objective(candidate)
            ) <= float(old_objective):
                beta = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
        if float(torch.linalg.vector_norm(scale * step)) <= tolerance * (
            1.0 + float(torch.linalg.vector_norm(beta))
        ):
            converged = True
            break

    return (
        beta[:-1].to(dtype=torch.float32),
        beta[-1].to(dtype=torch.float32),
        {
            "iterations": iterations,
            "converged": converged,
            "objective": float(objective(beta)),
        },
    )


@dataclass
class LinearProbe(ContrastiveLearner):
    normalize: str | None = "none"
    l2: float = 1e-4
    weight_decay: float | None = None
    max_iter: int = 100
    epochs: int | None = None
    tolerance: float = 1e-7
    learning_rate: float = 1.0
    lr: float | None = None
    standardize: bool = False
    method: ClassVar[str] = "linear_probe"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.weight_decay is not None:
            self.l2 = float(self.weight_decay)
        if self.epochs is not None:
            self.max_iter = int(self.epochs)
        if self.lr is not None:
            self.learning_rate = float(self.lr)
        self.l2 = float(self.l2)
        self.max_iter = int(self.max_iter)
        self.tolerance = float(self.tolerance)
        self.learning_rate = float(self.learning_rate)

    def fit(
        self,
        model: Any,
        data: ContrastivePairs,
        *,
        store: ActivationStore | None = None,
    ) -> Any:
        from repsteer.artifacts import ProbeArtifact

        positive_batch, negative_batch = self._capture_pair(model, data, store)
        positive = activation_matrix(positive_batch, name="positive activations")
        negative = activation_matrix(negative_batch, name="negative activations")
        features = torch.cat((positive, negative), dim=0)
        labels = torch.cat(
            (
                torch.ones(positive.shape[0], device=features.device),
                torch.zeros(negative.shape[0], device=features.device),
            )
        )
        weights = torch.cat(
            (
                sample_weights(positive_batch, positive),
                sample_weights(negative_batch, negative),
            )
        )

        feature_mean = torch.zeros(features.shape[1], dtype=features.dtype)
        feature_scale = torch.ones(features.shape[1], dtype=features.dtype)
        training_features = features
        if self.standardize:
            feature_mean = features.mean(dim=0)
            feature_scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
            training_features = (features - feature_mean) / feature_scale

        weight, bias, diagnostics = _fit_logistic_probe(
            training_features,
            labels,
            sample_weight=weights,
            l2=self.l2,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            learning_rate=self.learning_rate,
        )
        if self.standardize:
            feature_mean = feature_mean.to(device="cpu", dtype=torch.float32)
            feature_scale = feature_scale.to(device="cpu", dtype=torch.float32)
            raw_weight = weight / feature_scale
            bias = bias - torch.dot(raw_weight, feature_mean)
            weight = raw_weight
        if self.normalize not in {None, "none", "identity", "raw", "false"}:
            norm = torch.linalg.vector_norm(weight)
            if float(norm) > 0:
                weight = normalize_vector(weight, self.normalize)
                bias = bias / norm
        weight = weight.detach().to(device="cpu", dtype=torch.float32)
        bias = bias.detach().to(device="cpu", dtype=torch.float32)
        metadata = artifact_metadata(
            learner=self,
            model=model,
            data=data,
            artifact_type="probe",
            hidden_size=weight.numel(),
            extra_config={
                "solver": "deterministic_newton",
                "l2": self.l2,
                "max_iter": self.max_iter,
                "tolerance": self.tolerance,
                "learning_rate": self.learning_rate,
                "standardize": self.standardize,
                "positive_samples": positive.shape[0],
                "negative_samples": negative.shape[0],
                "training": diagnostics,
            },
        )
        return ProbeArtifact(metadata=metadata, weight=weight, bias=bias)


LogisticProbe = LinearProbe


__all__ = ["LinearProbe", "LogisticProbe", "fit_logistic_probe"]
