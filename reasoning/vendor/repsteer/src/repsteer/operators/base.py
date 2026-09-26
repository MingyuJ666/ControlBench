"""Operator protocol and numeric helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor

from repsteer.artifacts import DirectionArtifact, ProbeArtifact, SubspaceArtifact
from repsteer.core import StepContext
from repsteer.schedules import as_strength_tensor


class Operator(ABC):
    """Transform an activation without owning position/gate semantics."""

    is_additive: bool = False

    @abstractmethod
    def apply(
        self,
        activation: Tensor,
        artifact: Any,
        strength: Tensor | float,
        context: StepContext,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def __call__(
        self,
        activation: Tensor,
        artifact: Any,
        strength: Tensor | float,
        context: StepContext,
    ) -> Tensor:
        return self.apply(activation, artifact, strength, context)


def direction_tensor(artifact: Any, activation: Tensor) -> Tensor:
    if isinstance(artifact, DirectionArtifact) or hasattr(artifact, "direction"):
        value = artifact.direction
    elif isinstance(artifact, ProbeArtifact) or hasattr(artifact, "weight"):
        value = artifact.weight
    elif isinstance(artifact, SubspaceArtifact):
        if artifact.rank != 1:
            raise ValueError(
                "A direction operator can only use a rank-1 subspace artifact; "
                f"got rank {artifact.rank}. Select a component explicitly or use "
                "RemoveProjection for the full subspace."
            )
        value = artifact.basis[0]
    elif isinstance(artifact, Tensor):
        value = artifact
    else:
        raise TypeError(
            f"{type(artifact).__name__} does not expose a direction or probe weight"
        )
    value = torch.as_tensor(value, device=activation.device, dtype=activation.dtype)
    if value.ndim != 1:
        raise ValueError(
            "A direction operator requires a one-dimensional artifact tensor; "
            f"got {tuple(value.shape)}"
        )
    if activation.shape[-1] != value.shape[-1]:
        raise ValueError(
            f"Artifact hidden size {value.shape[-1]} does not match activation "
            f"hidden size {activation.shape[-1]}"
        )
    return cast(Tensor, value)


def replacement_tensor(artifact: Any, activation: Tensor) -> Tensor:
    if isinstance(artifact, SubspaceArtifact) and artifact.mean is not None:
        value = artifact.mean
    elif hasattr(artifact, "replacement"):
        value = artifact.replacement
    else:
        return direction_tensor(artifact, activation)
    value = torch.as_tensor(value, device=activation.device, dtype=activation.dtype)
    if value.ndim != 1 or value.shape[-1] != activation.shape[-1]:
        raise ValueError(
            "Replacement tensor must be one-dimensional and match hidden size"
        )
    return value


def strength_tensor(value: Any, activation: Tensor) -> Tensor:
    return as_strength_tensor(value, activation)


def operator_from_dict(value: Mapping[str, Any]) -> Operator:
    from .add import Add, Subtract
    from .iti import ITIAdd
    from .projection import RemoveProjection
    from .replace import Replace
    from .sae import Ablate, Clamp, SAEAblate, SAEClamp

    kind = str(value.get("type", "")).lower()
    if kind == "add":
        return Add()
    if kind == "subtract":
        return Subtract()
    if kind == "iti_add":
        return ITIAdd(layer=value["layer"])
    if kind in ("remove_projection", "removeprojection"):
        return RemoveProjection(eps=value.get("eps", 1e-12))
    if kind == "replace":
        return Replace()
    if kind in ("clamp", "latent_clamp"):
        return Clamp(value["value"], feature_id=value.get("feature_id"))
    if kind in ("ablate", "latent_ablate"):
        return Ablate(feature_id=value.get("feature_id"))
    if kind == "sae_clamp":
        if value.get("sae_adapter") is not None:
            raise ValueError(
                "A serialized SAEClamp runtime handle must be rebound explicitly"
            )
        return SAEClamp(value["value"], feature_id=value.get("feature_id"))
    if kind == "sae_ablate":
        if value.get("sae_adapter") is not None:
            raise ValueError(
                "A serialized SAEAblate runtime handle must be rebound explicitly"
            )
        return SAEAblate(feature_id=value.get("feature_id"))
    raise ValueError(f"Unknown operator type {kind!r}")


__all__ = [
    "Operator",
    "direction_tensor",
    "operator_from_dict",
    "replacement_tensor",
    "strength_tensor",
]
