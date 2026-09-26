"""Pure latent-space SAE operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from repsteer.core import StepContext
from repsteer.sae.adapter import SAEAdapter, validate_adapter

from .base import Operator, strength_tensor


def _resolve_feature_id(
    artifact: Any,
    explicit: int | None,
    activation: Tensor,
) -> int:
    feature_id = explicit
    if feature_id is None:
        feature_id = getattr(artifact, "feature_id", None)
    if isinstance(feature_id, bool) or not isinstance(feature_id, int):
        raise TypeError(
            "A latent operator needs feature_id=... or an artifact exposing "
            "an integer feature_id"
        )
    if feature_id < 0 or feature_id >= activation.shape[-1]:
        raise IndexError(
            f"feature_id {feature_id} is outside latent width {activation.shape[-1]}"
        )
    metadata = getattr(artifact, "metadata", None)
    config = getattr(metadata, "config", {})
    expected = config.get("sae_num_features") if hasattr(config, "get") else None
    if expected is not None and int(expected) != activation.shape[-1]:
        raise ValueError(
            "Latent activation width does not match the feature artifact's SAE: "
            f"{activation.shape[-1]} != {int(expected)}. Encode with the matching "
            "SAE before applying a latent operator."
        )
    return feature_id


def _target_tensor(value: Any, selected: Tensor) -> Tensor:
    target = torch.as_tensor(value, device=selected.device, dtype=selected.dtype)
    try:
        return torch.broadcast_to(target, selected.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Clamp value shape {tuple(target.shape)} cannot broadcast to selected "
            f"latent shape {tuple(selected.shape)}"
        ) from exc


@dataclass(frozen=True, slots=True)
class Clamp(Operator):
    """Interpolate one latent feature toward a fixed value.

    ``strength=1`` performs an exact clamp, ``strength=0`` is identity, and
    intermediate strengths interpolate only the selected feature.
    """

    value: float | Tensor
    feature_id: int | None = None

    def __post_init__(self) -> None:
        if self.feature_id is not None and (
            isinstance(self.feature_id, bool) or not isinstance(self.feature_id, int)
        ):
            raise TypeError("Clamp.feature_id must be an int or None")

    def apply(
        self,
        activation: Tensor,
        artifact: Any,
        strength: Tensor | float,
        context: StepContext,
    ) -> Tensor:
        del context
        if activation.ndim < 1:
            raise ValueError("Clamp requires an activation feature dimension")
        feature_id = _resolve_feature_id(artifact, self.feature_id, activation)
        selected = activation[..., feature_id]
        target = _target_tensor(self.value, selected)
        candidate = activation.clone()
        candidate[..., feature_id] = target
        alpha = strength_tensor(strength, activation)
        return activation + alpha * (candidate - activation)

    def to_dict(self) -> dict[str, Any]:
        value = torch.as_tensor(self.value)
        if value.numel() != 1:
            raise ValueError("Only scalar Clamp values can be represented in plan JSON")
        return {
            "type": "clamp",
            "value": float(value.detach().cpu()),
            "feature_id": self.feature_id,
        }


@dataclass(frozen=True, slots=True)
class Ablate(Operator):
    """Interpolate one latent feature toward zero."""

    feature_id: int | None = None

    def __post_init__(self) -> None:
        if self.feature_id is not None and (
            isinstance(self.feature_id, bool) or not isinstance(self.feature_id, int)
        ):
            raise TypeError("Ablate.feature_id must be an int or None")

    def apply(
        self,
        activation: Tensor,
        artifact: Any,
        strength: Tensor | float,
        context: StepContext,
    ) -> Tensor:
        del context
        if activation.ndim < 1:
            raise ValueError("Ablate requires an activation feature dimension")
        feature_id = _resolve_feature_id(artifact, self.feature_id, activation)
        candidate = activation.clone()
        candidate[..., feature_id] = 0
        alpha = strength_tensor(strength, activation)
        return activation + alpha * (candidate - activation)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ablate", "feature_id": self.feature_id}


LatentClamp = Clamp
LatentAblate = Ablate


def _sae_delta(
    activation: Tensor,
    artifact: Any,
    sae: SAEAdapter,
    feature_id: int | None,
    *,
    value: Any,
) -> Tensor:
    sae = validate_adapter(sae)
    if activation.shape[-1] != sae.input_dim:
        raise ValueError(
            f"Residual activation width {activation.shape[-1]} does not match "
            f"SAE input_dim {sae.input_dim}"
        )
    resolved = feature_id
    if resolved is None:
        resolved = getattr(artifact, "feature_id", None)
    if isinstance(resolved, bool) or not isinstance(resolved, int):
        raise TypeError(
            "An SAE operator needs feature_id=... or an artifact exposing "
            "an integer feature_id"
        )
    if resolved < 0 or resolved >= sae.num_features:
        raise IndexError(
            f"feature_id {resolved} is outside SAE feature range "
            f"[0, {sae.num_features})"
        )
    metadata = getattr(artifact, "metadata", None)
    config = getattr(metadata, "config", {})
    expected_features = (
        config.get("sae_num_features") if hasattr(config, "get") else None
    )
    if expected_features is not None and int(expected_features) != sae.num_features:
        raise ValueError(
            "Runtime SAE feature count does not match the feature artifact: "
            f"{sae.num_features} != {int(expected_features)}"
        )
    expected_input = config.get("sae_input_dim") if hasattr(config, "get") else None
    if expected_input is not None and int(expected_input) != sae.input_dim:
        raise ValueError(
            "Runtime SAE input dimension does not match the feature artifact: "
            f"{sae.input_dim} != {int(expected_input)}"
        )
    latents = sae.encode(activation)
    if not isinstance(latents, Tensor):
        raise TypeError("SAEAdapter.encode must return a torch.Tensor")
    expected_shape = (*activation.shape[:-1], sae.num_features)
    if tuple(latents.shape) != expected_shape:
        raise ValueError(
            f"SAEAdapter.encode returned {tuple(latents.shape)}, expected "
            f"{expected_shape}"
        )
    modified = latents.clone()
    modified[..., resolved] = _target_tensor(value, latents[..., resolved])
    baseline_reconstruction = sae.decode(latents)
    modified_reconstruction = sae.decode(modified)
    for name, reconstruction in (
        ("decode(encode(h))", baseline_reconstruction),
        ("decode(modified_z)", modified_reconstruction),
    ):
        if not isinstance(reconstruction, Tensor):
            raise TypeError(f"SAEAdapter.{name} must return a torch.Tensor")
        if reconstruction.shape != activation.shape:
            raise ValueError(
                f"SAEAdapter.{name} returned {tuple(reconstruction.shape)}, "
                f"expected {tuple(activation.shape)}"
            )
    baseline_reconstruction = baseline_reconstruction.to(
        device=activation.device, dtype=activation.dtype
    )
    modified_reconstruction = modified_reconstruction.to(
        device=activation.device, dtype=activation.dtype
    )
    return modified_reconstruction - baseline_reconstruction


@dataclass(frozen=True, slots=True)
class SAEClamp(Operator):
    """Clamp a feature either in latent space or through an SAE residual delta.

    When ``sae`` is supplied, ``strength=1`` returns
    ``h + decode(z_clamped) - decode(z)`` so the SAE reconstruction residual is
    preserved.  ``strength=0`` is exactly ``h``.  Without ``sae``, this behaves
    like :class:`Clamp` on a pre-encoded latent activation.
    """

    value: float | Tensor
    sae: SAEAdapter | None = None
    feature_id: int | None = None

    def __post_init__(self) -> None:
        if self.sae is not None:
            validate_adapter(self.sae)
        if self.feature_id is not None and (
            isinstance(self.feature_id, bool) or not isinstance(self.feature_id, int)
        ):
            raise TypeError("SAEClamp.feature_id must be an int or None")

    def apply(
        self,
        activation: Tensor,
        artifact: Any,
        strength: Tensor | float,
        context: StepContext,
    ) -> Tensor:
        if self.sae is None:
            return Clamp(self.value, self.feature_id).apply(
                activation,
                artifact,
                strength,
                context,
            )
        delta = _sae_delta(
            activation,
            artifact,
            self.sae,
            self.feature_id,
            value=self.value,
        )
        return activation + strength_tensor(strength, activation) * delta

    def to_dict(self) -> dict[str, Any]:
        value = torch.as_tensor(self.value)
        if value.numel() != 1:
            raise ValueError(
                "Only scalar SAEClamp values can be represented in plan JSON"
            )
        return {
            "type": "sae_clamp",
            "value": float(value.detach().cpu()),
            "feature_id": self.feature_id,
            "sae_adapter": (
                None
                if self.sae is None
                else f"{type(self.sae).__module__}.{type(self.sae).__qualname__}"
            ),
        }


@dataclass(frozen=True, slots=True)
class SAEAblate(Operator):
    """Ablate a feature with optional residual-preserving SAE reconstruction."""

    sae: SAEAdapter | None = None
    feature_id: int | None = None

    def __post_init__(self) -> None:
        if self.sae is not None:
            validate_adapter(self.sae)
        if self.feature_id is not None and (
            isinstance(self.feature_id, bool) or not isinstance(self.feature_id, int)
        ):
            raise TypeError("SAEAblate.feature_id must be an int or None")

    def apply(
        self,
        activation: Tensor,
        artifact: Any,
        strength: Tensor | float,
        context: StepContext,
    ) -> Tensor:
        if self.sae is None:
            return Ablate(self.feature_id).apply(
                activation,
                artifact,
                strength,
                context,
            )
        delta = _sae_delta(
            activation,
            artifact,
            self.sae,
            self.feature_id,
            value=0.0,
        )
        return activation + strength_tensor(strength, activation) * delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sae_ablate",
            "feature_id": self.feature_id,
            "sae_adapter": (
                None
                if self.sae is None
                else f"{type(self.sae).__module__}.{type(self.sae).__qualname__}"
            ),
        }


__all__ = [
    "Ablate",
    "Clamp",
    "LatentAblate",
    "LatentClamp",
    "SAEAblate",
    "SAEClamp",
]
