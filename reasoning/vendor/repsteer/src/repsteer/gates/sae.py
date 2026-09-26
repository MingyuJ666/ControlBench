"""SAE-feature activation gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from repsteer.artifacts import SAEFeatureArtifact
from repsteer.core import GateContext
from repsteer.sae import SAEAdapter, validate_adapter, validate_feature_id

from .activation import context_tensor, reduce_token_scores
from .base import Gate


@dataclass(frozen=True, slots=True)
class SAEActivationGate(Gate):
    """Trigger when one SAE latent reaches a threshold.

    With ``sae=...``, the context tensor is a residual-stream activation and is
    encoded by the adapter.  Without it, the gate reads pre-encoded latents,
    preferring ``metadata['sae_latents']`` over ``metadata['activation']``.
    """

    feature: SAEFeatureArtifact | int
    threshold: float
    sae: SAEAdapter | None = None
    evaluate_at: Any | None = None
    positions: Any | None = None
    reduction: str = "max"
    activation_key: str | None = None
    absolute: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.feature, SAEFeatureArtifact | int) or isinstance(
            self.feature, bool
        ):
            raise TypeError(
                "SAEActivationGate.feature must be an SAEFeatureArtifact or int"
            )
        threshold = float(self.threshold)
        if not math.isfinite(threshold):
            raise ValueError("SAEActivationGate.threshold must be finite")
        if self.sae is not None:
            validate_adapter(self.sae)
            validate_feature_id(self.sae, self.feature_id)
        elif self.feature_id < 0:
            raise ValueError("SAE feature_id cannot be negative")
        object.__setattr__(self, "threshold", threshold)

    @property
    def feature_id(self) -> int:
        return (
            self.feature.feature_id
            if isinstance(self.feature, SAEFeatureArtifact)
            else int(self.feature)
        )

    def _latents(self, context: GateContext) -> Tensor:
        if self.sae is not None:
            activation = context_tensor(context, key=self.activation_key)
            if activation.shape[-1] != self.sae.input_dim:
                raise ValueError(
                    f"SAE gate input width {activation.shape[-1]} does not match "
                    f"SAE input_dim {self.sae.input_dim}"
                )
            latents = self.sae.encode(activation)
        else:
            latent_key = self.activation_key
            if latent_key is None and context.metadata.get("sae_latents") is not None:
                latent_key = "sae_latents"
            latents = context_tensor(context, key=latent_key)
            expected = None
            if isinstance(self.feature, SAEFeatureArtifact):
                expected = self.feature.metadata.config.get("sae_num_features")
            if expected is not None and latents.shape[-1] != int(expected):
                raise ValueError(
                    "SAE gate latent width does not match its feature artifact: "
                    f"{latents.shape[-1]} != {int(expected)}"
                )
        if not isinstance(latents, Tensor):
            raise TypeError("SAEAdapter.encode must return a torch.Tensor")
        if self.feature_id >= latents.shape[-1]:
            raise IndexError(
                f"feature_id {self.feature_id} is outside latent width "
                f"{latents.shape[-1]}"
            )
        return latents

    def evaluate(self, context: GateContext) -> Tensor:
        latents = self._latents(context)
        scores = latents[..., self.feature_id]
        if self.absolute:
            scores = scores.abs()
        pooled, valid = reduce_token_scores(
            scores,
            latents,
            context,
            positions=self.positions,
            reduction=self.reduction,
        )
        return (pooled >= self.threshold) & valid

    def to_dict(self) -> dict[str, Any]:
        fingerprint = (
            self.feature.fingerprint()
            if isinstance(self.feature, SAEFeatureArtifact)
            else None
        )
        evaluate_at = self.evaluate_at
        positions = self.positions
        return {
            "type": "sae_activation",
            "feature_id": self.feature_id,
            "feature_fingerprint": fingerprint,
            "threshold": self.threshold,
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
            "absolute": self.absolute,
        }


__all__ = ["SAEActivationGate"]
