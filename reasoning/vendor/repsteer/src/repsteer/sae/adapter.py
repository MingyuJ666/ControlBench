"""Backend-independent sparse-autoencoder protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class SAEAdapter(Protocol):
    """The small SAE surface used by repsteer.

    Implementations may wrap SAELens, a Top-K SAE, or a project-local model.
    The last dimension is always the feature dimension for encoded values and
    the model hidden dimension for decoded values.
    """

    @property
    def input_dim(self) -> int: ...

    @property
    def num_features(self) -> int: ...

    def encode(self, x: Tensor) -> Tensor: ...

    def decode(self, z: Tensor) -> Tensor: ...

    def decoder_direction(self, feature_id: int) -> Tensor: ...


def validate_feature_id(sae: SAEAdapter, feature_id: int) -> int:
    """Validate and normalize a feature index."""

    if isinstance(feature_id, bool) or not isinstance(feature_id, int):
        raise TypeError("feature_id must be an int")
    count = int(sae.num_features)
    if count <= 0:
        raise ValueError("SAEAdapter.num_features must be positive")
    if feature_id < 0 or feature_id >= count:
        raise IndexError(
            f"feature_id {feature_id} is outside the valid range [0, {count})"
        )
    return feature_id


def validate_adapter(sae: SAEAdapter) -> SAEAdapter:
    """Fail early when an integration does not implement the SAE protocol."""

    if not isinstance(sae, SAEAdapter):
        missing = [
            name
            for name in (
                "input_dim",
                "num_features",
                "encode",
                "decode",
                "decoder_direction",
            )
            if not hasattr(sae, name)
        ]
        suffix = f"; missing {', '.join(missing)}" if missing else ""
        raise TypeError(f"Object does not implement SAEAdapter{suffix}")
    if isinstance(sae.input_dim, bool) or int(sae.input_dim) <= 0:
        raise ValueError("SAEAdapter.input_dim must be positive")
    if isinstance(sae.num_features, bool) or int(sae.num_features) <= 0:
        raise ValueError("SAEAdapter.num_features must be positive")
    return sae


__all__ = ["SAEAdapter", "validate_adapter", "validate_feature_id"]
