"""Helpers for adapting project-local sparse autoencoders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from repsteer.core import Site

from .adapter import validate_feature_id


@dataclass(frozen=True, slots=True)
class FunctionalSAEAdapter:
    """Build an :class:`SAEAdapter` from three ordinary callables."""

    input_dim: int
    num_features: int
    encode_fn: Callable[[Tensor], Tensor]
    decode_fn: Callable[[Tensor], Tensor]
    decoder_direction_fn: Callable[[int], Tensor]
    site: Site | None = None

    def __post_init__(self) -> None:
        if isinstance(self.input_dim, bool) or int(self.input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if isinstance(self.num_features, bool) or int(self.num_features) <= 0:
            raise ValueError("num_features must be positive")
        for name in ("encode_fn", "decode_fn", "decoder_direction_fn"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if self.site is not None and not isinstance(self.site, Site):
            raise TypeError("FunctionalSAEAdapter.site must be a Site or None")
        object.__setattr__(self, "input_dim", int(self.input_dim))
        object.__setattr__(self, "num_features", int(self.num_features))

    def encode(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"SAE input has width {x.shape[-1]}, expected {self.input_dim}"
            )
        encoded = self.encode_fn(x)
        if not isinstance(encoded, Tensor):
            raise TypeError("encode_fn must return a torch.Tensor")
        if encoded.shape[:-1] != x.shape[:-1] or encoded.shape[-1] != self.num_features:
            raise ValueError(
                "encode_fn must preserve leading dimensions and return "
                f"{self.num_features} features; got {tuple(encoded.shape)}"
            )
        return encoded

    def decode(self, z: Tensor) -> Tensor:
        if z.shape[-1] != self.num_features:
            raise ValueError(
                f"SAE latents have width {z.shape[-1]}, expected {self.num_features}"
            )
        decoded = self.decode_fn(z)
        if not isinstance(decoded, Tensor):
            raise TypeError("decode_fn must return a torch.Tensor")
        if decoded.shape[:-1] != z.shape[:-1] or decoded.shape[-1] != self.input_dim:
            raise ValueError(
                "decode_fn must preserve leading dimensions and return "
                f"width {self.input_dim}; got {tuple(decoded.shape)}"
            )
        return decoded

    def decoder_direction(self, feature_id: int) -> Tensor:
        feature_id = validate_feature_id(self, feature_id)
        direction = self.decoder_direction_fn(feature_id)
        if not isinstance(direction, Tensor):
            direction = torch.as_tensor(direction)
        if direction.ndim != 1 or direction.numel() != self.input_dim:
            raise ValueError(
                "decoder_direction_fn must return one vector of length "
                f"{self.input_dim}; got {tuple(direction.shape)}"
            )
        return direction


CustomSAEAdapter = FunctionalSAEAdapter


__all__ = ["CustomSAEAdapter", "FunctionalSAEAdapter"]
