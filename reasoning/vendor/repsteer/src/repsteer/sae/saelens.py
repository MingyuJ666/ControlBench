"""Optional SAELens integration.

Importing :mod:`repsteer.sae` never imports SAELens.  The external package is
only required when ``SAELensAdapter.load`` is called.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from repsteer.core import Site
from repsteer.core.errors import MissingOptionalDependencyError

from .adapter import validate_feature_id


def _positive_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            result = int(value)
        except (TypeError, ValueError):
            continue
        if result > 0:
            return result
    return None


def _config_value(model: Any, name: str) -> Any:
    for config in (
        getattr(model, "cfg", None),
        getattr(model, "config", None),
        model,
    ):
        if isinstance(config, dict):
            value = config.get(name)
        else:
            value = getattr(config, name, None)
        if value is not None:
            return value
    return None


_HOOK_SITE = re.compile(
    r"(?:^|\.)blocks\.(?P<layer>[0-9]+)\."
    r"hook_(?P<component>resid_pre|attn_out|resid_mid|mlp_out|resid_post)$"
)


def _site_from_hook_name(value: Any) -> Site | None:
    """Map only unambiguous TransformerLens-style hook names to public sites."""

    if not isinstance(value, str):
        return None
    match = _HOOK_SITE.search(value.strip())
    if match is None:
        return None
    return Site(
        stream="language",
        component=match.group("component"),
        layer=int(match.group("layer")),
        io="output",
    )


def _unwrap_tensor(value: Any, *, operation: str) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, tuple | list) and value and isinstance(value[0], Tensor):
        return value[0]
    for name in ("sae_acts", "feature_acts", "hidden_acts", "reconstruction"):
        tensor = getattr(value, name, None)
        if isinstance(tensor, Tensor):
            return tensor
    raise TypeError(f"SAELens {operation} must return a torch.Tensor")


@dataclass(frozen=True, slots=True)
class SAELensAdapter:
    """Adapt a loaded ``sae_lens.SAE`` without exposing it in public artifacts."""

    sae: Any
    release: str | None = None
    sae_id: str | None = None
    site: Site | None = None

    def __post_init__(self) -> None:
        if not callable(getattr(self.sae, "encode", None)):
            raise TypeError("SAELens object must expose encode(x)")
        if not callable(getattr(self.sae, "decode", None)):
            raise TypeError("SAELens object must expose decode(z)")
        site = self.site
        if site is not None and not isinstance(site, Site):
            raise TypeError("SAELensAdapter.site must be a Site or None")
        if site is None:
            site = _site_from_hook_name(_config_value(self.sae, "hook_name"))
            object.__setattr__(self, "site", site)
        # Resolve dimensions eagerly so malformed third-party objects fail at
        # construction rather than halfway through generation.
        _ = self.input_dim
        _ = self.num_features

    @property
    def input_dim(self) -> int:
        weight = getattr(self.sae, "W_dec", None)
        shape: tuple[int, ...] = tuple(
            int(item) for item in getattr(weight, "shape", ())
        )
        value = _positive_int(
            _config_value(self.sae, "d_in"),
            shape[-1] if len(shape) == 2 else None,
        )
        if value is None:
            raise ValueError("Cannot infer SAELens SAE input dimension")
        return value

    @property
    def num_features(self) -> int:
        weight = getattr(self.sae, "W_dec", None)
        shape: tuple[int, ...] = tuple(
            int(item) for item in getattr(weight, "shape", ())
        )
        configured = _positive_int(
            _config_value(self.sae, "d_sae"),
            _config_value(self.sae, "num_features"),
        )
        if configured is not None:
            return configured
        if len(shape) != 2:
            raise ValueError("Cannot infer SAELens SAE feature count")
        if int(shape[-1]) == self.input_dim:
            return int(shape[0])
        if int(shape[0]) == self.input_dim:
            return int(shape[1])
        raise ValueError("SAELens W_dec shape is incompatible with its input dimension")

    def encode(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"SAE input has width {x.shape[-1]}, expected {self.input_dim}"
            )
        encoded = _unwrap_tensor(self.sae.encode(x), operation="encode")
        if encoded.shape[:-1] != x.shape[:-1] or encoded.shape[-1] != self.num_features:
            raise ValueError(
                f"SAELens encode returned an incompatible shape: {tuple(encoded.shape)}"
            )
        return encoded

    def decode(self, z: Tensor) -> Tensor:
        if z.shape[-1] != self.num_features:
            raise ValueError(
                f"SAE latents have width {z.shape[-1]}, expected {self.num_features}"
            )
        decoded = _unwrap_tensor(self.sae.decode(z), operation="decode")
        if decoded.shape[:-1] != z.shape[:-1] or decoded.shape[-1] != self.input_dim:
            raise ValueError(
                f"SAELens decode returned an incompatible shape: {tuple(decoded.shape)}"
            )
        return decoded

    def decoder_direction(self, feature_id: int) -> Tensor:
        feature_id = validate_feature_id(self, feature_id)
        weight = getattr(self.sae, "W_dec", None)
        if weight is None:
            getter = getattr(self.sae, "get_decoder_direction", None)
            if not callable(getter):
                raise AttributeError(
                    "SAELens object has neither W_dec nor get_decoder_direction()"
                )
            direction = torch.as_tensor(getter(feature_id))
        else:
            weight = torch.as_tensor(weight)
            if tuple(weight.shape) == (self.num_features, self.input_dim):
                direction = weight[feature_id]
            elif tuple(weight.shape) == (self.input_dim, self.num_features):
                direction = weight[:, feature_id]
            else:
                raise ValueError(
                    f"Unsupported SAELens W_dec shape {tuple(weight.shape)}"
                )
        if direction.ndim != 1 or direction.numel() != self.input_dim:
            raise ValueError("SAELens decoder direction has an incompatible shape")
        return direction.detach().clone()

    @classmethod
    def load(
        cls,
        *,
        release: str,
        sae_id: str,
        device: str | torch.device | None = None,
        site: Site | None = None,
        **kwargs: Any,
    ) -> SAELensAdapter:
        try:
            from sae_lens import SAE  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise MissingOptionalDependencyError(
                "SAELens support requires the 'sae-lens' package; install repsteer[sae]"
            ) from exc
        load_kwargs = dict(kwargs)
        if device is not None:
            load_kwargs["device"] = str(device)
        loaded = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            **load_kwargs,
        )
        # SAELens releases have returned either SAE or
        # (SAE, config_dict, sparsity) across supported versions.
        sae = loaded[0] if isinstance(loaded, tuple | list) else loaded
        return cls(sae=sae, release=release, sae_id=sae_id, site=site)


__all__ = ["SAELensAdapter"]
