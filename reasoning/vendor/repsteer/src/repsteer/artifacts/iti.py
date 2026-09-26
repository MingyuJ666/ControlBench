"""Portable profile artifacts for Inference-Time Intervention (ITI)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import torch
from torch import Tensor

from repsteer.sites import head_result

from .base import (
    ArtifactMetadata,
    SteeringArtifact,
    _coerce_tensor,
    _metadata_for_tensor,
)
from .compatibility import assert_compatible, check_compatibility


@dataclass(frozen=True, slots=True)
class ITIHead:
    """One selected query-head intervention in an :class:`ITIArtifact`.

    ``directions[index]`` is the unit mass-mean (or probe) direction for this
    entry.  At runtime it is multiplied by ``projected_std`` and the recipe's
    user-controlled ``strength`` (the paper's :math:`\\alpha`).
    """

    rank: int
    layer: int
    head: int
    validation_accuracy: float
    projected_std: float

    def __post_init__(self) -> None:
        for name in ("rank", "layer", "head"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"ITIHead.{name} must be a non-negative integer")
        for name in ("validation_accuracy", "projected_std"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"ITIHead.{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.validation_accuracy <= 1.0:
            raise ValueError("ITIHead.validation_accuracy must be in [0, 1]")
        if self.projected_std <= 0.0:
            raise ValueError("ITIHead.projected_std must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "rank": self.rank,
            "layer": self.layer,
            "head": self.head,
            "validation_accuracy": self.validation_accuracy,
            "projected_std": self.projected_std,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ITIHead:
        required = {"rank", "layer", "head", "validation_accuracy", "projected_std"}
        unknown = sorted(set(value) - required)
        missing = sorted(required - set(value))
        if unknown or missing:
            details: list[str] = []
            if unknown:
                details.append("unknown fields: " + ", ".join(unknown))
            if missing:
                details.append("missing fields: " + ", ".join(missing))
            raise ValueError("invalid ITI head metadata (" + "; ".join(details) + ")")
        return cls(
            rank=int(value["rank"]),
            layer=int(value["layer"]),
            head=int(value["head"]),
            validation_accuracy=float(value["validation_accuracy"]),
            projected_std=float(value["projected_std"]),
        )


@dataclass(frozen=True, slots=True)
class ITIArtifact(SteeringArtifact):
    """A complete, portable ITI profile.

    ITI selects heads using held-out linear-probe accuracy, but stores only the
    selected direction and its calibration scale.  The profile is intentionally
    not tied to one layer site: ``recipes.iti`` maps every entry to that layer's
    pre-``o_proj`` ``head_result`` surface.
    """

    metadata: ArtifactMetadata
    directions: Tensor
    heads: tuple[ITIHead, ...] | list[ITIHead] | tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        directions = _coerce_tensor(self.directions, name="directions", ndim=(2,))
        if directions.shape[0] == 0 or directions.shape[1] == 0:
            raise ValueError(
                "ITI directions must have at least one head and one feature"
            )
        if not bool(torch.isfinite(directions).all().item()):
            raise ValueError("ITI directions must be finite")
        if directions.is_complex():
            raise ValueError("ITI directions must be real-valued")
        direction_norms = torch.linalg.vector_norm(directions, dim=1)
        if not bool(torch.allclose(direction_norms, torch.ones_like(direction_norms))):
            raise ValueError("every ITI direction must have unit L2 norm")

        heads = tuple(
            item if isinstance(item, ITIHead) else ITIHead.from_dict(item)
            for item in self.heads
        )
        if len(heads) != directions.shape[0]:
            raise ValueError("ITI directions need exactly one ITIHead entry per row")
        if tuple(entry.rank for entry in heads) != tuple(range(len(heads))):
            raise ValueError(
                "ITIHead ranks must be contiguous and match direction rows"
            )
        locations = tuple((entry.layer, entry.head) for entry in heads)
        if len(set(locations)) != len(locations):
            raise ValueError("ITI profile cannot select the same layer/head twice")

        config = dict(self.metadata.config)
        configured_heads = config.get("heads")
        serialized_heads = [entry.to_dict() for entry in heads]
        if configured_heads is not None:
            if not isinstance(configured_heads, tuple | list) or any(
                not isinstance(item, Mapping) for item in configured_heads
            ):
                raise ValueError("metadata heads must be a sequence of JSON objects")
            if [dict(item) for item in configured_heads] != serialized_heads:
                raise ValueError("metadata heads do not match ITIArtifact.heads")
        configured_count = config.get("head_count")
        if configured_count is not None and (
            isinstance(configured_count, bool)
            or not isinstance(configured_count, int)
            or configured_count != len(heads)
        ):
            raise ValueError("metadata head_count does not match ITIArtifact.heads")
        config["heads"] = serialized_heads
        config["head_count"] = len(heads)
        metadata = self.metadata.with_updates(config=config)
        metadata = _metadata_for_tensor(
            metadata,
            artifact_type="iti",
            hidden_size=int(directions.shape[-1]),
            dtype=directions.dtype,
        )
        if metadata.site is not None:
            raise ValueError(
                "ITIArtifact metadata.site must be None; an ITI profile spans layers"
            )
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "heads", heads)
        object.__setattr__(self, "metadata", metadata)

    @property
    def head_dim(self) -> int:
        return int(self.directions.shape[-1])

    @property
    def layers(self) -> tuple[int, ...]:
        # __post_init__ always normalizes heads to tuple[ITIHead, ...] via
        # object.__setattr__, which mypy cannot see through from the
        # constructor's wider (JSON-friendly) declared field type.
        heads = cast("tuple[ITIHead, ...]", self.heads)
        return tuple(sorted({entry.layer for entry in heads}))

    def entries_for_layer(self, layer: int) -> tuple[tuple[int, ITIHead], ...]:
        heads = cast("tuple[ITIHead, ...]", self.heads)
        return tuple(
            (index, entry) for index, entry in enumerate(heads) if entry.layer == layer
        )

    def bind(self, model: Any, *, compatibility: str = "exact") -> ITIArtifact:
        """Bind every selected head to its concrete pre-``o_proj`` surface."""

        resolve_site = getattr(model, "resolve_site", None)
        if not callable(resolve_site):
            raise TypeError("ITI binding requires a model with resolve_site()")
        for layer in self.layers:
            resolved = resolve_site(head_result(layer))
            resolved_site = getattr(resolved, "site", None)
            resolved_hidden_size = getattr(resolved, "hidden_dim", None)
            if resolved_site != head_result(layer):
                raise TypeError(
                    "model.resolve_site() must preserve the requested head_result site"
                )
            if (
                isinstance(resolved_hidden_size, bool)
                or not isinstance(resolved_hidden_size, int)
                or resolved_hidden_size <= 0
            ):
                raise TypeError(
                    "model.resolve_site() must return a positive integer hidden_dim"
                )
            assert_compatible(
                self,
                model,
                compatibility=compatibility,
                site=resolved_site,
                hidden_size=resolved_hidden_size,
            )
        return self

    def check_compatibility(self, target: Any, *, site: Any = None) -> Any:
        """Check this profile using its head dimension rather than model width."""

        if site is not None:
            return check_compatibility(self, target, site=site)
        resolve_site = getattr(target, "resolve_site", None)
        if not callable(resolve_site):
            return check_compatibility(self, target)
        result = None
        for layer in self.layers:
            resolved = resolve_site(head_result(layer))
            resolved_site = getattr(resolved, "site", None)
            resolved_hidden_size = getattr(resolved, "hidden_dim", None)
            if resolved_site != head_result(layer):
                raise TypeError(
                    "model.resolve_site() must preserve the requested head_result site"
                )
            if (
                isinstance(resolved_hidden_size, bool)
                or not isinstance(resolved_hidden_size, int)
                or resolved_hidden_size <= 0
            ):
                raise TypeError(
                    "model.resolve_site() must return a positive integer hidden_dim"
                )
            result = check_compatibility(
                self,
                target,
                site=resolved_site,
                hidden_size=resolved_hidden_size,
            )
            if not result.compatible:
                return result
        if result is None:  # pragma: no cover - ITIArtifact cannot be empty
            raise RuntimeError("ITIArtifact has no selected layers")
        return result

    def tensors(self) -> Mapping[str, Tensor]:
        return MappingProxyType({"directions": self.directions})


__all__ = ["ITIArtifact", "ITIHead"]
