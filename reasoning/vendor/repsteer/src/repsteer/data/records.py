"""Small immutable-ish records shared by steering datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExampleRecord:
    """One model input with a weight and optional user metadata."""

    input: Any
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str | None = None

    def __post_init__(self) -> None:
        weight = float(self.weight)
        if weight < 0:
            raise ValueError("example weights must be non-negative")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def value(self) -> Any:
        return self.input

    @property
    def content(self) -> Any:
        return self.input


@dataclass(frozen=True)
class ContrastiveRecord:
    """A paired record, or one side of an unpaired contrastive dataset."""

    positive: Any | None = None
    negative: Any | None = None
    weight: float = 1.0
    positive_weight: float | None = None
    negative_weight: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    positive_metadata: Mapping[str, Any] = field(default_factory=dict)
    negative_metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str | None = None

    def __post_init__(self) -> None:
        if self.positive is None and self.negative is None:
            raise ValueError(
                "a contrastive record must contain a positive or negative input"
            )
        weight = float(self.weight)
        positive_weight = (
            weight if self.positive_weight is None else float(self.positive_weight)
        )
        negative_weight = (
            weight if self.negative_weight is None else float(self.negative_weight)
        )
        if min(weight, positive_weight, negative_weight) < 0:
            raise ValueError("contrastive weights must be non-negative")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "positive_weight", positive_weight)
        object.__setattr__(self, "negative_weight", negative_weight)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "positive_metadata", dict(self.positive_metadata))
        object.__setattr__(self, "negative_metadata", dict(self.negative_metadata))

    @property
    def paired(self) -> bool:
        return self.positive is not None and self.negative is not None


__all__ = ["ContrastiveRecord", "ExampleRecord"]
