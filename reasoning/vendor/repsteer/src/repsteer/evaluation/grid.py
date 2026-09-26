"""Explicit and reproducible sweep search spaces."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from repsteer.core.site import Site
    from repsteer.positions import PositionSelector


@dataclass(frozen=True)
class GridPoint:
    """A single layer/site, strength, and token-policy combination."""

    site: Site
    strength: float
    positions: PositionSelector


@dataclass(frozen=True)
class Grid:
    """Finite search space used by :func:`repsteer.evaluation.sweep`."""

    sites: Sequence[Site]
    strengths: Sequence[float]
    positions: Sequence[PositionSelector]

    def __post_init__(self) -> None:
        if not self.sites:
            raise ValueError("Grid.sites must contain at least one site")
        if not self.strengths:
            raise ValueError("Grid.strengths must contain at least one strength")
        if not self.positions:
            raise ValueError("Grid.positions must contain at least one selector")

    def __iter__(self) -> Iterator[GridPoint]:
        for site, strength, positions in product(
            self.sites, self.strengths, self.positions
        ):
            yield GridPoint(site=site, strength=float(strength), positions=positions)

    def __len__(self) -> int:
        return len(self.sites) * len(self.strengths) * len(self.positions)
