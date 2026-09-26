"""Dataset protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .fingerprint import Fingerprint


@runtime_checkable
class SteeringDataset(Protocol):
    @property
    def fingerprint(self) -> Fingerprint: ...

    def __len__(self) -> int: ...


__all__ = ["SteeringDataset"]
