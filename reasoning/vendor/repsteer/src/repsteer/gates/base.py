"""Gate protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from torch import Tensor

from repsteer.core import GateContext


class Gate(ABC):
    """Return bool or [0, 1] weights per batch/token."""

    @abstractmethod
    def evaluate(self, context: GateContext) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def __call__(self, context: GateContext) -> Tensor:
        return self.evaluate(context)


def gate_from_dict(value: Mapping[str, Any]) -> Gate:
    from .always import Always
    from .logical import AndGate, NotGate, OrGate

    kind = str(value.get("type", "")).lower()
    if kind == "always":
        return Always()
    if kind == "and":
        return AndGate(*(gate_from_dict(item) for item in value.get("gates", ())))
    if kind == "or":
        return OrGate(*(gate_from_dict(item) for item in value.get("gates", ())))
    if kind == "not":
        nested = value.get("gate")
        if not isinstance(nested, Mapping):
            raise ValueError("NotGate configuration needs a nested 'gate'")
        return NotGate(gate_from_dict(nested))
    raise ValueError(f"Unknown gate type {kind!r}")


__all__ = ["Gate", "gate_from_dict"]
