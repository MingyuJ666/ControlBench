"""Boolean/fuzzy composition for gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from repsteer.core import GateContext

from .base import Gate


def _value(gate: Gate, context: GateContext) -> Tensor:
    value = gate.evaluate(context)
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, device=context.runtime_device)
    if value.dtype != torch.bool:
        if not value.is_floating_point():
            value = value.float()
        if not bool(torch.isfinite(value).all()):
            raise ValueError("composed gate returned non-finite weights")
        if bool(((value < 0) | (value > 1)).any()):
            raise ValueError("composed gate weights must lie in [0, 1]")
    return value


def _broadcast(values: list[Tensor]) -> list[Tensor]:
    try:
        return list(torch.broadcast_tensors(*values))
    except RuntimeError as exc:
        shapes = ", ".join(str(tuple(value.shape)) for value in values)
        raise ValueError(f"gate output shapes cannot broadcast: {shapes}") from exc


@dataclass(frozen=True, slots=True, init=False)
class AndGate(Gate):
    gates: tuple[Gate, ...]

    def __init__(self, *gates: Gate) -> None:
        if len(gates) == 1 and isinstance(gates[0], tuple | list):
            gates = tuple(gates[0])
        if len(gates) < 2 or not all(isinstance(gate, Gate) for gate in gates):
            raise TypeError("AndGate requires at least two Gate objects")
        object.__setattr__(self, "gates", tuple(gates))

    def evaluate(self, context: GateContext) -> Tensor:
        values = _broadcast([_value(gate, context) for gate in self.gates])
        if all(value.dtype == torch.bool for value in values):
            result = values[0]
            for value in values[1:]:
                result = result & value
            return result
        result = values[0].float()
        for value in values[1:]:
            result = torch.minimum(result, value.float())
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"type": "and", "gates": [gate.to_dict() for gate in self.gates]}


@dataclass(frozen=True, slots=True, init=False)
class OrGate(Gate):
    gates: tuple[Gate, ...]

    def __init__(self, *gates: Gate) -> None:
        if len(gates) == 1 and isinstance(gates[0], tuple | list):
            gates = tuple(gates[0])
        if len(gates) < 2 or not all(isinstance(gate, Gate) for gate in gates):
            raise TypeError("OrGate requires at least two Gate objects")
        object.__setattr__(self, "gates", tuple(gates))

    def evaluate(self, context: GateContext) -> Tensor:
        values = _broadcast([_value(gate, context) for gate in self.gates])
        if all(value.dtype == torch.bool for value in values):
            result = values[0]
            for value in values[1:]:
                result = result | value
            return result
        result = values[0].float()
        for value in values[1:]:
            result = torch.maximum(result, value.float())
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"type": "or", "gates": [gate.to_dict() for gate in self.gates]}


@dataclass(frozen=True, slots=True)
class NotGate(Gate):
    gate: Gate

    def __post_init__(self) -> None:
        if not isinstance(self.gate, Gate):
            raise TypeError("NotGate.gate must be a Gate")

    def evaluate(self, context: GateContext) -> Tensor:
        value = _value(self.gate, context)
        return ~value if value.dtype == torch.bool else 1.0 - value

    def to_dict(self) -> dict[str, Any]:
        return {"type": "not", "gate": self.gate.to_dict()}


__all__ = ["AndGate", "NotGate", "OrGate"]
