"""Semantic model sites."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeAlias

SiteStream: TypeAlias = Literal["language", "vision", "projector", "fusion", "logits"]
SiteIO: TypeAlias = Literal["input", "output"]
SiteUnit: TypeAlias = int | slice | tuple[int, ...] | None


def _unit_to_json(unit: SiteUnit) -> Any:
    if isinstance(unit, slice):
        return {
            "kind": "slice",
            "start": unit.start,
            "stop": unit.stop,
            "step": unit.step,
        }
    if isinstance(unit, tuple):
        return {"kind": "indices", "values": list(unit)}
    return unit


def _unit_from_json(value: Any) -> SiteUnit:
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind == "slice":
            return slice(value.get("start"), value.get("stop"), value.get("step"))
        if kind == "indices":
            return tuple(int(item) for item in value.get("values", ()))
    if value is None or isinstance(value, int | slice):
        return value
    if isinstance(value, list | tuple):
        return tuple(int(item) for item in value)
    raise TypeError(f"Unsupported site unit encoding: {value!r}")


@dataclass(frozen=True, slots=True)
class Site:
    """A model-independent location at which activations are read or written."""

    stream: SiteStream | str
    component: str
    layer: int | None = None
    unit: SiteUnit = None
    io: SiteIO = "output"
    tensor_path: tuple[int | str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stream, str) or not self.stream:
            raise ValueError("Site.stream must be a non-empty string")
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("Site.component must be a non-empty string")
        if self.layer is not None and not isinstance(self.layer, int):
            raise TypeError("Site.layer must be an int or None")
        if self.io not in ("input", "output"):
            raise ValueError("Site.io must be 'input' or 'output'")
        path = tuple(self.tensor_path)
        if not all(isinstance(item, int | str) for item in path):
            raise TypeError("Site.tensor_path entries must be int or str")
        object.__setattr__(self, "tensor_path", path)
        if isinstance(self.unit, list):
            object.__setattr__(self, "unit", tuple(int(item) for item in self.unit))

    @property
    def name(self) -> str:
        """Return a compact stable human-readable identifier."""

        layer = "" if self.layer is None else f"[layer={self.layer}]"
        unit = "" if self.unit is None else f"[unit={self.unit!r}]"
        io = "" if self.io == "output" else f"@{self.io}"
        return f"{self.stream}.{self.component}{layer}{unit}{io}"

    def __str__(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "component": self.component,
            "layer": self.layer,
            "unit": _unit_to_json(self.unit),
            "io": self.io,
            "tensor_path": list(self.tensor_path),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Site:
        return cls(
            stream=str(value.get("stream", "language")),
            component=str(value["component"]),
            layer=value.get("layer"),
            unit=_unit_from_json(value.get("unit")),
            io=value.get("io", "output"),
            tensor_path=tuple(value.get("tensor_path", ())),
        )

    def with_layer(self, layer: int | None) -> Site:
        return replace(self, layer=layer)

    def with_stream(self, stream: SiteStream | str) -> Site:
        return replace(self, stream=stream)

    def with_io(self, io: SiteIO) -> Site:
        return replace(self, io=io)

    def with_unit(self, unit: SiteUnit) -> Site:
        return replace(self, unit=unit)

    def with_tensor_path(self, *path: int | str) -> Site:
        return replace(self, tensor_path=tuple(path))


__all__ = ["Site", "SiteIO", "SiteStream", "SiteUnit"]
