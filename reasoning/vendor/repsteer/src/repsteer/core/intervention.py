"""Immutable intervention and plan composition objects."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias, cast

from .site import Site

InterventionPhase: TypeAlias = Literal["prefill", "decode", "both"]


def _always_gate() -> Any:
    # Delayed import avoids a core -> gates -> core import cycle.
    from repsteer.gates import Always

    return Always()


@dataclass(frozen=True, slots=True)
class Intervention:
    """One ordered activation modification at one semantic site."""

    artifact: Any
    operator: Any
    positions: Any
    strength: Any
    gate: Any = field(default_factory=_always_gate)
    site: Site | None = None
    phase: InterventionPhase = "both"
    priority: int = 0

    def __post_init__(self) -> None:
        if self.phase not in ("prefill", "decode", "both"):
            raise ValueError(
                "Intervention.phase must be 'prefill', 'decode', or 'both'"
            )
        if not isinstance(self.priority, int):
            raise TypeError("Intervention.priority must be an int")

    @property
    def resolved_site(self) -> Site:
        if self.site is not None:
            return self.site
        metadata = getattr(self.artifact, "metadata", None)
        artifact_site = getattr(metadata, "site", None)
        if artifact_site is None:
            raise ValueError(
                "Intervention.site is None and the artifact metadata has no source site"
            )
        if not isinstance(artifact_site, Site):
            raise TypeError("The artifact metadata site is not a Site object")
        return artifact_site

    def applies_in(self, phase: str) -> bool:
        return self.phase == "both" or self.phase == phase

    def with_strength(self, value: Any) -> Intervention:
        from repsteer.schedules import Constant, StrengthSchedule

        schedule = (
            value
            if isinstance(value, StrengthSchedule)
            or callable(getattr(value, "value", None))
            else Constant(value)
        )
        return replace(self, strength=schedule)

    def with_gate(self, gate: Any) -> Intervention:
        return replace(self, gate=gate)

    def with_site(self, site: Site | None) -> Intervention:
        return replace(self, site=site)

    def with_priority(self, priority: int) -> Intervention:
        return replace(self, priority=priority)

    def with_positions(self, positions: Any) -> Intervention:
        return replace(self, positions=positions)

    def with_operator(self, operator: Any) -> Intervention:
        return replace(self, operator=operator)

    def with_phase(self, phase: InterventionPhase) -> Intervention:
        return replace(self, phase=phase)

    def with_artifact(self, artifact: Any) -> Intervention:
        return replace(self, artifact=artifact)

    def to_dict(self) -> dict[str, Any]:
        metadata = getattr(self.artifact, "metadata", None)
        return {
            "artifact": (
                cast(Any, metadata).to_dict()
                if metadata is not None and hasattr(metadata, "to_dict")
                else None
            ),
            "operator": (
                self.operator.to_dict()
                if hasattr(self.operator, "to_dict")
                else {"type": type(self.operator).__name__}
            ),
            "positions": (
                self.positions.to_dict()
                if hasattr(self.positions, "to_dict")
                else {"type": type(self.positions).__name__}
            ),
            "strength": (
                self.strength.to_dict()
                if hasattr(self.strength, "to_dict")
                else {"type": type(self.strength).__name__}
            ),
            "gate": (
                self.gate.to_dict()
                if hasattr(self.gate, "to_dict")
                else {"type": type(self.gate).__name__}
            ),
            "site": self.site.to_dict() if self.site is not None else None,
            "phase": self.phase,
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True, init=False)
class SteeringPlan:
    """An immutable, declaration-order-preserving collection of interventions."""

    interventions: tuple[Intervention, ...]

    def __init__(self, interventions: Iterable[Intervention] = ()) -> None:
        values = tuple(interventions)
        if not all(isinstance(value, Intervention) for value in values):
            raise TypeError("SteeringPlan only accepts Intervention objects")
        object.__setattr__(self, "interventions", values)

    def __iter__(self) -> Iterator[Intervention]:
        return iter(self.interventions)

    def __len__(self) -> int:
        return len(self.interventions)

    def __getitem__(
        self, index: int | slice
    ) -> Intervention | tuple[Intervention, ...]:
        return self.interventions[index]

    def ordered(self) -> tuple[Intervention, ...]:
        """Return stable execution order: priority, then declaration order."""

        return tuple(
            value
            for _, value in sorted(
                enumerate(self.interventions),
                key=lambda item: (item[1].priority, item[0]),
            )
        )

    # A descriptive alias is convenient in compilers and notebooks.
    sorted_interventions = ordered

    def for_phase(self, phase: str) -> tuple[Intervention, ...]:
        return tuple(item for item in self.ordered() if item.applies_in(phase))

    def with_intervention(self, intervention: Intervention) -> SteeringPlan:
        return SteeringPlan((*self.interventions, intervention))

    def with_interventions(self, interventions: Iterable[Intervention]) -> SteeringPlan:
        return SteeringPlan((*self.interventions, *tuple(interventions)))

    def to_dict(self) -> dict[str, Any]:
        return {"interventions": [item.to_dict() for item in self.interventions]}


def as_plan(
    value: Intervention | SteeringPlan | Sequence[Intervention],
) -> SteeringPlan:
    if isinstance(value, SteeringPlan):
        return value
    if isinstance(value, Intervention):
        return SteeringPlan((value,))
    return SteeringPlan(value)


__all__ = ["Intervention", "InterventionPhase", "SteeringPlan", "as_plan"]
