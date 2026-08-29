"""Shared types every steering method uses to describe its dev search.

A method never runs the search itself: it declares the configurations it wants
evaluated, and the kernel generates, judges, and selects. Fairness across
methods is equal budget (the number of dev configurations), not an identical
search shape — axes interact differently for different methods, so forcing one
shape would compare doses rather than the axis under study.
"""

from dataclasses import dataclass, field
from typing import Any



@dataclass(frozen=True)
class SearchStage:
    """One judged round of a dev search.

    ``configs`` are evaluated together; ``select`` names the kernel-side rule
    applied to the judged results; ``carry`` names the fields copied from the
    winning configuration into every configuration of later stages, which is
    how the classic two-stage search fixes a layer before sweeping strength.
    """

    name: str
    configs: list[dict[str, Any]]
    carry: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("search stage needs a name")
        if not self.configs:
            raise ValueError(f"search stage {self.name!r} has no configurations")
        keys = {frozenset(config) for config in self.configs}
        if len(keys) != 1:
            raise ValueError(
                f"search stage {self.name!r} mixes configuration axes: {keys}"
            )
        missing = [key for key in self.carry if key not in self.configs[0]]
        if missing:
            raise ValueError(
                f"search stage {self.name!r} carries unknown axes: {missing}"
            )


@dataclass(frozen=True)
class SearchPlan:
    """The full dev search a method requests, plus what it cost."""

    stages: list[SearchStage]
    budget: int
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_configs(self) -> int:
        return sum(len(stage.configs) for stage in self.stages)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a search plan needs at least one stage")
        if self.budget < 1:
            raise ValueError(f"budget must be positive, got {self.budget}")
        if self.n_configs > self.budget:
            raise ValueError(
                f"search plan evaluates {self.n_configs} configurations, "
                f"over the budget of {self.budget}"
            )
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"search stages must have distinct names: {names}")

    def as_record(self) -> dict:
        """Serializable description stored with calibration results."""
        return {
            "budget": self.budget,
            "n_configs": self.n_configs,
            "notes": self.notes,
            "metadata": dict(self.metadata),
            "stages": [
                {
                    "name": stage.name,
                    "carry": list(stage.carry),
                    "configs": [dict(config) for config in stage.configs],
                }
                for stage in self.stages
            ],
        }


def joint_plan(structural_axis: str, structural_values: list,
               strength_axis: str, strength_values: list,
               budget: int, notes: str = "") -> SearchPlan:
    """One stage over every (site, strength) pair.

    The two-stage alternative — choose a site at one fixed strength, then
    sweep strength there — buys a finer grid per axis by assuming the axes
    separate: that the best site at the coarse strength is the best site at
    every strength. Two things break that. The coarse strength is a
    hand-picked guess, one per baseline, so a bad guess handicaps that
    baseline instead of the search noticing. And when the coarse strength is
    too weak everywhere, the first stage returns a tie, the site is chosen
    arbitrarily, and the entire second stage is spent on it.

    A joint grid costs resolution — three values an axis rather than five and
    four — and assumes nothing. Every baseline then searches the same shape,
    which is also what makes the equal-budget claim easy to state.
    """
    configs = [
        {structural_axis: value, strength_axis: strength}
        for value in structural_values
        for strength in strength_values
    ]
    return SearchPlan(
        stages=[SearchStage(name="search", configs=configs, carry=())],
        budget=budget,
        notes=notes,
        metadata={
            "shape": "joint",
            "structural_axis": structural_axis,
            "strength_axis": strength_axis,
        },
    )
