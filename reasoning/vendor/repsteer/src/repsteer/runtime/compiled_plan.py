from __future__ import annotations

import json
import weakref
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from repsteer.models.hf.adapters.base import ResolvedSite


def _name(value: Any) -> str:
    return type(value).__name__


def _description(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        try:
            result = value.to_dict()
            if isinstance(result, dict):
                return result
        except (TypeError, ValueError):
            pass
    return {"type": _name(value)}


@dataclass(frozen=True)
class CompiledIntervention:
    declaration_index: int
    execution_index: int
    intervention: Any
    site: Any
    resolved_site: ResolvedSite
    gate_site: ResolvedSite | None
    compatibility: str
    statically_disabled: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def conflict_key(self) -> tuple[int, str, str]:
        return self.resolved_site.conflict_key

    def to_dict(self) -> dict[str, Any]:
        metadata = getattr(self.intervention.artifact, "metadata", None)
        artifact = (
            metadata.to_dict()
            if metadata is not None and hasattr(metadata, "to_dict")
            else {
                "type": _name(self.intervention.artifact),
                "model_id": getattr(metadata, "model_id", None),
                "model_revision": getattr(metadata, "model_revision", None),
                "hidden_size": getattr(metadata, "hidden_size", None),
            }
        )
        site = self.site.to_dict() if hasattr(self.site, "to_dict") else str(self.site)
        return {
            "declaration_index": self.declaration_index,
            "execution_index": self.execution_index,
            "priority": int(getattr(self.intervention, "priority", 0)),
            "phase": getattr(self.intervention, "phase", "both"),
            "site": site,
            "resolved": {
                "architecture": self.resolved_site.architecture_name,
                "module_path": self.resolved_site.module_path,
                "module_type": type(self.resolved_site.module).__name__,
                "hook_kind": self.resolved_site.hook_kind,
                "tensor_path": self.resolved_site.tensor_accessor.description,
                "hidden_size": self.resolved_site.hidden_dim,
            },
            "artifact": artifact,
            "operator": _description(self.intervention.operator),
            "positions": _description(self.intervention.positions),
            "strength": _description(self.intervention.strength),
            "gate": _description(self.intervention.gate),
            "gate_dependency": (
                {
                    "site": (
                        self.gate_site.site.to_dict()
                        if hasattr(self.gate_site.site, "to_dict")
                        else str(self.gate_site.site)
                    ),
                    "module_path": self.gate_site.module_path,
                    "hook_kind": self.gate_site.hook_kind,
                    "tensor_path": self.gate_site.tensor_accessor.description,
                    "hidden_size": self.gate_site.hidden_dim,
                    "cache": "prefill_sequence",
                }
                if self.gate_site is not None and not self.statically_disabled
                else None
            ),
            "compatibility": self.compatibility,
            "statically_disabled": self.statically_disabled,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CompiledPlan:
    """A validated plan with concrete modules but no registered hooks."""

    interventions: tuple[CompiledIntervention, ...]
    model_id: str
    revision: str | None
    architecture: str
    compatibility: str = "exact"
    warnings: tuple[str, ...] = ()
    _model_ref: weakref.ReferenceType[Any] | None = field(
        default=None, repr=False, compare=False
    )

    def __iter__(self) -> Iterator[CompiledIntervention]:
        return iter(self.interventions)

    def __len__(self) -> int:
        return len(self.interventions)

    @property
    def source_model(self) -> Any | None:
        return self._model_ref() if self._model_ref is not None else None

    @property
    def conflict_keys(self) -> frozenset[tuple[int, str, str]]:
        return frozenset(
            item.conflict_key
            for item in self.interventions
            if not item.statically_disabled
        )

    def groups(self) -> tuple[tuple[CompiledIntervention, ...], ...]:
        """Group callbacks by module/hook while preserving execution order."""

        ordered: list[list[CompiledIntervention]] = []
        positions: dict[tuple[int, str], int] = {}
        for item in self.interventions:
            if item.statically_disabled:
                continue
            key = (id(item.resolved_site.module), item.resolved_site.hook_kind)
            if key not in positions:
                positions[key] = len(ordered)
                ordered.append([])
            ordered[positions[key]].append(item)
        return tuple(tuple(group) for group in ordered)

    def gate_groups(self) -> tuple[tuple[CompiledIntervention, ...], ...]:
        """Group sequence-gate reads by concrete module/hook/accessor."""

        ordered: list[list[CompiledIntervention]] = []
        positions: dict[tuple[int, str, str], int] = {}
        for item in self.interventions:
            if item.gate_site is None or item.statically_disabled:
                continue
            key = item.gate_site.conflict_key
            if key not in positions:
                positions[key] = len(ordered)
                ordered.append([])
            ordered[positions[key]].append(item)
        return tuple(tuple(group) for group in ordered)

    def diagnostics(self) -> Any:
        """Return direction, fusion and operator-order diagnostics."""

        from .composition import diagnose_composition

        return diagnose_composition(self)

    def to_dict(self) -> dict[str, Any]:
        non_commutative = _non_commutative_diagnostics(self.interventions)
        return {
            "schema_version": "1.0",
            "model": {
                "id": self.model_id,
                "revision": self.revision,
                "architecture": self.architecture,
            },
            "compatibility": self.compatibility,
            "hook_count": len(self.groups()) + len(self.gate_groups()),
            "interventions": [item.to_dict() for item in self.interventions],
            "composition": self.diagnostics().to_dict(),
            "non_commutative_combinations": non_commutative,
            "warnings": list(self.warnings),
        }

    def explain(
        self,
        format: Literal["text", "dict", "json"] = "text",
        *,
        machine_readable: bool | None = None,
        indent: int = 2,
    ) -> str | dict[str, Any]:
        """Explain resolved execution without changing model hook state.

        ``format='dict'`` (or ``machine_readable=True``) is stable enough for
        reports and tests.  The default text form is intentionally compact for
        notebooks and logs.
        """

        if machine_readable is True:
            format = "dict"
        data = self.to_dict()
        if format == "dict":
            return data
        if format == "json":
            return json.dumps(
                data,
                allow_nan=False,
                indent=indent,
                sort_keys=True,
            )
        if format != "text":
            raise ValueError("format must be 'text', 'dict', or 'json'")
        lines = [
            f"CompiledPlan(model={self.model_id}@{self.revision or '<unknown>'}, "
            f"architecture={self.architecture}, interventions={len(self)})"
        ]
        for item in self.interventions:
            value = item.to_dict()
            resolved = value["resolved"]
            lines.append(
                "  "
                f"[{item.execution_index}] priority={value['priority']} "
                f"phase={value['phase']} site={self.site_name(item.site)} -> "
                f"{resolved['module_path']}:{resolved['hook_kind']} "
                f"tensor={resolved['tensor_path']} hidden={resolved['hidden_size']}"
            )
            if item.statically_disabled:
                lines.append("      disabled: strength schedule is statically zero")
            lines.append(
                "      "
                f"operator={_name(item.intervention.operator)} "
                f"positions={_name(item.intervention.positions)} "
                f"strength={_name(item.intervention.strength)} "
                f"gate={_name(item.intervention.gate)}"
            )
            dependency = value["gate_dependency"]
            if dependency is not None:
                lines.append(
                    "      gate dependency="
                    f"{dependency['module_path']}:{dependency['hook_kind']} "
                    f"cache={dependency['cache']}"
                )
        for diagnostic in data["non_commutative_combinations"]:
            lines.append(
                "  warning: non-commutative order "
                f"{diagnostic['first']} -> {diagnostic['second']} at "
                f"{diagnostic['module_path']}"
            )
        for group in data["composition"]["additive_fusion_groups"]:
            lines.append(
                "  additive fusion candidate: execution indices "
                + ", ".join(str(index) for index in group)
            )
        condition = data["composition"]["condition_number"]
        if condition is not None:
            lines.append(
                "  direction diagnostics: "
                f"rank={data['composition']['effective_rank']} "
                f"condition_number={condition:.6g}"
            )
        elif data["composition"]["condition_number_infinite"]:
            lines.append(
                "  direction diagnostics: "
                f"rank={data['composition']['effective_rank']} "
                "condition_number=inf"
            )
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)

    @staticmethod
    def site_name(site: Any) -> str:
        return getattr(site, "name", str(site))


def _operator_kind(value: Any) -> str:
    name = type(value).__name__.lower()
    if name in {"add", "subtract"} or name.startswith("add"):
        return "additive"
    return name


def _non_commutative_diagnostics(
    values: Iterable[CompiledIntervention],
) -> list[dict[str, Any]]:
    items = tuple(values)
    diagnostics: list[dict[str, Any]] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left.conflict_key != right.conflict_key:
                continue
            left_kind = _operator_kind(left.intervention.operator)
            right_kind = _operator_kind(right.intervention.operator)
            if left_kind == right_kind == "additive":
                continue
            diagnostics.append(
                {
                    "first": type(left.intervention.operator).__name__,
                    "second": type(right.intervention.operator).__name__,
                    "first_execution_index": left.execution_index,
                    "second_execution_index": right.execution_index,
                    "module_path": left.resolved_site.module_path,
                    "reason": "operator order is preserved and was not fused",
                }
            )
    return diagnostics


__all__ = ["CompiledIntervention", "CompiledPlan"]
