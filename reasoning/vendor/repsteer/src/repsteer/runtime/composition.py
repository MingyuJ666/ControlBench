"""Deterministic composition ordering and direction diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def stable_order(interventions: Iterable[Any]) -> tuple[Any, ...]:
    """Return priority/declaration order without reordering operators by kind."""

    return tuple(
        value
        for _, value in sorted(
            enumerate(interventions),
            key=lambda pair: (int(getattr(pair[1], "priority", 0)), pair[0]),
        )
    )


def is_additive(operator: Any) -> bool:
    """Return whether an operator declares a purely additive transformation."""

    declared = getattr(operator, "is_additive", None)
    if declared is not None:
        return bool(declared)
    return type(operator).__name__.lower() in {"add", "subtract"}


def _intervention(value: Any) -> Any:
    return getattr(value, "intervention", value)


def _execution_index(value: Any, fallback: int) -> int:
    return int(getattr(value, "execution_index", fallback))


def _target_key(value: Any) -> tuple[Any, ...]:
    conflict_key = getattr(value, "conflict_key", None)
    if conflict_key is not None:
        return ("resolved", *tuple(conflict_key))
    intervention = _intervention(value)
    try:
        site = intervention.resolved_site
    except (AttributeError, TypeError, ValueError):
        site = getattr(intervention, "site", None)
    if hasattr(site, "to_dict"):
        raw = site.to_dict()
        return (
            "semantic",
            raw.get("stream"),
            raw.get("component"),
            raw.get("layer"),
            raw.get("unit"),
            raw.get("io"),
            tuple(raw.get("tensor_path", ())),
        )
    return ("object", id(site))


def _configuration_key(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        try:
            raw = value.to_dict()
            if isinstance(raw, Mapping):
                return _freeze(raw)
        except (TypeError, ValueError):
            pass
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    return value


def _direction(value: Any) -> Tensor | None:
    intervention = _intervention(value)
    artifact = getattr(intervention, "artifact", None)
    candidate: Any | None = None
    for name in (
        "direction",
        "decoder_direction",
        "feature_direction",
        "weight",
    ):
        candidate = getattr(artifact, name, None)
        if candidate is not None:
            break
    if candidate is None:
        basis = getattr(artifact, "basis", None)
        if isinstance(basis, Tensor) and basis.ndim == 2 and basis.shape[0] == 1:
            candidate = basis[0]
    if candidate is None:
        return None
    tensor = torch.as_tensor(candidate).detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim != 1 or not bool(torch.isfinite(tensor).all()):
        return None
    return tensor


@dataclass(frozen=True)
class CompositionDiagnostics:
    """Machine-readable diagnostics for an ordered steering composition.

    Matrix entries are ``None`` when either intervention has no direction-like
    tensor or when the tensors have incompatible hidden dimensions.
    """

    execution_indices: tuple[int, ...]
    operator_names: tuple[str, ...]
    direction_norms: tuple[float | None, ...]
    pairwise_cosine: tuple[tuple[float | None, ...], ...]
    additive_fusion_groups: tuple[tuple[int, ...], ...]
    non_commutative_pairs: tuple[tuple[int, int], ...]
    effective_rank: int | None
    condition_number: float | None
    orthogonal_basis: Tensor | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_indices": list(self.execution_indices),
            "operator_names": list(self.operator_names),
            "direction_norms": list(self.direction_norms),
            "pairwise_cosine": [list(row) for row in self.pairwise_cosine],
            "additive_fusion_groups": [
                list(group) for group in self.additive_fusion_groups
            ],
            "non_commutative_pairs": [
                list(pair) for pair in self.non_commutative_pairs
            ],
            "effective_rank": self.effective_rank,
            "condition_number": (
                self.condition_number
                if self.condition_number is None or math.isfinite(self.condition_number)
                else None
            ),
            "condition_number_infinite": bool(
                self.condition_number is not None
                and not math.isfinite(self.condition_number)
            ),
            "orthogonal_basis_shape": (
                list(self.orthogonal_basis.shape)
                if self.orthogonal_basis is not None
                else None
            ),
        }


def _fusion_groups(values: Sequence[Any]) -> tuple[tuple[int, ...], ...]:
    groups: dict[tuple[Any, ...], list[int]] = {}
    for fallback, value in enumerate(values):
        intervention = _intervention(value)
        if not is_additive(getattr(intervention, "operator", None)):
            continue
        key = (
            *_target_key(value),
            getattr(intervention, "phase", "both"),
            _configuration_key(getattr(intervention, "positions", None)),
            _configuration_key(getattr(intervention, "gate", None)),
        )
        groups.setdefault(key, []).append(_execution_index(value, fallback))
    return tuple(tuple(indices) for indices in groups.values() if len(indices) > 1)


def diagnose_composition(plan: Any, *, eps: float = 1e-12) -> CompositionDiagnostics:
    """Inspect ordering, conflicts and linear relationships without execution."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    raw_values = getattr(plan, "interventions", plan)
    values = tuple(raw_values)
    # Raw SteeringPlan values still need stable priority ordering. Compiled
    # interventions already carry their final execution indices.
    if values and not hasattr(values[0], "execution_index"):
        values = stable_order(values)

    indices = tuple(_execution_index(value, i) for i, value in enumerate(values))
    operators = tuple(
        type(getattr(_intervention(value), "operator", None)).__name__
        for value in values
    )
    target_keys = tuple(_target_key(value) for value in values)
    directions = tuple(_direction(value) for value in values)
    norms = tuple(
        None if direction is None else float(torch.linalg.vector_norm(direction))
        for direction in directions
    )

    cosine_rows: list[tuple[float | None, ...]] = []
    for left_index, (left, left_norm) in enumerate(zip(directions, norms, strict=True)):
        row: list[float | None] = []
        for right_index, (right, right_norm) in enumerate(
            zip(directions, norms, strict=True)
        ):
            if (
                left is None
                or right is None
                or target_keys[left_index] != target_keys[right_index]
                or left.shape != right.shape
                or left_norm is None
                or right_norm is None
                or left_norm <= eps
                or right_norm <= eps
            ):
                row.append(None)
            else:
                value = torch.dot(left, right) / (left_norm * right_norm)
                row.append(float(value.clamp(-1, 1)))
        cosine_rows.append(tuple(row))

    non_commutative: list[tuple[int, int]] = []
    for offset, left in enumerate(values):
        left_intervention = _intervention(left)
        for right in values[offset + 1 :]:
            right_intervention = _intervention(right)
            if _target_key(left) != _target_key(right):
                continue
            if is_additive(
                getattr(left_intervention, "operator", None)
            ) and is_additive(getattr(right_intervention, "operator", None)):
                continue
            non_commutative.append(
                (
                    _execution_index(left, offset),
                    _execution_index(right, offset + 1),
                )
            )

    comparable = [
        (target, direction)
        for target, direction in zip(target_keys, directions, strict=True)
        if direction is not None
    ]
    basis: Tensor | None = None
    rank: int | None = None
    condition: float | None = None
    if comparable:
        diagnostic_targets = {target for target, _ in comparable}
        hidden_sizes = {int(value.numel()) for _, value in comparable}
        if len(diagnostic_targets) == 1 and len(hidden_sizes) == 1:
            matrix = torch.stack([value for _, value in comparable])
            singular_values = torch.linalg.svdvals(matrix)
            tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps
            tolerance *= float(singular_values.max()) if singular_values.numel() else 1
            rank = int((singular_values > tolerance).sum())
            if singular_values.numel():
                smallest = float(singular_values.min())
                condition = (
                    math.inf
                    if smallest <= eps
                    else float(singular_values.max()) / smallest
                )
            # QR on columns returns orthonormal rows spanning all directions.
            if rank:
                q, _ = torch.linalg.qr(matrix.transpose(0, 1), mode="reduced")
                basis = q[:, :rank].transpose(0, 1).contiguous()

    return CompositionDiagnostics(
        execution_indices=indices,
        operator_names=operators,
        direction_norms=norms,
        pairwise_cosine=tuple(cosine_rows),
        additive_fusion_groups=_fusion_groups(values),
        non_commutative_pairs=tuple(non_commutative),
        effective_rank=rank,
        condition_number=condition,
        orthogonal_basis=basis,
    )


__all__ = [
    "CompositionDiagnostics",
    "diagnose_composition",
    "is_additive",
    "stable_order",
]
