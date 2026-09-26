"""Serializable sweep results and explicit selection helpers."""

from __future__ import annotations

import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SweepRecord:
    site: Mapping[str, Any]
    strength: float
    positions: str
    phase: str
    metrics: Mapping[str, float]
    samples: int
    elapsed_seconds: float
    outputs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SweepReport:
    records: tuple[SweepRecord, ...]
    search_space: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "search_space": dict(self.search_space),
            "provenance": dict(self.provenance),
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        payload = json.dumps(
            _finite_json(self.to_dict()),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload + "\n", encoding="utf-8")
        return payload

    def to_markdown(self, path: str | Path | None = None) -> str:
        metric_names = sorted(
            {name for record in self.records for name in record.metrics}
        )
        headers = [
            "site",
            "strength",
            "positions",
            "phase",
            *metric_names,
            "samples",
            "seconds",
        ]
        rows = [
            [
                _site_label(record.site),
                f"{record.strength:g}",
                record.positions,
                record.phase,
                *[_format_number(record.metrics.get(name)) for name in metric_names],
                str(record.samples),
                f"{record.elapsed_seconds:.4f}",
            ]
            for record in self.records
        ]
        lines = [
            "# repsteer sweep report",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
            "",
        ]
        markdown = "\n".join(lines)
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(markdown, encoding="utf-8")
        return markdown

    def select(
        self,
        *,
        maximize: str | None = None,
        minimize: str | None = None,
        subject_to: Mapping[str, str | float] | None = None,
    ) -> SweepRecord:
        """Select one record with an explicit objective and optional constraints."""

        if (maximize is None) == (minimize is None):
            raise ValueError("provide exactly one of maximize or minimize")
        candidates = [
            record
            for record in self.records
            if _satisfies(record.metrics, subject_to or {})
        ]
        if not candidates:
            raise ValueError("no sweep record satisfies the requested constraints")
        objective = maximize or minimize
        assert objective is not None
        missing = [record for record in candidates if objective not in record.metrics]
        if missing:
            raise KeyError(
                f"metric {objective!r} is missing from one or more sweep records"
            )
        key = lambda record: record.metrics[objective]  # noqa: E731
        return (max if maximize else min)(candidates, key=key)

    def pareto_frontier(
        self,
        *,
        maximize: Sequence[str] = (),
        minimize: Sequence[str] = (),
    ) -> SweepReport:
        if not maximize and not minimize:
            raise ValueError("provide at least one Pareto objective")
        objectives = [*maximize, *minimize]
        for record in self.records:
            missing = [name for name in objectives if name not in record.metrics]
            if missing:
                raise KeyError(f"record is missing Pareto metrics: {missing}")
        frontier = tuple(
            candidate
            for candidate in self.records
            if not any(
                _dominates(other, candidate, maximize=maximize, minimize=minimize)
                for other in self.records
                if other is not candidate
            )
        )
        return SweepReport(frontier, self.search_space, self.provenance)


def _dominates(
    left: SweepRecord,
    right: SweepRecord,
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> bool:
    weakly_better = all(left.metrics[name] >= right.metrics[name] for name in maximize)
    weakly_better &= all(left.metrics[name] <= right.metrics[name] for name in minimize)
    strictly_better = any(left.metrics[name] > right.metrics[name] for name in maximize)
    strictly_better |= any(
        left.metrics[name] < right.metrics[name] for name in minimize
    )
    return weakly_better and strictly_better


_COMPARISONS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


def _satisfies(
    metrics: Mapping[str, float], constraints: Mapping[str, str | float]
) -> bool:
    for name, expression in constraints.items():
        if name not in metrics:
            return False
        if isinstance(expression, int | float):
            if not math.isclose(metrics[name], float(expression)):
                return False
            continue
        stripped = expression.strip()
        for prefix, comparison in _COMPARISONS.items():
            if stripped.startswith(prefix):
                try:
                    threshold = float(stripped[len(prefix) :].strip())
                except ValueError as error:
                    raise ValueError(
                        f"invalid constraint {name}={expression!r}"
                    ) from error
                if not comparison(metrics[name], threshold):
                    return False
                break
        else:
            raise ValueError(f"constraint must start with one of {tuple(_COMPARISONS)}")
    return True


def _site_label(site: Mapping[str, Any]) -> str:
    stream = site.get("stream", "language")
    component = site.get("component", "unknown")
    layer = site.get("layer")
    return f"{stream}.{component}" + (f"[{layer}]" if layer is not None else "")


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def _finite_json(value: Any) -> Any:
    """Map non-finite metric values to JSON null instead of emitting invalid JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite_json(item) for item in value]
    return value
