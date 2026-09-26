"""Portable evaluation records for multimodal steering experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repsteer.core.serialization import json_safe


def _metrics(values: Mapping[str, float], *, group: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, raw in values.items():
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{group} metric {name!r} must be finite")
        result[str(name)] = value
    return result


@dataclass(frozen=True)
class EvaluationMetricGroups:
    """Metrics separated by the three causal-evaluation layers in the RFC."""

    representation: Mapping[str, float] = field(default_factory=dict)
    causal_behavior: Mapping[str, float] = field(default_factory=dict)
    capability_quality: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "representation",
            _metrics(self.representation, group="representation"),
        )
        object.__setattr__(
            self,
            "causal_behavior",
            _metrics(self.causal_behavior, group="causal_behavior"),
        )
        object.__setattr__(
            self,
            "capability_quality",
            _metrics(self.capability_quality, group="capability_quality"),
        )

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "representation": dict(self.representation),
            "causal_behavior": dict(self.causal_behavior),
            "capability_quality": dict(self.capability_quality),
        }


@dataclass(frozen=True)
class MultimodalEvaluationRecord:
    """One auditable VLM steering result.

    ``modality`` should contain image indices, patch grids/token counts, and any
    processor-specific mapping details needed to reproduce selector behavior.
    """

    sample_id: str
    metrics: EvaluationMetricGroups
    modality: Mapping[str, Any]
    interventions: tuple[Mapping[str, Any], ...] = ()
    input_fingerprints: Mapping[str, str] = field(default_factory=dict)
    output: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        object.__setattr__(self, "modality", dict(self.modality))
        object.__setattr__(
            self, "interventions", tuple(dict(value) for value in self.interventions)
        )
        object.__setattr__(self, "input_fingerprints", dict(self.input_fingerprints))
        object.__setattr__(self, "output", dict(self.output))

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "sample_id": self.sample_id,
                "input_fingerprints": self.input_fingerprints,
                "modality": self.modality,
                "interventions": self.interventions,
                "metrics": self.metrics.to_dict(),
                "output": self.output,
            }
        )


@dataclass(frozen=True)
class MultimodalEvaluationReport:
    """JSON-safe schema for VLM representation and causal evaluation."""

    model_id: str
    model_revision: str | None
    processor_id: str
    processor_revision: str | None
    records: tuple[MultimodalEvaluationRecord, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.processor_id:
            raise ValueError("processor_id cannot be empty")
        if not self.schema_version:
            raise ValueError("schema_version cannot be empty")
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "schema_version": self.schema_version,
                "kind": "multimodal_steering_evaluation",
                "model": {
                    "id": self.model_id,
                    "revision": self.model_revision,
                },
                "processor": {
                    "id": self.processor_id,
                    "revision": self.processor_revision,
                },
                "provenance": self.provenance,
                "records": [record.to_dict() for record in self.records],
            }
        )

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload + "\n", encoding="utf-8")
        return payload


__all__ = [
    "EvaluationMetricGroups",
    "MultimodalEvaluationRecord",
    "MultimodalEvaluationReport",
]
