"""Small deterministic metrics suitable for local smoke tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class EvaluationBatch:
    """One generated sample delivered to an incremental metric."""

    prompt: Any
    result: Any
    configuration: Mapping[str, Any]


@runtime_checkable
class Metric(Protocol):
    """Incremental metric contract.

    ``prepare`` and ``reset`` are intentionally optional at runtime; a metric
    only needs ``update`` and ``compute``.
    """

    name: str

    def update(self, batch: EvaluationBatch) -> None: ...

    def compute(self) -> Mapping[str, float] | float: ...


class CallableMetric:
    """Average a pure per-example metric callable."""

    def __init__(self, name: str, function: Callable[[EvaluationBatch], float]) -> None:
        self.name = name
        self.function = function
        self._values: list[float] = []

    def reset(self) -> None:
        self._values.clear()

    def update(self, batch: EvaluationBatch) -> None:
        self._values.append(float(self.function(batch)))

    def compute(self) -> Mapping[str, float]:
        if not self._values:
            return {self.name: float("nan")}
        return {self.name: sum(self._values) / len(self._values)}


class ContainsText:
    """Fraction of generated texts containing a target substring."""

    def __init__(
        self, target: str, *, case_sensitive: bool = False, name: str = "contains"
    ):
        self.target = target
        self.case_sensitive = case_sensitive
        self.name = name
        self._matches = 0
        self._total = 0

    def reset(self) -> None:
        self._matches = 0
        self._total = 0

    def update(self, batch: EvaluationBatch) -> None:
        text = _result_text(batch.result)
        target = self.target
        if not self.case_sensitive:
            text, target = text.casefold(), target.casefold()
        self._matches += int(target in text)
        self._total += 1

    def compute(self) -> Mapping[str, float]:
        return {self.name: self._matches / self._total if self._total else float("nan")}


class MeanOutputLength:
    """Mean number of generated token ids (or whitespace words as fallback)."""

    name = "output_length"

    def __init__(self) -> None:
        self._lengths: list[int] = []

    def reset(self) -> None:
        self._lengths.clear()

    def update(self, batch: EvaluationBatch) -> None:
        token_ids = getattr(batch.result, "token_ids", None)
        if token_ids is not None:
            try:
                length = int(token_ids.shape[-1])
            except (AttributeError, IndexError):
                length = len(token_ids)
        else:
            length = len(_result_text(batch.result).split())
        self._lengths.append(length)

    def compute(self) -> Mapping[str, float]:
        value = (
            sum(self._lengths) / len(self._lengths) if self._lengths else float("nan")
        )
        return {self.name: value}


def _result_text(result: Any) -> str:
    text = getattr(result, "text", result)
    if isinstance(text, list):
        return "\n".join(str(item) for item in text)
    return str(text)


__all__ = [
    "CallableMetric",
    "ContainsText",
    "EvaluationBatch",
    "MeanOutputLength",
    "Metric",
]
