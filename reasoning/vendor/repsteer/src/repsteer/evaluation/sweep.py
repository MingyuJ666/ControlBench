"""Layer × strength × token-policy evaluation."""

from __future__ import annotations

import copy
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast

from repsteer.core.intervention import Intervention
from repsteer.evaluation.grid import Grid
from repsteer.evaluation.metrics import EvaluationBatch
from repsteer.evaluation.report import SweepRecord, SweepReport
from repsteer.operators import Add
from repsteer.schedules import Constant


def sweep(
    *,
    model: Any,
    artifact: Any,
    data: Iterable[Any],
    search: Grid,
    metrics: Sequence[Any],
    operator: Any | None = None,
    generate_kwargs: Mapping[str, Any] | None = None,
    include_outputs: bool = False,
) -> SweepReport:
    """Evaluate every explicit grid point and return a serializable report.

    Metrics follow the incremental ``update``/``compute`` protocol. A metric is
    deep-copied per grid point so its state cannot leak between configurations.
    """

    samples = tuple(data)
    if not samples:
        raise ValueError("sweep data must contain at least one sample")
    kwargs = dict(generate_kwargs or {})
    records: list[SweepRecord] = []
    selected_operator = operator if operator is not None else Add()

    for point in search:
        phase = _selector_phase(point.positions)
        intervention = Intervention(
            artifact=artifact,
            operator=selected_operator,
            positions=point.positions,
            strength=Constant(point.strength),
            site=point.site,
            phase=phase,
        )
        point_metrics = [_fresh_metric(metric, model) for metric in metrics]
        outputs: list[str] = []
        configuration = {
            "site": _site_dict(point.site),
            "strength": point.strength,
            "positions": type(point.positions).__name__,
            "phase": phase,
        }
        started = time.perf_counter()
        with model.steer(intervention):
            for sample in samples:
                result = _generate_sample(model, sample, kwargs)
                if include_outputs:
                    outputs.append(_output_text(result))
                batch = EvaluationBatch(sample, result, configuration)
                for metric in point_metrics:
                    metric.update(batch)
        elapsed = time.perf_counter() - started
        values: dict[str, float] = {}
        for metric in point_metrics:
            computed = metric.compute()
            if isinstance(computed, Mapping):
                values.update(
                    {str(key): float(value) for key, value in computed.items()}
                )
            else:
                values[str(getattr(metric, "name", type(metric).__name__))] = float(
                    computed
                )
        records.append(
            SweepRecord(
                site=_site_dict(point.site),
                strength=point.strength,
                positions=type(point.positions).__name__,
                phase=phase,
                metrics=values,
                samples=len(samples),
                elapsed_seconds=elapsed,
                outputs=tuple(outputs),
            )
        )

    metadata = getattr(artifact, "metadata", None)
    return SweepReport(
        records=tuple(records),
        search_space={
            "sites": [_site_dict(site) for site in search.sites],
            "strengths": [float(value) for value in search.strengths],
            "positions": [type(selector).__name__ for selector in search.positions],
        },
        provenance={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_id": getattr(model, "model_id", None),
            "model_revision": getattr(model, "revision", None),
            "artifact_method": getattr(metadata, "method", None),
        },
    )


def _fresh_metric(metric: Any, model: Any) -> Any:
    try:
        candidate = copy.deepcopy(metric)
    except Exception:  # pragma: no cover - defensive for user-owned metric objects
        candidate = metric
    reset = getattr(candidate, "reset", None)
    if callable(reset):
        reset()
    prepare = getattr(candidate, "prepare", None)
    if callable(prepare):
        prepare(model)
    if not callable(getattr(candidate, "update", None)) or not callable(
        getattr(candidate, "compute", None)
    ):
        raise TypeError("each metric must implement update(batch) and compute()")
    return candidate


def _generate_sample(model: Any, sample: Any, kwargs: Mapping[str, Any]) -> Any:
    # A single chat message is a prompt payload, not a mapping of generate()
    # keyword arguments.  Conversations represented as lists already take the
    # positional branch below; this special case keeps the compact one-message
    # form consistent with the model wrapper's chat API.
    if _is_chat_message(sample):
        return model.generate(sample, **kwargs)
    if isinstance(sample, Mapping):
        merged = dict(sample)
        merged.update(kwargs)
        return model.generate(**merged)
    return model.generate(sample, **kwargs)


def _is_chat_message(value: Any) -> bool:
    return isinstance(value, Mapping) and "role" in value and "content" in value


def _selector_phase(selector: Any) -> Literal["prefill", "decode", "both"]:
    name = type(selector).__name__
    if name in {"GeneratedTokens"}:
        return "decode"
    if name in {"AllTokens"}:
        return "both"
    return "prefill"


def _site_dict(site: Any) -> dict[str, Any]:
    if hasattr(site, "to_dict"):
        value = site.to_dict()
    elif is_dataclass(site) and not isinstance(site, type):
        value = asdict(cast(Any, site))
    else:
        value = {
            "stream": getattr(site, "stream", "language"),
            "component": getattr(site, "component", str(site)),
            "layer": getattr(site, "layer", None),
        }
    return {
        str(key): _jsonable(item)
        for key, item in value.items()
        if key != "unit" or item is not None
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def _output_text(result: Any) -> str:
    text = getattr(result, "text", result)
    if isinstance(text, list):
        return "\n".join(str(item) for item in text)
    return str(text)
