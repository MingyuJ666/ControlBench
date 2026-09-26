"""Deterministic, data-free candidate selection records.

This module deliberately contains only split records, candidate evaluation,
selection, and safe report serialization.  It does not load models, run
generation, train learners, or retain evaluator callables.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from repsteer._version import __version__
from repsteer.data import stable_fingerprint

_SELECTION_SCHEMA_VERSION = "1.0"
_CANONICAL_CANDIDATE_ORDER = "canonical_id_ascending"


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_seed(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int or None, got {type(value).__name__}")
    return value


def _freeze_json(value: Any, *, path: str) -> Any:
    """Return a recursively immutable, finite-only JSON value.

    The standard ``json`` module would accept several values that are unsafe
    for an auditable report (notably NaN and Infinity).  Keeping the validation
    here separate from the project's broader fingerprint utility is deliberate:
    fingerprints may describe arbitrary Python configuration, while selection
    records must remain portable JSON data only.
    """

    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite, got {value!r}")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key in value:
            if not isinstance(key, str):
                raise TypeError(
                    f"{path} contains non-string JSON key {key!r} "
                    f"({type(key).__name__})"
                )
        for key in sorted(value):
            frozen[key] = _freeze_json(value[key], path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only JSON primitives, mappings, or sequences; "
        f"got {type(value).__name__}"
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _checksum(value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _validate_checksum(value: Any, *, name: str) -> str:
    """Require the exact digest format emitted by :func:`_checksum`."""

    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{name} must be a sha256 checksum, got {value!r}")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} is not a valid sha256 checksum: {value!r}")
    return value


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object, got {type(value).__name__}")
    return value


def _reject_unknown_fields(
    value: Mapping[str, Any], *, name: str, allowed: set[str]
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name} has unsupported fields: {', '.join(unknown)}")


def _normalise_ids(value: Iterable[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{name} must be an iterable of IDs, not one string")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of string IDs") from exc
    for index, item in enumerate(values):
        _require_nonempty_string(item, name=f"{name}[{index}]")
    duplicates = sorted({item for item in values if values.count(item) > 1})
    if duplicates:
        raise ValueError(f"{name} contains duplicate IDs: {', '.join(duplicates)}")
    return values


@dataclass(frozen=True, slots=True)
class HoldoutSplit:
    """An immutable, explicit train/validation partition of record IDs.

    Both partitions must be non-empty.  Use :func:`make_holdout_split` for a
    deterministic hash-ranked partition, or instantiate this record when the
    split IDs are already known.
    """

    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    seed: int | None = None
    strategy: str = "explicit"
    source_fingerprint: str | None = None

    def __post_init__(self) -> None:
        train_ids = _normalise_ids(self.train_ids, name="HoldoutSplit.train_ids")
        validation_ids = _normalise_ids(
            self.validation_ids, name="HoldoutSplit.validation_ids"
        )
        if not train_ids:
            raise ValueError("HoldoutSplit.train_ids cannot be empty")
        if not validation_ids:
            raise ValueError("HoldoutSplit.validation_ids cannot be empty")
        overlap = sorted(set(train_ids) & set(validation_ids))
        if overlap:
            raise ValueError(
                "HoldoutSplit train_ids and validation_ids overlap: "
                + ", ".join(overlap)
            )
        source_fingerprint = self.source_fingerprint
        if source_fingerprint is not None:
            _require_nonempty_string(
                source_fingerprint, name="HoldoutSplit.source_fingerprint"
            )
        object.__setattr__(self, "train_ids", train_ids)
        object.__setattr__(self, "validation_ids", validation_ids)
        object.__setattr__(self, "seed", _require_seed(self.seed, name="seed"))
        object.__setattr__(
            self,
            "strategy",
            _require_nonempty_string(self.strategy, name="HoldoutSplit.strategy"),
        )

    @property
    def ids(self) -> tuple[str, ...]:
        """Return all split IDs in explicit train-then-validation order."""

        return (*self.train_ids, *self.validation_ids)

    @property
    def fingerprint(self) -> str:
        """Return a canonical digest for referring to this split."""

        return _checksum(self.to_dict())

    def validate_against(self, source_ids: Iterable[str]) -> HoldoutSplit:
        """Validate that this split is a complete partition of ``source_ids``.

        Raises:
            ValueError: If source IDs are duplicated, missing from the split, or
                if the split references an unknown source ID.
        """

        source = _normalise_ids(source_ids, name="source_ids")
        source_set = set(source)
        split_set = set(self.ids)
        unknown = sorted(split_set - source_set)
        if unknown:
            raise ValueError(
                "HoldoutSplit references unknown source IDs: " + ", ".join(unknown)
            )
        missing = sorted(source_set - split_set)
        if missing:
            raise ValueError(
                "HoldoutSplit does not cover source IDs: " + ", ".join(missing)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a finite JSON-safe split record."""

        return {
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "seed": self.seed,
            "strategy": self.strategy,
            "source_fingerprint": self.source_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HoldoutSplit:
        """Build a split from :meth:`to_dict` output."""

        document = _as_mapping(value, name="HoldoutSplit")
        _reject_unknown_fields(
            document,
            name="HoldoutSplit",
            allowed={
                "train_ids",
                "validation_ids",
                "seed",
                "strategy",
                "source_fingerprint",
            },
        )
        required = {
            "train_ids",
            "validation_ids",
            "seed",
            "strategy",
            "source_fingerprint",
        }
        missing = required - set(document)
        if missing:
            raise ValueError(
                "HoldoutSplit is missing required fields: " + ", ".join(sorted(missing))
            )
        return cls(
            train_ids=_normalise_ids(document["train_ids"], name="train_ids"),
            validation_ids=_normalise_ids(
                document["validation_ids"], name="validation_ids"
            ),
            seed=document["seed"],
            strategy=document["strategy"],
            source_fingerprint=document["source_fingerprint"],
        )


def make_holdout_split(
    ids: Iterable[str],
    *,
    validation_ratio: float = 0.2,
    seed: int | None = 0,
    source_fingerprint: str | None = None,
) -> HoldoutSplit:
    """Create a deterministic, non-overlapping held-out split.

    IDs are ranked with SHA-256 rather than Python's randomized ``hash()``.
    A ``None`` seed is still deterministic; it selects the documented fixed
    seed namespace rather than entropy from the current process.

    Args:
        ids: Unique, stable source-record IDs.
        validation_ratio: Strictly between zero and one.
        seed: Optional deterministic ranking seed.
        source_fingerprint: Dataset fingerprint to retain in the split.  When
            omitted, a fingerprint of the sorted IDs is recorded.

    Raises:
        ValueError: If IDs cannot form a useful split or the ratio is invalid.

    Returns:
        A checked immutable split with a retained source fingerprint.
    """

    source_ids = _normalise_ids(ids, name="ids")
    if len(source_ids) < 2:
        raise ValueError("at least two unique IDs are required for a holdout split")
    if isinstance(validation_ratio, bool) or not isinstance(
        validation_ratio, int | float
    ):
        raise TypeError("validation_ratio must be a finite float between 0 and 1")
    ratio = float(validation_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError(
            "validation_ratio must be finite and strictly between 0 and 1, "
            f"got {ratio!r}"
        )
    normalized_seed = _require_seed(seed, name="seed")
    validation_count = min(
        len(source_ids) - 1,
        max(1, int(round(len(source_ids) * ratio))),
    )
    seed_text = "none" if normalized_seed is None else str(normalized_seed)

    def rank(identifier: str) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"repsteer.holdout.v1\x00{seed_text}\x00{identifier}".encode()
        ).hexdigest()
        return digest, identifier

    ranked = sorted(source_ids, key=rank)
    validation_ids = tuple(sorted(ranked[:validation_count]))
    train_ids = tuple(sorted(ranked[validation_count:]))
    resolved_fingerprint = (
        str(stable_fingerprint({"record_ids": sorted(source_ids)}))
        if source_fingerprint is None
        else _require_nonempty_string(source_fingerprint, name="source_fingerprint")
    )
    return HoldoutSplit(
        train_ids=train_ids,
        validation_ids=validation_ids,
        seed=normalized_seed,
        strategy="sha256_rank_v1",
        source_fingerprint=resolved_fingerprint,
    ).validate_against(source_ids)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A stable JSON-only descriptor for one selectable configuration."""

    id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier = _require_nonempty_string(self.id, name="Candidate.id")
        parameters = _freeze_json(self.parameters, path="Candidate.parameters")
        if not isinstance(parameters, Mapping):  # pragma: no cover - helper guard
            raise TypeError("Candidate.parameters must be a JSON object")
        object.__setattr__(self, "id", identifier)
        object.__setattr__(self, "parameters", parameters)

    @property
    def canonical(self) -> str:
        """Return the deterministic descriptor used for audit records."""

        return _canonical_json(self.to_dict())

    @property
    def fingerprint(self) -> str:
        """Return a content digest independent of mapping insertion order."""

        return _checksum(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe candidate descriptor."""

        return {"id": self.id, "parameters": _json_value(self.parameters)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Candidate:
        """Build a candidate from :meth:`to_dict` output."""

        document = _as_mapping(value, name="Candidate")
        _reject_unknown_fields(document, name="Candidate", allowed={"id", "parameters"})
        required = {"id", "parameters"}
        missing = sorted(required - set(document))
        if missing:
            raise ValueError(
                "Candidate is missing required fields: " + ", ".join(missing)
            )
        parameters = document["parameters"]
        if not isinstance(parameters, Mapping):
            raise TypeError("Candidate.parameters must be a JSON object")
        return cls(id=document["id"], parameters=parameters)


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Finite metric values recorded for one :class:`Candidate`."""

    candidate: Candidate
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, Candidate):
            raise TypeError(
                "CandidateEvaluation.candidate must be a Candidate, got "
                f"{type(self.candidate).__name__}"
            )
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise ValueError("CandidateEvaluation.metrics must be a non-empty mapping")
        metrics: dict[str, float] = {}
        for name in self.metrics:
            _require_nonempty_string(name, name="metric name")
        for name in sorted(self.metrics):
            value = self.metrics[name]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(
                    f"metric {name!r} for candidate {self.candidate.id!r} "
                    "must be numeric"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"metric {name!r} for candidate {self.candidate.id!r} "
                    f"must be finite, got {value!r}"
                )
            metrics[name] = 0.0 if numeric == 0.0 else numeric
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    @property
    def candidate_id(self) -> str:
        """Return the stable ID of the evaluated candidate."""

        return self.candidate.id

    def to_dict(self) -> dict[str, Any]:
        """Return a finite JSON-safe evaluation record."""

        return {"candidate": self.candidate.to_dict(), "metrics": dict(self.metrics)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateEvaluation:
        """Build an evaluation from :meth:`to_dict` output."""

        document = _as_mapping(value, name="CandidateEvaluation")
        _reject_unknown_fields(
            document,
            name="CandidateEvaluation",
            allowed={"candidate", "metrics"},
        )
        missing = {"candidate", "metrics"} - set(document)
        if missing:
            raise ValueError(
                "CandidateEvaluation is missing required fields: "
                + ", ".join(sorted(missing))
            )
        return cls(
            candidate=Candidate.from_dict(
                _as_mapping(document["candidate"], name="candidate")
            ),
            metrics=_as_mapping(document["metrics"], name="metrics"),
        )


@dataclass(frozen=True, slots=True)
class SecondaryMetric:
    """One explicit secondary metric used only when prior metrics tie."""

    name: str
    maximize: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _require_nonempty_string(self.name, name="SecondaryMetric.name"),
        )
        if not isinstance(self.maximize, bool):
            raise TypeError("SecondaryMetric.maximize must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "maximize": self.maximize}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SecondaryMetric:
        document = _as_mapping(value, name="SecondaryMetric")
        _reject_unknown_fields(
            document, name="SecondaryMetric", allowed={"name", "maximize"}
        )
        required = {"name", "maximize"}
        missing = sorted(required - set(document))
        if missing:
            raise ValueError(
                "SecondaryMetric is missing required fields: " + ", ".join(missing)
            )
        return cls(name=document["name"], maximize=document["maximize"])


def _normalise_secondary_metrics(
    secondary_metrics: Sequence[
        SecondaryMetric | str | tuple[str, bool] | Mapping[str, Any]
    ]
    | str = (),
    *,
    primary_maximize: bool,
    secondary_metric: str | None = None,
    secondary_maximize: bool | None = None,
) -> tuple[SecondaryMetric, ...]:
    if isinstance(secondary_metrics, str):
        values: Sequence[
            SecondaryMetric | str | tuple[str, bool] | Mapping[str, Any]
        ] = (secondary_metrics,)
    else:
        values = secondary_metrics
    normalized: list[SecondaryMetric] = []
    if secondary_metric is not None:
        normalized.append(
            SecondaryMetric(
                secondary_metric,
                primary_maximize if secondary_maximize is None else secondary_maximize,
            )
        )
    elif secondary_maximize is not None:
        raise ValueError("secondary_maximize requires secondary_metric")
    for value in values:
        if isinstance(value, SecondaryMetric):
            normalized.append(value)
        elif isinstance(value, str):
            normalized.append(SecondaryMetric(value, primary_maximize))
        elif isinstance(value, tuple) and len(value) == 2:
            normalized.append(SecondaryMetric(value[0], value[1]))
        elif isinstance(value, Mapping):
            normalized.append(SecondaryMetric.from_dict(value))
        else:
            raise TypeError(
                "secondary_metrics items must be SecondaryMetric, metric name, "
                "(name, maximize), or JSON object"
            )
    names = [item.name for item in normalized]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "secondary metric names must be unique: " + ", ".join(duplicates)
        )
    return tuple(normalized)


def _normalise_evaluations(
    evaluations: Iterable[CandidateEvaluation], *, name: str = "evaluations"
) -> tuple[CandidateEvaluation, ...]:
    values: tuple[CandidateEvaluation, ...]
    if isinstance(evaluations, CandidateEvaluation):
        values = (evaluations,)
    else:
        values = tuple(evaluations)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    for index, value in enumerate(values):
        if not isinstance(value, CandidateEvaluation):
            raise TypeError(
                f"{name}[{index}] must be CandidateEvaluation, got "
                f"{type(value).__name__}"
            )
    candidate_ids = [value.candidate.id for value in values]
    duplicates = sorted(
        {item for item in candidate_ids if candidate_ids.count(item) > 1}
    )
    if duplicates:
        raise ValueError("candidate IDs must be unique: " + ", ".join(duplicates))
    return tuple(sorted(values, key=lambda evaluation: evaluation.candidate.id))


def _validate_order(value: str) -> str:
    if value != _CANONICAL_CANDIDATE_ORDER:
        raise ValueError(
            f"candidate_order must be 'canonical_id_ascending'; got {value!r}"
        )
    return value


def select_best(
    evaluations: Iterable[CandidateEvaluation],
    *,
    objective: str,
    maximize: bool = True,
    secondary_metrics: Sequence[
        SecondaryMetric | str | tuple[str, bool] | Mapping[str, Any]
    ]
    | str = (),
    secondary_metric: str | None = None,
    secondary_maximize: bool | None = None,
    candidate_order: str = _CANONICAL_CANDIDATE_ORDER,
) -> CandidateEvaluation:
    """Select one evaluation with explicit, deterministic tie-breaking.

    The order is primary objective, declared secondary metrics, then ascending
    canonical candidate ID.  The final rule never depends on input iteration,
    Python hash randomization, identity, or wall-clock time.

    Args:
        evaluations: Candidate metrics to compare; candidate IDs must be unique.
        objective: Required finite metric name used as the primary ordering key.
        maximize: Whether a larger primary metric is better.
        secondary_metrics: Optional ordered secondary metric declarations.
        secondary_metric: Convenience alias for one secondary metric.
        secondary_maximize: Direction for ``secondary_metric``.
        candidate_order: Recorded final deterministic candidate ordering rule.

    Returns:
        The selected existing evaluation; inputs are never mutated.

    Raises:
        KeyError: If a required objective metric is missing.
        ValueError: If selection semantics are invalid or candidate IDs repeat.
    """

    objective_name = _require_nonempty_string(objective, name="objective")
    if not isinstance(maximize, bool):
        raise TypeError("maximize must be a bool")
    order = _validate_order(candidate_order)
    del order  # Validation makes the recorded ordering rule explicit.
    values = _normalise_evaluations(evaluations)
    secondary = _normalise_secondary_metrics(
        secondary_metrics,
        primary_maximize=maximize,
        secondary_metric=secondary_metric,
        secondary_maximize=secondary_maximize,
    )
    if objective_name in {item.name for item in secondary}:
        raise ValueError("objective cannot also be a secondary metric")
    required_metrics = (objective_name, *(item.name for item in secondary))
    for evaluation in values:
        missing = [name for name in required_metrics if name not in evaluation.metrics]
        if missing:
            raise KeyError(
                f"candidate {evaluation.candidate.id!r} is missing required "
                f"metric(s): {', '.join(missing)}"
            )

    def ordering_key(evaluation: CandidateEvaluation) -> tuple[Any, ...]:
        primary = evaluation.metrics[objective_name]
        metric_values: list[float] = [-primary if maximize else primary]
        for item in secondary:
            value = evaluation.metrics[item.name]
            metric_values.append(-value if item.maximize else value)
        return (*metric_values, evaluation.candidate.id)

    return min(values, key=ordering_key)


@dataclass(frozen=True, slots=True)
class SelectionReport:
    """A checksummed, replayable record of deterministic candidate selection."""

    objective: str
    maximize: bool
    evaluations: tuple[CandidateEvaluation, ...]
    selected_id: str
    secondary_metrics: tuple[SecondaryMetric, ...] = ()
    candidate_order: str = _CANONICAL_CANDIDATE_ORDER
    split: HoldoutSplit | None = None
    split_fingerprint: str | None = None
    source_fingerprint: str | None = None
    selection_seed: int | None = None
    provenance: Mapping[str, Any] = field(
        default_factory=lambda: {"package": "repsteer", "version": __version__}
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = _SELECTION_SCHEMA_VERSION
    checksum: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _SELECTION_SCHEMA_VERSION:
            raise ValueError(
                "unsupported SelectionReport schema "
                f"{self.schema_version!r}; supported: {_SELECTION_SCHEMA_VERSION!r}"
            )
        objective = _require_nonempty_string(self.objective, name="objective")
        if not isinstance(self.maximize, bool):
            raise TypeError("maximize must be a bool")
        evaluations = _normalise_evaluations(self.evaluations)
        secondary = _normalise_secondary_metrics(
            self.secondary_metrics, primary_maximize=self.maximize
        )
        if objective in {item.name for item in secondary}:
            raise ValueError("objective cannot also be a secondary metric")
        candidate_order = _validate_order(self.candidate_order)
        selected_id = _require_nonempty_string(self.selected_id, name="selected_id")
        selected = select_best(
            evaluations,
            objective=objective,
            maximize=self.maximize,
            secondary_metrics=secondary,
            candidate_order=candidate_order,
        )
        if selected_id != selected.candidate.id:
            raise ValueError(
                "SelectionReport.selected_id does not match replayed selection: "
                f"expected {selected.candidate.id!r}, got {selected_id!r}"
            )
        if self.split is not None and not isinstance(self.split, HoldoutSplit):
            raise TypeError("split must be a HoldoutSplit or None")
        split_fingerprint = self.split_fingerprint
        if split_fingerprint is not None:
            _require_nonempty_string(split_fingerprint, name="split_fingerprint")
        if self.split is not None:
            computed_split_fingerprint = self.split.fingerprint
            if (
                split_fingerprint is not None
                and split_fingerprint != computed_split_fingerprint
            ):
                raise ValueError(
                    "split_fingerprint does not match the embedded HoldoutSplit: "
                    f"expected {computed_split_fingerprint!r}, got "
                    f"{split_fingerprint!r}"
                )
            split_fingerprint = computed_split_fingerprint
        if self.split is None and split_fingerprint is None:
            raise ValueError(
                "SelectionReport requires an embedded split or split_fingerprint"
            )
        source_fingerprint = self.source_fingerprint
        if (
            self.split is not None
            and source_fingerprint is not None
            and self.split.source_fingerprint is not None
            and source_fingerprint != self.split.source_fingerprint
        ):
            raise ValueError(
                "source_fingerprint does not match the embedded HoldoutSplit: "
                f"expected {self.split.source_fingerprint!r}, got "
                f"{source_fingerprint!r}"
            )
        if source_fingerprint is None and self.split is not None:
            source_fingerprint = self.split.source_fingerprint
        if source_fingerprint is not None:
            _require_nonempty_string(source_fingerprint, name="source_fingerprint")
        else:
            raise ValueError("SelectionReport requires source_fingerprint")
        provenance = _freeze_json(self.provenance, path="SelectionReport.provenance")
        metadata = _freeze_json(self.metadata, path="SelectionReport.metadata")
        if not isinstance(provenance, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError(
                "SelectionReport provenance and metadata must be JSON objects"
            )
        provenance_values = dict(provenance)
        provenance_values.setdefault("package", "repsteer")
        provenance_values.setdefault("version", __version__)
        _require_nonempty_string(
            provenance_values["package"], name="SelectionReport.provenance.package"
        )
        _require_nonempty_string(
            provenance_values["version"], name="SelectionReport.provenance.version"
        )
        provenance = _freeze_json(provenance_values, path="SelectionReport.provenance")
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "selected_id", selected_id)
        object.__setattr__(self, "secondary_metrics", secondary)
        object.__setattr__(self, "candidate_order", candidate_order)
        object.__setattr__(self, "split_fingerprint", split_fingerprint)
        object.__setattr__(self, "source_fingerprint", source_fingerprint)
        selection_seed = (
            self.split.seed
            if self.selection_seed is None and self.split is not None
            else self.selection_seed
        )
        object.__setattr__(
            self,
            "selection_seed",
            _require_seed(selection_seed, name="selection_seed"),
        )
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "metadata", metadata)
        expected_checksum = self._computed_checksum()
        if self.checksum != "":
            supplied_checksum = _validate_checksum(
                self.checksum, name="SelectionReport checksum"
            )
            if supplied_checksum != expected_checksum:
                raise ValueError(
                    "SelectionReport checksum mismatch: "
                    f"expected {expected_checksum}, got {supplied_checksum}"
                )
        object.__setattr__(self, "checksum", expected_checksum)

    @property
    def seed(self) -> int | None:
        """Alias for :attr:`selection_seed` for concise report consumers."""

        return self.selection_seed

    @property
    def tie_break_rule(self) -> tuple[str, ...]:
        """Return the exact ordering rule persisted in :meth:`to_dict`."""

        rules = [
            f"objective:{self.objective}:{'maximize' if self.maximize else 'minimize'}"
        ]
        rules.extend(
            f"secondary:{metric.name}:{'maximize' if metric.maximize else 'minimize'}"
            for metric in self.secondary_metrics
        )
        rules.append("candidate_id:canonical_ascending")
        return tuple(rules)

    @property
    def selected(self) -> CandidateEvaluation:
        """Return the selected candidate evaluation after replay validation."""

        return self.validate_selection()

    def validate_selection(self) -> CandidateEvaluation:
        """Replay selection and raise if this record is internally inconsistent."""

        selected = select_best(
            self.evaluations,
            objective=self.objective,
            maximize=self.maximize,
            secondary_metrics=self.secondary_metrics,
            candidate_order=self.candidate_order,
        )
        if selected.candidate.id != self.selected_id:
            raise ValueError(
                "SelectionReport selected candidate is not reproducible: "
                f"expected {selected.candidate.id!r}, got {self.selected_id!r}"
            )
        if self.checksum != self._computed_checksum():
            raise ValueError("SelectionReport checksum does not match report content")
        return selected

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "objective": self.objective,
            "maximize": self.maximize,
            "secondary_metrics": [item.to_dict() for item in self.secondary_metrics],
            "candidate_order": self.candidate_order,
            "tie_break_rule": list(self.tie_break_rule),
            "evaluations": [item.to_dict() for item in self.evaluations],
            "selected_id": self.selected_id,
            "split": None if self.split is None else self.split.to_dict(),
            "split_fingerprint": self.split_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "selection_seed": self.selection_seed,
            "provenance": _json_value(self.provenance),
            "metadata": _json_value(self.metadata),
        }

    def _computed_checksum(self) -> str:
        return _checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        """Return the checksummed JSON record, with no evaluator payload."""

        return {**self._payload(), "checksum": self.checksum}

    @property
    def canonical(self) -> str:
        """Return the canonical serialization used for the report checksum."""

        return _canonical_json(self.to_dict())

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize the report to finite JSON, optionally writing ``path``."""

        payload = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload + "\n", encoding="utf-8")
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SelectionReport:
        """Load and validate a report, rejecting unknown schemas and tampering."""

        document = _as_mapping(value, name="SelectionReport")
        allowed = {
            "schema_version",
            "objective",
            "maximize",
            "secondary_metrics",
            "candidate_order",
            "tie_break_rule",
            "evaluations",
            "selected_id",
            "split",
            "split_fingerprint",
            "source_fingerprint",
            "selection_seed",
            "provenance",
            "metadata",
            "checksum",
        }
        _reject_unknown_fields(document, name="SelectionReport", allowed=allowed)
        required = allowed
        missing = sorted(required - set(document))
        if missing:
            raise ValueError(
                "SelectionReport is missing required fields: " + ", ".join(missing)
            )
        if document["schema_version"] != _SELECTION_SCHEMA_VERSION:
            raise ValueError(
                "unsupported SelectionReport schema "
                f"{document['schema_version']!r}; supported: "
                f"{_SELECTION_SCHEMA_VERSION!r}"
            )
        _validate_checksum(document["checksum"], name="SelectionReport checksum")
        raw_evaluations = document["evaluations"]
        if not isinstance(raw_evaluations, list):
            raise TypeError("SelectionReport.evaluations must be a JSON list")
        raw_secondary = document["secondary_metrics"]
        if not isinstance(raw_secondary, list):
            raise TypeError("SelectionReport.secondary_metrics must be a JSON list")
        supplied_tie_break = document["tie_break_rule"]
        if not isinstance(supplied_tie_break, list) or not all(
            isinstance(rule, str) for rule in supplied_tie_break
        ):
            raise TypeError(
                "SelectionReport.tie_break_rule must be a JSON list of strings"
            )
        raw_split = document["split"]
        if raw_split is not None and not isinstance(raw_split, Mapping):
            raise TypeError("SelectionReport.split must be a JSON object or null")
        report = cls(
            objective=document["objective"],
            maximize=document["maximize"],
            evaluations=tuple(
                CandidateEvaluation.from_dict(_as_mapping(item, name="evaluation"))
                for item in raw_evaluations
            ),
            selected_id=document["selected_id"],
            secondary_metrics=tuple(
                SecondaryMetric.from_dict(_as_mapping(item, name="secondary metric"))
                for item in raw_secondary
            ),
            candidate_order=document["candidate_order"],
            split=None if raw_split is None else HoldoutSplit.from_dict(raw_split),
            split_fingerprint=document["split_fingerprint"],
            source_fingerprint=document["source_fingerprint"],
            selection_seed=document["selection_seed"],
            provenance=_as_mapping(document["provenance"], name="provenance"),
            metadata=_as_mapping(document["metadata"], name="metadata"),
            schema_version=document["schema_version"],
            checksum=document["checksum"],
        )
        if tuple(supplied_tie_break) != report.tie_break_rule:
            raise ValueError(
                "SelectionReport tie_break_rule does not match its selection semantics"
            )
        return report

    @classmethod
    def from_json(cls, value: str | Path) -> SelectionReport:
        """Load a report from a JSON string or an existing JSON file path."""

        if isinstance(value, Path):
            payload = value.read_text(encoding="utf-8")
        elif isinstance(value, str):
            if value.lstrip().startswith(("{", "[")):
                payload = value
            else:
                try:
                    candidate_path = Path(value)
                    payload = (
                        candidate_path.read_text(encoding="utf-8")
                        if candidate_path.is_file()
                        else value
                    )
                except OSError:
                    # An inline JSON record can exceed platform filename limits.
                    payload = value
        else:  # pragma: no cover - type guard for public callers
            raise TypeError("SelectionReport.from_json expects a JSON string or Path")
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SelectionReport JSON is invalid: {exc}") from exc
        return cls.from_dict(_as_mapping(document, name="SelectionReport JSON"))


def grid_search(
    candidates: Iterable[Candidate],
    evaluator: Callable[[Candidate], CandidateEvaluation | Mapping[str, float]],
    *,
    objective: str,
    maximize: bool = True,
    secondary_metrics: Sequence[
        SecondaryMetric | str | tuple[str, bool] | Mapping[str, Any]
    ]
    | str = (),
    secondary_metric: str | None = None,
    secondary_maximize: bool | None = None,
    split: HoldoutSplit | None = None,
    split_fingerprint: str | None = None,
    source_fingerprint: str | None = None,
    seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    allow_partial: bool = False,
) -> SelectionReport:
    """Evaluate every candidate synchronously and return an auditable report.

    ``evaluator`` is invoked but never serialized.  Any evaluator failure is
    wrapped with its candidate ID and aborts the entire search; partial reports
    are intentionally unsupported in this release.

    Args:
        candidates: JSON-only candidate descriptors with unique stable IDs.
        evaluator: Synchronous callback returning metrics for one candidate.
        objective: Required primary metric used for selection.
        maximize: Whether a larger primary metric is better.
        secondary_metrics: Optional ordered secondary metric declarations.
        secondary_metric: Convenience alias for one secondary metric.
        secondary_maximize: Direction for ``secondary_metric``.
        split: Explicit held-out partition retained in the report when available.
        split_fingerprint: Required if ``split`` is not embedded.
        source_fingerprint: Required dataset fingerprint, checked against ``split``.
        seed: Optional selection/split seed retained in the report.
        metadata: Finite JSON-only caller metadata.
        provenance: Finite JSON-only package or experiment provenance.
        allow_partial: Must remain false; partial reports are unsupported.

    Returns:
        A checksummed :class:`SelectionReport` containing every evaluation.

    Raises:
        RuntimeError: If the evaluator fails, with the failed candidate ID.
        TypeError: If records, metrics, or metadata violate the JSON contract.
        ValueError: If selection semantics or report context are inconsistent.
    """

    if not callable(evaluator):
        raise TypeError("grid_search evaluator must be callable")
    if allow_partial:
        raise ValueError(
            "partial grid-search reports are not supported; set allow_partial=False"
        )
    objective_name = _require_nonempty_string(objective, name="objective")
    if not isinstance(maximize, bool):
        raise TypeError("maximize must be a bool")
    normalized_secondary = _normalise_secondary_metrics(
        secondary_metrics,
        primary_maximize=maximize,
        secondary_metric=secondary_metric,
        secondary_maximize=secondary_maximize,
    )
    if objective_name in {item.name for item in normalized_secondary}:
        raise ValueError("objective cannot also be a secondary metric")
    if split is not None and not isinstance(split, HoldoutSplit):
        raise TypeError("split must be a HoldoutSplit or None")
    if split_fingerprint is not None:
        _require_nonempty_string(split_fingerprint, name="split_fingerprint")
    if split is not None and split_fingerprint not in (None, split.fingerprint):
        raise ValueError(
            "split_fingerprint does not match the embedded HoldoutSplit: "
            f"expected {split.fingerprint!r}, got {split_fingerprint!r}"
        )
    if split is None and split_fingerprint is None:
        raise ValueError("grid_search requires split or split_fingerprint")
    if source_fingerprint is not None:
        _require_nonempty_string(source_fingerprint, name="source_fingerprint")
    if (
        split is not None
        and source_fingerprint is not None
        and split.source_fingerprint is not None
        and source_fingerprint != split.source_fingerprint
    ):
        raise ValueError(
            "source_fingerprint does not match the embedded HoldoutSplit: "
            f"expected {split.source_fingerprint!r}, got {source_fingerprint!r}"
        )
    if source_fingerprint is None and (
        split is None or split.source_fingerprint is None
    ):
        raise ValueError("grid_search requires source_fingerprint")
    normalized_seed = _require_seed(seed, name="seed")
    for name, value in (("metadata", metadata), ("provenance", provenance)):
        if value is not None:
            frozen = _freeze_json(value, path=f"grid_search.{name}")
            if not isinstance(frozen, Mapping):
                raise TypeError(f"grid_search {name} must be a JSON object")
    values = tuple(candidates)
    if not values:
        raise ValueError("grid_search candidates cannot be empty")
    for index, candidate in enumerate(values):
        if not isinstance(candidate, Candidate):
            raise TypeError(
                f"grid_search candidates[{index}] must be Candidate, got "
                f"{type(candidate).__name__}"
            )
    candidate_ids = [candidate.id for candidate in values]
    duplicates = sorted(
        {item for item in candidate_ids if candidate_ids.count(item) > 1}
    )
    if duplicates:
        raise ValueError(
            "grid_search candidate IDs must be unique: " + ", ".join(duplicates)
        )
    values = tuple(sorted(values, key=lambda candidate: candidate.id))
    evaluations: list[CandidateEvaluation] = []
    for candidate in values:
        try:
            result = evaluator(candidate)
            if isinstance(result, CandidateEvaluation):
                if result.candidate != candidate:
                    raise ValueError(
                        "evaluator returned an evaluation for a different candidate "
                        f"({result.candidate.id!r})"
                    )
                evaluation = result
            elif isinstance(result, Mapping):
                evaluation = CandidateEvaluation(candidate=candidate, metrics=result)
            else:
                raise TypeError(
                    "evaluator must return CandidateEvaluation or a metrics mapping, "
                    f"got {type(result).__name__}"
                )
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover - process control
            raise
        except Exception as exc:
            raise RuntimeError(
                f"grid_search evaluator failed for candidate {candidate.id!r}: {exc}"
            ) from exc
        evaluations.append(evaluation)
    selected = select_best(
        evaluations,
        objective=objective_name,
        maximize=maximize,
        secondary_metrics=normalized_secondary,
    )
    return SelectionReport(
        objective=objective_name,
        maximize=maximize,
        evaluations=tuple(evaluations),
        selected_id=selected.candidate.id,
        secondary_metrics=normalized_secondary,
        split=split,
        split_fingerprint=split_fingerprint,
        source_fingerprint=source_fingerprint,
        selection_seed=normalized_seed,
        metadata={} if metadata is None else metadata,
        provenance=(
            {"package": "repsteer", "version": __version__}
            if provenance is None
            else provenance
        ),
    )


__all__ = [
    "Candidate",
    "CandidateEvaluation",
    "HoldoutSplit",
    "SecondaryMetric",
    "SelectionReport",
    "grid_search",
    "make_holdout_split",
    "select_best",
]
