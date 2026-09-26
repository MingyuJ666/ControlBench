"""Contrastive text/chat/multimodal input records."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from .fingerprint import Fingerprint, stable_fingerprint
from .records import ContrastiveRecord, ExampleRecord


def _record_from_mapping(record: Mapping[str, Any]) -> ContrastiveRecord:
    positive = record.get("positive")
    negative = record.get("negative")
    # Explicit chat columns are convenient when records also contain raw-text
    # columns for display or evaluation.
    if positive is None:
        positive = record.get("positive_messages", record.get("chosen"))
    if negative is None:
        negative = record.get("negative_messages", record.get("rejected"))
    known = {
        "positive",
        "negative",
        "positive_messages",
        "negative_messages",
        "chosen",
        "rejected",
        "weight",
        "positive_weight",
        "negative_weight",
        "metadata",
        "positive_metadata",
        "negative_metadata",
        "id",
    }
    metadata = dict(record.get("metadata", {}))
    # Preserve extra columns as sample metadata rather than silently dropping
    # information supplied by a benchmark dataset.
    metadata.update(
        {str(key): value for key, value in record.items() if key not in known}
    )
    return ContrastiveRecord(
        positive=positive,
        negative=negative,
        weight=record.get("weight", 1.0),
        positive_weight=record.get("positive_weight"),
        negative_weight=record.get("negative_weight"),
        metadata=metadata,
        positive_metadata=record.get("positive_metadata", {}),
        negative_metadata=record.get("negative_metadata", {}),
        id=record.get("id"),
    )


class ContrastivePairs:
    """Positive and negative model inputs, paired or grouped independently.

    Inputs are deliberately opaque.  A value may be raw text, chat messages, or a
    processor-ready multimodal mapping.  Rendering belongs to the model adapter and
    its settings are captured by :class:`repsteer.capture.CaptureRequest`.
    """

    def __init__(
        self,
        records: Iterable[ContrastiveRecord | Mapping[str, Any]] | None = None,
        *,
        positives: Sequence[Any] | None = None,
        negatives: Sequence[Any] | None = None,
        paired: bool | None = None,
        weights: Sequence[float] | None = None,
        positive_weights: Sequence[float] | None = None,
        negative_weights: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if records is not None and (positives is not None or negatives is not None):
            raise ValueError("pass records or positive/negative groups, not both")

        converted: list[ContrastiveRecord]
        if records is not None:
            converted = [
                (
                    item
                    if isinstance(item, ContrastiveRecord)
                    else _record_from_mapping(item)
                )
                for item in records
            ]
        else:
            positives = tuple(positives or ())
            negatives = tuple(negatives or ())
            inferred_paired = len(positives) == len(negatives) and bool(positives)
            use_pairs = inferred_paired if paired is None else paired
            if use_pairs:
                if len(positives) != len(negatives):
                    raise ValueError(
                        "paired positives and negatives must have equal length"
                    )
                pair_weights = _weights(weights, len(positives), "weights")
                pos_weights = _weights(
                    positive_weights, len(positives), "positive_weights"
                )
                neg_weights = _weights(
                    negative_weights, len(negatives), "negative_weights"
                )
                converted = [
                    ContrastiveRecord(
                        positive=positive,
                        negative=negative,
                        weight=pair_weights[index] if pair_weights else 1.0,
                        positive_weight=pos_weights[index] if pos_weights else None,
                        negative_weight=neg_weights[index] if neg_weights else None,
                    )
                    for index, (positive, negative) in enumerate(
                        zip(positives, negatives, strict=True)
                    )
                ]
            else:
                if weights is not None:
                    raise ValueError(
                        "shared weights are ambiguous for unpaired groups; "
                        "use side weights"
                    )
                pos_weights = _weights(
                    positive_weights, len(positives), "positive_weights"
                )
                neg_weights = _weights(
                    negative_weights, len(negatives), "negative_weights"
                )
                converted = [
                    ContrastiveRecord(
                        positive=value,
                        positive_weight=pos_weights[index] if pos_weights else None,
                    )
                    for index, value in enumerate(positives)
                ]
                converted.extend(
                    ContrastiveRecord(
                        negative=value,
                        negative_weight=neg_weights[index] if neg_weights else None,
                    )
                    for index, value in enumerate(negatives)
                )

        if not converted:
            raise ValueError("contrastive data cannot be empty")
        if not any(record.positive is not None for record in converted):
            raise ValueError("contrastive data needs at least one positive input")
        if not any(record.negative is not None for record in converted):
            raise ValueError("contrastive data needs at least one negative input")

        inferred_paired = all(record.paired for record in converted)
        if paired is True and not inferred_paired:
            raise ValueError("paired=True requires every record to contain both sides")
        self._records = tuple(converted)
        self._paired = inferred_paired if paired is None else bool(paired)
        self.metadata = dict(metadata or {})
        self._fingerprint = stable_fingerprint(
            {
                "type": "contrastive_pairs",
                "paired": self._paired,
                "records": self._records,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[ContrastiveRecord | Mapping[str, Any]],
        *,
        paired: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContrastivePairs:
        return cls(records, paired=paired, metadata=metadata)

    @classmethod
    def from_pairs(
        cls,
        positives: Sequence[Any],
        negatives: Sequence[Any],
        *,
        weights: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContrastivePairs:
        return cls(
            positives=positives,
            negatives=negatives,
            paired=True,
            weights=weights,
            metadata=metadata,
        )

    @classmethod
    def from_groups(
        cls,
        positives: Sequence[Any],
        negatives: Sequence[Any],
        *,
        positive_weights: Sequence[float] | None = None,
        negative_weights: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContrastivePairs:
        return cls(
            positives=positives,
            negatives=negatives,
            paired=False,
            positive_weights=positive_weights,
            negative_weights=negative_weights,
            metadata=metadata,
        )

    unpaired = from_groups

    @property
    def records(self) -> tuple[ContrastiveRecord, ...]:
        return self._records

    @property
    def paired(self) -> bool:
        return self._paired

    @property
    def is_paired(self) -> bool:
        return self._paired

    @property
    def positives(self) -> tuple[Any, ...]:
        return tuple(
            record.positive for record in self._records if record.positive is not None
        )

    @property
    def negatives(self) -> tuple[Any, ...]:
        return tuple(
            record.negative for record in self._records if record.negative is not None
        )

    @property
    def positive_examples(self) -> tuple[ExampleRecord, ...]:
        return tuple(
            ExampleRecord(
                record.positive,
                weight=(
                    record.weight
                    if record.positive_weight is None
                    else record.positive_weight
                ),
                metadata={**record.metadata, **record.positive_metadata},
                id=record.id,
            )
            for record in self._records
            if record.positive is not None
        )

    @property
    def negative_examples(self) -> tuple[ExampleRecord, ...]:
        return tuple(
            ExampleRecord(
                record.negative,
                weight=(
                    record.weight
                    if record.negative_weight is None
                    else record.negative_weight
                ),
                metadata={**record.metadata, **record.negative_metadata},
                id=record.id,
            )
            for record in self._records
            if record.negative is not None
        )

    @property
    def pair_weights(self) -> tuple[float, ...]:
        if not self.paired:
            raise ValueError("pair weights are undefined for unpaired data")
        return tuple(record.weight for record in self._records)

    @property
    def fingerprint(self) -> Fingerprint:
        return self._fingerprint

    @property
    def is_chat(self) -> bool:
        values = self.positives + self.negatives
        return bool(values) and all(_looks_like_chat(value) for value in values)

    def __len__(self) -> int:
        return (
            len(self._records)
            if self.paired
            else len(self.positives) + len(self.negatives)
        )

    def __iter__(self) -> Iterator[ContrastiveRecord]:
        return iter(self._records)

    def __repr__(self) -> str:
        mode = "paired" if self.paired else "unpaired"
        return (
            f"ContrastivePairs({mode}, positives={len(self.positives)}, "
            f"negatives={len(self.negatives)})"
        )


def _weights(
    values: Sequence[float] | None, expected: int, name: str
) -> tuple[float, ...]:
    if values is None:
        return ()
    result = tuple(float(value) for value in values)
    if len(result) != expected:
        raise ValueError(f"{name} has length {len(result)}, expected {expected}")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must be non-negative")
    return result


def _looks_like_chat(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | bytearray)
        and all(
            isinstance(message, Mapping) and "role" in message and "content" in message
            for message in value
        )
    )


__all__ = ["ContrastivePairs", "ContrastiveRecord", "ExampleRecord"]
