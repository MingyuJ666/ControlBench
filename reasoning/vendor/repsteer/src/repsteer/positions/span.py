"""Selectors that require tokenizer/processor alignment metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from repsteer.core import PositionResolutionError, StepContext

from .base import PositionSelector, empty_mask

Occurrence = Literal["first", "last", "all"]


def _context_value(context: StepContext, attribute: str, metadata_key: str) -> Any:
    value = getattr(context, attribute, None)
    return value if value is not None else context.metadata.get(metadata_key)


def _batch_values(value: Any, batch: int, *, name: str) -> list[Any]:
    if value is None:
        raise PositionResolutionError(f"TextSpan requires StepContext.{name}")
    if batch == 1 and isinstance(value, str):
        return [value]
    values = list(value)
    if len(values) != batch:
        raise PositionResolutionError(
            f"{name} has {len(values)} entries for a batch of size {batch}"
        )
    return values


def _occurrences(
    haystack: str, needle: str, occurrence: Occurrence
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    offset = 0
    while True:
        start = haystack.find(needle, offset)
        if start < 0:
            break
        matches.append((start, start + len(needle)))
        offset = start + max(1, len(needle))
    if occurrence == "first":
        return matches[:1]
    if occurrence == "last":
        return matches[-1:]
    return matches


@dataclass(frozen=True, slots=True)
class TextSpan(PositionSelector):
    text: str
    occurrence: Occurrence = "first"
    case_sensitive: bool = True

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("TextSpan.text cannot be empty")
        if self.occurrence not in ("first", "last", "all"):
            raise ValueError("TextSpan.occurrence must be 'first', 'last', or 'all'")

    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        mask = empty_mask(activation, context)
        batch, sequence = mask.shape
        texts = _batch_values(
            _context_value(context, "texts", "texts"), batch, name="texts"
        )
        offsets_value = _context_value(context, "token_offsets", "offset_mapping")
        token_strings_value = _context_value(context, "token_strings", "token_strings")

        if offsets_value is not None:
            offsets = torch.as_tensor(offsets_value, device=activation.device)
            if offsets.ndim == 2 and batch == 1:
                offsets = offsets.unsqueeze(0)
            if offsets.ndim != 3 or offsets.shape[0] != batch or offsets.shape[-1] != 2:
                raise PositionResolutionError(
                    "token_offsets must have shape [batch, sequence, 2]"
                )
            if offsets.shape[1] < sequence:
                raise PositionResolutionError(
                    "token_offsets is shorter than the current activation sequence"
                )
            if offsets.shape[1] > sequence:
                offsets = offsets[:, -sequence:]
            for row, source_text in enumerate(texts):
                source = source_text if self.case_sensitive else source_text.casefold()
                needle = self.text if self.case_sensitive else self.text.casefold()
                spans = _occurrences(source, needle, self.occurrence)
                for start, stop in spans:
                    token_start = offsets[row, :, 0]
                    token_stop = offsets[row, :, 1]
                    mask[row] |= (token_stop > start) & (token_start < stop)
            return mask

        if token_strings_value is not None:
            token_rows = _batch_values(token_strings_value, batch, name="token_strings")
            for row, raw_tokens in enumerate(token_rows):
                tokens = list(raw_tokens)
                if len(tokens) < sequence:
                    raise PositionResolutionError(
                        "token_strings is shorter than activation"
                    )
                if len(tokens) > sequence:
                    tokens = tokens[-sequence:]
                joined = "".join(tokens)
                source = joined if self.case_sensitive else joined.casefold()
                needle = self.text if self.case_sensitive else self.text.casefold()
                spans = _occurrences(source, needle, self.occurrence)
                cursor = 0
                token_spans: list[tuple[int, int]] = []
                for token in tokens:
                    token_spans.append((cursor, cursor + len(token)))
                    cursor += len(token)
                for start, stop in spans:
                    for index, (char_token_start, char_token_stop) in enumerate(
                        token_spans
                    ):
                        if char_token_stop > start and char_token_start < stop:
                            mask[row, index] = True
            return mask

        raise PositionResolutionError(
            "TextSpan cannot map characters to tokens. Supply texts plus "
            "token_offsets (preferred) or token_strings in StepContext."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "text_span",
            "text": self.text,
            "occurrence": self.occurrence,
            "case_sensitive": self.case_sensitive,
        }


def _indices_to_mask(value: Any, mask: Tensor) -> bool:
    tensor = torch.as_tensor(value, device=mask.device)
    if tensor.dtype == torch.bool:
        if tensor.ndim == 1 and mask.shape[0] == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 2 and tensor.shape[0] == mask.shape[0]:
            if tensor.shape[1] < mask.shape[1]:
                raise PositionResolutionError(
                    "Special-token mask is shorter than activation"
                )
            if tensor.shape[1] > mask.shape[1]:
                tensor = tensor[:, -mask.shape[1] :]
            mask |= tensor
            return True
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.ndim == 1:
        # A 1-D vector with one value per batch is interpreted as one index
        # for each row; otherwise it is a shared index list.
        if mask.shape[0] != 1 and tensor.numel() == mask.shape[0]:
            for row, index in enumerate(tensor.tolist()):
                if 0 <= int(index) < mask.shape[1]:
                    mask[row, int(index)] = True
            return True
        for index in tensor.tolist():
            if 0 <= int(index) < mask.shape[1]:
                mask[:, int(index)] = True
        return True
    if tensor.ndim == 2 and tensor.shape[0] == mask.shape[0]:
        for row in range(mask.shape[0]):
            for index in tensor[row].tolist():
                if 0 <= int(index) < mask.shape[1]:
                    mask[row, int(index)] = True
        return True
    return False


@dataclass(frozen=True, slots=True)
class SpecialToken(PositionSelector):
    token: str
    token_id: int | None = None

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("SpecialToken.token cannot be empty")

    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        mask = empty_mask(activation, context)
        token_id = self.token_id
        if token_id is None:
            token_ids_by_name = context.metadata.get("special_token_ids", {})
            if isinstance(token_ids_by_name, Mapping):
                token_id = token_ids_by_name.get(self.token)
        if token_id is not None and context.token_ids is not None:
            ids = torch.as_tensor(context.token_ids, device=activation.device)
            if ids.ndim == 1 and mask.shape[0] == 1:
                ids = ids.unsqueeze(0)
            if ids.ndim != 2 or ids.shape[0] != mask.shape[0]:
                raise PositionResolutionError(
                    "token_ids must have shape [batch, sequence]"
                )
            if ids.shape[1] < mask.shape[1]:
                raise PositionResolutionError("token_ids is shorter than activation")
            if ids.shape[1] > mask.shape[1]:
                ids = ids[:, -mask.shape[1] :]
            return ids == int(token_id)

        modality_map = context.modality_map
        indices = None
        if modality_map is not None:
            mapping = (
                modality_map.get("special_token_indices")
                if isinstance(modality_map, Mapping)
                else getattr(modality_map, "special_token_indices", None)
            )
            if isinstance(mapping, Mapping):
                indices = mapping.get(self.token)
        if indices is not None and _indices_to_mask(indices, mask):
            return mask

        token_strings = _context_value(context, "token_strings", "token_strings")
        if token_strings is not None:
            rows = _batch_values(token_strings, mask.shape[0], name="token_strings")
            for row, values in enumerate(rows):
                tokens = list(values)
                if len(tokens) > mask.shape[1]:
                    tokens = tokens[-mask.shape[1] :]
                if len(tokens) != mask.shape[1]:
                    raise PositionResolutionError(
                        "token_strings does not align to activation"
                    )
                mask[row] = torch.tensor(
                    [value == self.token for value in tokens],
                    device=activation.device,
                    dtype=torch.bool,
                )
            return mask
        raise PositionResolutionError(
            f"Cannot resolve special token {self.token!r}. Supply token_id/token_ids, "
            "a modality map, or token_strings."
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": "special_token", "token": self.token, "token_id": self.token_id}


__all__ = ["SpecialToken", "TextSpan"]
