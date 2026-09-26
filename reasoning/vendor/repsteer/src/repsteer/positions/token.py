"""Text token selectors with explicit generation-phase semantics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from repsteer.core import PositionResolutionError, StepContext

from .base import PositionSelector, empty_mask, first_n_true, last_true, nth_true


@dataclass(frozen=True, slots=True)
class LastNonPaddingToken(PositionSelector):
    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        return last_true(context.current_attention_mask(activation))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "last_non_padding_token"}


@dataclass(frozen=True, slots=True)
class LastPromptToken(PositionSelector):
    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        if context.phase == "decode":
            return empty_mask(activation, context)
        mask = context.current_attention_mask(activation)
        lengths = context.prompt_lengths_for(mask.shape[0], activation.device)
        if torch.any(lengths > mask.sum(dim=-1)):
            raise PositionResolutionError(
                "prompt_lengths exceeds the number of non-padding tokens in the "
                "current prefill activation"
            )
        # Selecting by rank among valid tokens works for both left and right
        # padding and remains correct if a forward context has trailing tokens.
        return nth_true(mask, lengths - 1)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "last_prompt_token"}


@dataclass(frozen=True, slots=True)
class PromptTokens(PositionSelector):
    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        if context.phase == "decode":
            return empty_mask(activation, context)
        mask = context.current_attention_mask(activation)
        lengths = context.prompt_lengths_for(mask.shape[0], activation.device)
        if torch.any(lengths > mask.sum(dim=-1)):
            raise PositionResolutionError(
                "prompt_lengths exceeds the number of non-padding tokens in the "
                "current prefill activation"
            )
        return first_n_true(mask, lengths)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "prompt_tokens"}


@dataclass(frozen=True, slots=True)
class GeneratedTokens(PositionSelector):
    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        if context.phase != "decode":
            return empty_mask(activation, context)
        # The RFC defines GeneratedTokens during decode as the current decode
        # token.  This remains true for a no-cache runtime that passes history.
        return last_true(context.current_attention_mask(activation))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "generated_tokens"}


@dataclass(frozen=True, slots=True)
class AllTokens(PositionSelector):
    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        mask = context.current_attention_mask(activation)
        return last_true(mask) if context.phase == "decode" else mask

    def to_dict(self) -> dict[str, Any]:
        return {"type": "all_tokens"}


@dataclass(frozen=True, slots=True, init=False)
class TokenIndices(PositionSelector):
    """Select current-forward sequence indices; negative indices are supported."""

    indices: tuple[int, ...]
    strict: bool

    def __init__(self, indices: Iterable[int], *, strict: bool = False) -> None:
        values = tuple(int(index) for index in indices)
        object.__setattr__(self, "indices", values)
        object.__setattr__(self, "strict", bool(strict))

    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        mask = empty_mask(activation, context)
        sequence = mask.shape[1]
        normalized: list[int] = []
        invalid: list[int] = []
        for index in self.indices:
            current = index + sequence if index < 0 else index
            if 0 <= current < sequence:
                normalized.append(current)
            else:
                invalid.append(index)
        if invalid and self.strict:
            raise PositionResolutionError(
                f"Token indices {invalid} are out of range for current "
                f"length {sequence}"
            )
        if normalized:
            mask[:, torch.as_tensor(normalized, device=mask.device)] = True
        return mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "token_indices",
            "indices": list(self.indices),
            "strict": self.strict,
        }


__all__ = [
    "AllTokens",
    "GeneratedTokens",
    "LastNonPaddingToken",
    "LastPromptToken",
    "PromptTokens",
    "TokenIndices",
]
