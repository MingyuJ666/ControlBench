"""Position selector protocol and mask utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from repsteer.core import PositionResolutionError, StepContext


class PositionSelector(ABC):
    """Resolve semantic positions to a ``[batch, sequence]`` bool mask."""

    @abstractmethod
    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def __call__(self, activation: Tensor, context: StepContext) -> Tensor:
        return self.select(activation, context)


def batch_sequence_shape(activation: Tensor, context: StepContext) -> tuple[int, int]:
    return (
        context.resolved_batch_size(activation),
        context.resolved_sequence_length(activation),
    )


def empty_mask(activation: Tensor, context: StepContext) -> Tensor:
    batch, sequence = batch_sequence_shape(activation, context)
    return torch.zeros((batch, sequence), dtype=torch.bool, device=activation.device)


def last_true(mask: Tensor) -> Tensor:
    """Select the final true entry per row, leaving all-false rows empty."""

    if mask.ndim != 2:
        raise PositionResolutionError(f"Expected a 2-D mask, got {tuple(mask.shape)}")
    result = torch.zeros_like(mask, dtype=torch.bool)
    if mask.shape[1] == 0:
        return result
    reversed_index = mask.flip(-1).to(torch.int64).argmax(dim=-1)
    index = mask.shape[-1] - 1 - reversed_index
    rows = torch.arange(mask.shape[0], device=mask.device)
    valid = mask.any(dim=-1)
    result[rows[valid], index[valid]] = True
    return result


def first_n_true(mask: Tensor, counts: Tensor) -> Tensor:
    """Select the first ``counts[b]`` true positions in each row."""

    if mask.ndim != 2 or counts.ndim != 1 or counts.shape[0] != mask.shape[0]:
        raise PositionResolutionError("first_n_true received incompatible shapes")
    ranks = mask.to(torch.long).cumsum(dim=-1)
    return mask & (ranks <= counts[:, None]) & (counts[:, None] > 0)


def nth_true(mask: Tensor, zero_based_indices: Tensor) -> Tensor:
    """Select a per-row zero-based rank among true entries."""

    ranks = mask.to(torch.long).cumsum(dim=-1) - 1
    return (
        mask
        & (ranks == zero_based_indices[:, None])
        & (zero_based_indices[:, None] >= 0)
    )


def validate_mask(mask: Tensor, activation: Tensor, context: StepContext) -> Tensor:
    expected = batch_sequence_shape(activation, context)
    if tuple(mask.shape) != expected:
        raise PositionResolutionError(
            f"Selector returned mask shape {tuple(mask.shape)}, expected {expected}"
        )
    return mask.to(device=activation.device, dtype=torch.bool)


def apply_mask(original: Tensor, modified: Tensor, mask: Tensor) -> Tensor:
    """Merge an operator result using a selector's bool mask.

    This helper intentionally lives outside operators: operators define the
    numeric transformation, while the runtime owns position and gate semantics.
    """

    if original.shape != modified.shape:
        raise PositionResolutionError(
            f"Operator changed activation shape {tuple(original.shape)} -> "
            f"{tuple(modified.shape)}"
        )
    if original.ndim == 2:
        if tuple(mask.shape) != (1, original.shape[0]):
            raise PositionResolutionError(
                f"Unbatched activation expects mask [1, {original.shape[0]}]"
            )
        broadcast = mask[0, :, None]
    elif original.ndim >= 3:
        if tuple(mask.shape) != (original.shape[0], original.shape[-2]):
            raise PositionResolutionError(
                "Mask must match activation batch and penultimate sequence axes"
            )
        broadcast = mask
        while broadcast.ndim < original.ndim:
            broadcast = broadcast.unsqueeze(-1)
    else:
        raise PositionResolutionError("Cannot position-mask an activation below rank 2")
    return torch.where(broadcast.to(device=original.device), modified, original)


def position_selector_from_dict(value: Mapping[str, Any]) -> PositionSelector:
    """Deserialize built-in selectors from safe configuration data."""

    from .modality import ImagePatches, ImageTokens, ObjectPatches
    from .span import SpecialToken, TextSpan
    from .token import (
        AllTokens,
        GeneratedTokens,
        LastNonPaddingToken,
        LastPromptToken,
        PromptTokens,
        TokenIndices,
    )

    kind = str(value.get("type", "")).lower()
    if kind in ("last_non_padding_token", "lastnonpaddingtoken"):
        return LastNonPaddingToken()
    if kind in ("last_prompt_token", "lastprompttoken"):
        return LastPromptToken()
    if kind in ("prompt_tokens", "prompttokens"):
        return PromptTokens()
    if kind in ("generated_tokens", "generatedtokens"):
        return GeneratedTokens()
    if kind in ("all_tokens", "alltokens"):
        return AllTokens()
    if kind in ("token_indices", "tokenindices"):
        return TokenIndices(
            value.get("indices", ()), strict=bool(value.get("strict", False))
        )
    if kind in ("text_span", "textspan"):
        return TextSpan(
            str(value["text"]),
            occurrence=value.get("occurrence", "first"),
            case_sensitive=bool(value.get("case_sensitive", True)),
        )
    if kind in ("special_token", "specialtoken"):
        return SpecialToken(str(value["token"]), token_id=value.get("token_id"))
    if kind in ("image_tokens", "imagetokens"):
        return ImageTokens(image_index=value.get("image_index", 0))
    if kind in ("image_patches", "imagepatches"):
        return ImagePatches(value["mask"], image_index=value.get("image_index", 0))
    if kind in ("object_patches", "objectpatches"):
        return ObjectPatches(value["boxes"], image_index=value.get("image_index", 0))
    raise PositionResolutionError(f"Unknown position selector type {kind!r}")


__all__ = [
    "PositionSelector",
    "apply_mask",
    "batch_sequence_shape",
    "empty_mask",
    "first_n_true",
    "last_true",
    "nth_true",
    "position_selector_from_dict",
    "validate_mask",
]
