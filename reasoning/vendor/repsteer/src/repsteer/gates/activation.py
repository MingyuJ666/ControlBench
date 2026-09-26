"""Shared activation lookup and sequence reduction for activation gates."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from repsteer.core import GateContext
from repsteer.positions.base import last_true, validate_mask


def context_tensor(
    context: GateContext,
    *,
    key: str | None = None,
    fallbacks: tuple[str, ...] = ("activation", "gate_activation"),
) -> Tensor:
    """Read gate input without coupling the gate package to one runtime."""

    if key is not None:
        value = getattr(context, key, None)
        if value is None:
            value = context.metadata.get(key)
        if value is None:
            raise ValueError(f"GateContext has no activation under key {key!r}")
    else:
        value = getattr(context, "activation", None)
        if value is None:
            for name in fallbacks:
                value = context.metadata.get(name)
                if value is not None:
                    break
    if value is None:
        raise ValueError(
            "Activation gate needs a tensor on GateContext.activation or "
            "GateContext.metadata['activation']"
        )
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, device=context.runtime_device)
    if value.ndim < 1:
        raise ValueError("Gate activation must have a hidden/feature dimension")
    return value


def _looks_batched_pooled(
    scores: Tensor,
    activation: Tensor,
    context: GateContext,
) -> bool:
    if activation.ndim != 2 or scores.ndim != 1:
        return False
    try:
        batch = context.resolved_batch_size()
    except Exception:
        return False
    return batch == activation.shape[0] and (
        batch > 1 or context.sequence_length in (None, 1)
    )


def reduce_token_scores(
    scores: Tensor,
    activation: Tensor,
    context: GateContext,
    *,
    positions: Any | None,
    reduction: str,
) -> tuple[Tensor, Tensor]:
    """Reduce ``[..., sequence]`` scores and return score + validity masks."""

    reduction = str(reduction).lower().replace("-", "_")
    if reduction not in {"mean", "max", "last", "none"}:
        raise ValueError("gate reduction must be 'mean', 'max', 'last', or 'none'")
    if scores.ndim == 0:
        return scores.reshape(1), torch.ones(1, dtype=torch.bool, device=scores.device)
    if (
        scores.ndim == 1
        and positions is None
        and _looks_batched_pooled(scores, activation, context)
    ):
        return scores, torch.ones_like(scores, dtype=torch.bool)

    if activation.ndim == 1:
        if scores.numel() != 1:
            raise ValueError("A vector activation must produce one gate score")
        return scores.reshape(1), torch.ones(1, dtype=torch.bool, device=scores.device)
    if activation.ndim == 2:
        token_scores = scores.reshape(1, -1)
    elif activation.ndim >= 3:
        if scores.ndim != 2:
            raise ValueError(
                "Token gate scores must be [batch, sequence], got "
                f"{tuple(scores.shape)}"
            )
        token_scores = scores
    else:  # pragma: no cover - guarded by context_tensor
        raise ValueError("Gate activation has no sequence or hidden dimension")

    if positions is not None:
        raw_mask = positions.select(activation, context)
        mask = validate_mask(torch.as_tensor(raw_mask), activation, context)
    else:
        mask = context.current_attention_mask(activation)
    mask = mask.to(device=token_scores.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(token_scores.shape):
        raise ValueError(
            f"Gate mask shape {tuple(mask.shape)} does not match scores "
            f"{tuple(token_scores.shape)}"
        )
    if reduction == "none":
        return token_scores, mask
    valid = mask.any(dim=-1)
    if reduction == "mean":
        total = torch.where(mask, token_scores, torch.zeros_like(token_scores)).sum(
            dim=-1
        )
        pooled = total / mask.sum(dim=-1).clamp_min(1)
    elif reduction == "max":
        floor = torch.full_like(token_scores, -torch.inf)
        pooled = torch.where(mask, token_scores, floor).max(dim=-1).values
    else:
        final = last_true(mask)
        pooled = torch.where(
            final,
            token_scores,
            torch.zeros_like(token_scores),
        ).sum(dim=-1)
    # Never let an all-padding/all-unselected row trigger, even for a negative
    # threshold.  Gate implementations combine their comparison with `valid`.
    return pooled, valid


__all__ = ["context_tensor", "reduce_token_scores"]
