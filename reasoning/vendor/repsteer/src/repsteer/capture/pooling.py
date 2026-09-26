"""Reusable sequence pooling for captured activations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Pooling(Protocol):
    def pool(
        self,
        activations: Tensor,
        *,
        attention_mask: Tensor | None = None,
        token_weights: Tensor | None = None,
        spans: Sequence[tuple[int, int]] | Tensor | None = None,
    ) -> Tensor: ...


@dataclass(frozen=True)
class IdentityPooling:
    name: str = "identity"

    def pool(self, activations: Tensor, **_: Any) -> Tensor:
        return activations

    __call__ = pool


@dataclass(frozen=True)
class LastTokenPooling:
    name: str = "last_token"

    def pool(
        self,
        activations: Tensor,
        *,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        if activations.ndim == 2:
            return activations
        _validate_sequence_activations(activations)
        batch, sequence = activations.shape[:2]
        if attention_mask is None:
            indices = torch.full(
                (batch,), sequence - 1, dtype=torch.long, device=activations.device
            )
        else:
            mask = _mask(attention_mask, activations)
            positions = torch.arange(sequence, device=activations.device).expand(
                batch, -1
            )
            indices = torch.where(mask, positions, -1).amax(dim=1)
            if bool((indices < 0).any()):
                raise ValueError("cannot pool a sample with no selected tokens")
        batch_indices = torch.arange(batch, device=activations.device)
        return activations[batch_indices, indices]

    __call__ = pool


@dataclass(frozen=True)
class MeanPooling:
    name: str = "mean"

    def pool(
        self,
        activations: Tensor,
        *,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        if activations.ndim == 2:
            return activations
        _validate_sequence_activations(activations)
        if attention_mask is None:
            return activations.mean(dim=1)
        mask = _mask(attention_mask, activations)
        weights = mask.to(dtype=activations.dtype)
        denominator = weights.sum(dim=1, keepdim=True)
        if bool((denominator == 0).any()):
            raise ValueError("cannot pool a sample with no selected tokens")
        return (activations * weights.unsqueeze(-1)).sum(dim=1) / denominator

    __call__ = pool


@dataclass(frozen=True)
class MaxPooling:
    name: str = "max"

    def pool(
        self,
        activations: Tensor,
        *,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        if activations.ndim == 2:
            return activations
        _validate_sequence_activations(activations)
        if attention_mask is None:
            return activations.amax(dim=1)
        mask = _mask(attention_mask, activations)
        if bool((~mask).all(dim=1).any()):
            raise ValueError("cannot pool a sample with no selected tokens")
        fill = torch.finfo(activations.dtype).min
        return activations.masked_fill(~mask.unsqueeze(-1), fill).amax(dim=1)

    __call__ = pool


@dataclass(frozen=True)
class WeightedMeanPooling:
    weights: Tensor | Sequence[Sequence[float]] | None = None
    name: str = "weighted_mean"

    def pool(
        self,
        activations: Tensor,
        *,
        attention_mask: Tensor | None = None,
        token_weights: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        if activations.ndim == 2:
            return activations
        _validate_sequence_activations(activations)
        raw_weights = token_weights if token_weights is not None else self.weights
        if raw_weights is None:
            raw_weights = attention_mask
        if raw_weights is None:
            raise ValueError("weighted mean pooling requires token_weights")
        weights = torch.as_tensor(
            raw_weights, device=activations.device, dtype=activations.dtype
        )
        if weights.shape != activations.shape[:2]:
            raise ValueError("token_weights must have shape [batch, sequence]")
        if attention_mask is not None:
            weights = weights * _mask(attention_mask, activations).to(weights.dtype)
        denominator = weights.sum(dim=1, keepdim=True)
        if bool((denominator == 0).any()):
            raise ValueError("weighted pooling weights must have a non-zero sum")
        return (activations * weights.unsqueeze(-1)).sum(dim=1) / denominator

    __call__ = pool


@dataclass(frozen=True)
class SpanMeanPooling:
    spans: Sequence[tuple[int, int]] | Tensor | None = None
    name: str = "span_mean"

    def pool(
        self,
        activations: Tensor,
        *,
        spans: Sequence[tuple[int, int]] | Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        if activations.ndim == 2:
            return activations
        _validate_sequence_activations(activations)
        selected_spans = spans if spans is not None else self.spans
        if selected_spans is None:
            raise ValueError(
                "span mean pooling requires one (start, end) span per sample"
            )
        span_tensor = torch.as_tensor(
            selected_spans, device=activations.device, dtype=torch.long
        )
        if span_tensor.shape != (activations.shape[0], 2):
            raise ValueError("spans must have shape [batch, 2]")
        sequence = activations.shape[1]
        rows: list[Tensor] = []
        for row, (start_tensor, end_tensor) in zip(
            activations, span_tensor, strict=True
        ):
            start, end = int(start_tensor), int(end_tensor)
            if start < 0:
                start += sequence
            if end < 0:
                end += sequence
            if not 0 <= start < end <= sequence:
                raise ValueError(f"invalid half-open token span ({start}, {end})")
            rows.append(row[start:end].mean(dim=0))
        return torch.stack(rows)

    __call__ = pool


def resolve_pooling(pooling: str | Pooling | None) -> Pooling:
    if pooling is None:
        return IdentityPooling()
    if isinstance(pooling, str):
        normalized = pooling.lower().replace("-", "_").replace(" ", "_")
        choices: dict[str, Pooling] = {
            "identity": IdentityPooling(),
            "none": IdentityPooling(),
            "last": LastTokenPooling(),
            "last_token": LastTokenPooling(),
            "last_non_padding_token": LastTokenPooling(),
            "mean": MeanPooling(),
            "max": MaxPooling(),
            "weighted": WeightedMeanPooling(),
            "weighted_mean": WeightedMeanPooling(),
            "span": SpanMeanPooling(),
            "span_mean": SpanMeanPooling(),
        }
        if normalized not in choices:
            raise ValueError(f"unknown activation pooling: {pooling!r}")
        return choices[normalized]
    if isinstance(pooling, Pooling):
        return pooling
    if callable(pooling):
        return _CallablePooling(pooling)
    raise TypeError("pooling must be a name, Pooling object, or callable")


def pool_activations(
    activations: Tensor,
    pooling: str | Pooling | None,
    *,
    attention_mask: Tensor | None = None,
    token_weights: Tensor | None = None,
    spans: Sequence[tuple[int, int]] | Tensor | None = None,
) -> Tensor:
    return resolve_pooling(pooling).pool(
        activations,
        attention_mask=attention_mask,
        token_weights=token_weights,
        spans=spans,
    )


@dataclass(frozen=True)
class _CallablePooling:
    function: Any

    def pool(self, activations: Tensor, **kwargs: Any) -> Tensor:
        try:
            result = cast(Tensor, self.function(activations, **kwargs))
        except TypeError:
            result = cast(Tensor, self.function(activations))
        if not isinstance(result, Tensor):
            raise TypeError("a pooling callable must return a torch.Tensor")
        return result


def _validate_sequence_activations(activations: Tensor) -> None:
    if activations.ndim != 3:
        raise ValueError(
            "sequence pooling expects [batch, sequence, hidden] activations, "
            f"got shape {tuple(activations.shape)}"
        )


def _mask(attention_mask: Tensor, activations: Tensor) -> Tensor:
    mask = torch.as_tensor(attention_mask, device=activations.device).bool()
    if mask.shape != activations.shape[:2]:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    return mask


# Concise public aliases.
Identity = IdentityPooling
LastToken = LastTokenPooling
Mean = MeanPooling
Max = MaxPooling
WeightedMean = WeightedMeanPooling
SpanMean = SpanMeanPooling


__all__ = [
    "Identity",
    "IdentityPooling",
    "LastToken",
    "LastTokenPooling",
    "Max",
    "MaxPooling",
    "Mean",
    "MeanPooling",
    "Pooling",
    "SpanMean",
    "SpanMeanPooling",
    "WeightedMean",
    "WeightedMeanPooling",
    "pool_activations",
    "resolve_pooling",
]
