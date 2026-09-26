"""Capture orchestration independent of any model backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import torch
from torch import Tensor

from .cache import capture_cache_key
from .request import ActivationBatch, CaptureRequest
from .store import ActivationStore


class CaptureRunner:
    def __init__(self, store: ActivationStore | None = None) -> None:
        self.store = store

    def capture(self, model: Any, request: CaptureRequest) -> ActivationBatch:
        key = str(capture_cache_key(model, request))
        cached = _store_get(self.store, key)
        if cached is not None:
            return cached

        if request.batch_size is not None and len(request.inputs) > request.batch_size:
            batches: list[ActivationBatch] = []
            for start in range(0, len(request.inputs), request.batch_size):
                stop = min(start + request.batch_size, len(request.inputs))
                weights = (
                    request.sample_weights[start:stop]
                    if request.sample_weights is not None
                    else None
                )
                child = replace(
                    request,
                    inputs=request.inputs[start:stop],
                    sample_weights=weights,
                    batch_size=None,
                )
                batches.append(self.capture(model, child))
            batch = _concatenate_batches(batches, request)
            metadata = dict(batch.metadata)
            metadata.update(
                {
                    "capture_cache_key": key,
                    "capture_batches": len(batches),
                    "requested_batch_size": request.batch_size,
                }
            )
            batch.metadata = metadata
            _store_put(self.store, key, batch)
            return batch

        capture_method = getattr(model, "capture", None)
        if not callable(capture_method):
            capture_method = getattr(model, "capture_activations", None)
        if not callable(capture_method):
            raise TypeError(
                "model must implement capture(request) or capture_activations(request)"
            )

        result = capture_method(request)
        batch = _coerce_activation_batch(result, request)
        if not batch.pooled:
            batch = batch.pool(request.pooling)
        if request.detach and batch.activations.requires_grad:
            batch = batch.detach()
        metadata = dict(batch.metadata)
        metadata.setdefault("capture_cache_key", key)
        batch.metadata = metadata
        _store_put(self.store, key, batch)
        return batch

    __call__ = capture


def capture_activations(
    model: Any,
    request: CaptureRequest | Sequence[Any] | None = None,
    *,
    inputs: Sequence[Any] | None = None,
    site: Any = None,
    positions: Any = None,
    pooling: Any = "identity",
    store: ActivationStore | None = None,
    **request_kwargs: Any,
) -> ActivationBatch:
    """Capture via the model contract, with a convenience request constructor."""

    if isinstance(request, CaptureRequest):
        if (
            inputs is not None
            or site is not None
            or positions is not None
            or request_kwargs
        ):
            raise ValueError(
                "do not combine a CaptureRequest with request constructor fields"
            )
        capture_request = request
    else:
        if request is not None:
            if inputs is not None:
                raise ValueError("inputs were passed both positionally and by keyword")
            inputs = request
        if inputs is None:
            raise TypeError("capture_activations requires a CaptureRequest or inputs")
        if site is None:
            raise TypeError("site is required when constructing a CaptureRequest")
        if positions is None:
            raise TypeError("positions is required when constructing a CaptureRequest")
        capture_request = CaptureRequest(
            inputs=inputs,
            site=site,
            positions=positions,
            pooling=pooling,
            **request_kwargs,
        )
    return CaptureRunner(store).capture(model, capture_request)


run_capture = capture_activations


def _coerce_activation_batch(result: Any, request: CaptureRequest) -> ActivationBatch:
    if isinstance(result, ActivationBatch):
        if result.request is None:
            result.request = request
        if result.sample_weights is None and request.sample_weights is not None:
            result.sample_weights = torch.as_tensor(
                request.sample_weights,
                dtype=torch.float32,
                device=result.activations.device,
            )
        return result
    if isinstance(result, Tensor):
        return ActivationBatch(result, request=request)
    if isinstance(result, Mapping):
        activations = result.get("activations", result.get("activation"))
        if activations is None:
            raise ValueError("capture mapping must contain 'activations'")
        return ActivationBatch(
            activations=activations,
            request=request,
            attention_mask=result.get("attention_mask", result.get("mask")),
            token_ids=result.get("token_ids", result.get("input_ids")),
            sample_weights=result.get("sample_weights", request.sample_weights),
            sample_metadata=result.get("sample_metadata", ()),
            metadata=result.get("metadata", {}),
            pooled=bool(result.get("pooled", False)),
        )
    if isinstance(result, tuple) and result and isinstance(result[0], Tensor):
        return ActivationBatch(
            result[0],
            request=request,
            attention_mask=result[1] if len(result) > 1 else None,
        )
    if (
        isinstance(result, Sequence)
        and result
        and all(isinstance(item, ActivationBatch) for item in result)
    ):
        return _concatenate_batches(result, request)
    raise TypeError(
        "model capture must return ActivationBatch, Tensor, mapping, or batches"
    )


def _concatenate_batches(
    batches: Sequence[ActivationBatch], request: CaptureRequest
) -> ActivationBatch:
    if not batches:
        raise ValueError("cannot concatenate an empty capture batch sequence")

    def concatenate(name: str, *, pad_value: int | float = 0) -> Tensor | None:
        values = [getattr(batch, name) for batch in batches]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"cannot concatenate partially missing {name}")
        tensors = [value for value in values if isinstance(value, Tensor)]
        if len(tensors) != len(values):
            raise TypeError(f"{name} values must be tensors")
        return _concatenate_tensors(tensors, name=name, pad_value=pad_value)

    metadata: dict[str, Any] = {}
    for batch in batches:
        metadata.update(batch.metadata)
    activations = _concatenate_tensors(
        [batch.activations for batch in batches], name="activations", pad_value=0
    )
    attention_mask = concatenate("attention_mask", pad_value=0)
    if attention_mask is None and activations.ndim >= 3:
        sequence_lengths = [int(batch.activations.shape[1]) for batch in batches]
        if len(set(sequence_lengths)) > 1:
            attention_mask = torch.zeros(
                (activations.shape[0], activations.shape[1]),
                dtype=torch.bool,
                device=activations.device,
            )
            row = 0
            for batch, length in zip(batches, sequence_lengths, strict=True):
                attention_mask[row : row + len(batch), :length] = True
                row += len(batch)
    return ActivationBatch(
        activations=activations,
        request=request,
        attention_mask=attention_mask,
        token_ids=concatenate("token_ids", pad_value=0),
        sample_weights=concatenate("sample_weights"),
        sample_metadata=tuple(
            item for batch in batches for item in batch.sample_metadata
        ),
        metadata=metadata,
        pooled=all(batch.pooled for batch in batches),
    )


def _concatenate_tensors(
    tensors: Sequence[Tensor], *, name: str, pad_value: int | float
) -> Tensor:
    first = tensors[0]
    if any(tensor.device != first.device for tensor in tensors):
        raise ValueError(f"cannot concatenate {name} from different devices")
    if any(tensor.dtype != first.dtype for tensor in tensors):
        raise ValueError(f"cannot concatenate {name} with different dtypes")
    if any(tensor.ndim != first.ndim for tensor in tensors):
        raise ValueError(f"cannot concatenate {name} with different ranks")
    if all(tensor.shape[1:] == first.shape[1:] for tensor in tensors):
        return torch.cat(tuple(tensors), dim=0)
    if first.ndim < 2 or any(tensor.shape[2:] != first.shape[2:] for tensor in tensors):
        shapes = [tuple(tensor.shape) for tensor in tensors]
        raise ValueError(f"cannot concatenate incompatible {name} shapes: {shapes}")

    maximum = max(int(tensor.shape[1]) for tensor in tensors)
    padded: list[Tensor] = []
    for tensor in tensors:
        shape = (tensor.shape[0], maximum, *tensor.shape[2:])
        value = torch.full(
            shape,
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        value[:, : tensor.shape[1]] = tensor
        padded.append(value)
    return torch.cat(padded, dim=0)


def _store_get(store: ActivationStore | None, key: str) -> ActivationBatch | None:
    if store is None:
        return None
    getter = getattr(store, "get", None)
    if callable(getter):
        value = getter(key)
        if value is not None and not isinstance(value, ActivationBatch):
            raise TypeError("activation store returned a non-ActivationBatch value")
        return value
    try:
        value = cast(Any, store)[key]
        if not isinstance(value, ActivationBatch):
            raise TypeError("activation store returned a non-ActivationBatch value")
        return value
    except KeyError:
        return None


def _store_put(store: ActivationStore | None, key: str, value: ActivationBatch) -> None:
    if store is None:
        return
    putter = getattr(store, "put", None)
    if callable(putter):
        putter(key, value)
    else:
        cast(Any, store)[key] = value


__all__ = ["CaptureRunner", "capture_activations", "run_capture"]
