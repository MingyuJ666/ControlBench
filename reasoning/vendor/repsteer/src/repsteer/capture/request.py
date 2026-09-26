"""Public request/result types for activation capture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor

from repsteer.data.fingerprint import Fingerprint, stable_fingerprint


@dataclass(frozen=True)
class CaptureRequest:
    """A model-independent description of an activation capture operation."""

    inputs: Sequence[Any]
    site: Any
    positions: Any
    pooling: Any = "identity"
    batch_size: int | None = None
    sample_weights: Sequence[float] | Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dataset_fingerprint: str | None = None
    apply_chat_template: bool | None = None
    add_generation_prompt: bool = False
    system_prompt: str | None = None
    chat_template_kwargs: Mapping[str, Any] | None = None
    special_tokens: bool | Mapping[str, Any] | None = None
    dtype: str | torch.dtype | None = None
    model_mode: str = "eval"
    seed: int = 42
    detach: bool = True
    gradient: bool = False

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        if not inputs:
            raise ValueError("capture inputs cannot be empty")
        object.__setattr__(self, "inputs", inputs)
        if self.apply_chat_template is not None and not isinstance(
            self.apply_chat_template, bool
        ):
            raise TypeError("apply_chat_template must be True, False, or None")
        if not isinstance(self.add_generation_prompt, bool):
            raise TypeError("add_generation_prompt must be a bool")
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise TypeError("system_prompt must be a string or None")
        if self.chat_template_kwargs is not None:
            if not isinstance(self.chat_template_kwargs, Mapping):
                raise TypeError("chat_template_kwargs must be a mapping or None")
            object.__setattr__(
                self, "chat_template_kwargs", dict(self.chat_template_kwargs)
            )
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.sample_weights is not None:
            if isinstance(self.sample_weights, Tensor):
                weights = tuple(
                    float(item) for item in self.sample_weights.detach().cpu().flatten()
                )
            else:
                weights = tuple(float(item) for item in self.sample_weights)
            if len(weights) != len(inputs):
                raise ValueError(
                    f"sample_weights has length {len(weights)}, expected {len(inputs)}"
                )
            if any(weight < 0 for weight in weights):
                raise ValueError("sample_weights must be non-negative")
            object.__setattr__(self, "sample_weights", weights)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "seed", int(self.seed))
        if self.gradient and self.detach:
            # Gradient capture and detachment are mutually exclusive.  Treat the
            # explicit gradient request as authoritative for ergonomic construction.
            object.__setattr__(self, "detach", False)

    @property
    def fingerprint(self) -> Fingerprint:
        return stable_fingerprint(
            {
                "inputs": self.inputs,
                "site": self.site,
                "positions": self.positions,
                "pooling": self.pooling,
                "batch_size": self.batch_size,
                "sample_weights": self.sample_weights,
                "metadata": self.metadata,
                "dataset_fingerprint": self.dataset_fingerprint,
                "rendering": {
                    "apply_chat_template": self.apply_chat_template,
                    "add_generation_prompt": self.add_generation_prompt,
                    "system_prompt": self.system_prompt,
                    "chat_template_kwargs": self.chat_template_kwargs,
                    "special_tokens": self.special_tokens,
                },
                "dtype": self.dtype,
                "model_mode": self.model_mode,
                "seed": self.seed,
                "detach": self.detach,
                "gradient": self.gradient,
            }
        )

    def with_inputs(
        self,
        inputs: Sequence[Any],
        *,
        sample_weights: Sequence[float] | Tensor | None = None,
    ) -> CaptureRequest:
        return replace(self, inputs=inputs, sample_weights=sample_weights)


@dataclass
class ActivationBatch:
    """Captured activations and the masks/provenance needed to pool them."""

    activations: Tensor
    request: CaptureRequest | None = None
    attention_mask: Tensor | None = None
    token_ids: Tensor | None = None
    sample_weights: Tensor | Sequence[float] | None = None
    sample_metadata: Sequence[Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    pooled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.activations, Tensor):
            self.activations = torch.as_tensor(self.activations)
        if self.activations.ndim == 0:
            raise ValueError("captured activations need a batch dimension")
        batch_size = int(self.activations.shape[0])
        if self.sample_weights is None and self.request is not None:
            self.sample_weights = self.request.sample_weights
        if self.sample_weights is not None:
            weights = torch.as_tensor(
                self.sample_weights,
                dtype=torch.float32,
                device=self.activations.device,
            ).flatten()
            if weights.numel() != batch_size:
                raise ValueError(
                    f"sample_weights has length {weights.numel()}, "
                    f"expected {batch_size}"
                )
            if bool((weights < 0).any()):
                raise ValueError("sample_weights must be non-negative")
            self.sample_weights = weights
        if self.attention_mask is not None:
            self.attention_mask = torch.as_tensor(
                self.attention_mask, device=self.activations.device
            )
            if self.attention_mask.shape[0] != batch_size:
                raise ValueError(
                    "attention_mask batch dimension does not match activations"
                )
        if self.token_ids is not None:
            self.token_ids = torch.as_tensor(
                self.token_ids, device=self.activations.device
            )
            if self.token_ids.shape[0] != batch_size:
                raise ValueError("token_ids batch dimension does not match activations")
        self.sample_metadata = tuple(dict(item) for item in self.sample_metadata)
        if self.sample_metadata and len(self.sample_metadata) != batch_size:
            raise ValueError(
                "sample_metadata batch dimension does not match activations"
            )
        self.metadata = dict(self.metadata)

    @property
    def tensor(self) -> Tensor:
        return self.activations

    @property
    def weights(self) -> Tensor | None:
        return self.sample_weights if isinstance(self.sample_weights, Tensor) else None

    @property
    def shape(self) -> torch.Size:
        return self.activations.shape

    @property
    def device(self) -> torch.device:
        return self.activations.device

    @property
    def dtype(self) -> torch.dtype:
        return self.activations.dtype

    def __len__(self) -> int:
        return int(self.activations.shape[0])

    def pool(self, pooling: Any | None = None) -> ActivationBatch:
        from .pooling import pool_activations

        pooling = self.request.pooling if pooling is None and self.request else pooling
        pooling = "identity" if pooling is None else pooling
        activations = pool_activations(
            self.activations,
            pooling,
            attention_mask=self.attention_mask,
            token_weights=self.metadata.get("token_weights"),
            spans=self.metadata.get("spans"),
        )
        return ActivationBatch(
            activations=activations,
            request=self.request,
            attention_mask=self.attention_mask,
            token_ids=self.token_ids,
            sample_weights=self.sample_weights,
            sample_metadata=self.sample_metadata,
            metadata=self.metadata,
            pooled=True,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ActivationBatch:
        activations = self.activations.to(device=device, dtype=dtype)
        mask = (
            self.attention_mask.to(device=device)
            if self.attention_mask is not None
            else None
        )
        token_ids = (
            self.token_ids.to(device=device) if self.token_ids is not None else None
        )
        weights = (
            self.sample_weights.to(device=device)
            if isinstance(self.sample_weights, Tensor)
            else self.sample_weights
        )
        return ActivationBatch(
            activations=activations,
            request=self.request,
            attention_mask=mask,
            token_ids=token_ids,
            sample_weights=weights,
            sample_metadata=self.sample_metadata,
            metadata=self.metadata,
            pooled=self.pooled,
        )

    def detach(self) -> ActivationBatch:
        result = self.to()
        result.activations = result.activations.detach()
        return result

    def cpu(self) -> ActivationBatch:
        return self.to(device="cpu")

    def clone(self) -> ActivationBatch:
        result = self.to()
        result.activations = result.activations.clone()
        if result.attention_mask is not None:
            result.attention_mask = result.attention_mask.clone()
        if result.token_ids is not None:
            result.token_ids = result.token_ids.clone()
        if isinstance(result.sample_weights, Tensor):
            result.sample_weights = result.sample_weights.clone()
        return result


__all__ = ["ActivationBatch", "CaptureRequest"]
