"""Shared capture, numerical, and artifact helpers for learners."""

from __future__ import annotations

import platform
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import torch
from torch import Tensor

from repsteer._version import __version__
from repsteer.capture import (
    ActivationBatch,
    ActivationStore,
    CaptureRequest,
    capture_activations,
)
from repsteer.data import ContrastivePairs, canonicalize, stable_fingerprint


@runtime_checkable
class Learner(Protocol):
    def fit(
        self,
        model: Any,
        data: ContrastivePairs,
        *,
        store: ActivationStore | None = None,
    ) -> Any: ...


@dataclass
class ContrastiveLearner:
    """Configuration shared by learners that consume contrastive activations."""

    site: Any
    positions: Any
    pooling: Any = "identity"
    normalize: str | None = "l2"
    seed: int = 42
    batch_size: int | None = None
    apply_chat_template: bool | None = None
    add_generation_prompt: bool = False
    system_prompt: str | None = None
    chat_template_kwargs: Mapping[str, Any] | None = None
    special_tokens: bool | Mapping[str, Any] | None = None
    capture_dtype: str | torch.dtype | None = None

    method: ClassVar[str] = "contrastive"

    def __post_init__(self) -> None:
        self.seed = int(self.seed)
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
            self.chat_template_kwargs = dict(self.chat_template_kwargs)
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def _capture_pair(
        self,
        model: Any,
        data: ContrastivePairs,
        store: ActivationStore | None,
    ) -> tuple[ActivationBatch, ActivationBatch]:
        positive_examples = data.positive_examples
        negative_examples = data.negative_examples
        common = {
            "site": self.site,
            "positions": self.positions,
            "pooling": self.pooling,
            "batch_size": self.batch_size,
            "dataset_fingerprint": str(data.fingerprint),
            "apply_chat_template": self._resolved_apply_chat_template(data),
            "add_generation_prompt": self.add_generation_prompt,
            "system_prompt": self.system_prompt,
            "chat_template_kwargs": self.chat_template_kwargs,
            "special_tokens": self.special_tokens,
            "dtype": self.capture_dtype,
            "seed": self.seed,
        }
        positive_request = CaptureRequest(
            inputs=[example.input for example in positive_examples],
            sample_weights=[example.weight for example in positive_examples],
            metadata={"contrastive_side": "positive"},
            **common,
        )
        negative_request = CaptureRequest(
            inputs=[example.input for example in negative_examples],
            sample_weights=[example.weight for example in negative_examples],
            metadata={"contrastive_side": "negative"},
            **common,
        )
        return (
            capture_activations(model, positive_request, store=store),
            capture_activations(model, negative_request, store=store),
        )

    def _resolved_apply_chat_template(
        self, data: ContrastivePairs | None = None
    ) -> bool | None:
        """Record the effective rendering choice instead of an ambiguous auto flag."""

        if self.apply_chat_template is not None:
            return self.apply_chat_template
        if self.system_prompt is not None or self.chat_template_kwargs is not None:
            return True
        if data is not None:
            return data.is_chat
        return None

    def _capture_config(self, data: ContrastivePairs | None = None) -> dict[str, Any]:
        return {
            "site": _description(self.site),
            "positions": _description(self.positions),
            "pooling": _description(self.pooling),
            "normalize": self.normalize,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "dtype": (
                str(self.capture_dtype).removeprefix("torch.")
                if self.capture_dtype is not None
                else None
            ),
            "rendering": {
                "apply_chat_template": self._resolved_apply_chat_template(data),
                "requested_apply_chat_template": self.apply_chat_template,
                "add_generation_prompt": self.add_generation_prompt,
                "system_prompt": self.system_prompt,
                "chat_template_kwargs": _description(self.chat_template_kwargs),
                "special_tokens": _description(self.special_tokens),
            },
        }


def activation_matrix(batch: ActivationBatch, *, name: str = "activations") -> Tensor:
    """Return a detached [samples, hidden] float32 matrix."""

    matrix = batch.activations
    if matrix.ndim == 3 and matrix.shape[1] == 1:
        matrix = matrix[:, 0]
    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must be [samples, hidden] after pooling; "
            f"got {tuple(matrix.shape)}. "
            "Choose mean/last/max pooling when capturing multiple positions."
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not matrix.is_floating_point():
        matrix = matrix.float()
    return matrix.detach().to(dtype=torch.float32)


def sample_weights(batch: ActivationBatch, matrix: Tensor) -> Tensor:
    if batch.sample_weights is None:
        return torch.ones(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    weights = torch.as_tensor(
        batch.sample_weights, dtype=matrix.dtype, device=matrix.device
    ).flatten()
    if weights.numel() != matrix.shape[0]:
        raise ValueError("sample weights do not match captured activations")
    return weights


def weighted_mean(matrix: Tensor, weights: Tensor | None = None) -> Tensor:
    if matrix.ndim != 2:
        raise ValueError("weighted_mean expects a [samples, features] matrix")
    if weights is None:
        return matrix.mean(dim=0)
    weights = torch.as_tensor(
        weights, dtype=matrix.dtype, device=matrix.device
    ).flatten()
    if weights.numel() != matrix.shape[0]:
        raise ValueError("weights must have one value per sample")
    total = weights.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0:
        raise ValueError("sample weights must have a positive finite sum")
    return (matrix * weights[:, None]).sum(dim=0) / total


def normalize_vector(vector: Tensor, normalize: str | None) -> Tensor:
    normalized = (
        "none" if normalize is None else str(normalize).lower().replace("-", "_")
    )
    if normalized in {"none", "identity", "false", "raw"}:
        return vector
    if normalized in {"l2", "unit", "unit_l2"}:
        norm = torch.linalg.vector_norm(vector)
        if not bool(torch.isfinite(norm)):
            raise ValueError("cannot normalize a non-finite direction")
        if float(norm) == 0.0:
            return vector
        return cast(Tensor, vector / norm)
    raise ValueError(f"unsupported direction normalization: {normalize!r}")


def canonicalize_component_signs(basis: Tensor) -> Tensor:
    """Resolve arbitrary SVD signs by making each largest-magnitude entry positive."""

    if basis.ndim != 2:
        raise ValueError("basis must be a matrix")
    result = basis.clone()
    for index in range(result.shape[0]):
        component = result[index]
        pivot = component[component.abs().argmax()]
        if float(pivot) < 0:
            result[index] = -component
    return result


def artifact_metadata(
    *,
    learner: ContrastiveLearner,
    model: Any,
    data: ContrastivePairs,
    artifact_type: str,
    hidden_size: int,
    extra_config: Mapping[str, Any] | None = None,
) -> Any:
    """Construct core ArtifactMetadata while remaining tolerant during prototyping."""

    from repsteer.artifacts import ArtifactMetadata

    model_config = getattr(model, "config", None)
    model_id = _first(
        getattr(model, "model_id", None),
        getattr(model, "name_or_path", None),
        getattr(model_config, "_name_or_path", None),
        type(model).__qualname__,
    )
    revision = _first(
        getattr(model, "revision", None), getattr(model_config, "_commit_hash", None)
    )
    architecture = _first(
        getattr(model, "architecture", None),
        getattr(model_config, "model_type", None),
        type(model).__qualname__,
    )
    config = learner._capture_config(data)
    config.update(dict(extra_config or {}))
    provenance = {
        "capture": learner._capture_config(data),
        "dataset": {
            "type": type(data).__qualname__,
            "paired": data.paired,
            "positive_samples": len(data.positives),
            "negative_samples": len(data.negatives),
            "fingerprint": str(data.fingerprint),
        },
        "environment": {
            "python": platform.python_version(),
            "dependencies": _dependency_versions(),
        },
    }
    kwargs: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": artifact_type,
        "model_id": str(model_id),
        "model_revision": None if revision is None else str(revision),
        "site": learner.site,
        "hidden_size": int(hidden_size),
        "method": learner.method,
        "config": config,
        "architecture": str(architecture),
        "tokenizer": _tokenizer_metadata(model, str(model_id), revision),
        "processor": _processor_metadata(model),
        "modality": _modality_metadata(model, learner.site),
        "dataset_fingerprint": str(data.fingerprint),
        "dtype": "float32",
        "normalization": learner.normalize,
        "seed": learner.seed,
        "library": {"name": "repsteer", "version": __version__},
        "provenance": provenance,
    }
    return ArtifactMetadata(**kwargs)


def _processor_metadata(model: Any) -> dict[str, Any]:
    from repsteer.artifacts.processor import processor_metadata

    return processor_metadata(model)


def _modality_metadata(model: Any, site: Any) -> dict[str, Any]:
    if getattr(model, "processor", None) is None:
        return {}
    return {
        "kind": "image",
        "stream": getattr(site, "stream", None),
        "component": getattr(site, "component", None),
        "modality_map_schema": "1.0",
    }


def _tokenizer_metadata(
    model: Any, model_id: str, model_revision: Any
) -> dict[str, Any]:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return {}
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(init_kwargs, Mapping):
        init_kwargs = {}
    tokenizer_id = _first(
        getattr(tokenizer, "name_or_path", None),
        init_kwargs.get("name_or_path"),
        init_kwargs.get("pretrained_model_name_or_path"),
        model_id,
    )
    tokenizer_revision = _first(
        getattr(tokenizer, "_commit_hash", None),
        init_kwargs.get("_commit_hash"),
        init_kwargs.get("revision"),
        model_revision,
    )
    result: dict[str, Any] = {
        "id": str(tokenizer_id),
        "revision": None if tokenizer_revision is None else str(tokenizer_revision),
    }
    # Text models usually own the template on the tokenizer, while some VLM
    # processors own a distinct multimodal template.  Record the renderer that
    # would actually be selected as far as metadata makes that observable.
    processor = getattr(model, "processor", None)
    chat_template = _first(
        getattr(tokenizer, "chat_template", None),
        getattr(processor, "chat_template", None),
    )
    if chat_template is not None:
        result["chat_template_sha256"] = str(
            stable_fingerprint(chat_template)
        ).removeprefix("sha256:")
    return result


def _dependency_versions() -> dict[str, str]:
    dependencies: dict[str, str] = {}
    for distribution in ("torch", "numpy", "safetensors", "transformers"):
        try:
            dependencies[distribution] = version(distribution)
        except PackageNotFoundError:
            continue
    return dependencies


def _description(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except (TypeError, ValueError):
            pass
    return canonicalize(value)


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


__all__ = [
    "ContrastiveLearner",
    "Learner",
    "activation_matrix",
    "artifact_metadata",
    "canonicalize_component_signs",
    "normalize_vector",
    "sample_weights",
    "weighted_mean",
]
