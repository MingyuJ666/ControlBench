"""Immutable, model-independent steering artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, TypeAlias, TypeVar, cast

import torch
from torch import Tensor

from repsteer._version import __version__
from repsteer.core.serialization import json_safe
from repsteer.core.site import Site


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Portable provenance and compatibility metadata.

    Defaults make hand-created research artifacts convenient, while learned
    artifacts should fill model revision, dataset fingerprint and seed.
    """

    schema_version: str = "1.0"
    artifact_type: str = ""
    model_id: str = ""
    model_revision: str | None = None
    site: Site | None = None
    hidden_size: int = 0
    method: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    architecture: str | None = None
    tokenizer: Mapping[str, Any] = field(default_factory=dict)
    processor: Mapping[str, Any] = field(default_factory=dict)
    modality: Mapping[str, Any] = field(default_factory=dict)
    dataset_fingerprint: str | None = None
    dtype: str | None = None
    normalization: str | None = None
    seed: int | None = None
    library: Mapping[str, Any] = field(
        default_factory=lambda: {"name": "repsteer", "version": __version__}
    )
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("ArtifactMetadata.schema_version cannot be empty")
        if self.hidden_size < 0:
            raise ValueError("ArtifactMetadata.hidden_size cannot be negative")
        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError("ArtifactMetadata.seed must be an int or None")
        object.__setattr__(self, "config", _freeze(self.config))
        object.__setattr__(self, "tokenizer", _freeze(self.tokenizer))
        object.__setattr__(self, "processor", _freeze(self.processor))
        object.__setattr__(self, "modality", _freeze(self.modality))
        object.__setattr__(self, "library", _freeze(self.library))
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        if self.dtype is not None:
            object.__setattr__(self, "dtype", str(self.dtype).removeprefix("torch."))

    @property
    def model(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "id": self.model_id,
                "revision": self.model_revision,
                "architecture": self.architecture,
                "hidden_size": self.hidden_size,
            }
        )

    def with_updates(self, **changes: Any) -> ArtifactMetadata:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat JSON-safe representation useful in Python APIs."""

        return cast(
            dict[str, Any],
            json_safe(
                {
                    "schema_version": self.schema_version,
                    "artifact_type": self.artifact_type,
                    "model_id": self.model_id,
                    "model_revision": self.model_revision,
                    "site": self.site.to_dict() if self.site is not None else None,
                    "hidden_size": self.hidden_size,
                    "method": self.method,
                    "config": self.config,
                    "architecture": self.architecture,
                    "tokenizer": self.tokenizer,
                    "processor": self.processor,
                    "modality": self.modality,
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "dtype": self.dtype,
                    "normalization": self.normalization,
                    "seed": self.seed,
                    "library": self.library,
                    "provenance": self.provenance,
                }
            ),
        )

    def to_manifest(self) -> dict[str, Any]:
        """Return the RFC bundle manifest layout."""

        return cast(
            dict[str, Any],
            json_safe(
                {
                    "schema_version": self.schema_version,
                    "artifact_type": self.artifact_type,
                    "method": self.method,
                    "model": {
                        "id": self.model_id,
                        "revision": self.model_revision,
                        "architecture": self.architecture,
                        "hidden_size": self.hidden_size,
                    },
                    "tokenizer": self.tokenizer,
                    "processor": self.processor,
                    "modality": self.modality,
                    "site": self.site.to_dict() if self.site is not None else None,
                    "normalization": self.normalization,
                    "dtype": self.dtype,
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "seed": self.seed,
                    "config": self.config,
                    "library": self.library,
                    "provenance": self.provenance,
                }
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactMetadata:
        model = value.get("model")
        if isinstance(model, Mapping):
            model_id = model.get("id", "")
            revision = model.get("revision")
            architecture = model.get("architecture")
            hidden_size = model.get("hidden_size", 0)
        else:
            model_id = value.get("model_id", "")
            revision = value.get("model_revision")
            architecture = value.get("architecture")
            hidden_size = value.get("hidden_size", 0)
        site_value = value.get("site")
        site = (
            Site.from_dict(site_value)
            if isinstance(site_value, Mapping)
            else site_value
            if isinstance(site_value, Site)
            else None
        )
        return cls(
            schema_version=str(value.get("schema_version", "1.0")),
            artifact_type=str(value.get("artifact_type", "")),
            model_id=str(model_id or ""),
            model_revision=None if revision is None else str(revision),
            site=site,
            hidden_size=int(hidden_size or 0),
            method=str(value.get("method", "")),
            config=value.get("config", {}),
            architecture=None if architecture is None else str(architecture),
            tokenizer=value.get("tokenizer", {}),
            processor=value.get("processor", {}),
            modality=value.get("modality", {}),
            dataset_fingerprint=value.get("dataset_fingerprint"),
            dtype=value.get("dtype"),
            normalization=value.get("normalization"),
            seed=value.get("seed"),
            library=value.get("library", {"name": "repsteer", "version": __version__}),
            provenance=value.get("provenance", {}),
        )


ArtifactT = TypeVar("ArtifactT", bound="SteeringArtifact")


class SteeringArtifact(ABC):
    """Base class for immutable tensor artifacts (never executable code)."""

    metadata: ArtifactMetadata

    @abstractmethod
    def tensors(self) -> Mapping[str, Tensor]:
        """Return named tensors to persist in a safetensors bundle."""

    def save(self, path: str) -> None:
        from .io import save_artifact

        save_artifact(self, path)

    @classmethod
    def load(cls: type[ArtifactT], path: str, **kwargs: Any) -> ArtifactT:
        from .io import load_artifact

        artifact = load_artifact(path, **kwargs)
        if cls is not SteeringArtifact and not isinstance(artifact, cls):
            raise TypeError(
                f"Bundle contains {type(artifact).__name__}, not {cls.__name__}"
            )
        return artifact  # type: ignore[return-value]

    def bind(self: ArtifactT, model: Any, *, compatibility: str = "exact") -> ArtifactT:
        """Validate a target model and return this immutable artifact."""

        from .compatibility import assert_compatible

        compatibility_site: Site | None = None
        compatibility_hidden_size: int | None = None
        resolve_site = getattr(model, "resolve_site", None)
        if self.metadata.site is not None and callable(resolve_site):
            # Resolving first is deliberate: a matching model identity and global
            # hidden size must never let an invalid layer/component bind.  It also
            # makes compatibility use the dimension of the concrete hook site,
            # which need not equal a model's default hidden size.
            resolved = resolve_site(self.metadata.site)
            compatibility_site = getattr(resolved, "site", None)
            compatibility_hidden_size = getattr(resolved, "hidden_dim", None)
            if not isinstance(compatibility_site, Site):
                raise TypeError(
                    "model.resolve_site() must return an object whose 'site' is a Site"
                )
            if (
                not isinstance(compatibility_hidden_size, int)
                or isinstance(compatibility_hidden_size, bool)
                or compatibility_hidden_size <= 0
            ):
                raise TypeError(
                    "model.resolve_site() must return an object whose 'hidden_dim' "
                    "is a positive int"
                )

        assert_compatible(
            self,
            model,
            compatibility=compatibility,
            site=compatibility_site,
            hidden_size=compatibility_hidden_size,
        )
        return self

    def fingerprint(self) -> str:
        """Content fingerprint independent of a bundle's filesystem layout."""

        digest = hashlib.sha256()
        metadata = json.dumps(
            self.metadata.to_manifest(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(metadata)
        for name, tensor in sorted(self.tensors().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            # Flatten first because PyTorch cannot reinterpret a scalar tensor
            # as bytes while changing element size.
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return f"sha256:{digest.hexdigest()}"

    def check_compatibility(self, target: Any, *, site: Site | None = None) -> Any:
        from .compatibility import check_compatibility

        return check_compatibility(self, target, site=site)


def _coerce_tensor(value: Any, *, name: str, ndim: tuple[int, ...]) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim not in ndim:
        expected = " or ".join(str(item) for item in ndim)
        raise ValueError(f"{name} must have {expected} dimensions, got {tensor.ndim}")
    if not (tensor.is_floating_point() or tensor.is_complex()):
        tensor = tensor.to(dtype=torch.get_default_dtype())
    return tensor.detach().clone().contiguous()


def _metadata_for_tensor(
    metadata: ArtifactMetadata,
    *,
    artifact_type: str,
    hidden_size: int,
    dtype: torch.dtype,
) -> ArtifactMetadata:
    if metadata.artifact_type and metadata.artifact_type != artifact_type:
        raise ValueError(
            f"{artifact_type} artifact received metadata for {metadata.artifact_type!r}"
        )
    if metadata.hidden_size not in (0, hidden_size):
        raise ValueError(
            f"metadata hidden_size {metadata.hidden_size} does not match tensor "
            f"hidden size {hidden_size}"
        )
    tensor_dtype = _dtype_name(dtype)
    metadata_dtype = (
        None if metadata.dtype is None else str(metadata.dtype).removeprefix("torch.")
    )
    if metadata_dtype is not None and metadata_dtype != tensor_dtype:
        raise ValueError(
            f"metadata dtype {metadata_dtype!r} does not match tensor dtype "
            f"{tensor_dtype!r}"
        )
    return metadata.with_updates(
        artifact_type=artifact_type,
        hidden_size=hidden_size,
        dtype=metadata_dtype or tensor_dtype,
    )


def _coerce_tensor_like(
    value: Any,
    *,
    name: str,
    ndim: tuple[int, ...],
    reference: Tensor,
) -> Tensor:
    """Normalize an attached tensor to its artifact's primary tensor.

    Artifact manifests expose one dtype and bundles should remain portable as a
    coherent unit.  Normalizing secondary statistics/parameters here avoids
    device-mismatch failures and prevents the manifest dtype from describing
    only an arbitrary subset of the stored tensors.
    """

    tensor = _coerce_tensor(value, name=name, ndim=ndim)
    if tensor.is_complex() and not reference.is_complex():
        raise ValueError(f"{name} is complex but the artifact's primary tensor is real")
    return tensor.to(device=reference.device, dtype=reference.dtype).contiguous()


@dataclass(frozen=True, slots=True)
class DirectionArtifact(SteeringArtifact):
    metadata: ArtifactMetadata
    direction: Tensor

    def __post_init__(self) -> None:
        direction = _coerce_tensor(self.direction, name="direction", ndim=(1,))
        metadata = _metadata_for_tensor(
            self.metadata,
            artifact_type="direction",
            hidden_size=int(direction.shape[-1]),
            dtype=direction.dtype,
        )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "metadata", metadata)

    @property
    def vector(self) -> Tensor:
        return self.direction

    def tensors(self) -> Mapping[str, Tensor]:
        return MappingProxyType({"direction": self.direction})

    def normalized(self, eps: float = 1e-12) -> DirectionArtifact:
        norm = self.direction.norm().clamp_min(eps)
        metadata = self.metadata.with_updates(normalization="l2")
        return DirectionArtifact(metadata, self.direction / norm)

    def with_metadata(self, metadata: ArtifactMetadata) -> DirectionArtifact:
        return DirectionArtifact(metadata, self.direction)


@dataclass(frozen=True, slots=True)
class SubspaceArtifact(SteeringArtifact):
    metadata: ArtifactMetadata
    basis: Tensor
    mean: Tensor | None = None
    explained_variance: Tensor | None = None

    def __post_init__(self) -> None:
        basis = _coerce_tensor(self.basis, name="basis", ndim=(2,))
        metadata = _metadata_for_tensor(
            self.metadata,
            artifact_type="subspace",
            hidden_size=int(basis.shape[-1]),
            dtype=basis.dtype,
        )
        mean = None
        if self.mean is not None:
            mean = _coerce_tensor_like(
                self.mean,
                name="mean",
                ndim=(1,),
                reference=basis,
            )
            if mean.shape[0] != basis.shape[-1]:
                raise ValueError("mean length must equal the subspace hidden size")
        explained = None
        if self.explained_variance is not None:
            explained = _coerce_tensor_like(
                self.explained_variance,
                name="explained_variance",
                ndim=(1,),
                reference=basis,
            )
            if explained.shape[0] != basis.shape[0]:
                raise ValueError(
                    "explained_variance length must equal the number of basis vectors"
                )
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "explained_variance", explained)
        object.__setattr__(self, "metadata", metadata)

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])

    def tensors(self) -> Mapping[str, Tensor]:
        values: dict[str, Tensor] = {"basis": self.basis}
        if self.mean is not None:
            values["mean"] = self.mean
        if self.explained_variance is not None:
            values["explained_variance"] = self.explained_variance
        return MappingProxyType(values)

    def with_metadata(self, metadata: ArtifactMetadata) -> SubspaceArtifact:
        return SubspaceArtifact(
            metadata, self.basis, self.mean, self.explained_variance
        )


@dataclass(frozen=True, slots=True)
class ProbeArtifact(SteeringArtifact):
    metadata: ArtifactMetadata
    weight: Tensor
    bias: Tensor | float | None = None

    def __post_init__(self) -> None:
        weight = _coerce_tensor(self.weight, name="weight", ndim=(1, 2))
        metadata = _metadata_for_tensor(
            self.metadata,
            artifact_type="probe",
            hidden_size=int(weight.shape[-1]),
            dtype=weight.dtype,
        )
        if self.bias is None:
            bias = torch.zeros((), dtype=weight.dtype, device=weight.device)
        else:
            bias = _coerce_tensor_like(
                self.bias,
                name="bias",
                ndim=(0, 1),
                reference=weight,
            )
        if weight.ndim == 1 and bias.numel() != 1:
            raise ValueError("A binary probe's bias must be scalar")
        if weight.ndim == 2 and bias.numel() not in (1, weight.shape[0]):
            raise ValueError("Probe bias must be scalar or have one value per output")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "metadata", metadata)

    @property
    def weights(self) -> Tensor:
        """Compatibility alias for integrations that use plural naming."""

        return self.weight

    def tensors(self) -> Mapping[str, Tensor]:
        bias = cast(Tensor, self.bias)
        return MappingProxyType({"weight": self.weight, "bias": bias})

    def logits(self, activation: Tensor) -> Tensor:
        weight = self.weight.to(device=activation.device, dtype=activation.dtype)
        bias = cast(Tensor, self.bias).to(
            device=activation.device, dtype=activation.dtype
        )
        if weight.ndim == 1:
            return torch.einsum("...d,d->...", activation, weight) + bias
        return torch.einsum("...d,od->...o", activation, weight) + bias

    def probabilities(self, activation: Tensor) -> Tensor:
        logits = self.logits(activation)
        return logits.sigmoid() if self.weight.ndim == 1 else logits.softmax(dim=-1)

    def with_metadata(self, metadata: ArtifactMetadata) -> ProbeArtifact:
        return ProbeArtifact(metadata, self.weight, self.bias)


@dataclass(frozen=True, slots=True)
class SAEFeatureArtifact(SteeringArtifact):
    """Portable identity and decoder direction for one SAE feature.

    The SAE module itself is deliberately not retained: artifacts remain safe
    tensor + JSON bundles, while runtime encoding is supplied by an
    :class:`repsteer.sae.SAEAdapter`.
    """

    metadata: ArtifactMetadata
    feature_id: int
    decoder_direction: Tensor
    score: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.feature_id, bool) or not isinstance(self.feature_id, int):
            raise TypeError("SAEFeatureArtifact.feature_id must be an int")
        if self.feature_id < 0:
            raise ValueError("SAEFeatureArtifact.feature_id cannot be negative")
        direction = _coerce_tensor(
            self.decoder_direction,
            name="decoder_direction",
            ndim=(1,),
        )
        config = dict(self.metadata.config)
        configured_id = config.get("feature_id")
        if configured_id is not None and int(configured_id) != self.feature_id:
            raise ValueError(
                "metadata feature_id does not match SAEFeatureArtifact.feature_id"
            )
        configured_score = config.get("selection_score")
        score = self.score
        if score is None and configured_score is not None:
            score = float(configured_score)
        if score is not None:
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("SAE feature selection score must be finite")
        config["feature_id"] = self.feature_id
        if score is not None:
            config["selection_score"] = score
        metadata = self.metadata.with_updates(config=config)
        metadata = _metadata_for_tensor(
            metadata,
            artifact_type="sae_feature",
            hidden_size=int(direction.shape[-1]),
            dtype=direction.dtype,
        )
        object.__setattr__(self, "decoder_direction", direction)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "score", score)

    @property
    def direction(self) -> Tensor:
        """Make decoder-direction add work with the ordinary ``Add`` operator."""

        return self.decoder_direction

    @property
    def vector(self) -> Tensor:
        return self.decoder_direction

    def tensors(self) -> Mapping[str, Tensor]:
        return MappingProxyType({"decoder_direction": self.decoder_direction})

    def as_direction_artifact(self) -> DirectionArtifact:
        """Return a plain direction view while preserving provenance."""

        metadata = self.metadata.with_updates(artifact_type="direction")
        return DirectionArtifact(metadata, self.decoder_direction)

    def with_metadata(self, metadata: ArtifactMetadata) -> SAEFeatureArtifact:
        return SAEFeatureArtifact(
            metadata,
            self.feature_id,
            self.decoder_direction,
            self.score,
        )


# All concrete public artifacts inherit this base. Keeping the alias broad also
# lets artifact I/O return newer, independently-defined artifact subclasses.
Artifact: TypeAlias = SteeringArtifact


__all__ = [
    "Artifact",
    "ArtifactMetadata",
    "DirectionArtifact",
    "ProbeArtifact",
    "SAEFeatureArtifact",
    "SteeringArtifact",
    "SubspaceArtifact",
]
