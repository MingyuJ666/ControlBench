"""Explicit artifact compatibility levels; exact revision is the default."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from repsteer.core.errors import ArtifactCompatibilityError
from repsteer.core.site import Site
from repsteer.data import stable_fingerprint

from .base import ArtifactMetadata, SteeringArtifact
from .processor import processor_metadata


class CompatibilityLevel(str, Enum):
    EXACT = "exact"
    ARCHITECTURE_COMPATIBLE = "architecture"
    DIMENSION_ONLY = "dimension"
    INCOMPATIBLE = "incompatible"

    @classmethod
    def parse(cls, value: CompatibilityLevel | str) -> CompatibilityLevel:
        if isinstance(value, cls):
            return value
        aliases = {
            "exact": cls.EXACT,
            "architecture": cls.ARCHITECTURE_COMPATIBLE,
            "architecture_compatible": cls.ARCHITECTURE_COMPATIBLE,
            "dimension": cls.DIMENSION_ONLY,
            "dimension_only": cls.DIMENSION_ONLY,
            "incompatible": cls.INCOMPATIBLE,
        }
        try:
            return aliases[str(value).lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown compatibility level {value!r}") from exc


_RANK = {
    CompatibilityLevel.INCOMPATIBLE: 0,
    CompatibilityLevel.DIMENSION_ONLY: 1,
    CompatibilityLevel.ARCHITECTURE_COMPATIBLE: 2,
    CompatibilityLevel.EXACT: 3,
}


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    level: CompatibilityLevel
    reasons: tuple[str, ...]
    artifact_model: str
    target_model: str
    artifact_hidden_size: int
    target_hidden_size: int | None
    artifact_site: Site | None
    target_site: Site | None
    artifact_processor: str | None = None
    target_processor: str | None = None

    @property
    def compatible(self) -> bool:
        return self.level is not CompatibilityLevel.INCOMPATIBLE

    def allows(
        self, minimum: CompatibilityLevel | str = CompatibilityLevel.EXACT
    ) -> bool:
        return _RANK[self.level] >= _RANK[CompatibilityLevel.parse(minimum)]

    def explain(self) -> str:
        reasons = "; ".join(self.reasons) if self.reasons else "all checks passed"
        return f"{self.level.value}: {reasons}"


def _get(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            result = getattr(value, name)
            if result is not None:
                return result
    return None


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _tokenizer_template_hash(target: Any) -> str | None:
    """Return the stable hash of the template used by an HF wrapper.

    Text models usually expose it on ``tokenizer``.  Some VLM processors own a
    distinct template, so inspect both the processor and its tokenizer as a
    compatibility fallback.  Absence remains compatible with legacy artifacts
    that predate template provenance.
    """

    tokenizer = _get(target, "tokenizer")
    processor = _get(target, "processor")
    processor_tokenizer = _get(processor, "tokenizer")
    chat_template = _first(
        _get(tokenizer, "chat_template"),
        _get(processor, "chat_template"),
        _get(processor_tokenizer, "chat_template"),
    )
    if chat_template is None:
        return None
    return str(stable_fingerprint(chat_template)).removeprefix("sha256:")


def _target_attributes(target: Any) -> tuple[str, str | None, str | None, int | None]:
    model_id = _get(target, "model_id", "id", "name_or_path")
    revision = _get(target, "revision", "model_revision")
    architecture = _get(target, "architecture", "model_type")
    hidden_size = _get(target, "hidden_size", "d_model", "n_embd")

    config = _get(target, "config")
    if config is None:
        inner = _get(target, "model")
        config = _get(inner, "config") if inner is not None else None
    if config is not None:
        model_id = model_id or _get(config, "_name_or_path", "name_or_path")
        architectures = _get(config, "architectures")
        architecture = architecture or (
            architectures[0]
            if isinstance(architectures, list | tuple) and architectures
            else None
        )
        architecture = architecture or _get(config, "model_type")
        hidden_size = hidden_size or _get(config, "hidden_size", "d_model", "n_embd")
        revision = revision or _get(config, "_commit_hash", "revision")
    return (
        str(model_id or ""),
        None if revision is None else str(revision),
        None if architecture is None else str(architecture),
        None if hidden_size is None else int(hidden_size),
    )


def _same_site_mapping(left: Site | None, right: Site | None) -> bool:
    if left is None or right is None:
        return True
    return (
        left.stream == right.stream
        and left.component == right.component
        and left.layer == right.layer
        and left.io == right.io
        and left.unit == right.unit
    )


def _processor_attributes(
    target: Any,
) -> tuple[str | None, str | None, str | None]:
    values = processor_metadata(target)
    processor_id = values.get("id")
    processor_revision = values.get("revision")
    preprocess_fingerprint = values.get("preprocess_fingerprint")
    return (
        None if processor_id is None else str(processor_id),
        None if processor_revision is None else str(processor_revision),
        (None if preprocess_fingerprint is None else str(preprocess_fingerprint)),
    )


def check_compatibility(
    artifact: SteeringArtifact | ArtifactMetadata,
    target: Any,
    *,
    site: Site | None = None,
    hidden_size: int | None = None,
) -> CompatibilityResult:
    metadata = artifact if isinstance(artifact, ArtifactMetadata) else artifact.metadata
    target_id, target_revision, target_architecture, target_hidden = _target_attributes(
        target
    )
    (
        target_processor_id,
        target_processor_revision,
        target_preprocess_fingerprint,
    ) = _processor_attributes(target)
    target_template_hash = _tokenizer_template_hash(target)
    if hidden_size is not None:
        target_hidden = int(hidden_size)
    target_site = site or _get(target, "site")
    if target_site is not None and not isinstance(target_site, Site):
        if isinstance(target_site, Mapping):
            target_site = Site.from_dict(target_site)
        else:
            target_site = None

    reasons: list[str] = []
    dimensions_match = (
        target_hidden is not None and metadata.hidden_size == target_hidden
    )
    if target_hidden is None:
        reasons.append("target hidden size is unknown")
    elif not dimensions_match:
        reasons.append(
            f"hidden size differs ({metadata.hidden_size} != {target_hidden})"
        )
    sites_match = _same_site_mapping(metadata.site, target_site)
    if not sites_match:
        reasons.append(f"site mapping differs ({metadata.site} != {target_site})")

    exact_id = bool(metadata.model_id and target_id and metadata.model_id == target_id)
    exact_revision = bool(
        metadata.model_revision
        and target_revision
        and metadata.model_revision == target_revision
    )
    if not exact_id:
        reasons.append(f"model id differs ({metadata.model_id!r} != {target_id!r})")
    if not metadata.model_revision or not target_revision:
        reasons.append(
            "model revision is unknown; exact compatibility requires explicit revisions"
        )
    elif not exact_revision:
        reasons.append(
            "model revision differs "
            f"({metadata.model_revision!r} != {target_revision!r})"
        )
    architecture_match = bool(
        metadata.architecture
        and target_architecture
        and metadata.architecture == target_architecture
    )
    architecture_exact_or_unknown = (
        not (metadata.architecture and target_architecture) or architecture_match
    )
    if metadata.architecture and target_architecture and not architecture_match:
        reasons.append(
            "architecture differs "
            f"({metadata.architecture!r} != {target_architecture!r})"
        )

    artifact_processor_id = metadata.processor.get("id")
    artifact_processor_revision = metadata.processor.get("revision")
    artifact_preprocess_fingerprint = metadata.processor.get("preprocess_fingerprint")
    processor_match = True
    if (
        artifact_processor_id
        or artifact_processor_revision
        or artifact_preprocess_fingerprint
    ):
        processor_match = bool(
            artifact_processor_id
            and target_processor_id
            and str(artifact_processor_id) == target_processor_id
            and artifact_processor_revision
            and target_processor_revision
            and str(artifact_processor_revision) == target_processor_revision
        )
        if artifact_preprocess_fingerprint:
            processor_match = bool(
                processor_match
                and target_preprocess_fingerprint
                and str(artifact_preprocess_fingerprint)
                == target_preprocess_fingerprint
            )
        if not processor_match:
            reasons.append(
                "processor identity differs or preprocessing differs "
                f"({artifact_processor_id!r}@{artifact_processor_revision!r} != "
                f"{target_processor_id!r}@{target_processor_revision!r}; "
                f"preprocess {artifact_preprocess_fingerprint!r} != "
                f"{target_preprocess_fingerprint!r})"
            )

    artifact_template_hash = metadata.tokenizer.get("chat_template_sha256")
    tokenizer_template_match = True
    if artifact_template_hash:
        tokenizer_template_match = bool(
            target_template_hash
            and str(artifact_template_hash).removeprefix("sha256:")
            == target_template_hash
        )
        if not tokenizer_template_match:
            reasons.append(
                "tokenizer chat template differs "
                f"({artifact_template_hash!r} != {target_template_hash!r})"
            )

    if (
        dimensions_match
        and sites_match
        and exact_id
        and exact_revision
        and architecture_exact_or_unknown
        and processor_match
        and tokenizer_template_match
    ):
        level = CompatibilityLevel.EXACT
    elif dimensions_match and sites_match and architecture_match:
        level = CompatibilityLevel.ARCHITECTURE_COMPATIBLE
    elif dimensions_match:
        level = CompatibilityLevel.DIMENSION_ONLY
    else:
        level = CompatibilityLevel.INCOMPATIBLE
    return CompatibilityResult(
        level=level,
        reasons=tuple(reasons),
        artifact_model=f"{metadata.model_id}@{metadata.model_revision or '<unknown>'}",
        target_model=f"{target_id}@{target_revision or '<unknown>'}",
        artifact_hidden_size=metadata.hidden_size,
        target_hidden_size=target_hidden,
        artifact_site=metadata.site,
        target_site=target_site,
        artifact_processor=(
            None
            if not (artifact_processor_id or artifact_processor_revision)
            else f"{artifact_processor_id or '<unknown>'}"
            f"@{artifact_processor_revision or '<unknown>'}"
        ),
        target_processor=(
            None
            if not (target_processor_id or target_processor_revision)
            else f"{target_processor_id or '<unknown>'}"
            f"@{target_processor_revision or '<unknown>'}"
        ),
    )


def assert_compatible(
    artifact: SteeringArtifact | ArtifactMetadata,
    target: Any,
    *,
    compatibility: CompatibilityLevel | str = CompatibilityLevel.EXACT,
    site: Site | None = None,
    hidden_size: int | None = None,
) -> CompatibilityResult:
    required = CompatibilityLevel.parse(compatibility)
    result = check_compatibility(artifact, target, site=site, hidden_size=hidden_size)
    if not result.allows(required):
        raise ArtifactCompatibilityError(
            "Artifact compatibility check failed\n"
            f"  required: {required.value}\n"
            f"  actual: {result.level.value}\n"
            f"  artifact hidden_size: {result.artifact_hidden_size}\n"
            f"  target hidden_size: {result.target_hidden_size}\n"
            f"  artifact model: {result.artifact_model}\n"
            f"  target model: {result.target_model}\n"
            f"  artifact site: {result.artifact_site}\n"
            f"  target site: {result.target_site}\n"
            f"  artifact processor: {result.artifact_processor}\n"
            f"  target processor: {result.target_processor}\n"
            f"  details: {'; '.join(result.reasons)}\n"
            "  suggestion: relearn the artifact or explicitly request a weaker "
            "compatibility level after validating the target"
        )
    return result


__all__ = [
    "CompatibilityLevel",
    "CompatibilityResult",
    "assert_compatible",
    "check_compatibility",
]
