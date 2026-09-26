"""Safe, role-keyed collections of independently valid steering artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from repsteer._version import __version__
from repsteer.core.errors import (
    ArtifactCompatibilityError,
    ArtifactFormatError,
    UnsafeArtifactError,
)
from repsteer.core.site import Site
from repsteer.selection import SelectionReport

from .base import (
    DirectionArtifact,
    ProbeArtifact,
    SAEFeatureArtifact,
    SteeringArtifact,
    SubspaceArtifact,
)
from .compatibility import (
    CompatibilityLevel,
    CompatibilityResult,
)
from .compatibility import (
    check_compatibility as check_artifact_compatibility,
)
from .io import (
    CHECKSUMS_FILENAME,
    MANIFEST_FILENAME,
    TENSORS_FILENAME,
    _read_json,
    _sha256,
    _write_json,
    load_artifact,
    save_artifact,
)
from .iti import ITIArtifact

_BUNDLE_SCHEMA_VERSION = "1.0"
_BUNDLE_KIND = "artifact_bundle"
_COMPONENTS_DIRECTORY = "components"
_SUPPORTED_COMPONENT_TYPES = frozenset(
    {"direction", "subspace", "probe", "sae_feature", "iti"}
)
_SUPPORTED_COMPONENT_CLASSES = (
    DirectionArtifact,
    SubspaceArtifact,
    ProbeArtifact,
    SAEFeatureArtifact,
    ITIArtifact,
)
_COMPATIBILITY_RANK = {
    CompatibilityLevel.INCOMPATIBLE: 0,
    CompatibilityLevel.DIMENSION_ONLY: 1,
    CompatibilityLevel.ARCHITECTURE_COMPATIBLE: 2,
    CompatibilityLevel.EXACT: 3,
}


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _freeze_json(value: Any, *, path: str) -> Any:
    """Validate finite JSON-only metadata and freeze mappings recursively."""

    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite, got {value!r}")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        values: dict[str, Any] = {}
        for key in value:
            if not isinstance(key, str):
                raise TypeError(
                    f"{path} contains non-string JSON key {key!r} "
                    f"({type(key).__name__})"
                )
        for key in sorted(value):
            values[key] = _freeze_json(value[key], path=f"{path}.{key}")
        return MappingProxyType(values)
    if isinstance(value, list | tuple):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only JSON primitives, mappings, or sequences; "
        f"got {type(value).__name__}"
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: Any) -> str:
    return (
        "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    )


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactFormatError(
            f"{name} must be a JSON object, got {type(value).__name__}"
        )
    return value


def _reject_unknown_fields(
    value: Mapping[str, Any], *, name: str, allowed: set[str]
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ArtifactFormatError(
            f"{name} has unsupported fields: {', '.join(unknown)}"
        )


def _component_relative_path(key: str) -> str:
    """Map an arbitrary logical key to one controlled relative directory."""

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"{_COMPONENTS_DIRECTORY}/{digest}"


def _validate_relative_component_path(value: Any, *, key: str, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise UnsafeArtifactError(f"component {key!r} has no valid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise UnsafeArtifactError(
            f"component {key!r} path must be relative and cannot traverse directories: "
            f"{value!r}"
        )
    expected = _component_relative_path(key)
    if value != expected:
        raise UnsafeArtifactError(
            f"component {key!r} path must be the canonical bundle path "
            f"{expected!r}, got {value!r}"
        )
    target = root.joinpath(*path.parts)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise UnsafeArtifactError(
            f"component {key!r} path escapes the bundle directory: {value!r}"
        ) from exc
    return target


def _validate_digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ArtifactFormatError(f"{name} must be a sha256 digest, got {value!r}")
    payload = value.removeprefix("sha256:")
    if len(payload) != 64 or any(
        character not in "0123456789abcdef" for character in payload
    ):
        raise ArtifactFormatError(f"{name} is not a valid sha256 digest: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactBundleComponent:
    """One named component and its semantic role in an :class:`ArtifactBundle`."""

    key: str
    role: str
    artifact: SteeringArtifact
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = _require_nonempty_string(self.key, name="ArtifactBundleComponent.key")
        role = _require_nonempty_string(self.role, name="ArtifactBundleComponent.role")
        if not isinstance(self.artifact, _SUPPORTED_COMPONENT_CLASSES):
            raise TypeError(
                "ArtifactBundle components must be supported repsteer artifacts "
                "(DirectionArtifact, SubspaceArtifact, ProbeArtifact, or "
                "SAEFeatureArtifact, or ITIArtifact), got "
                f"{type(self.artifact).__name__}"
            )
        artifact_type = self.artifact.metadata.artifact_type
        if artifact_type not in _SUPPORTED_COMPONENT_TYPES:
            raise ValueError(
                f"component {key!r} has unsupported artifact type {artifact_type!r}"
            )
        metadata = _freeze_json(self.metadata, path=f"component {key!r} metadata")
        if not isinstance(metadata, Mapping):  # pragma: no cover - helper guard
            raise TypeError("ArtifactBundleComponent.metadata must be a JSON object")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "metadata", metadata)

    @property
    def artifact_type(self) -> str:
        """Return the loader-supported artifact type for this component."""

        return self.artifact.metadata.artifact_type

    @property
    def digest(self) -> str:
        """Return the component artifact's stable content fingerprint."""

        return self.artifact.fingerprint()

    @property
    def component_digest(self) -> str:
        """Alias for :attr:`digest` used by bundle-manifest consumers."""

        return self.digest

    @property
    def relative_path(self) -> str:
        """Return the bundle-controlled directory for this component."""

        return _component_relative_path(self.key)

    def to_manifest_record(self) -> dict[str, Any]:
        """Return the safe manifest record without embedding executable state."""

        return {
            "key": self.key,
            "role": self.role,
            "artifact_type": self.artifact_type,
            "artifact_digest": self.digest,
            "metadata": _json_value(self.metadata),
            "path": self.relative_path,
        }


@dataclass(frozen=True, slots=True)
class BundleCompatibilityResult:
    """Aggregate compatibility diagnostics keyed by bundle component key."""

    level: CompatibilityLevel
    required: CompatibilityLevel
    component_results: Mapping[str, CompatibilityResult]

    def __post_init__(self) -> None:
        if not isinstance(self.level, CompatibilityLevel):
            object.__setattr__(self, "level", CompatibilityLevel.parse(self.level))
        if not isinstance(self.required, CompatibilityLevel):
            object.__setattr__(
                self, "required", CompatibilityLevel.parse(self.required)
            )
        results: dict[str, CompatibilityResult] = {}
        for key in sorted(self.component_results):
            _require_nonempty_string(key, name="component compatibility key")
            result = self.component_results[key]
            if not isinstance(result, CompatibilityResult):
                raise TypeError(
                    "component_results values must be CompatibilityResult, got "
                    f"{type(result).__name__} for component {key!r}"
                )
            results[key] = result
        if not results:
            raise ValueError(
                "BundleCompatibilityResult requires at least one component"
            )
        if self.required is CompatibilityLevel.INCOMPATIBLE:
            raise ValueError(
                "BundleCompatibilityResult.required cannot be incompatible"
            )
        computed_level = _aggregate_level(results.values())
        if self.level is not computed_level:
            raise ValueError(
                "BundleCompatibilityResult.level does not match component results: "
                f"expected {computed_level.value!r}, got {self.level.value!r}"
            )
        object.__setattr__(self, "component_results", MappingProxyType(results))

    @property
    def compatible(self) -> bool:
        """Whether every component meets the bundle's requested minimum level."""

        return all(
            result.allows(self.required) for result in self.component_results.values()
        )

    @property
    def components(self) -> Mapping[str, CompatibilityResult]:
        """Alias for component-keyed diagnostics."""

        return self.component_results

    def allows(self, minimum: CompatibilityLevel | str) -> bool:
        """Whether every component allows ``minimum`` compatibility."""

        required = CompatibilityLevel.parse(minimum)
        return all(
            result.allows(required) for result in self.component_results.values()
        )

    def explain(self) -> str:
        """Render component-specific diagnostics for compiler or user errors."""

        lines = [
            "ArtifactBundle compatibility: "
            f"required={self.required.value}, aggregate={self.level.value}, "
            f"compatible={self.compatible}"
        ]
        lines.extend(
            f"  {key}: {result.explain()}"
            for key, result in self.component_results.items()
        )
        return "\n".join(lines)


def _aggregate_level(
    values: Iterable[CompatibilityResult],
) -> CompatibilityLevel:
    results = tuple(values)
    if all(result.level is CompatibilityLevel.EXACT for result in results):
        return CompatibilityLevel.EXACT
    if all(
        result.allows(CompatibilityLevel.ARCHITECTURE_COMPATIBLE) for result in results
    ):
        return CompatibilityLevel.ARCHITECTURE_COMPATIBLE
    if all(result.allows(CompatibilityLevel.DIMENSION_ONLY) for result in results):
        return CompatibilityLevel.DIMENSION_ONLY
    return CompatibilityLevel.INCOMPATIBLE


def _failed_component_compatibility(
    component: ArtifactBundleComponent, target: Any, error: Exception
) -> CompatibilityResult:
    metadata = component.artifact.metadata
    target_id = getattr(target, "model_id", getattr(target, "id", ""))
    target_revision = getattr(
        target, "revision", getattr(target, "model_revision", None)
    )
    return CompatibilityResult(
        level=CompatibilityLevel.INCOMPATIBLE,
        reasons=(
            f"component {component.key!r} could not resolve its required site: {error}",
        ),
        artifact_model=(
            f"{metadata.model_id}@{metadata.model_revision or '<unknown>'}"
        ),
        target_model=f"{target_id or ''}@{target_revision or '<unknown>'}",
        artifact_hidden_size=metadata.hidden_size,
        target_hidden_size=None,
        artifact_site=metadata.site,
        target_site=None,
    )


def _check_component_compatibility(
    component: ArtifactBundleComponent, target: Any
) -> CompatibilityResult:
    """Run the existing component contract after resolving its semantic site."""

    if isinstance(component.artifact, ITIArtifact):
        try:
            return component.artifact.check_compatibility(target)
        except Exception as exc:
            return _failed_component_compatibility(component, target, exc)

    site: Site | None = None
    hidden_size: int | None = None
    metadata_site = component.artifact.metadata.site
    resolve_site = getattr(target, "resolve_site", None)
    if metadata_site is not None and callable(resolve_site):
        try:
            resolved = resolve_site(metadata_site)
            site = getattr(resolved, "site", None)
            hidden_size = getattr(resolved, "hidden_dim", None)
            if not isinstance(site, Site):
                raise TypeError("resolve_site() returned no semantic Site")
            if (
                not isinstance(hidden_size, int)
                or isinstance(hidden_size, bool)
                or hidden_size <= 0
            ):
                raise TypeError("resolve_site() returned no positive hidden_dim")
        except Exception as exc:
            return _failed_component_compatibility(component, target, exc)
    return check_artifact_compatibility(
        component.artifact,
        target,
        site=site,
        hidden_size=hidden_size,
    )


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """A checksummed, role-keyed collection of safe repsteer artifacts.

    Components remain independently loadable artifacts; the bundle only adds
    role, provenance, optional selection evidence, and aggregate compatibility
    checks.  It does not encode a method-specific learner or runtime policy.
    """

    components: tuple[ArtifactBundleComponent, ...]
    kind: str = "generic"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    selection_report: SelectionReport | None = None
    compatibility_policy: CompatibilityLevel | str = CompatibilityLevel.EXACT
    created_with: Mapping[str, Any] = field(
        default_factory=lambda: {"name": "repsteer", "version": __version__}
    )
    schema_version: str = _BUNDLE_SCHEMA_VERSION
    bundle_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported ArtifactBundle schema "
                f"{self.schema_version!r}; supported: {_BUNDLE_SCHEMA_VERSION!r}"
            )
        if isinstance(self.components, Mapping):
            for key, component in self.components.items():
                if not isinstance(component, ArtifactBundleComponent):
                    continue
                if key != component.key:
                    raise ValueError(
                        "ArtifactBundle mapping key must match component key: "
                        f"{key!r} != {component.key!r}"
                    )
            values = tuple(self.components.values())
        else:
            values = tuple(self.components)
        if not values:
            raise ValueError("ArtifactBundle.components cannot be empty")
        for index, component in enumerate(values):
            if not isinstance(component, ArtifactBundleComponent):
                raise TypeError(
                    f"ArtifactBundle.components[{index}] must be "
                    "ArtifactBundleComponent, got "
                    f"{type(component).__name__}"
                )
        keys = [component.key for component in values]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(
                "ArtifactBundle component keys must be unique: " + ", ".join(duplicates)
            )
        components = tuple(sorted(values, key=lambda component: component.key))
        kind = _require_nonempty_string(self.kind, name="ArtifactBundle.kind")
        if self.selection_report is not None and not isinstance(
            self.selection_report, SelectionReport
        ):
            raise TypeError("selection_report must be a SelectionReport or None")
        policy = CompatibilityLevel.parse(self.compatibility_policy)
        if policy is CompatibilityLevel.INCOMPATIBLE:
            raise ValueError("compatibility_policy cannot be 'incompatible'")
        provenance = _freeze_json(self.provenance, path="ArtifactBundle.provenance")
        created_with = _freeze_json(
            self.created_with, path="ArtifactBundle.created_with"
        )
        if not isinstance(provenance, Mapping) or not isinstance(created_with, Mapping):
            raise TypeError(
                "ArtifactBundle provenance and created_with must be JSON objects"
            )
        created_with_values = dict(created_with)
        created_with_values.setdefault("name", "repsteer")
        created_with_values.setdefault("version", __version__)
        if created_with_values["name"] != "repsteer":
            raise ValueError("ArtifactBundle.created_with.name must be 'repsteer'")
        _require_nonempty_string(
            created_with_values["version"],
            name="ArtifactBundle.created_with.version",
        )
        created_with = _freeze_json(
            created_with_values, path="ArtifactBundle.created_with"
        )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "compatibility_policy", policy.value)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "created_with", created_with)
        expected_digest = self._computed_digest()
        if self.bundle_digest != "":
            supplied_digest = _validate_digest(
                self.bundle_digest, name="ArtifactBundle bundle_digest"
            )
            if supplied_digest != expected_digest:
                raise ValueError(
                    "ArtifactBundle digest mismatch: "
                    f"expected {expected_digest}, got {supplied_digest}"
                )
        object.__setattr__(self, "bundle_digest", expected_digest)

    @property
    def bundle_type(self) -> str:
        """Alias for :attr:`kind` used in manifest-oriented integrations."""

        return self.kind

    @property
    def fingerprint(self) -> str:
        """Alias for the canonical bundle digest."""

        return self.bundle_digest

    @property
    def manifest(self) -> Mapping[str, Any]:
        """Return the safe canonical manifest as a read-only mapping."""

        return MappingProxyType(self.to_dict())

    @property
    def by_key(self) -> Mapping[str, ArtifactBundleComponent]:
        """Return an immutable mapping of component key to component."""

        return MappingProxyType(
            {component.key: component for component in self.components}
        )

    def component(self, key: str) -> ArtifactBundleComponent:
        """Return one component or raise a clear ``KeyError``."""

        try:
            return self.by_key[key]
        except KeyError as exc:
            raise KeyError(f"ArtifactBundle has no component {key!r}") from exc

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "components": [
                component.to_manifest_record() for component in self.components
            ],
            "provenance": _json_value(self.provenance),
            "selection_report": (
                None
                if self.selection_report is None
                else self.selection_report.to_dict()
            ),
            "compatibility_policy": self.compatibility_policy,
            "created_with": _json_value(self.created_with),
        }

    def _computed_digest(self) -> str:
        return _digest(self._payload())

    def _assert_current_digest(self) -> str:
        """Reject mutable component changes made after bundle construction."""

        expected_digest = self._computed_digest()
        if not hmac.compare_digest(self.bundle_digest, expected_digest):
            raise ValueError(
                "ArtifactBundle content changed after construction; create a new "
                "ArtifactBundle before serializing or saving it"
            )
        return expected_digest

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical bundle manifest, including its digest."""

        self._assert_current_digest()
        return {**self._payload(), "bundle_digest": self.bundle_digest}

    @property
    def canonical(self) -> str:
        """Return the stable bundle manifest serialization."""

        return _canonical_json(self.to_dict())

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Safely write component artifacts plus a strict bundle manifest.

        Existing bundle directories may be overwritten only when they contain
        the same controlled layout.  Unknown or stale files are rejected rather
        than silently ignored.

        Args:
            path: Destination directory owned by this bundle save operation.

        Returns:
            The normalized destination path after all component and root records
            have been written and revalidated.

        Raises:
            ArtifactFormatError: If the existing layout is not controlled.
            UnsafeArtifactError: If a symlink could redirect a bundle write.
        """

        self._assert_current_digest()
        destination = Path(path)
        if destination.is_symlink():
            raise UnsafeArtifactError("ArtifactBundle destination cannot be a symlink")
        if destination.exists() and not destination.is_dir():
            raise ArtifactFormatError(
                f"ArtifactBundle path is not a directory: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        components_root = destination / _COMPONENTS_DIRECTORY
        if components_root.is_symlink():
            raise UnsafeArtifactError(
                "ArtifactBundle components directory cannot be a symlink"
            )
        if components_root.exists() and not components_root.is_dir():
            raise ArtifactFormatError(
                f"ArtifactBundle components path is not a directory: {components_root}"
            )
        components_root.mkdir(exist_ok=True)
        self._validate_existing_layout(destination)
        for component in self.components:
            component_path = destination / component.relative_path
            if component_path.is_symlink():
                raise UnsafeArtifactError(
                    f"component {component.key!r} directory cannot be a symlink"
                )
            if component_path.exists() and not component_path.is_dir():
                raise ArtifactFormatError(
                    f"component {component.key!r} path is not a directory: "
                    f"{component_path}"
                )
            component_path.mkdir(parents=True, exist_ok=True)
            save_artifact(component.artifact, component_path)
        manifest_path = destination / MANIFEST_FILENAME
        _write_json(manifest_path, self.to_dict())
        _write_json(
            destination / CHECKSUMS_FILENAME,
            {
                "algorithm": "sha256",
                "files": {MANIFEST_FILENAME: _sha256(manifest_path)},
            },
        )
        _validate_file_layout(destination)
        return destination

    def _validate_existing_layout(self, destination: Path) -> None:
        """Reject stale files before an overwrite can make them ambiguous."""

        if not any(destination.iterdir()):
            return
        allowed_top_level = {
            MANIFEST_FILENAME,
            CHECKSUMS_FILENAME,
            _COMPONENTS_DIRECTORY,
        }
        unknown_top_level = sorted(
            entry.name
            for entry in destination.iterdir()
            if entry.name not in allowed_top_level
        )
        if unknown_top_level:
            raise ArtifactFormatError(
                "ArtifactBundle destination contains unexpected files: "
                + ", ".join(unknown_top_level)
            )
        for filename in (MANIFEST_FILENAME, CHECKSUMS_FILENAME):
            candidate = destination / filename
            if candidate.is_symlink():
                raise UnsafeArtifactError(
                    f"ArtifactBundle {filename} cannot be a symlink"
                )
            if candidate.exists() and not candidate.is_file():
                raise ArtifactFormatError(
                    f"ArtifactBundle {filename} is not a regular file: {candidate}"
                )
        components_root = destination / _COMPONENTS_DIRECTORY
        expected_directories = {
            Path(component.relative_path).name for component in self.components
        }
        if components_root.exists():
            unexpected = sorted(
                entry.name
                for entry in components_root.iterdir()
                if entry.name not in expected_directories
            )
            if unexpected:
                raise ArtifactFormatError(
                    "ArtifactBundle destination has stale or unknown component "
                    "directories: " + ", ".join(unexpected)
                )
        for component in self.components:
            component_path = destination / component.relative_path
            if component_path.is_symlink():
                raise UnsafeArtifactError(
                    f"component {component.key!r} directory cannot be a symlink"
                )
            if component_path.exists() and not component_path.is_dir():
                raise ArtifactFormatError(
                    f"component {component.key!r} path is not a directory: "
                    f"{component_path}"
                )
            if component_path.exists():
                expected_files = {
                    MANIFEST_FILENAME,
                    TENSORS_FILENAME,
                    CHECKSUMS_FILENAME,
                }
                entries = tuple(component_path.iterdir())
                unexpected_files = sorted(
                    entry.name for entry in entries if entry.name not in expected_files
                )
                if unexpected_files:
                    raise ArtifactFormatError(
                        f"component {component.key!r} has unexpected files: "
                        + ", ".join(unexpected_files)
                    )
                for entry in entries:
                    if entry.is_symlink():
                        raise UnsafeArtifactError(
                            f"component {component.key!r} contains a symlink"
                        )
                    if not entry.is_file():
                        raise ArtifactFormatError(
                            f"component {component.key!r} contains a non-file entry: "
                            f"{entry.name}"
                        )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        verify_checksums: bool = True,
        device: str = "cpu",
    ) -> ArtifactBundle:
        """Load a bundle after verifying root and every component artifact.

        Args:
            path: Bundle root directory.
            verify_checksums: Verify root and component checksums before decode.
            device: Device forwarded to the existing safetensors artifact loader.

        Returns:
            A canonical, role-keyed bundle with independently loaded components.

        Raises:
            ArtifactFormatError: If JSON, layout, or artifact records are invalid.
            UnsafeArtifactError: If a checksum, digest, or path safety check fails.
        """

        source = Path(path)
        if not source.is_dir():
            raise ArtifactFormatError(
                f"ArtifactBundle path is not a directory: {source}"
            )
        if source.is_symlink():
            raise UnsafeArtifactError("ArtifactBundle source cannot be a symlink")
        if verify_checksums:
            verify_bundle_checksums(source)
        manifest = _read_json(source / MANIFEST_FILENAME)
        document = _parse_bundle_manifest(manifest, root=source)
        _validate_bundle_file_layout(source, document["components"])
        components: list[ArtifactBundleComponent] = []
        for record in document["components"]:
            key = record["key"]
            component_path = _validate_relative_component_path(
                record["path"], key=key, root=source
            )
            _validate_component_files(component_path, key=key)
            artifact = load_artifact(
                component_path,
                verify_checksums=verify_checksums,
                device=device,
            )
            if artifact.metadata.artifact_type != record["artifact_type"]:
                raise ArtifactFormatError(
                    f"component {key!r} artifact type mismatch: manifest declares "
                    f"{record['artifact_type']!r}, component contains "
                    f"{artifact.metadata.artifact_type!r}"
                )
            actual_digest = artifact.fingerprint()
            if not hmac.compare_digest(actual_digest, record["artifact_digest"]):
                raise UnsafeArtifactError(
                    f"component {key!r} digest mismatch: expected "
                    f"{record['artifact_digest']}, got {actual_digest}"
                )
            components.append(
                ArtifactBundleComponent(
                    key=key,
                    role=record["role"],
                    artifact=artifact,
                    metadata=record["metadata"],
                )
            )
        selection_value = document["selection_report"]
        selection_report = (
            None
            if selection_value is None
            else SelectionReport.from_dict(selection_value)
        )
        return cls(
            components=tuple(components),
            kind=document["kind"],
            provenance=document["provenance"],
            selection_report=selection_report,
            compatibility_policy=document["compatibility_policy"],
            created_with=document["created_with"],
            schema_version=document["schema_version"],
            bundle_digest=document["bundle_digest"],
        )

    def check_compatibility(
        self,
        target: Any,
        *,
        compatibility: CompatibilityLevel | str | None = None,
    ) -> BundleCompatibilityResult:
        """Run the existing compatibility contract independently per component.

        ``compatibility`` may request a stricter level, but cannot weaken the
        policy recorded in the bundle manifest.

        Returns:
            Aggregate and component-keyed compatibility diagnostics.

        Raises:
            ValueError: If the requested level is invalid or weaker than policy.
        """

        self._assert_current_digest()
        recorded = CompatibilityLevel.parse(self.compatibility_policy)
        required = CompatibilityLevel.parse(
            recorded if compatibility is None else compatibility
        )
        if required is CompatibilityLevel.INCOMPATIBLE:
            raise ValueError("compatibility cannot be 'incompatible'")
        if _COMPATIBILITY_RANK[required] < _COMPATIBILITY_RANK[recorded]:
            raise ValueError(
                "cannot weaken ArtifactBundle compatibility_policy "
                f"from {recorded.value!r} to {required.value!r}"
            )
        results = {
            component.key: _check_component_compatibility(component, target)
            for component in self.components
        }
        return BundleCompatibilityResult(
            level=_aggregate_level(results.values()),
            required=required,
            component_results=results,
        )

    def bind(
        self,
        target: Any,
        *,
        compatibility: CompatibilityLevel | str | None = None,
    ) -> ArtifactBundle:
        """Require every component to meet compatibility and return this bundle.

        Raises:
            ArtifactCompatibilityError: If any named component does not meet the
                recorded compatibility policy.
        """

        result = self.check_compatibility(target, compatibility=compatibility)
        if not result.compatible:
            raise ArtifactCompatibilityError(
                "ArtifactBundle compatibility check failed\n" + result.explain()
            )
        return self


def _parse_bundle_manifest(
    manifest: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    document = _as_mapping(manifest, name="ArtifactBundle manifest")
    allowed = {
        "schema_version",
        "kind",
        "components",
        "provenance",
        "selection_report",
        "compatibility_policy",
        "created_with",
        "bundle_digest",
    }
    _reject_unknown_fields(document, name="ArtifactBundle manifest", allowed=allowed)
    required = allowed
    missing = sorted(required - set(document))
    if missing:
        raise ArtifactFormatError(
            "ArtifactBundle manifest is missing required fields: " + ", ".join(missing)
        )
    if document["schema_version"] != _BUNDLE_SCHEMA_VERSION:
        raise ArtifactFormatError(
            "unsupported ArtifactBundle schema "
            f"{document['schema_version']!r}; supported: {_BUNDLE_SCHEMA_VERSION!r}"
        )
    kind = _require_nonempty_string(document["kind"], name="ArtifactBundle.kind")
    policy = CompatibilityLevel.parse(document["compatibility_policy"])
    if policy is CompatibilityLevel.INCOMPATIBLE:
        raise ArtifactFormatError(
            "ArtifactBundle compatibility_policy cannot be incompatible"
        )
    raw_components = document["components"]
    if not isinstance(raw_components, list | tuple) or not raw_components:
        raise ArtifactFormatError(
            "ArtifactBundle manifest components must be a non-empty list"
        )
    components = [_parse_component_record(item, root=root) for item in raw_components]
    keys = [item["key"] for item in components]
    if keys != sorted(keys):
        raise ArtifactFormatError(
            "ArtifactBundle manifest components must be ordered by canonical key"
        )
    if len(keys) != len(set(keys)):
        raise ArtifactFormatError(
            "ArtifactBundle manifest component keys must be unique"
        )
    provenance = _freeze_json(document["provenance"], path="ArtifactBundle.provenance")
    created_with = _freeze_json(
        document["created_with"], path="ArtifactBundle.created_with"
    )
    if not isinstance(provenance, Mapping) or not isinstance(created_with, Mapping):
        raise ArtifactFormatError(
            "ArtifactBundle provenance and created_with must be JSON objects"
        )
    try:
        created_with_name = _require_nonempty_string(
            created_with.get("name"), name="ArtifactBundle.created_with.name"
        )
        _require_nonempty_string(
            created_with.get("version"), name="ArtifactBundle.created_with.version"
        )
    except ValueError as exc:
        raise ArtifactFormatError(str(exc)) from exc
    if created_with_name != "repsteer":
        raise ArtifactFormatError("ArtifactBundle.created_with.name must be 'repsteer'")
    selection_report = document["selection_report"]
    if selection_report is not None:
        try:
            selection_report = SelectionReport.from_dict(
                _as_mapping(selection_report, name="selection_report")
            ).to_dict()
        except (TypeError, ValueError) as exc:
            raise ArtifactFormatError(
                f"ArtifactBundle selection_report is invalid: {exc}"
            ) from exc
    expected_payload = {
        "schema_version": document["schema_version"],
        "kind": kind,
        "components": components,
        "provenance": _json_value(provenance),
        "selection_report": selection_report,
        "compatibility_policy": policy.value,
        "created_with": _json_value(created_with),
    }
    expected_digest = _digest(expected_payload)
    supplied_digest = _validate_digest(
        document["bundle_digest"], name="ArtifactBundle bundle_digest"
    )
    if not hmac.compare_digest(expected_digest, supplied_digest):
        raise UnsafeArtifactError(
            "ArtifactBundle digest mismatch: "
            f"expected {expected_digest}, got {supplied_digest}"
        )
    return {
        **expected_payload,
        "bundle_digest": supplied_digest,
    }


def _parse_component_record(value: Any, *, root: Path) -> dict[str, Any]:
    document = _as_mapping(value, name="ArtifactBundle component")
    allowed = {"key", "role", "artifact_type", "artifact_digest", "metadata", "path"}
    _reject_unknown_fields(document, name="ArtifactBundle component", allowed=allowed)
    missing = sorted(allowed - set(document))
    if missing:
        raise ArtifactFormatError(
            "ArtifactBundle component is missing required fields: " + ", ".join(missing)
        )
    key = _require_nonempty_string(document["key"], name="component key")
    role = _require_nonempty_string(document["role"], name=f"component {key!r} role")
    artifact_type = document["artifact_type"]
    if artifact_type not in _SUPPORTED_COMPONENT_TYPES:
        raise ArtifactFormatError(
            f"component {key!r} has unknown artifact type {artifact_type!r}"
        )
    artifact_digest = _validate_digest(
        document["artifact_digest"], name=f"component {key!r} artifact_digest"
    )
    metadata = _freeze_json(document["metadata"], path=f"component {key!r} metadata")
    if not isinstance(metadata, Mapping):
        raise ArtifactFormatError(f"component {key!r} metadata must be a JSON object")
    path = document["path"]
    _validate_relative_component_path(path, key=key, root=root)
    return {
        "key": key,
        "role": role,
        "artifact_type": artifact_type,
        "artifact_digest": artifact_digest,
        "metadata": _json_value(metadata),
        "path": path,
    }


def _validate_component_manifest(component_path: Path, *, key: str) -> None:
    manifest = _read_json(component_path / MANIFEST_FILENAME)
    allowed = {
        "schema_version",
        "artifact_type",
        "method",
        "model",
        "tokenizer",
        "processor",
        "modality",
        "site",
        "normalization",
        "dtype",
        "dataset_fingerprint",
        "seed",
        "config",
        "library",
        "provenance",
        "tensor_keys",
        "artifact_fingerprint",
    }
    _reject_unknown_fields(
        manifest,
        name=f"component {key!r} artifact manifest",
        allowed=allowed,
    )


def _validate_component_checksums(component_path: Path, *, key: str) -> None:
    checksums = _read_json(component_path / CHECKSUMS_FILENAME)
    _reject_unknown_fields(
        checksums,
        name=f"component {key!r} checksums",
        allowed={"algorithm", "files"},
    )
    if checksums.get("algorithm") != "sha256":
        raise UnsafeArtifactError(
            f"component {key!r} uses unsupported checksum algorithm "
            f"{checksums.get('algorithm')!r}"
        )
    files = checksums.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        MANIFEST_FILENAME,
        TENSORS_FILENAME,
    }:
        raise UnsafeArtifactError(
            f"component {key!r} checksums must cover exactly "
            f"{MANIFEST_FILENAME} and {TENSORS_FILENAME}"
        )


def _validate_component_files(component_path: Path, *, key: str) -> None:
    if not component_path.is_dir() or component_path.is_symlink():
        raise UnsafeArtifactError(
            f"component {key!r} directory is missing or unsafe: {component_path}"
        )
    expected = {MANIFEST_FILENAME, TENSORS_FILENAME, CHECKSUMS_FILENAME}
    actual = {entry.name for entry in component_path.iterdir()}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ArtifactFormatError(
            f"component {key!r} has invalid file layout: {'; '.join(details)}"
        )
    if any(entry.is_symlink() for entry in component_path.iterdir()):
        raise UnsafeArtifactError(f"component {key!r} contains a symlink")
    _validate_component_manifest(component_path, key=key)
    _validate_component_checksums(component_path, key=key)


def _validate_bundle_file_layout(
    root: Path, components: Iterable[Mapping[str, Any]]
) -> None:
    expected_files = {MANIFEST_FILENAME, CHECKSUMS_FILENAME}
    expected_directories = {_COMPONENTS_DIRECTORY}
    for record in components:
        relative = PurePosixPath(str(record["path"]))
        expected_directories.add(relative.as_posix())
        for name in (MANIFEST_FILENAME, TENSORS_FILENAME, CHECKSUMS_FILENAME):
            expected_files.add((relative / name).as_posix())
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for entry in root.rglob("*"):
        actual_relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise UnsafeArtifactError(
                f"ArtifactBundle contains unsafe symlink {actual_relative!r}"
            )
        if entry.is_dir():
            actual_directories.add(actual_relative)
        else:
            actual_files.add(actual_relative)
    missing_files = sorted(expected_files - actual_files)
    unexpected_files = sorted(actual_files - expected_files)
    missing_directories = sorted(expected_directories - actual_directories)
    unexpected_directories = sorted(actual_directories - expected_directories)
    if (
        missing_files
        or unexpected_files
        or missing_directories
        or unexpected_directories
    ):
        details: list[str] = []
        if missing_files:
            details.append("missing files: " + ", ".join(missing_files))
        if unexpected_files:
            details.append("unexpected files: " + ", ".join(unexpected_files))
        if missing_directories:
            details.append("missing directories: " + ", ".join(missing_directories))
        if unexpected_directories:
            details.append(
                "unexpected directories: " + ", ".join(unexpected_directories)
            )
        raise ArtifactFormatError(
            "ArtifactBundle has invalid file layout: " + "; ".join(details)
        )
    for record in components:
        component_path = _validate_relative_component_path(
            record["path"], key=str(record["key"]), root=root
        )
        _validate_component_files(component_path, key=str(record["key"]))


def _validate_file_layout(root: Path) -> None:
    """Validate this saved bundle by parsing its own just-written manifest."""

    manifest = _read_json(root / MANIFEST_FILENAME)
    document = _parse_bundle_manifest(manifest, root=root)
    _validate_bundle_file_layout(root, document["components"])


def verify_bundle_checksums(path: str | os.PathLike[str]) -> None:
    """Verify the root bundle manifest checksum before loading components.

    Args:
        path: Bundle root directory containing ``manifest.json`` and its checksum.

    Raises:
        UnsafeArtifactError: If the root checksum, required files, or symlink
            safety checks fail.
    """

    source = Path(path)
    if source.is_symlink():
        raise UnsafeArtifactError("ArtifactBundle source cannot be a symlink")
    checksums_path = source / CHECKSUMS_FILENAME
    if checksums_path.is_symlink():
        raise UnsafeArtifactError(
            f"ArtifactBundle {CHECKSUMS_FILENAME} cannot be a symlink"
        )
    if not checksums_path.is_file():
        raise UnsafeArtifactError(f"ArtifactBundle has no {CHECKSUMS_FILENAME}")
    checksums = _read_json(checksums_path)
    _reject_unknown_fields(
        checksums,
        name="ArtifactBundle checksums",
        allowed={"algorithm", "files"},
    )
    if checksums.get("algorithm") != "sha256":
        raise UnsafeArtifactError(
            "Unsupported ArtifactBundle checksum algorithm "
            f"{checksums.get('algorithm')!r}"
        )
    files = checksums.get("files")
    if not isinstance(files, Mapping) or set(files) != {MANIFEST_FILENAME}:
        raise UnsafeArtifactError(
            f"ArtifactBundle checksums must cover exactly {MANIFEST_FILENAME}"
        )
    expected = str(files[MANIFEST_FILENAME]).removeprefix("sha256:")
    manifest_path = source / MANIFEST_FILENAME
    if manifest_path.is_symlink():
        raise UnsafeArtifactError(
            f"ArtifactBundle {MANIFEST_FILENAME} cannot be a symlink"
        )
    if not manifest_path.is_file():
        raise UnsafeArtifactError(f"ArtifactBundle is missing {MANIFEST_FILENAME}")
    actual = _sha256(manifest_path)
    if not hmac.compare_digest(expected, actual):
        raise UnsafeArtifactError(
            f"Checksum mismatch for {MANIFEST_FILENAME}: "
            f"expected {expected}, got {actual}"
        )


def save_artifact_bundle(bundle: ArtifactBundle, path: str | os.PathLike[str]) -> Path:
    """Save ``bundle`` through :meth:`ArtifactBundle.save` and return its path."""

    if not isinstance(bundle, ArtifactBundle):
        raise TypeError("save_artifact_bundle expects an ArtifactBundle")
    return bundle.save(path)


def load_artifact_bundle(
    path: str | os.PathLike[str],
    *,
    verify_checksums: bool = True,
    device: str = "cpu",
) -> ArtifactBundle:
    """Load a role-keyed bundle through :meth:`ArtifactBundle.load`."""

    return ArtifactBundle.load(
        path,
        verify_checksums=verify_checksums,
        device=device,
    )


__all__ = [
    "ArtifactBundle",
    "ArtifactBundleComponent",
    "BundleCompatibilityResult",
    "load_artifact_bundle",
    "save_artifact_bundle",
    "verify_bundle_checksums",
]
