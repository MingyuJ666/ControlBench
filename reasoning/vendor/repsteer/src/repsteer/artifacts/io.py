"""Safe safetensors + JSON artifact bundle I/O."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from repsteer.core.errors import (
    ArtifactFormatError,
    MissingOptionalDependencyError,
    UnsafeArtifactError,
)

from .base import (
    Artifact,
    ArtifactMetadata,
    DirectionArtifact,
    ProbeArtifact,
    SAEFeatureArtifact,
    SteeringArtifact,
    SubspaceArtifact,
)

MANIFEST_FILENAME = "manifest.json"
TENSORS_FILENAME = "tensors.safetensors"
CHECKSUMS_FILENAME = "checksums.json"


def _safetensors() -> tuple[Any, Any]:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - dependency contract guard
        raise MissingOptionalDependencyError(
            "Artifact I/O requires safetensors; install the core repsteer dependencies"
        ) from exc
    return load_file, save_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    _assert_safe_artifact_file(path, action="write")
    _assert_safe_artifact_file(temporary, action="write")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    _assert_safe_artifact_file(path, action="read")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactFormatError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactFormatError(f"Expected a JSON object in {path}")
    return value


def _assert_safe_artifact_file(path: Path, *, action: str) -> None:
    """Reject symlinks and non-files at controlled artifact paths.

    Artifact directories intentionally contain only a small fixed set of files.
    Following a symlink at one of those paths could redirect an otherwise local
    artifact read or write outside the caller-selected directory.
    """

    if path.is_symlink():
        raise UnsafeArtifactError(f"Artifact file cannot be a symlink: {path}")
    if path.exists() and not path.is_file():
        raise ArtifactFormatError(
            f"Artifact {action} path is not a regular file: {path}"
        )


def _assert_safe_artifact_directory(path: Path, *, action: str) -> None:
    if path.is_symlink():
        raise UnsafeArtifactError(f"Artifact {action} directory cannot be a symlink")
    if path.exists() and not path.is_dir():
        raise ArtifactFormatError(f"Artifact path is not a directory: {path}")


def save_artifact(artifact: SteeringArtifact, path: str | os.PathLike[str]) -> Path:
    """Write an artifact bundle and checksums without pickle or executable code."""

    if not isinstance(artifact, SteeringArtifact):
        raise TypeError("save_artifact expects a SteeringArtifact")
    destination = Path(path)
    _assert_safe_artifact_directory(destination, action="destination")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_FILENAME
    tensors_path = destination / TENSORS_FILENAME
    checksums_path = destination / CHECKSUMS_FILENAME
    temporary_tensors = tensors_path.with_name(f".{tensors_path.name}.tmp")
    for candidate in (
        manifest_path,
        tensors_path,
        checksums_path,
        temporary_tensors,
    ):
        _assert_safe_artifact_file(candidate, action="write")

    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in artifact.tensors().items()
    }
    if not tensors:
        raise ArtifactFormatError("An artifact must contain at least one tensor")
    if not all(isinstance(name, str) and name for name in tensors):
        raise ArtifactFormatError("Artifact tensor names must be non-empty strings")
    _, save_file = _safetensors()
    save_file(tensors, str(temporary_tensors))
    os.replace(temporary_tensors, tensors_path)

    manifest = artifact.metadata.to_manifest()
    manifest["tensor_keys"] = sorted(tensors)
    manifest["artifact_fingerprint"] = artifact.fingerprint()
    _write_json(manifest_path, manifest)
    checksums = {
        "algorithm": "sha256",
        "files": {
            MANIFEST_FILENAME: _sha256(manifest_path),
            TENSORS_FILENAME: _sha256(tensors_path),
        },
    }
    _write_json(checksums_path, checksums)
    return destination


def verify_artifact_checksums(path: str | os.PathLike[str]) -> None:
    source = Path(path)
    _assert_safe_artifact_directory(source, action="source")
    checksums_path = source / CHECKSUMS_FILENAME
    _assert_safe_artifact_file(checksums_path, action="read")
    if not checksums_path.is_file():
        raise UnsafeArtifactError(f"Artifact bundle has no {CHECKSUMS_FILENAME}")
    document = _read_json(checksums_path)
    if document.get("algorithm") != "sha256":
        raise UnsafeArtifactError(
            f"Unsupported checksum algorithm {document.get('algorithm')!r}"
        )
    files = document.get("files")
    if not isinstance(files, Mapping):
        raise UnsafeArtifactError("checksums.json has no valid 'files' mapping")
    required = {MANIFEST_FILENAME, TENSORS_FILENAME}
    if not required.issubset(files):
        missing = ", ".join(sorted(required - set(files)))
        raise UnsafeArtifactError(f"Artifact checksums are missing: {missing}")
    for filename in required:
        expected = str(files[filename]).removeprefix("sha256:")
        file_path = source / filename
        _assert_safe_artifact_file(file_path, action="read")
        if not file_path.is_file():
            raise UnsafeArtifactError(f"Artifact bundle is missing {filename}")
        actual = _sha256(file_path)
        if not hmac.compare_digest(actual, expected):
            raise UnsafeArtifactError(
                f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
            )


def _build_artifact(
    metadata: ArtifactMetadata, tensors: Mapping[str, torch.Tensor]
) -> Artifact:
    artifact_type = metadata.artifact_type
    if artifact_type == "direction":
        if "direction" not in tensors:
            raise ArtifactFormatError("Direction artifact has no 'direction' tensor")
        return DirectionArtifact(metadata, tensors["direction"])
    if artifact_type == "subspace":
        if "basis" not in tensors:
            raise ArtifactFormatError("Subspace artifact has no 'basis' tensor")
        return SubspaceArtifact(
            metadata,
            tensors["basis"],
            tensors.get("mean"),
            tensors.get("explained_variance"),
        )
    if artifact_type == "probe":
        if "weight" not in tensors:
            raise ArtifactFormatError("Probe artifact has no 'weight' tensor")
        return ProbeArtifact(metadata, tensors["weight"], tensors.get("bias"))
    if artifact_type == "sae_feature":
        if "decoder_direction" not in tensors:
            raise ArtifactFormatError(
                "SAE feature artifact has no 'decoder_direction' tensor"
            )
        feature_id = metadata.config.get("feature_id")
        if feature_id is None:
            raise ArtifactFormatError(
                "SAE feature artifact metadata has no 'feature_id'"
            )
        try:
            normalized_feature_id = int(feature_id)
        except (TypeError, ValueError) as exc:
            raise ArtifactFormatError(
                "SAE feature artifact feature_id must be an int"
            ) from exc
        score = metadata.config.get("selection_score")
        try:
            normalized_score = None if score is None else float(score)
        except (TypeError, ValueError) as exc:
            raise ArtifactFormatError(
                "SAE feature artifact selection_score must be numeric"
            ) from exc
        return SAEFeatureArtifact(
            metadata,
            normalized_feature_id,
            tensors["decoder_direction"],
            normalized_score,
        )
    if artifact_type == "iti":
        if "directions" not in tensors:
            raise ArtifactFormatError("ITI artifact has no 'directions' tensor")
        heads = metadata.config.get("heads")
        if not isinstance(heads, tuple | list):
            raise ArtifactFormatError("ITI artifact metadata has no valid 'heads' list")
        from .iti import ITIArtifact

        return ITIArtifact(metadata, tensors["directions"], tuple(heads))
    raise ArtifactFormatError(f"Unsupported artifact_type {artifact_type!r}")


def load_artifact(
    path: str | os.PathLike[str],
    *,
    verify_checksums: bool = True,
    device: str | torch.device = "cpu",
) -> Artifact:
    """Load a bundle after integrity verification (enabled by default)."""

    source = Path(path)
    _assert_safe_artifact_directory(source, action="source")
    if not source.is_dir():
        raise ArtifactFormatError(f"Artifact path is not a directory: {source}")
    if verify_checksums:
        verify_artifact_checksums(source)
    manifest_path = source / MANIFEST_FILENAME
    tensors_path = source / TENSORS_FILENAME
    _assert_safe_artifact_file(manifest_path, action="read")
    _assert_safe_artifact_file(tensors_path, action="read")
    manifest = _read_json(manifest_path)
    schema_version = str(manifest.get("schema_version", ""))
    if schema_version.split(".", 1)[0] != "1":
        raise ArtifactFormatError(
            f"Unsupported artifact schema {schema_version!r}; this version reads 1.x"
        )
    metadata = ArtifactMetadata.from_dict(manifest)
    load_file, _ = _safetensors()
    try:
        tensors = load_file(str(tensors_path), device=str(device))
    except Exception as exc:
        raise ArtifactFormatError(f"Cannot load artifact tensors: {exc}") from exc
    expected_keys = manifest.get("tensor_keys")
    if expected_keys is not None:
        if not isinstance(expected_keys, list) or not all(
            isinstance(key, str) and key for key in expected_keys
        ):
            raise ArtifactFormatError("Manifest tensor_keys must be a list of strings")
        if len(set(expected_keys)) != len(expected_keys):
            raise ArtifactFormatError("Manifest tensor_keys cannot contain duplicates")
        if set(expected_keys) != set(tensors):
            raise ArtifactFormatError(
                "Manifest tensor_keys do not match tensors.safetensors: "
                f"{sorted(expected_keys)} != {sorted(tensors)}"
            )
    artifact = _build_artifact(metadata, tensors)
    expected_fingerprint = manifest.get("artifact_fingerprint")
    if expected_fingerprint and artifact.fingerprint() != expected_fingerprint:
        raise UnsafeArtifactError(
            "Artifact content fingerprint does not match the manifest"
        )
    return artifact


__all__ = [
    "CHECKSUMS_FILENAME",
    "MANIFEST_FILENAME",
    "TENSORS_FILENAME",
    "load_artifact",
    "save_artifact",
    "verify_artifact_checksums",
]
