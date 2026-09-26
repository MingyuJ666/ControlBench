"""CPU-only adversarial tests for ordinary artifact bundle I/O."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from repsteer.artifacts import ArtifactMetadata, DirectionArtifact, load_artifact
from repsteer.core import ArtifactFormatError, Site, UnsafeArtifactError


def _artifact() -> DirectionArtifact:
    return DirectionArtifact(
        ArtifactMetadata(
            model_id="tiny/artifact-io",
            model_revision="r1",
            architecture="TinyArtifactModel",
            site=Site("language", "resid_post", 0),
            hidden_size=2,
            method="unit",
        ),
        torch.tensor([1.0, -1.0]),
    )


def _symlink_or_skip(link, target, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:  # pragma: no cover - platforms without symlink support
        pytest.skip(f"symlinks unavailable: {exc}")


def _rewrite_manifest_checksum(destination) -> None:
    manifest = destination / "manifest.json"
    checksums = destination / "checksums.json"
    document = json.loads(checksums.read_text(encoding="utf-8"))
    document["files"]["manifest.json"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    checksums.write_text(json.dumps(document), encoding="utf-8")


def test_save_rejects_symlinked_destination_before_external_write(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    destination = tmp_path / "artifact-link"
    _symlink_or_skip(destination, external, target_is_directory=True)

    with pytest.raises(
        UnsafeArtifactError, match="destination directory cannot be a symlink"
    ):
        _artifact().save(destination)

    assert list(external.iterdir()) == []


def test_save_rejects_symlinked_temporary_tensor_path_before_write(tmp_path):
    destination = tmp_path / "artifact"
    destination.mkdir()
    external = tmp_path / "external.safetensors"
    external.write_bytes(b"must not be overwritten")
    temporary = destination / ".tensors.safetensors.tmp"
    _symlink_or_skip(temporary, external)

    with pytest.raises(UnsafeArtifactError, match="Artifact file cannot be a symlink"):
        _artifact().save(destination)

    assert external.read_bytes() == b"must not be overwritten"
    assert not (destination / "tensors.safetensors").exists()


@pytest.mark.parametrize(
    "filename", ["checksums.json", "manifest.json", "tensors.safetensors"]
)
def test_load_rejects_symlinked_controlled_files(tmp_path, filename):
    destination = tmp_path / "artifact"
    _artifact().save(destination)
    source = destination / filename
    external = tmp_path / f"external-{filename}"
    external.write_bytes(source.read_bytes())
    source.unlink()
    _symlink_or_skip(source, external)

    with pytest.raises(UnsafeArtifactError, match="Artifact file cannot be a symlink"):
        load_artifact(destination)


def test_load_rejects_a_symlinked_artifact_root(tmp_path):
    source = tmp_path / "actual-artifact"
    _artifact().save(source)
    linked = tmp_path / "artifact-link"
    _symlink_or_skip(linked, source, target_is_directory=True)

    with pytest.raises(
        UnsafeArtifactError, match="source directory cannot be a symlink"
    ):
        load_artifact(linked)


@pytest.mark.parametrize(
    ("tensor_keys", "message"),
    [
        (["direction", "direction"], "cannot contain duplicates"),
        ({"direction": True}, "must be a list of strings"),
        (["wrong"], "do not match"),
    ],
)
def test_load_rejects_noncanonical_manifest_tensor_keys(tmp_path, tensor_keys, message):
    destination = tmp_path / "artifact"
    _artifact().save(destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensor_keys"] = tensor_keys
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rewrite_manifest_checksum(destination)

    with pytest.raises(ArtifactFormatError, match=message):
        load_artifact(destination)
