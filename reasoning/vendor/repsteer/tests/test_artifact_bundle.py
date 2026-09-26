import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from repsteer.artifacts import (
    ArtifactBundle,
    ArtifactBundleComponent,
    ArtifactMetadata,
    BundleCompatibilityResult,
    CompatibilityLevel,
    DirectionArtifact,
    ProbeArtifact,
)
from repsteer.core import (
    ArtifactCompatibilityError,
    ArtifactFormatError,
    Site,
    SiteResolutionError,
    UnsafeArtifactError,
)
from repsteer.selection import Candidate, grid_search, make_holdout_split


class _Target:
    model_id = "tiny/model"
    revision = "rev-a"
    architecture = "TinyForCausalLM"
    hidden_size = 3

    def resolve_site(self, site):
        if site != Site("language", "resid_post", 0):
            raise SiteResolutionError(f"unsupported site {site}")
        return SimpleNamespace(site=site, hidden_dim=3)


def _artifact(kind: str):
    metadata = ArtifactMetadata(
        model_id="tiny/model",
        model_revision="rev-a",
        architecture="TinyForCausalLM",
        site=Site("language", "resid_post", 0),
        method="unit",
    )
    if kind == "direction":
        return DirectionArtifact(metadata, torch.tensor([1.0, 2.0, 3.0]))
    return ProbeArtifact(metadata, torch.tensor([1.0, 0.0, -1.0]))


def _selection_report():
    return grid_search(
        (Candidate("low", {"threshold": 0.2}), Candidate("high", {"threshold": 0.8})),
        lambda candidate: {"f1": 1.0 if candidate.id == "low" else 0.5},
        objective="f1",
        split=make_holdout_split(["a", "b", "c", "d"], seed=8),
        seed=8,
    )


def _bundle():
    return ArtifactBundle(
        components=(
            ArtifactBundleComponent(
                "condition/layer_0", "condition", _artifact("probe"), {"layer": 0}
            ),
            ArtifactBundleComponent(
                "behavior/layer_0", "behavior", _artifact("direction"), {"layer": 0}
            ),
        ),
        provenance={"fixture": "unit"},
        selection_report=_selection_report(),
    )


def _rewrite_root_checksum(path):
    manifest = path / "manifest.json"
    checksum = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (path / "checksums.json").write_text(
        json.dumps({"algorithm": "sha256", "files": {"manifest.json": checksum}}),
        encoding="utf-8",
    )


def test_artifact_bundle_round_trips_with_stable_role_key_order_and_component_checks(
    tmp_path,
):
    bundle = _bundle()
    bundle.save(tmp_path)
    restored = ArtifactBundle.load(tmp_path)

    assert tuple(component.key for component in restored.components) == (
        "behavior/layer_0",
        "condition/layer_0",
    )
    assert restored.component("condition/layer_0").role == "condition"
    assert restored.component("behavior/layer_0").role == "behavior"
    assert restored.to_dict() == bundle.to_dict()
    assert restored.bind(_Target()) is restored

    tensors = next((tmp_path / "components").glob("*/tensors.safetensors"))
    content = tensors.read_bytes()
    tensors.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(UnsafeArtifactError, match="Checksum mismatch"):
        ArtifactBundle.load(tmp_path)


def test_artifact_bundle_root_digest_detects_role_and_report_tampering(tmp_path):
    _bundle().save(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["components"][0]["role"] = "wrong-role"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(UnsafeArtifactError, match="Checksum mismatch"):
        ArtifactBundle.load(tmp_path)

    _rewrite_root_checksum(tmp_path)

    with pytest.raises(UnsafeArtifactError, match="digest mismatch"):
        ArtifactBundle.load(tmp_path)

    _bundle().save(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection_report"]["metadata"] = {"tampered": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rewrite_root_checksum(tmp_path)
    with pytest.raises(ArtifactFormatError, match="selection_report is invalid"):
        ArtifactBundle.load(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", "/absolute/component", "path must be relative"),
        ("path", "components/../escape", "path must be relative"),
        ("artifact_type", "unknown", "unknown artifact type"),
    ],
)
def test_artifact_bundle_rejects_unsafe_component_manifest_records(
    tmp_path, field, value, message
):
    _bundle().save(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["components"][0][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rewrite_root_checksum(tmp_path)

    with pytest.raises((UnsafeArtifactError, ArtifactFormatError), match=message):
        ArtifactBundle.load(tmp_path)


def test_artifact_bundle_rejects_missing_and_extra_component_files(tmp_path):
    _bundle().save(tmp_path)
    tensor_path = next((tmp_path / "components").glob("*/tensors.safetensors"))
    tensor_path.unlink()
    with pytest.raises(ArtifactFormatError, match="missing"):
        ArtifactBundle.load(tmp_path)

    _bundle().save(tmp_path)
    extra_path = tmp_path / "unexpected.txt"
    extra_path.write_text("not part of a bundle", encoding="utf-8")
    with pytest.raises(ArtifactFormatError, match="unexpected files"):
        ArtifactBundle.load(tmp_path)


def test_artifact_bundle_save_rejects_component_symlink_before_any_write(tmp_path):
    bundle = _bundle()
    destination = tmp_path / "bundle"
    component = bundle.components[0]
    external = tmp_path / "outside"
    external.mkdir()
    component_path = destination / component.relative_path
    component_path.parent.mkdir(parents=True)
    try:
        component_path.symlink_to(external, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platforms without symlink support
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(UnsafeArtifactError, match="cannot be a symlink"):
        bundle.save(destination)

    assert not (external / "manifest.json").exists()
    assert not (external / "tensors.safetensors").exists()
    assert not (external / "checksums.json").exists()


def test_artifact_bundle_created_with_is_recorded_and_mutation_fails_before_write(
    tmp_path,
):
    bundle = ArtifactBundle(components=_bundle().components, created_with={})

    assert bundle.created_with["name"] == "repsteer"
    assert bundle.created_with["version"]
    bundle.save(tmp_path / "created-with")
    restored = ArtifactBundle.load(tmp_path / "created-with")
    assert restored.created_with["name"] == "repsteer"
    assert restored.created_with["version"]

    with pytest.raises(ValueError, match="created_with.name"):
        ArtifactBundle(
            components=_bundle().components,
            created_with={"name": "other", "version": "1.0"},
        )

    mutable = _bundle()
    mutable.component("behavior/layer_0").artifact.direction.add_(1)
    destination = tmp_path / "mutated"
    with pytest.raises(ValueError, match="content changed after construction"):
        mutable.bind(_Target())
    with pytest.raises(ValueError, match="content changed after construction"):
        mutable.save(destination)
    assert not destination.exists()


def test_bundle_aggregates_component_compatibility_without_weakening_exact():
    bundle = _bundle()
    target = _Target()
    result = bundle.check_compatibility(target)

    assert result.compatible
    assert set(result.component_results) == {"condition/layer_0", "behavior/layer_0"}
    with pytest.raises(ValueError, match="required cannot be incompatible"):
        BundleCompatibilityResult(
            level=CompatibilityLevel.INCOMPATIBLE,
            required=CompatibilityLevel.INCOMPATIBLE,
            component_results=result.component_results,
        )

    target.architecture = "WrongArchitecture"
    mismatch = bundle.check_compatibility(target)
    assert not mismatch.compatible
    assert "condition/layer_0" in mismatch.explain()
    with pytest.raises(ArtifactCompatibilityError, match="condition/layer_0"):
        bundle.bind(target)
    with pytest.raises(ValueError, match="cannot weaken"):
        bundle.bind(target, compatibility="dimension")
