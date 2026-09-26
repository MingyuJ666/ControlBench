from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from repsteer.artifacts import (
    ArtifactMetadata,
    DirectionArtifact,
    ProbeArtifact,
    SubspaceArtifact,
    load_artifact,
)
from repsteer.artifacts.processor import processor_metadata
from repsteer.core import (
    ArtifactCompatibilityError,
    Site,
    SiteResolutionError,
    UnsafeArtifactError,
)
from repsteer.data import stable_fingerprint


class _Target:
    model_id = "tiny/model"
    revision = "rev-a"
    architecture = "TinyForCausalLM"
    hidden_size = 3


def _artifact() -> DirectionArtifact:
    return DirectionArtifact(
        ArtifactMetadata(
            model_id="tiny/model",
            model_revision="rev-a",
            architecture="TinyForCausalLM",
            site=Site("language", "resid_post", 0),
            hidden_size=3,
            method="diff_mean",
            dataset_fingerprint="sha256:data",
            seed=42,
        ),
        torch.tensor([1.0, -2.0, 3.0]),
    )


def test_artifact_save_load_roundtrip_is_lossless_and_non_pickle(tmp_path):
    artifact = _artifact()
    artifact.save(tmp_path)
    restored = load_artifact(tmp_path)

    assert isinstance(restored, DirectionArtifact)
    assert torch.equal(restored.direction, artifact.direction)
    assert restored.metadata.to_dict() == artifact.metadata.to_dict()
    assert restored.fingerprint() == artifact.fingerprint()
    assert {path.name for path in tmp_path.iterdir()} == {
        "checksums.json",
        "manifest.json",
        "tensors.safetensors",
    }


def test_artifact_checksum_and_revision_mismatch_fail_closed(tmp_path):
    artifact = _artifact()
    artifact.save(tmp_path)
    target = _Target()
    artifact.bind(target)

    target.revision = "rev-b"
    with pytest.raises(ArtifactCompatibilityError, match="model revision differs"):
        artifact.bind(target)

    tensors = tmp_path / "tensors.safetensors"
    payload = tensors.read_bytes()
    tensors.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(UnsafeArtifactError, match="Checksum mismatch"):
        load_artifact(tmp_path)


def test_artifact_container_is_frozen():
    artifact = _artifact()
    with pytest.raises(FrozenInstanceError):
        artifact.direction = torch.zeros(3)  # type: ignore[misc]


class _ResolvingTarget(_Target):
    def __init__(self, *, hidden_dim: int = 3) -> None:
        self.hidden_dim = hidden_dim
        self.resolved: list[Site] = []

    def resolve_site(self, site: Site):
        self.resolved.append(site)
        if site.component != "resid_post" or site.layer != 0:
            raise SiteResolutionError(f"cannot resolve {site}")
        return SimpleNamespace(site=site, hidden_dim=self.hidden_dim)


def test_bind_resolves_metadata_site_and_uses_its_hidden_dimension():
    artifact = _artifact()
    target = _ResolvingTarget()

    assert artifact.bind(target) is artifact
    assert target.resolved == [artifact.metadata.site]

    with pytest.raises(
        ArtifactCompatibilityError, match=r"hidden size differs \(3 != 4\)"
    ):
        artifact.bind(_ResolvingTarget(hidden_dim=4), compatibility="dimension")


@pytest.mark.parametrize(
    "site",
    [
        Site("language", "not_a_component", 0),
        Site("language", "resid_post", 99),
    ],
)
def test_bind_invalid_resolved_site_fails_before_identity_compatibility(site):
    artifact = _artifact().with_metadata(_artifact().metadata.with_updates(site=site))
    target = _ResolvingTarget()
    target.revision = "wrong-revision"

    with pytest.raises(SiteResolutionError, match="cannot resolve"):
        artifact.bind(target)


def test_artifact_metadata_dtype_must_match_primary_tensor():
    with pytest.raises(ValueError, match="metadata dtype 'float16'.*'float32'"):
        DirectionArtifact(
            ArtifactMetadata(dtype="torch.float16"),
            torch.ones(3, dtype=torch.float32),
        )

    artifact = DirectionArtifact(
        ArtifactMetadata(dtype="torch.float32"),
        torch.ones(3, dtype=torch.float32),
    )
    assert artifact.metadata.dtype == "float32"


def test_attached_tensors_follow_primary_dtype_and_device():
    basis = torch.ones((2, 3), dtype=torch.float64)
    subspace = SubspaceArtifact(
        ArtifactMetadata(),
        basis,
        mean=torch.ones(3, dtype=torch.float32),
        explained_variance=torch.ones(2, dtype=torch.float16),
    )
    assert subspace.mean is not None
    assert subspace.explained_variance is not None
    assert subspace.mean.dtype == basis.dtype
    assert subspace.explained_variance.dtype == basis.dtype
    assert subspace.mean.device == basis.device
    assert subspace.explained_variance.device == basis.device

    weight = torch.ones(3, dtype=torch.float16)
    probe = ProbeArtifact(
        ArtifactMetadata(),
        weight,
        bias=torch.tensor(1.0, dtype=torch.float64),
    )
    assert probe.bias is not None
    assert probe.bias.dtype == weight.dtype
    assert probe.bias.device == weight.device


def test_multimodal_metadata_roundtrip_and_processor_compatibility(tmp_path):
    artifact = _artifact().with_metadata(
        _artifact().metadata.with_updates(
            site=Site("vision", "vision_resid", 0),
            processor={
                "id": "tiny/processor",
                "revision": "processor-r1",
                "preprocess_fingerprint": "sha256:preprocess",
            },
            modality={
                "kind": "image",
                "modality_map_schema": "1.0",
                "image_indices": [0],
            },
        )
    )
    artifact.save(tmp_path)
    restored = load_artifact(tmp_path)

    assert restored.metadata.processor["revision"] == "processor-r1"
    assert restored.metadata.modality["modality_map_schema"] == "1.0"

    target = _Target()
    target.processor_id = "tiny/processor"
    target.processor_revision = "processor-r1"
    target.processor_preprocess_fingerprint = "sha256:preprocess"
    restored.bind(target)
    target.processor_preprocess_fingerprint = "sha256:different-preprocess"
    with pytest.raises(ArtifactCompatibilityError, match="preprocessing differs"):
        restored.bind(target)
    target.processor_preprocess_fingerprint = "sha256:preprocess"
    target.processor_revision = "processor-r2"
    with pytest.raises(ArtifactCompatibilityError, match="processor identity differs"):
        restored.bind(target)


def test_exact_processor_compatibility_includes_preprocessing_fingerprint():
    class ImageProcessor:
        def __init__(self, size):
            self.size = size
            self.image_mean = [0.5, 0.5, 0.5]

    class Processor:
        name_or_path = "tiny/processor"
        _commit_hash = "processor-r1"

        def __init__(self, size):
            self.image_processor = ImageProcessor(size)

    source = _Target()
    source.processor = Processor({"height": 224, "width": 224})
    source.processor_id = "tiny/processor"
    source.processor_revision = "processor-r1"
    artifact = _artifact().with_metadata(
        _artifact().metadata.with_updates(processor=processor_metadata(source))
    )
    target = _Target()
    target.processor = Processor({"height": 448, "width": 448})
    target.processor_id = "tiny/processor"
    target.processor_revision = "processor-r1"

    with pytest.raises(ArtifactCompatibilityError, match="preprocessing differs"):
        artifact.bind(target)


def test_processor_chat_template_participates_in_exact_compatibility():
    class Tokenizer:
        def __init__(self, template):
            self.chat_template = template

    class Processor:
        name_or_path = "tiny/processor"
        _commit_hash = "processor-r1"

        def __init__(self, template):
            self.tokenizer = Tokenizer(template)

    source = _Target()
    source.processor = Processor("<chat-v1>")
    source.processor_id = "tiny/processor"
    source.processor_revision = "processor-r1"
    artifact = _artifact().with_metadata(
        _artifact().metadata.with_updates(processor=processor_metadata(source))
    )
    target = _Target()
    target.processor = Processor("<chat-v2>")
    target.processor_id = "tiny/processor"
    target.processor_revision = "processor-r1"

    with pytest.raises(ArtifactCompatibilityError, match="preprocessing differs"):
        artifact.bind(target)


def test_text_tokenizer_chat_template_participates_in_exact_compatibility():
    class Tokenizer:
        def __init__(self, template):
            self.chat_template = template

    source = _Target()
    source.tokenizer = Tokenizer("<chat-v1>")
    template_hash = str(
        stable_fingerprint(source.tokenizer.chat_template)
    ).removeprefix("sha256:")
    artifact = _artifact().with_metadata(
        _artifact().metadata.with_updates(
            tokenizer={"chat_template_sha256": template_hash}
        )
    )
    target = _Target()
    target.tokenizer = Tokenizer("<chat-v1>")

    artifact.bind(target)
    target.tokenizer = Tokenizer("<chat-v2>")
    with pytest.raises(ArtifactCompatibilityError, match="chat template differs"):
        artifact.bind(target)
