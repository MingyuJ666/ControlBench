import json
from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers

from repsteer.artifacts import ArtifactMetadata, SAEFeatureArtifact, load_artifact
from repsteer.capture import ActivationBatch
from repsteer.core import Intervention, StepContext
from repsteer.data import ContrastivePairs
from repsteer.models import from_model
from repsteer.operators import Ablate, Add, Clamp, SAEAblate, SAEClamp
from repsteer.positions import AllTokens, LastNonPaddingToken
from repsteer.sae import (
    FunctionalSAEAdapter,
    SAELensAdapter,
    feature_artifact,
    select_feature,
    select_supervised_features,
)
from repsteer.schedules import Constant
from repsteer.sites import mlp_out, resid_post, resid_pre


def _context(batch: int = 1, sequence: int = 1) -> StepContext:
    return StepContext(
        phase="forward",
        prompt_lengths=torch.full((batch,), sequence),
        attention_mask=torch.ones(batch, sequence),
    )


def _adapter() -> FunctionalSAEAdapter:
    encoder = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    decoder = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    return FunctionalSAEAdapter(
        input_dim=2,
        num_features=3,
        encode_fn=lambda x: x @ encoder,
        decode_fn=lambda z: z @ decoder + torch.tensor([7.0, -3.0]),
        decoder_direction_fn=lambda feature_id: decoder[feature_id],
    )


def _adapter_at(site) -> FunctionalSAEAdapter:
    encoder = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    decoder = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    return FunctionalSAEAdapter(
        input_dim=2,
        num_features=3,
        encode_fn=lambda x: x @ encoder,
        decode_fn=lambda z: z @ decoder,
        decoder_direction_fn=lambda feature_id: decoder[feature_id],
        site=site,
    )


class _TinyCaptureModel:
    model_id = "tiny/sae-capture"
    revision = "model-revision"
    architecture = "TinyForCausalLM"
    tokenizer = SimpleNamespace(
        name_or_path="tiny/tokenizer",
        _commit_hash="tokenizer-revision",
    )

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.values = {
            "positive one": torch.tensor([5.0, 0.0]),
            "positive two": torch.tensor([4.0, 1.0]),
            "negative one": torch.tensor([0.0, 3.0]),
            "negative two": torch.tensor([0.0, 2.0]),
        }

    def capture(self, request):
        self.requests.append(request)
        # A real wrapper has already applied request.positions before returning
        # its ActivationBatch. Keep a singleton sequence axis to exercise the
        # normal identity-pooling path used by CaptureRunner.
        values = torch.stack([self.values[value] for value in request.inputs])
        return ActivationBatch(
            values[:, None, :],
            request=request,
            sample_weights=request.sample_weights,
        )


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "</s>"
    pad_token = "<pad>"
    padding_side = "right"
    name_or_path = "tiny/sae-tokenizer"
    _commit_hash = "tokenizer-revision"

    vocabulary = {
        "kind words": [1, 8, 9],
        "help people": [1, 10, 11],
        "cruel words": [1, 12, 9],
        "hurt people": [1, 13, 11],
    }

    def __call__(self, texts, *, return_tensors="pt", padding=False, **_kwargs):
        assert return_tensors == "pt"
        values = [texts] if isinstance(texts, str) else list(texts)
        rows = [self.vocabulary[value] for value in values]
        width = max(map(len, rows))
        ids = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }


def _tiny_llama_wrapper():
    torch.manual_seed(7)
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    raw = transformers.LlamaForCausalLM(config).eval()
    return from_model(
        raw,
        _TinyTokenizer(),
        model_id="tiny/sae-llama",
        revision="model-revision",
    )


def test_saelens_adapter_exposes_stable_protocol_without_importing_saelens():
    encoder = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    decoder = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])

    class FakeSAE:
        cfg = SimpleNamespace(d_in=2, d_sae=3)
        W_dec = decoder

        def encode(self, x):
            return x @ encoder

        def decode(self, z):
            return z @ decoder

    adapter = SAELensAdapter(FakeSAE(), release="test-release", sae_id="layer_0")
    values = torch.tensor([[2.0, 3.0]])

    assert adapter.input_dim == 2
    assert adapter.num_features == 3
    assert torch.equal(adapter.encode(values), values @ encoder)
    assert torch.equal(
        adapter.decode(adapter.encode(values)), values @ encoder @ decoder
    )
    assert torch.equal(adapter.decoder_direction(2), decoder[2])


@pytest.mark.parametrize(
    ("hook_name", "expected_site"),
    [
        ("blocks.7.hook_resid_post", resid_post(7)),
        ("blocks.3.hook_mlp_out", mlp_out(3)),
    ],
)
def test_saelens_adapter_infers_semantic_site_from_common_hook_names(
    hook_name, expected_site
):
    class FakeSAE:
        cfg = SimpleNamespace(d_in=2, d_sae=3, hook_name=hook_name)
        W_dec = torch.ones(3, 2)

        def encode(self, x):
            return torch.zeros(*x.shape[:-1], 3)

        def decode(self, z):
            return torch.zeros(*z.shape[:-1], 2)

    assert SAELensAdapter(FakeSAE()).site == expected_site


def test_saelens_adapter_explicit_site_overrides_hook_name_inference():
    class FakeSAE:
        cfg = SimpleNamespace(
            d_in=2,
            d_sae=3,
            hook_name="blocks.7.hook_resid_post",
        )
        W_dec = torch.ones(3, 2)

        def encode(self, x):
            return torch.zeros(*x.shape[:-1], 3)

        def decode(self, z):
            return torch.zeros(*z.shape[:-1], 2)

    explicit = resid_pre(1)
    assert SAELensAdapter(FakeSAE(), site=explicit).site == explicit


def test_functional_sae_adapter_accepts_an_optional_site():
    assert _adapter().site is None
    assert _adapter_at(resid_post(4)).site == resid_post(4)


def test_sae_feature_artifact_add_direction_and_safe_roundtrip(tmp_path):
    artifact = feature_artifact(
        _adapter(),
        2,
        metadata=ArtifactMetadata(model_id="tiny", method="test"),
        score=1.25,
        criterion="supervised",
    )

    assert isinstance(artifact, SAEFeatureArtifact)
    assert artifact.feature_id == 2
    assert torch.equal(artifact.direction, torch.tensor([1.0, -1.0]))
    assert artifact.metadata.config["sae_num_features"] == 3
    activation = torch.zeros(1, 1, 2)
    assert torch.equal(
        Add().apply(activation, artifact, 2.0, _context()),
        torch.tensor([[[2.0, -2.0]]]),
    )

    artifact.save(tmp_path)
    restored = load_artifact(tmp_path)
    assert isinstance(restored, SAEFeatureArtifact)
    assert restored.feature_id == artifact.feature_id
    assert restored.score == artifact.score
    assert torch.equal(restored.decoder_direction, artifact.decoder_direction)
    assert restored.fingerprint() == artifact.fingerprint()


def test_latent_clamp_and_ablate_only_change_selected_feature():
    artifact = feature_artifact(_adapter(), 1)
    latents = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    context = _context(sequence=2)

    clamped = Clamp(10.0).apply(latents, artifact, 0.5, context)
    ablated = Ablate().apply(latents, artifact, 1.0, context)

    assert torch.equal(
        clamped,
        torch.tensor([[[1.0, 6.0, 3.0], [4.0, 7.5, 6.0]]]),
    )
    assert torch.equal(
        ablated,
        torch.tensor([[[1.0, 0.0, 3.0], [4.0, 0.0, 6.0]]]),
    )


def test_residual_sae_clamp_and_ablate_preserve_reconstruction_residual():
    sae = _adapter()
    artifact = feature_artifact(sae, 0)
    activation = torch.tensor([[[1.0, 2.0]]])
    context = _context()

    unchanged = SAEClamp(5.0, sae=sae).apply(activation, artifact, 0.0, context)
    halfway = SAEClamp(5.0, sae=sae).apply(activation, artifact, 0.5, context)
    clamped = SAEClamp(5.0, sae=sae).apply(activation, artifact, 1.0, context)
    ablated = SAEAblate(sae=sae).apply(activation, artifact, 1.0, context)

    # Decoder direction 0 is [2, 0]. Moving z0 from 1 to 5 therefore adds
    # [8, 0] to h, independent of the decoder bias/reconstruction error.
    assert torch.equal(unchanged, activation)
    assert torch.equal(halfway, torch.tensor([[[5.0, 2.0]]]))
    assert torch.equal(clamped, torch.tensor([[[9.0, 2.0]]]))
    assert torch.equal(ablated, torch.tensor([[[-1.0, 2.0]]]))


def test_supervised_shortlist_then_causal_selection():
    sae = _adapter()
    encoded = torch.tensor(
        [
            [5.0, 4.0, 0.0],
            [4.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 2.0],
        ]
    )
    labels = torch.tensor([1, 1, 0, 0])

    supervised = select_supervised_features(
        sae,
        encoded,
        labels,
        encoded=True,
        top_k=2,
    )
    assert [item.feature_id for item in supervised] == [0, 1]

    selected = select_feature(
        sae=sae,
        criterion="causal_effect",
        activations=encoded,
        labels=labels,
        encoded=True,
        top_k=2,
        evaluator={0: 0.25, 1: 1.5},
    )
    assert selected.feature_id == 1
    assert selected.score == 1.5
    assert selected.metadata.config["selection_criterion"] == "causal_effect"


def test_select_feature_captures_contrastive_data_with_default_position_and_metadata():
    model = _TinyCaptureModel()
    data = ContrastivePairs.from_records(
        [
            {
                "positive": "positive one",
                "negative": "negative one",
                "weight": 2.0,
            },
            {
                "positive": "positive two",
                "negative": "negative two",
                "positive_weight": 3.0,
                "negative_weight": 4.0,
            },
        ],
        metadata={"source": "sae-unit-test"},
    )
    site = resid_post(5)

    artifact = select_feature(
        sae=_adapter_at(site),
        model=model,
        data=data,
        criterion="supervised",
    )

    assert artifact.feature_id == 0
    assert len(model.requests) == 2
    positive, negative = model.requests
    assert positive.inputs == data.positives
    assert negative.inputs == data.negatives
    assert positive.site == negative.site == site
    assert isinstance(positive.positions, LastNonPaddingToken)
    assert isinstance(negative.positions, LastNonPaddingToken)
    assert positive.sample_weights == (2.0, 3.0)
    assert negative.sample_weights == (2.0, 4.0)
    assert positive.metadata["contrastive_side"] == "positive"
    assert negative.metadata["contrastive_side"] == "negative"
    assert positive.dataset_fingerprint == str(data.fingerprint)
    assert negative.dataset_fingerprint == str(data.fingerprint)

    metadata = artifact.metadata
    assert metadata.model_id == model.model_id
    assert metadata.model_revision == model.revision
    assert metadata.architecture == model.architecture
    assert metadata.tokenizer["id"] == model.tokenizer.name_or_path
    assert metadata.tokenizer["revision"] == model.tokenizer._commit_hash
    assert metadata.site == site
    assert metadata.hidden_size == 2
    assert metadata.dataset_fingerprint == str(data.fingerprint)
    assert metadata.seed == 42
    assert metadata.provenance["dataset"]["positive_samples"] == 2
    assert metadata.provenance["dataset"]["negative_samples"] == 2
    assert metadata.provenance["capture"]["positions"] == {
        "type": "last_non_padding_token"
    }


def test_select_feature_keeps_explicit_activations_api_and_artifact_is_data_only():
    sae = _adapter_at(resid_post(2))
    activations = torch.tensor(
        [
            [4.0, 0.0],
            [3.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ]
    )
    labels = torch.tensor([1, 1, 0, 0])

    artifact = select_feature(
        sae=sae,
        activations=activations,
        labels=labels,
        criterion="supervised",
    )

    assert artifact.feature_id == 0
    assert "sae" not in {field.name for field in fields(artifact)}
    assert set(artifact.tensors()) == {"decoder_direction"}
    assert all(value is not sae for value in artifact.tensors().values())
    # The manifest remains JSON-only: adapter identity may be recorded as a
    # string, but no executable SAE object crosses the artifact boundary.
    json.dumps(artifact.metadata.to_manifest())


def test_select_feature_real_hf_capture_compiles_and_applies_decoder_add():
    wrapper = _tiny_llama_wrapper()
    site = resid_post(0)
    directions = torch.zeros(3, wrapper.hidden_size)
    directions[0, 0] = 1.0
    directions[1, 0] = -1.0
    directions[2, 1] = 1.0
    sae = FunctionalSAEAdapter(
        input_dim=wrapper.hidden_size,
        num_features=3,
        encode_fn=lambda x: torch.stack(
            (x[..., 0], -x[..., 0], torch.zeros_like(x[..., 0])),
            dim=-1,
        ),
        decode_fn=lambda z: z @ directions,
        decoder_direction_fn=lambda feature_id: directions[feature_id],
        site=site,
    )
    data = ContrastivePairs.from_records(
        [
            {"positive": "kind words", "negative": "cruel words"},
            {"positive": "help people", "negative": "hurt people"},
        ]
    )

    artifact = select_feature(
        sae=sae,
        model=wrapper,
        data=data,
        criterion="supervised",
    )

    assert artifact.feature_id in (0, 1)
    assert artifact.metadata.model_id == wrapper.model_id
    assert artifact.metadata.model_revision == wrapper.revision
    assert artifact.metadata.site == site
    assert artifact.metadata.dataset_fingerprint == str(data.fingerprint)
    assert artifact.metadata.provenance["capture"]["positions"] == {
        "type": "last_non_padding_token"
    }

    control = Intervention(
        artifact=artifact,
        operator=Add(),
        positions=AllTokens(),
        strength=Constant(3.0),
        phase="both",
    )
    compiled = wrapper.compile(control)
    encoded = wrapper.tokenizer(["kind words"], return_tensors="pt", padding=True)
    with torch.no_grad():
        baseline = wrapper(**encoded).logits
        with wrapper.steer(compiled):
            steered = wrapper(**encoded).logits

    assert not torch.allclose(steered, baseline)
    assert wrapper.hook_manager.hook_count == 0
