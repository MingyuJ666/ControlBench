from types import SimpleNamespace

import pytest
import torch
import transformers

import repsteer as rs
from repsteer.artifacts import (
    ArtifactBundle,
    ArtifactBundleComponent,
    ArtifactMetadata,
    ITIArtifact,
    ITIHead,
    load_artifact,
)
from repsteer.capture import ActivationBatch
from repsteer.data import ContrastivePairs
from repsteer.learners import ITI
from repsteer.models import from_model
from repsteer.models.hf.adapters import AdapterCapabilities, HeadResultCapability
from repsteer.operators import ITIAdd
from repsteer.positions import AllTokens, GeneratedTokens
from repsteer.sites import head_result


class _ITIProbeModel:
    """Deterministic head-result source for learner-only tests."""

    model_id = "tiny/iti-probe"
    revision = "r1"
    architecture = "TinyITI"
    config = SimpleNamespace(num_hidden_layers=2, _commit_hash="r1")
    adapter = SimpleNamespace(
        capabilities=AdapterCapabilities(
            adapter_name="tiny-iti",
            head_result=HeadResultCapability(),
        )
    )

    def __init__(self) -> None:
        self.values = {
            "p1": torch.tensor([2.0, 0.0, 0.1, 1.0]),
            "p2": torch.tensor([3.0, 0.1, -0.1, 1.1]),
            "p3": torch.tensor([4.0, -0.1, 0.0, 0.9]),
            "n1": torch.tensor([-2.0, 0.0, 0.0, 0.0]),
            "n2": torch.tensor([-3.0, -0.1, 0.2, -0.1]),
            "n3": torch.tensor([-4.0, 0.1, -0.2, 0.1]),
        }

    def resolve_site(self, site):
        assert site.component == "head_result"
        return SimpleNamespace(site=site, hidden_dim=2)

    def capture(self, request):
        layer_shift = float(request.site.layer)
        values = torch.stack(
            [
                self.values[value] + torch.tensor([0.0, layer_shift, 0.0, 0.0])
                for value in request.inputs
            ]
        )
        return ActivationBatch(values, request=request)


def _iti_data():
    return ContrastivePairs.from_pairs(
        positives=("p1", "p2", "p3"), negatives=("n1", "n2", "n3")
    )


def test_iti_learner_selects_held_out_head_and_stores_calibrated_profile(tmp_path):
    artifact = ITI(top_k=1, validation_fraction=1 / 3, seed=7).fit(
        _ITIProbeModel(), _iti_data()
    )

    assert isinstance(artifact, ITIArtifact)
    assert artifact.metadata.artifact_type == "iti"
    assert artifact.metadata.site is None
    assert artifact.metadata.config["head_surface"] == "self_attn.o_proj.input"
    assert len(artifact.heads) == 1
    assert artifact.heads[0].head == 0
    assert artifact.heads[0].validation_accuracy == 1.0
    assert artifact.heads[0].projected_std > 0
    assert torch.allclose(artifact.directions[0].abs(), torch.tensor([1.0, 0.0]))

    artifact.save(tmp_path)
    restored = load_artifact(tmp_path)
    assert isinstance(restored, ITIArtifact)
    assert restored.heads == artifact.heads
    assert torch.equal(restored.directions, artifact.directions)


def test_iti_operator_changes_only_the_profile_selected_query_head():
    artifact = ITIArtifact(
        ArtifactMetadata(method="iti"),
        torch.tensor(
            [
                [2.0**-0.5, -(2.0**-0.5)],
                [2.0 / (5.0**0.5), 1.0 / (5.0**0.5)],
            ]
        ),
        (
            ITIHead(0, 2, 0, 0.9, 2.0),
            ITIHead(1, 2, 2, 0.8, 3.0),
        ),
    )
    activation = torch.zeros(1, 2, 6)
    result = ITIAdd(layer=2).apply(
        activation,
        artifact,
        strength=0.5,
        context=rs.StepContext(
            phase="prefill",
            attention_mask=torch.ones(1, 2),
            prompt_lengths=torch.tensor([2]),
        ),
    )

    heads = result.reshape(1, 2, 3, 2)
    assert torch.equal(heads[..., 1, :], torch.zeros(1, 2, 2))
    assert torch.allclose(
        heads[..., 0, :],
        torch.tensor([[[2.0**-0.5, -(2.0**-0.5)], [2.0**-0.5, -(2.0**-0.5)]]]),
    )
    assert torch.allclose(
        heads[..., 2, :],
        torch.tensor(
            [
                [
                    [3.0 / (5.0**0.5), 1.5 / (5.0**0.5)],
                    [3.0 / (5.0**0.5), 1.5 / (5.0**0.5)],
                ]
            ]
        ),
    )


def _hf_wrapper():
    raw = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
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
    ).eval()
    return from_model(raw, model_id="tiny/iti-runtime", revision="r1")


def test_iti_recipe_edits_o_proj_input_not_post_attention_residual():
    model = _hf_wrapper()
    artifact = ITIArtifact(
        ArtifactMetadata(
            model_id=model.model_id,
            model_revision=model.revision,
            architecture=model.architecture,
            method="iti",
        ),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        (ITIHead(0, 0, 1, 0.9, 2.0),),
    )
    resolved = model.resolve_site(head_result(0))
    baseline: list[torch.Tensor] = []
    baseline_hook = resolved.module.register_forward_pre_hook(
        lambda _module, args: baseline.append(args[0].detach().clone())
    )
    try:
        model(input_ids=torch.tensor([[1, 4, 5]]))
    finally:
        baseline_hook.remove()

    plan = rs.recipes.iti(artifact, positions=AllTokens(), phase="both", strength=1.0)
    compiled = model.compile(plan)
    assert compiled.interventions[0].resolved_site.module is resolved.module
    assert compiled.interventions[0].resolved_site.hook_kind == "forward_pre"
    assert compiled.interventions[0].resolved_site.module_path.endswith(
        "self_attn.o_proj"
    )
    assert artifact.bind(model) is artifact

    edited: list[torch.Tensor] = []
    with model.steer(compiled):
        # Register after the runtime hook so this observer sees its replacement.
        observer = resolved.module.register_forward_pre_hook(
            lambda _module, args: edited.append(args[0].detach().clone())
        )
        try:
            model(input_ids=torch.tensor([[1, 4, 5]]))
        finally:
            observer.remove()

    delta = (edited[0] - baseline[0]).reshape(1, 3, 4, 4)
    assert torch.allclose(delta[..., 0, :], torch.zeros_like(delta[..., 0, :]))
    assert torch.allclose(delta[..., 2:, :], torch.zeros_like(delta[..., 2:, :]))
    assert torch.allclose(delta[..., 1, 0], torch.full((1, 3), 2.0))
    assert torch.allclose(delta[..., 1, 1:], torch.zeros_like(delta[..., 1, 1:]))


def test_iti_recipe_defaults_to_decode_generated_tokens():
    artifact = ITIArtifact(
        ArtifactMetadata(method="iti"),
        torch.tensor([[1.0, 0.0]]),
        (ITIHead(0, 1, 0, 0.9, 1.0),),
    )

    plan = rs.recipes.iti(artifact)

    assert len(plan) == 1
    assert isinstance(plan[0].positions, GeneratedTokens)
    assert plan[0].phase == "decode"
    assert plan[0].site == head_result(1)


def test_iti_profile_is_a_safe_artifact_bundle_component(tmp_path):
    model = _hf_wrapper()
    artifact = ITIArtifact(
        ArtifactMetadata(
            model_id=model.model_id,
            model_revision=model.revision,
            architecture=model.architecture,
            method="iti",
        ),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        (ITIHead(0, 0, 0, 0.9, 1.0),),
    )
    bundle = ArtifactBundle(
        components=(ArtifactBundleComponent("truthfulness", "iti", artifact),),
        provenance={"fixture": "iti"},
    )

    bundle.save(tmp_path)
    restored = ArtifactBundle.load(tmp_path)

    assert isinstance(restored.component("truthfulness").artifact, ITIArtifact)
    assert restored.bind(model) is restored


def test_iti_rejects_more_heads_than_the_profile_can_produce():
    with pytest.raises(ValueError, match="exceeds the 4 usable heads"):
        ITI(top_k=5, validation_fraction=1 / 3).fit(_ITIProbeModel(), _iti_data())
