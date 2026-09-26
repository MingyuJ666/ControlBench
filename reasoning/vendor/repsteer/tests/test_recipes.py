from types import SimpleNamespace

import pytest
import torch
import transformers
from torch import nn

import repsteer as rs
from repsteer.artifacts import (
    ArtifactMetadata,
    DirectionArtifact,
    ProbeArtifact,
    SubspaceArtifact,
)
from repsteer.core import ArtifactCompatibilityError, PlanCompilationError, StepContext
from repsteer.gates import Always, CallableGate, ProbeGate
from repsteer.models import from_model
from repsteer.models.hf.adapters import (
    ArchitectureAdapter,
    ResolvedSite,
    RootOrFirstTensorAccessor,
)
from repsteer.operators import Add, RemoveProjection
from repsteer.positions import AllTokens, GeneratedTokens
from repsteer.sites import resid_post


def _context(activation: torch.Tensor) -> StepContext:
    return StepContext(
        phase="forward",
        prompt_lengths=torch.tensor([activation.shape[-2]]),
        attention_mask=torch.ones(1, activation.shape[-2]),
    )


def _direction(
    values: list[float],
    *,
    model_id: str = "tiny/recipes",
    revision: str = "r1",
    architecture: str | None = None,
    site=None,
) -> DirectionArtifact:
    return DirectionArtifact(
        ArtifactMetadata(
            model_id=model_id,
            model_revision=revision,
            architecture=architecture,
            site=site or resid_post(1),
            method="unit",
        ),
        torch.tensor(values, dtype=torch.float32),
    )


def test_directional_ablation_is_a_thin_plan_and_matches_direct_operator():
    artifact = _direction([1.0, 0.0, 0.0])
    plan = rs.recipes.directional_ablation(
        artifact,
        positions=AllTokens(),
        priority=-2,
    )
    activation = torch.tensor([[[3.0, 4.0, 5.0], [-2.0, 1.0, 7.0]]])
    context = _context(activation)
    item = plan[0]

    assert isinstance(plan, rs.SteeringPlan)
    assert isinstance(item.operator, RemoveProjection)
    assert item.priority == -2
    recipe_result = item.operator.apply(
        activation,
        artifact,
        item.strength.value(activation, context),
        context,
    )
    direct_result = RemoveProjection().apply(
        activation,
        artifact,
        torch.tensor(1.0),
        context,
    )

    assert torch.allclose(recipe_result, direct_result)
    assert torch.allclose(
        recipe_result,
        torch.tensor([[[0.0, 4.0, 5.0], [0.0, 1.0, 7.0]]]),
    )
    assert torch.equal(recipe_result[..., 1:], activation[..., 1:])


def test_directional_ablation_removes_a_multidimensional_subspace():
    artifact = SubspaceArtifact(
        ArtifactMetadata(
            model_id="tiny/recipes",
            model_revision="r1",
            site=resid_post(1),
            method="unit",
        ),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    activation = torch.tensor([[[3.0, -4.0, 5.0]]])
    context = _context(activation)
    plan = rs.recipes.directional_ablation(artifact, positions=AllTokens())

    result = plan[0].operator.apply(
        activation,
        artifact,
        plan[0].strength.value(activation, context),
        context,
    )

    assert torch.allclose(result, torch.tensor([[[0.0, 0.0, 5.0]]]))


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (_direction([0.0, 0.0]), "non-zero norm"),
        (_direction([float("nan"), 0.0]), "finite values"),
        (
            SubspaceArtifact(
                ArtifactMetadata(
                    model_id="tiny/recipes",
                    model_revision="r1",
                    site=resid_post(1),
                    method="unit",
                ),
                torch.empty(0, 2),
            ),
            "at least one non-empty basis vector",
        ),
        (
            SubspaceArtifact(
                ArtifactMetadata(
                    model_id="tiny/recipes",
                    model_revision="r1",
                    site=resid_post(1),
                    method="unit",
                ),
                torch.tensor([[float("inf"), 0.0]]),
            ),
            "finite values",
        ),
    ],
)
def test_directional_ablation_rejects_invalid_directional_artifacts(artifact, message):
    with pytest.raises(ValueError, match=message):
        rs.recipes.directional_ablation(artifact, positions=AllTokens())


def test_recipes_reject_bare_and_unsupported_direction_inputs():
    probe = ProbeArtifact(
        ArtifactMetadata(method="unit"),
        weight=torch.tensor([1.0, 0.0]),
    )

    with pytest.raises(TypeError, match="DirectionArtifact or SubspaceArtifact"):
        rs.recipes.directional_ablation(probe, positions=AllTokens())
    with pytest.raises(TypeError, match="artifact-backed direction"):
        rs.recipes.gated_direction(
            torch.tensor([1.0, 0.0]),
            gate=Always(),
            positions=AllTokens(),
        )
    with pytest.raises(ValueError, match="rank-1 SubspaceArtifact"):
        rs.recipes.gated_direction(
            SubspaceArtifact(
                ArtifactMetadata(method="unit"),
                torch.eye(2),
            ),
            gate=Always(),
            positions=AllTokens(),
        )


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Identity()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            hidden_size=2,
            _name_or_path="tiny/toy-recipes",
            _commit_hash="r1",
            architectures=["ToyRecipeModel"],
        )

    def forward(self, *, inputs_embeds):
        return self.block(inputs_embeds)


class _ToyAdapter(ArchitectureAdapter):
    architecture_name = "toy-recipes"

    def supports(self, model):
        return isinstance(model, _ToyModel)

    def hidden_size(self, model, site=None):
        del model, site
        return 2

    def resolve(self, model, site):
        return ResolvedSite(
            site=site,
            module=model.block,
            module_path="block",
            hook_kind="forward",
            tensor_accessor=RootOrFirstTensorAccessor(),
            hidden_dim=2,
            architecture_name=self.architecture_name,
        )


def _toy_wrapper():
    return from_model(
        _ToyModel(),
        adapter=_ToyAdapter(),
        model_id="tiny/toy-recipes",
        revision="r1",
    )


def _wrapper_artifact(wrapper, values: list[float], **metadata):
    return DirectionArtifact(
        ArtifactMetadata(
            model_id=metadata.pop("model_id", wrapper.model_id),
            model_revision=metadata.pop("model_revision", wrapper.revision),
            architecture=metadata.pop("architecture", wrapper.architecture),
            site=metadata.pop("site", resid_post(1)),
            method="unit",
            **metadata,
        ),
        torch.tensor(values, dtype=torch.float32),
    )


def test_gated_direction_applies_true_false_and_threshold_boundary_per_sample():
    wrapper = _toy_wrapper()
    artifact = _wrapper_artifact(wrapper, [1.0, 0.0], site=resid_post(0))
    plan = rs.recipes.gated_direction(
        artifact,
        gate=CallableGate(lambda _: torch.tensor([True, False])),
        positions=AllTokens(),
        strength=2.0,
    )

    assert isinstance(plan, rs.SteeringPlan)
    assert isinstance(plan[0].operator, Add)
    with wrapper.steer(plan):
        result = wrapper(inputs_embeds=torch.zeros(2, 1, 2))

    assert torch.equal(
        result,
        torch.tensor([[[2.0, 0.0]], [[0.0, 0.0]]]),
    )

    probe = ProbeArtifact(
        artifact.metadata.with_updates(artifact_type=""),
        weight=torch.tensor([1.0, 0.0]),
        bias=0.0,
    )
    boundary_plan = rs.recipes.gated_direction(
        artifact,
        gate=ProbeGate(probe, threshold=0.5),
        positions=AllTokens(),
    )
    with wrapper.steer(boundary_plan):
        boundary = wrapper(inputs_embeds=torch.zeros(1, 1, 2))

    # sigmoid(0) == threshold: ProbeGate's documented >= boundary applies.
    assert torch.equal(boundary, torch.tensor([[[1.0, 0.0]]]))


def _hf_wrapper():
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return from_model(
        transformers.LlamaForCausalLM(config).eval(),
        model_id="tiny/recipe-generation",
        revision="r1",
    )


class _PerSampleSequenceGate:
    def __init__(self, values: list[bool]) -> None:
        self.evaluate_at = resid_post(0)
        self.values = values
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def evaluate(self, context):
        activation = context.metadata["activation"]
        self.calls.append((context.phase, tuple(activation.shape)))
        return torch.tensor(self.values, dtype=torch.bool, device=activation.device)

    def to_dict(self):
        return {
            "type": "per_sample_sequence",
            "evaluate_at": self.evaluate_at.to_dict(),
        }


def test_cached_sequence_gate_rejects_non_language_target_without_row_mapping():
    wrapper = _toy_wrapper()
    plan = rs.recipes.gated_direction(
        _wrapper_artifact(
            wrapper,
            [1.0, 0.0],
            site=rs.sites.vision_resid(0),
        ),
        gate=_PerSampleSequenceGate([True]),
        positions=AllTokens(),
    )

    with pytest.raises(PlanCompilationError, match="only language-stream targets"):
        wrapper.compile(plan)


def test_gated_direction_compiles_and_reuses_prefill_decisions_for_generation():
    wrapper = _hf_wrapper()
    gate = _PerSampleSequenceGate([True, False])
    plan = rs.recipes.gated_direction(
        _wrapper_artifact(wrapper, [1.0] * wrapper.hidden_size),
        gate=gate,
        positions=GeneratedTokens(),
        strength=0.1,
        phase="decode",
    )
    compiled = wrapper.compile(plan)
    inputs = torch.tensor([[1, 4, 5], [1, 6, 7]])
    options = {
        "attention_mask": torch.ones_like(inputs),
        "min_new_tokens": 3,
        "max_new_tokens": 3,
        "use_cache": True,
    }

    with wrapper.steer(compiled):
        greedy = wrapper.generate(inputs, do_sample=False, **options)
        sampled = wrapper.generate(inputs, do_sample=True, seed=123, **options)

    assert greedy.token_ids.shape[0] == 2
    assert sampled.token_ids.shape[0] == 2
    assert gate.calls == [
        ("prefill", (2, 3, wrapper.hidden_size)),
        ("prefill", (2, 3, wrapper.hidden_size)),
    ]


def test_recipes_compile_and_reject_model_dimension_architecture_and_site_mismatch():
    wrapper = _hf_wrapper()
    valid = _wrapper_artifact(wrapper, [1.0] * wrapper.hidden_size)
    plan = rs.recipes.directional_ablation(valid, positions=AllTokens())

    assert len(wrapper.compile(plan)) == 1

    wrong_width = _direction(
        [1.0] * (wrapper.hidden_size - 1),
        model_id=wrapper.model_id,
        revision=wrapper.revision,
        architecture=wrapper.architecture,
    )
    with pytest.raises(ArtifactCompatibilityError, match="hidden size differs"):
        wrapper.compile(
            rs.recipes.directional_ablation(wrong_width, positions=AllTokens())
        )

    wrong_model = _wrapper_artifact(wrapper, [1.0] * wrapper.hidden_size)
    wrong_model = DirectionArtifact(
        wrong_model.metadata.with_updates(model_revision="other"),
        wrong_model.direction,
    )
    with pytest.raises(ArtifactCompatibilityError, match="model revision differs"):
        wrapper.compile(
            rs.recipes.directional_ablation(wrong_model, positions=AllTokens())
        )

    wrong_architecture = _wrapper_artifact(
        wrapper,
        [1.0] * wrapper.hidden_size,
        architecture="OtherArchitecture",
    )
    with pytest.raises(ArtifactCompatibilityError, match="architecture differs"):
        wrapper.compile(
            rs.recipes.directional_ablation(
                wrong_architecture,
                positions=AllTokens(),
            )
        )

    with pytest.raises(PlanCompilationError, match="out of range"):
        wrapper.compile(
            rs.recipes.directional_ablation(
                valid,
                site=resid_post(99),
                positions=AllTokens(),
            )
        )
