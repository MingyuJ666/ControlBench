import torch

from repsteer.artifacts import ArtifactMetadata, DirectionArtifact, SubspaceArtifact
from repsteer.core import Intervention, Site, SteeringPlan, StepContext
from repsteer.gates import Always
from repsteer.operators import Add, RemoveProjection, Replace, Subtract
from repsteer.positions import AllTokens
from repsteer.schedules import Constant, NormRelative


def _context(activation):
    return StepContext(
        phase="forward",
        prompt_lengths=torch.tensor([activation.shape[-2]]),
        attention_mask=torch.ones(1, activation.shape[-2]),
    )


def _artifact(direction):
    return DirectionArtifact(
        ArtifactMetadata(
            model_id="tiny",
            model_revision="r1",
            site=Site("language", "resid_post", 0),
            method="unit",
        ),
        torch.tensor(direction, dtype=torch.float32),
    )


def test_add_subtract_replace_and_projection_math():
    activation = torch.tensor([[[3.0, 4.0], [1.0, -2.0]]])
    artifact = _artifact([1.0, 0.0])
    context = _context(activation)

    assert torch.equal(
        Add().apply(activation, artifact, torch.tensor(2.0), context),
        activation + torch.tensor([2.0, 0.0]),
    )
    assert torch.equal(
        Subtract().apply(activation, artifact, torch.tensor(2.0), context),
        activation - torch.tensor([2.0, 0.0]),
    )
    assert torch.equal(
        Replace().apply(activation, artifact, torch.tensor(1.0), context),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    )
    assert torch.allclose(
        RemoveProjection().apply(activation, artifact, torch.tensor(1.0), context),
        torch.tensor([[[0.0, 4.0], [0.0, -2.0]]]),
    )


def test_constant_and_norm_relative_strength_broadcast():
    activation = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
    context = _context(activation)

    assert Constant(1.5).value(activation, context).shape == torch.Size([])
    assert torch.equal(
        NormRelative(0.5).value(activation, context), torch.tensor([[2.5, 1.0]])
    )


def test_rank_one_pca_subspace_can_be_used_as_an_add_direction():
    activation = torch.zeros(1, 1, 2)
    metadata = ArtifactMetadata(
        model_id="tiny",
        model_revision="r1",
        site=Site("language", "resid_post", 0),
        method="pca",
    )
    rank_one = SubspaceArtifact(metadata, torch.tensor([[0.0, 1.0]]))

    changed = Add().apply(activation, rank_one, torch.tensor(2.0), _context(activation))

    assert torch.equal(changed, torch.tensor([[[0.0, 2.0]]]))


def test_plan_order_is_priority_then_declaration_order():
    artifact = _artifact([1.0, 0.0])

    def control(priority):
        return Intervention(
            artifact=artifact,
            operator=Add(),
            positions=AllTokens(),
            strength=Constant(1),
            gate=Always(),
            priority=priority,
        )

    first, second, third = control(10), control(0), control(10)
    assert SteeringPlan([first, second, third]).ordered() == (second, first, third)
