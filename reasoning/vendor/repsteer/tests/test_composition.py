import json
import math
from types import SimpleNamespace

import torch
import transformers
from torch import nn

from repsteer.artifacts import ArtifactMetadata, DirectionArtifact
from repsteer.core import Intervention, Site, SteeringPlan
from repsteer.models import from_model
from repsteer.models.hf.adapters import (
    ArchitectureAdapter,
    ResolvedSite,
    RootOrFirstTensorAccessor,
)
from repsteer.operators import Add, RemoveProjection, Replace
from repsteer.positions import AllTokens
from repsteer.runtime import diagnose_composition
from repsteer.schedules import Constant


def _artifact(values):
    return DirectionArtifact(
        ArtifactMetadata(
            model_id="tiny",
            model_revision="r1",
            site=Site("language", "resid_post", 0),
            method="unit",
        ),
        torch.tensor(values, dtype=torch.float32),
    )


def _control(values, operator=None, *, priority=0):
    return Intervention(
        artifact=_artifact(values),
        operator=operator or Add(),
        positions=AllTokens(),
        strength=Constant(1),
        priority=priority,
    )


def test_composition_reports_cosines_rank_condition_and_fusion_candidates():
    first = _control([1, 0], priority=10)
    second = _control([0, 2], priority=0)
    diagnostics = diagnose_composition(SteeringPlan([first, second]))

    assert diagnostics.execution_indices == (0, 1)
    assert diagnostics.direction_norms == (2.0, 1.0)
    assert diagnostics.pairwise_cosine == ((1.0, 0.0), (0.0, 1.0))
    assert diagnostics.additive_fusion_groups == ((0, 1),)
    assert diagnostics.effective_rank == 2
    assert math.isclose(diagnostics.condition_number, 2.0)
    assert diagnostics.orthogonal_basis.shape == (2, 2)


def test_composition_marks_non_commutative_same_site_order():
    plan = SteeringPlan(
        [
            _control([1, 0], Add()),
            _control([1, 0], RemoveProjection()),
        ]
    )

    diagnostics = diagnose_composition(plan)

    assert diagnostics.non_commutative_pairs == ((0, 1),)
    assert diagnostics.additive_fusion_groups == ()


def test_singular_composition_diagnostics_remain_strict_json():
    diagnostics = diagnose_composition(
        SteeringPlan([_control([1, 0]), _control([2, 0])])
    ).to_dict()

    payload = json.loads(json.dumps(diagnostics, allow_nan=False))
    assert payload["condition_number"] is None
    assert payload["condition_number_infinite"] is True


def test_composition_geometry_does_not_mix_different_sites():
    first = _control([1, 0])
    second = _control([0, 1]).with_site(Site("language", "resid_post", 1))

    diagnostics = diagnose_composition(SteeringPlan([first, second]))

    assert diagnostics.pairwise_cosine == ((1.0, None), (None, 1.0))
    assert diagnostics.effective_rank is None
    assert diagnostics.condition_number is None
    assert diagnostics.orthogonal_basis is None
    assert diagnostics.additive_fusion_groups == ()


def test_compiled_plan_exposes_machine_readable_composition_diagnostics():
    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=2,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=2,
    )
    wrapper = from_model(
        transformers.LlamaForCausalLM(config),
        model_id="tiny",
        revision="r1",
    )
    compiled = wrapper.compile(SteeringPlan([_control([1, 0]), _control([0, 1])]))

    assert compiled.diagnostics().effective_rank == 2
    assert compiled.to_dict()["composition"]["additive_fusion_groups"] == [[0, 1]]
    assert "additive fusion candidate" in compiled.explain()


def test_nonfinite_direction_never_leaks_nan_into_compiled_plan_json():
    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=2,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=2,
    )
    wrapper = from_model(
        transformers.LlamaForCausalLM(config),
        model_id="tiny",
        revision="r1",
    )
    compiled = wrapper.compile(_control([float("nan"), 0]))

    payload = compiled.explain(format="json")
    assert isinstance(payload, str)
    assert "NaN" not in payload
    assert json.loads(payload)["composition"]["direction_norms"] == [None]


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Identity()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            hidden_size=2,
            _name_or_path="tiny",
            _commit_hash="r1",
            architectures=["ToyModel"],
        )

    def forward(self, *, inputs_embeds):
        return self.block(inputs_embeds)


class _ToyAdapter(ArchitectureAdapter):
    architecture_name = "toy"

    def supports(self, model):
        return isinstance(model, _ToyModel)

    def hidden_size(self, model, site=None):
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


def test_runtime_preserves_declared_non_commutative_operator_order():
    wrapper = from_model(
        _ToyModel(),
        adapter=_ToyAdapter(),
        model_id="tiny",
        revision="r1",
    )
    add = _control([1, 0], Add(), priority=0)
    replace = _control([0, 0], operator=Replace(), priority=10)
    compiled = wrapper.compile(SteeringPlan([replace, add]))
    activation = torch.tensor([[[2.0, 0.0]]])

    with wrapper.steer(compiled):
        result = wrapper(inputs_embeds=activation)

    # Priority executes Add first: (2 + 1), then Replace(alpha=1) -> zero.
    assert torch.equal(result, torch.zeros_like(activation))
    conflicts = compiled.to_dict()["non_commutative_combinations"]
    assert [(item["first"], item["second"]) for item in conflicts] == [
        ("Add", "Replace")
    ]
