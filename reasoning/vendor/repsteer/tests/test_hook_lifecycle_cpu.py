from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from repsteer.artifacts import ArtifactMetadata, DirectionArtifact
from repsteer.core import HookLifecycleError, Intervention, Site, SteeringPlan
from repsteer.models import from_model
from repsteer.models.hf.adapters import (
    ArchitectureAdapter,
    ResolvedSite,
    RootOrFirstTensorAccessor,
)
from repsteer.operators import Add
from repsteer.positions import AllTokens
from repsteer.schedules import Constant


class _TwoBlockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Identity()
        self.second = nn.Identity()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            hidden_size=2,
            _name_or_path="tiny/hook-lifecycle",
            _commit_hash="r1",
            architectures=["TinyHookLifecycleModel"],
        )

    def forward(self, *, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(inputs_embeds))


class _TwoBlockAdapter(ArchitectureAdapter):
    architecture_name = "tiny-hook-lifecycle"

    def supports(self, model: nn.Module) -> bool:
        return isinstance(model, _TwoBlockModel)

    def hidden_size(self, model: nn.Module, site: Site | None = None) -> int:
        del model, site
        return 2

    def resolve(self, model: nn.Module, site: Site) -> ResolvedSite:
        if not isinstance(model, _TwoBlockModel):
            raise TypeError("_TwoBlockAdapter requires a _TwoBlockModel")
        modules = {
            "first": (model.first, "first"),
            "second": (model.second, "second"),
        }
        try:
            module, module_path = modules[site.component]
        except KeyError as exc:
            raise ValueError(f"unsupported test site {site}") from exc
        return ResolvedSite(
            site=site,
            module=module,
            module_path=module_path,
            hook_kind="forward",
            tensor_accessor=RootOrFirstTensorAccessor(),
            hidden_dim=2,
            architecture_name=self.architecture_name,
        )


def _wrapper():
    return from_model(
        _TwoBlockModel(),
        adapter=_TwoBlockAdapter(),
        model_id="tiny/hook-lifecycle",
        revision="r1",
    )


def _control(wrapper, component: str, values: list[float]) -> Intervention:
    site = Site("language", component, 0)
    artifact = DirectionArtifact(
        ArtifactMetadata(
            model_id=wrapper.model_id,
            model_revision=wrapper.revision,
            architecture=wrapper.architecture,
            site=site,
            hidden_size=2,
            method="unit",
        ),
        torch.tensor(values),
    )
    return Intervention(
        artifact=artifact,
        operator=Add(),
        positions=AllTokens(),
        strength=Constant(1),
        phase="both",
    )


def test_nested_same_compiled_plan_is_idempotent_and_cleans_up():
    wrapper = _wrapper()
    compiled = wrapper.compile(_control(wrapper, "first", [1.0, 0.0]))

    with wrapper.steer(compiled):
        outer_hook_count = wrapper.hook_manager.hook_count
        with wrapper.steer(compiled):
            assert wrapper.hook_manager.depth == 2
            assert wrapper.hook_manager.hook_count == outer_hook_count
            output = wrapper(inputs_embeds=torch.zeros(1, 1, 2))
        assert wrapper.hook_manager.depth == 1
        assert wrapper.hook_manager.hook_count == outer_hook_count

    assert torch.equal(output, torch.tensor([[[1.0, 0.0]]]))
    assert wrapper.hook_manager.hook_count == 0
    assert not wrapper.hook_manager.active


def test_disjoint_nested_plans_apply_in_order_and_overlaps_fail_closed():
    wrapper = _wrapper()
    first = wrapper.compile(_control(wrapper, "first", [1.0, 0.0]))
    second = wrapper.compile(_control(wrapper, "second", [0.0, 2.0]))
    overlap = wrapper.compile(_control(wrapper, "first", [3.0, 0.0]))

    with wrapper.steer(first):
        with wrapper.steer(second):
            output = wrapper(inputs_embeds=torch.zeros(1, 1, 2))
        with (
            pytest.raises(HookLifecycleError, match="overlapping mutable tensors"),
            wrapper.steer(overlap),
        ):
            pass
        assert wrapper.hook_manager.depth == 1

    assert torch.equal(output, torch.tensor([[[1.0, 2.0]]]))
    assert wrapper.hook_manager.hook_count == 0


def test_hook_registration_failure_rolls_back_prior_hooks(monkeypatch):
    wrapper = _wrapper()
    compiled = wrapper.compile(
        SteeringPlan(
            [
                _control(wrapper, "first", [1.0, 0.0]),
                _control(wrapper, "second", [0.0, 1.0]),
            ]
        )
    )

    def fail_registration(_callback):
        raise RuntimeError("injected registration failure")

    monkeypatch.setattr(
        wrapper.model.second,
        "register_forward_hook",
        fail_registration,
    )

    with (
        pytest.raises(HookLifecycleError, match="injected registration failure"),
        wrapper.steer(compiled),
    ):
        pass

    assert wrapper.hook_manager.hook_count == 0
    assert not wrapper.model.first._forward_hooks
    assert not wrapper.model._forward_pre_hooks


def test_foreign_thread_cannot_use_an_active_steering_session():
    wrapper = _wrapper()
    compiled = wrapper.compile(_control(wrapper, "first", [1.0, 0.0]))
    failures: list[BaseException] = []

    def run_forward() -> None:
        try:
            wrapper(inputs_embeds=torch.zeros(1, 1, 2))
        except BaseException as exc:  # The asserted public error is captured below.
            failures.append(exc)

    with wrapper.steer(compiled):
        thread = threading.Thread(target=run_forward)
        thread.start()
        thread.join()

    assert len(failures) == 1
    assert isinstance(failures[0], HookLifecycleError)
    assert "another thread" in str(failures[0])
    assert wrapper.hook_manager.hook_count == 0
