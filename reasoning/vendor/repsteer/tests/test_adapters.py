from typing import Any

import pytest
import torch
import transformers

from repsteer import sites
from repsteer.core import Site
from repsteer.core.errors import SiteResolutionError
from repsteer.models.hf.adapters import PathTensorAccessor, Qwen3Adapter, get_adapter


def _tiny_model(family: str):
    common: dict[str, Any] = dict(
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
    if family == "llama":
        return transformers.LlamaForCausalLM(transformers.LlamaConfig(**common))
    if family == "mistral":
        return transformers.MistralForCausalLM(
            transformers.MistralConfig(**common, head_dim=4, sliding_window=32)
        )
    if family == "qwen2":
        return transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**common))
    if family == "qwen3":
        return transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(**common, head_dim=4)
        )
    if family == "gemma2":
        return transformers.Gemma2ForCausalLM(
            transformers.Gemma2Config(**common, head_dim=4)
        )
    raise AssertionError(family)


def test_tensor_accessor_rebuilds_tuple_without_losing_auxiliary_values():
    source = (torch.ones(1, 2, 3), "cache", {"attention": True})
    accessor = PathTensorAccessor((0,))
    replacement = torch.zeros_like(source[0])

    rebuilt = accessor.rebuild(source, replacement)

    assert torch.equal(rebuilt[0], replacement)
    assert rebuilt[1:] == source[1:]
    assert torch.equal(source[0], torch.ones_like(source[0]))


def test_qwen3_adapter_does_not_claim_other_qwen3_layouts():
    unsupported = type("Qwen3NextForCausalLM", (torch.nn.Module,), {})()
    unsupported.config = type("Config", (), {"model_type": "qwen3_next"})()

    assert not Qwen3Adapter().supports(unsupported)


@pytest.mark.parametrize("family", ["gemma2", "llama", "mistral", "qwen2", "qwen3"])
def test_architecture_adapter_contract_for_all_v010_sites(family):
    model = _tiny_model(family).eval()
    adapter = get_adapter(model)
    if family == "qwen3":
        assert isinstance(adapter, Qwen3Adapter)
    components = (
        "resid_pre",
        "attn_out",
        "head_result",
        "resid_mid",
        "mlp_out",
        "resid_post",
    )
    resolved_sites = {
        component: adapter.resolve(model, getattr(sites, component)(0))
        for component in components
    }
    seen = {}
    handles = []
    baselines = {}
    for component, resolved in resolved_sites.items():
        hooks = (
            resolved.module._forward_pre_hooks
            if resolved.hook_kind == "forward_pre"
            else resolved.module._forward_hooks
        )
        baselines[(id(resolved.module), resolved.hook_kind)] = len(hooks)

        if resolved.hook_kind == "forward_pre":

            def pre_hook(_module, inputs, *, key=component, site=resolved):
                activation = site.read(inputs)
                seen[key] = tuple(activation.shape)
                return site.rebuild(inputs, activation.clone())

            handles.append(resolved.module.register_forward_pre_hook(pre_hook))
        else:

            def forward_hook(_module, _inputs, output, *, key=component, site=resolved):
                activation = site.read(output)
                seen[key] = tuple(activation.shape)
                return site.rebuild(output, activation.clone())

            handles.append(resolved.module.register_forward_hook(forward_hook))
    try:
        result = model(input_ids=torch.tensor([[1, 4, 5]]), use_cache=True)
    finally:
        for handle in handles:
            handle.remove()

    assert result.logits.shape == (1, 3, 32)
    assert seen == {component: (1, 3, 16) for component in components}
    head = resolved_sites["head_result"]
    assert head.hidden_dim == 4
    assert head.hook_kind == "forward_pre"
    assert head.module_path.endswith("self_attn.o_proj")
    for resolved in resolved_sites.values():
        hooks = (
            resolved.module._forward_pre_hooks
            if resolved.hook_kind == "forward_pre"
            else resolved.module._forward_hooks
        )
        assert len(hooks) == baselines[(id(resolved.module), resolved.hook_kind)]


def test_gemma2_resid_mid_is_the_post_attention_residual_not_raw_attention_output():
    model = _tiny_model("gemma2")
    resolved = get_adapter(model).resolve(model, sites.resid_mid(0))

    assert resolved.hook_kind == "forward_pre"
    assert resolved.module_path == "model.layers.0.pre_feedforward_layernorm"
    assert resolved.module is model.model.layers[0].pre_feedforward_layernorm


@pytest.mark.parametrize("component", ["resid_post", "attn_out"])
def test_adapter_rejects_io_that_contradicts_semantic_site(component):
    model = _tiny_model("llama")

    with pytest.raises(SiteResolutionError, match="requires io='output'"):
        get_adapter(model).resolve(model, Site("language", component, 0, io="input"))


def test_standard_site_helpers_keep_semantic_output_io():
    assert sites.resid_pre(0).io == "output"
    assert sites.resid_mid(0).io == "output"
