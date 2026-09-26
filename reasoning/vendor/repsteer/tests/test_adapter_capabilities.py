from typing import cast

import pytest
import transformers

from repsteer import sites
from repsteer.models.hf.adapters import (
    AdapterCapabilities,
    GemmaAdapter,
    InternVLAdapter,
    LlamaAdapter,
    MistralAdapter,
    ModalityMappingCapability,
    Qwen2_5_VLAdapter,
    Qwen2Adapter,
    Qwen3Adapter,
    SampleMappingCapability,
    get_adapter,
)


def _tiny_llama():
    return transformers.LlamaForCausalLM(
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
    )


def test_text_adapter_capabilities_are_stable_and_match_resolved_residual_contract():
    adapter = get_adapter(_tiny_llama())
    capabilities = adapter.capabilities

    assert isinstance(adapter, LlamaAdapter)
    assert capabilities == adapter.capabilities
    assert AdapterCapabilities.from_dict(capabilities.to_dict()) == capabilities
    assert capabilities.supports("residual_read")
    assert capabilities.supports("residual_write")
    assert capabilities.sample_mapping is not None
    assert capabilities.sample_mapping.target_streams == ("language",)
    assert "language.resid_post" in capabilities.residual_sites
    assert adapter.resolve(_tiny_llama(), sites.resid_post(0)).hidden_dim == 16
    assert capabilities.head_result is not None
    assert capabilities.attention_bias is None
    assert capabilities.supports("head_result")
    assert not capabilities.supports("attention_bias")
    assert capabilities.explain() == adapter.capabilities.explain()


def test_vlm_capabilities_keep_language_sample_mapping_separate_from_modality_mapping():
    for adapter in (Qwen2_5_VLAdapter(), InternVLAdapter()):
        capabilities = adapter.capabilities

        assert capabilities.modality_mapping is not None
        assert capabilities.modality_mapping.streams == (
            "vision",
            "projector",
            "language",
        )
        assert capabilities.sample_mapping is not None
        assert capabilities.sample_mapping.target_streams == ("language",)
        assert "vision.vision_resid" in capabilities.residual_sites
        assert "projector.projector_in" in capabilities.residual_sites
        assert AdapterCapabilities.from_dict(capabilities.to_dict()) == capabilities
        assert capabilities.canonical == adapter.capabilities.canonical
        assert capabilities.head_result is not None
        assert capabilities.attention_bias is None


def test_all_builtin_text_adapters_declare_the_residual_and_head_result_surfaces():
    for adapter in (
        LlamaAdapter(),
        MistralAdapter(),
        Qwen2Adapter(),
        Qwen3Adapter(),
        GemmaAdapter(),
    ):
        capabilities = adapter.capabilities

        assert capabilities.supports_residual_read
        assert capabilities.supports_residual_write
        assert capabilities.sample_mapping is not None
        assert capabilities.modality_mapping is None
        assert capabilities.head_result is not None
        assert capabilities.attention_bias is None


def test_capability_requirements_declare_head_result_and_fail_closed_elsewhere():
    capabilities = LlamaAdapter().capabilities

    capabilities.require("head_result")
    with pytest.raises(ValueError, match="unknown adapter capability requirement"):
        capabilities.require("not_a_surface")
    assert not capabilities.supports("not_a_surface")


@pytest.mark.parametrize(
    "factory",
    (
        lambda: AdapterCapabilities(adapter_name="unit", schema_version=True),
        lambda: AdapterCapabilities(adapter_name="unit", schema_version=cast(int, 1.0)),
        lambda: SampleMappingCapability(schema_version=True),
        lambda: SampleMappingCapability(schema_version=cast(int, 1.0)),
        lambda: ModalityMappingCapability(schema_version=True),
        lambda: ModalityMappingCapability(schema_version=cast(int, 1.0)),
    ),
)
def test_capability_schema_version_is_a_strict_integer(factory):
    with pytest.raises(TypeError, match="integer schema version"):
        factory()


def test_capability_mapping_declarations_reject_empty_or_unordered_streams():
    mapping = SampleMappingCapability(
        condition_streams=["language"], target_streams=["language"]
    )
    capabilities = AdapterCapabilities(
        adapter_name="unit",
        residual_sites=["language.resid_post"],
        supports_residual_read=True,
        supports_residual_write=True,
        sample_mapping=mapping,
    )
    assert mapping.condition_streams == ("language",)
    assert capabilities.residual_sites == ("language.resid_post",)

    with pytest.raises(ValueError, match="cannot be empty"):
        SampleMappingCapability(condition_streams=())
    with pytest.raises(ValueError, match="cannot be empty"):
        SampleMappingCapability(target_streams=())
    with pytest.raises(ValueError, match="cannot be empty"):
        ModalityMappingCapability(streams=())
    with pytest.raises(TypeError, match="list or tuple"):
        AdapterCapabilities(adapter_name="unit", residual_sites={"language.resid_post"})


def test_capability_deserialization_requires_complete_versioned_records():
    document = LlamaAdapter().capabilities.to_dict()
    document["schema_version"] = True
    with pytest.raises(TypeError, match="integer schema version"):
        AdapterCapabilities.from_dict(document)

    document = LlamaAdapter().capabilities.to_dict()
    document.pop("head_result")
    with pytest.raises(ValueError, match="missing required fields"):
        AdapterCapabilities.from_dict(document)
