"""Offline loading tests using small fake Transformers entry points.

The tests patch repsteer's lazy Transformers import, so they exercise loader
selection and argument forwarding without downloading a config, tokenizer,
processor, or model checkpoint.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import repsteer.models.hf.model as hf_model


class _Config:
    def __init__(
        self,
        model_type: str,
        *,
        commit: str | None = None,
        multimodal: bool = False,
    ) -> None:
        self.model_type = model_type
        self.hidden_size = 8
        self._commit_hash = commit
        self.architectures = ["FakeArchitecture"]
        if multimodal:
            self.vision_config = object()
            self.text_config = SimpleNamespace(hidden_size=8)


class _RawModel(nn.Module):
    def __init__(self, config: _Config) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(4, config.hidden_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


class _FromPretrained:
    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, model_name_or_path: str, **kwargs):
        self.calls.append((model_name_or_path, kwargs))
        return self.result


class _Tokenizer:
    pass


class _Processor:
    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer


def _fake_transformers(*, config: _Config, raw_model: _RawModel):
    config_loader = _FromPretrained(config)
    text_model_loader = _FromPretrained(raw_model)
    multimodal_model_loader = _FromPretrained(raw_model)
    tokenizer = _Tokenizer()
    tokenizer_loader = _FromPretrained(tokenizer)
    processor = _Processor(tokenizer)
    processor_loader = _FromPretrained(processor)
    transformers = SimpleNamespace(
        AutoConfig=config_loader,
        AutoModelForCausalLM=text_model_loader,
        AutoModelForImageTextToText=multimodal_model_loader,
        AutoProcessor=processor_loader,
        AutoTokenizer=tokenizer_loader,
    )
    return (
        transformers,
        config_loader,
        text_model_loader,
        multimodal_model_loader,
        tokenizer_loader,
        processor_loader,
    )


def test_text_from_pretrained_uses_fake_config_model_and_tokenizer_loaders(monkeypatch):
    config = _Config("llama")
    raw_model = _RawModel(_Config("llama", commit="resolved-commit"))
    (
        transformers,
        config_loader,
        text_model_loader,
        multimodal_model_loader,
        tokenizer_loader,
        processor_loader,
    ) = _fake_transformers(config=config, raw_model=raw_model)
    monkeypatch.setattr(hf_model, "_transformers", lambda: transformers)

    wrapped = hf_model.from_pretrained(
        "example/text-model",
        revision="release-5",
        dtype="bf16",
        trust_remote_code=True,
        tokenizer_kwargs={"use_fast": False},
    )

    assert config_loader.calls == [
        (
            "example/text-model",
            {"revision": "release-5", "trust_remote_code": True},
        )
    ]
    assert len(text_model_loader.calls) == 1
    model_name, model_kwargs = text_model_loader.calls[0]
    assert model_name == "example/text-model"
    assert model_kwargs["config"] is config
    assert model_kwargs["revision"] == "release-5"
    assert model_kwargs["trust_remote_code"] is True
    assert model_kwargs["dtype"] is torch.bfloat16
    assert tokenizer_loader.calls == [
        ("example/text-model", {"use_fast": False, "revision": "release-5"})
    ]
    assert multimodal_model_loader.calls == []
    assert processor_loader.calls == []
    assert wrapped.raw_model is raw_model
    assert wrapped.model_id == "example/text-model"
    assert wrapped.revision == "resolved-commit"
    assert wrapped.processor is None


def test_vlm_from_pretrained_selects_image_text_loader_and_processor(monkeypatch):
    config = _Config("qwen2_5_vl", multimodal=True)
    raw_model = _RawModel(
        _Config("qwen2_5_vl", commit="resolved-vlm-commit", multimodal=True)
    )
    (
        transformers,
        config_loader,
        text_model_loader,
        multimodal_model_loader,
        tokenizer_loader,
        processor_loader,
    ) = _fake_transformers(config=config, raw_model=raw_model)
    monkeypatch.setattr(hf_model, "_transformers", lambda: transformers)

    wrapped = hf_model.from_pretrained(
        "example/vlm-model",
        revision="release-5",
        processor_kwargs={"image_size": 32},
    )

    assert config_loader.calls == [("example/vlm-model", {"revision": "release-5"})]
    assert text_model_loader.calls == []
    assert len(multimodal_model_loader.calls) == 1
    model_name, model_kwargs = multimodal_model_loader.calls[0]
    assert model_name == "example/vlm-model"
    assert model_kwargs["config"] is config
    assert model_kwargs["revision"] == "release-5"
    assert processor_loader.calls == [
        ("example/vlm-model", {"image_size": 32, "revision": "release-5"})
    ]
    assert tokenizer_loader.calls == []
    assert wrapped.raw_model is raw_model
    assert wrapped.processor is processor_loader.result
    assert wrapped.tokenizer is processor_loader.result.tokenizer
    assert wrapped.processor_id == "example/vlm-model"
    assert wrapped.processor_revision == "resolved-vlm-commit"


def test_loader_normalizes_legacy_torch_dtype_alias_to_the_v5_keyword(monkeypatch):
    config = _Config("llama")
    raw_model = _RawModel(_Config("llama"))
    (
        transformers,
        _config_loader,
        text_model_loader,
        _multimodal_model_loader,
        _tokenizer_loader,
        _processor_loader,
    ) = _fake_transformers(config=config, raw_model=raw_model)
    monkeypatch.setattr(hf_model, "_transformers", lambda: transformers)

    hf_model.from_pretrained(
        "example/text-model",
        config=config,
        torch_dtype="float32",
    )

    assert text_model_loader.calls[0][1]["dtype"] is torch.float32
    assert "torch_dtype" not in text_model_loader.calls[0][1]
    with pytest.raises(TypeError, match="either dtype=.*torch_dtype"):
        hf_model.from_pretrained(
            "example/text-model",
            config=config,
            dtype="float16",
            torch_dtype="float32",
        )


def test_supplied_model_bypasses_lazy_transformers_loading(monkeypatch):
    raw_model = _RawModel(_Config("llama"))
    tokenizer = _Tokenizer()

    def unexpected_transformers_access():
        raise AssertionError("a supplied nn.Module must not invoke Transformers")

    monkeypatch.setattr(hf_model, "_transformers", unexpected_transformers_access)

    wrapped = hf_model.from_pretrained(
        model=raw_model,
        tokenizer=tokenizer,
        revision="local-revision",
    )

    assert wrapped.raw_model is raw_model
    assert wrapped.tokenizer is tokenizer
    assert wrapped.revision == "local-revision"
