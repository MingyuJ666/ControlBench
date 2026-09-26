import pytest
import torch
import transformers

from repsteer.artifacts import ArtifactMetadata, DirectionArtifact, load_artifact
from repsteer.capture import CaptureRequest
from repsteer.core import (
    ArtifactCompatibilityError,
    HookLifecycleError,
    Intervention,
    MissingOptionalDependencyError,
)
from repsteer.data import ContrastivePairs, canonicalize
from repsteer.learners import DiffMean
from repsteer.models import from_model, from_pretrained
from repsteer.operators import Add
from repsteer.positions import AllTokens, GeneratedTokens, LastNonPaddingToken, TextSpan
from repsteer.schedules import Constant
from repsteer.sites import resid_post


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "</s>"
    pad_token = "<pad>"
    padding_side = "right"

    def __init__(self):
        self.vocabulary = {
            "kind words": [1, 8, 9],
            "help people": [1, 10, 11],
            "cruel words": [1, 12, 9],
            "hurt people": [1, 13, 11],
        }

    def __call__(self, texts, *, return_tensors="pt", padding=False, **kwargs):
        assert return_tensors == "pt"
        values = [texts] if isinstance(texts, str) else list(texts)
        rows = [self.vocabulary.get(value, [1, 3, 4]) for value in values]
        width = max(map(len, rows))
        ids = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        result = {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }
        if kwargs.get("return_offsets_mapping"):
            offsets = []
            for text, row in zip(values, rows, strict=True):
                cursor = 0
                current = [[0, 0]]
                for word in text.split():
                    start = text.index(word, cursor)
                    stop = start + len(word)
                    current.append([start, stop])
                    cursor = stop
                current.extend([[0, 0]] * (width - len(row)))
                offsets.append(current)
            result["offset_mapping"] = torch.tensor(offsets)
        return result

    def batch_decode(self, sequences, **_kwargs):
        return [" ".join(map(str, row.tolist())) for row in sequences]


class _ChatTokenizer(_Tokenizer):
    """Tiny tokenizer fake that makes chat rendering observable in runtime tests."""

    chat_template = "<test-chat-template-v1>"

    def __init__(self):
        super().__init__()
        self.template_calls = []
        self.encode_calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        snapshot = tuple(dict(message) for message in messages)
        self.template_calls.append(
            {
                "messages": snapshot,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        rendered = "\n".join(
            f"<{message['role']}>{message['content']}" for message in snapshot
        )
        return f"{rendered}\n<assistant>" if add_generation_prompt else rendered

    def __call__(self, texts, *, return_tensors="pt", padding=False, **kwargs):
        self.encode_calls.append(
            {
                "texts": texts,
                "return_tensors": return_tensors,
                "padding": padding,
                "kwargs": kwargs,
            }
        )
        return super().__call__(
            texts,
            return_tensors=return_tensors,
            padding=padding,
            **kwargs,
        )


def _wrapper():
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
    return from_model(raw, _Tokenizer(), model_id="tiny/llama", revision="rev-1")


def _chat_wrapper():
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
    tokenizer = _ChatTokenizer()
    return (
        from_model(raw, tokenizer, model_id="tiny/llama", revision="rev-1"),
        tokenizer,
    )


def test_public_dtype_uses_the_transformers_5_loading_keyword():
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    raw = transformers.LlamaForCausalLM(config).eval()

    class Loader:
        seen = None

        @classmethod
        def from_pretrained(cls, _model_id, **kwargs):
            cls.seen = kwargs
            return raw

    wrapped = from_pretrained(
        "tiny/llama",
        config=config,
        model_class=Loader,
        tokenizer=_Tokenizer(),
        revision="rev-1",
        dtype="float32",
    )

    assert wrapped.raw_model is raw
    assert Loader.seen["dtype"] is torch.float32
    assert "torch_dtype" not in Loader.seen


def _artifact(wrapper, *, revision="rev-1"):
    site = resid_post(0)
    return DirectionArtifact(
        ArtifactMetadata(
            model_id=wrapper.model_id,
            model_revision=revision,
            architecture=wrapper.architecture,
            site=site,
            hidden_size=wrapper.hidden_size,
            method="unit",
        ),
        torch.linspace(-1, 1, wrapper.hidden_size),
    )


def test_compile_is_pure_zero_strength_is_baseline_and_exception_cleans_hooks():
    wrapper = _wrapper()
    artifact = _artifact(wrapper)
    resolved = wrapper.resolve_site(artifact.metadata.site)
    control = Intervention(
        artifact=artifact,
        operator=Add(),
        positions=AllTokens(),
        strength=Constant(0),
        phase="both",
    )
    ids = torch.tensor([[1, 5, 6]])
    baseline = wrapper(input_ids=ids).logits
    module_hooks = len(resolved.module._forward_hooks)

    compiled = wrapper.compile(control)
    assert len(resolved.module._forward_hooks) == module_hooks
    assert "model.layers.0" in compiled.explain()
    try:
        with wrapper.steer(compiled):
            # A statically zero intervention is removed from runtime hook groups,
            # including the otherwise-unneeded root phase tracker.
            assert wrapper.hook_manager.hook_count == 0
            steered = wrapper(input_ids=ids).logits
            assert torch.equal(steered, baseline)
            raise RuntimeError("body failure")
    except RuntimeError as error:
        assert str(error) == "body failure"

    assert wrapper.hook_manager.hook_count == 0
    assert len(resolved.module._forward_hooks) == module_hooks


def test_cached_generate_tracks_decode_and_zero_control_matches_baseline():
    wrapper = _wrapper()
    artifact = _artifact(wrapper)
    ids = torch.tensor([[1, 5, 6]])
    attention_mask = torch.ones_like(ids)
    control = Intervention(
        artifact=artifact,
        operator=Add(),
        positions=GeneratedTokens(),
        strength=Constant(0),
        phase="decode",
    )

    baseline = wrapper.generate(
        ids,
        attention_mask=attention_mask,
        max_new_tokens=3,
        do_sample=False,
        use_cache=True,
        seed=42,
    )
    with wrapper.steer(control):
        steered = wrapper.generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            seed=42,
        )

    assert torch.equal(steered.token_ids, baseline.token_ids)
    assert wrapper.hook_manager.hook_count == 0


def test_generation_tracker_exposes_decode_steps_to_operator():
    wrapper = _wrapper()
    artifact = _artifact(wrapper)

    class RecordingAdd:
        def __init__(self):
            self.seen = []

        def apply(self, activation, artifact, strength, context):
            self.seen.append(
                (context.phase, context.generation_step, activation.shape[-2])
            )
            return Add().apply(activation, artifact, strength, context)

    operator = RecordingAdd()
    control = Intervention(
        artifact=artifact,
        operator=operator,
        positions=GeneratedTokens(),
        strength=Constant(0.25),
        phase="decode",
    )
    with wrapper.steer(control):
        wrapper.generate(
            torch.tensor([[1, 5, 6]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            min_new_tokens=3,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )

    assert operator.seen == [("decode", 0, 1), ("decode", 1, 1)]


def test_zero_strength_does_not_invoke_rng_consuming_custom_operator():
    wrapper = _wrapper()
    artifact = _artifact(wrapper)

    class RandomConsumingOperator:
        def __init__(self):
            self.calls = 0

        def apply(self, activation, artifact, strength, context):
            del artifact, strength, context
            self.calls += 1
            torch.rand(1024, device=activation.device)
            return activation

    operator = RandomConsumingOperator()
    control = Intervention(
        artifact=artifact,
        operator=operator,
        positions=GeneratedTokens(),
        strength=Constant(0),
        phase="decode",
    )
    kwargs = {
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "min_new_tokens": 4,
        "max_new_tokens": 4,
        "do_sample": True,
        "seed": 123,
    }
    input_ids = torch.tensor([[1, 5, 6]])

    baseline = wrapper.generate(input_ids, **kwargs)
    with wrapper.steer(control):
        steered = wrapper.generate(input_ids, **kwargs)

    assert operator.calls == 0
    assert torch.equal(steered.token_ids, baseline.token_ids)


def test_diffmean_vertical_slice_capture_save_load_and_cross_layer_compile(tmp_path):
    wrapper = _wrapper()
    data = ContrastivePairs.from_records(
        [
            {"positive": "kind words", "negative": "cruel words"},
            {"positive": "help people", "negative": "hurt people"},
        ]
    )
    artifact = DiffMean(
        site=resid_post(0),
        positions=LastNonPaddingToken(),
    ).fit(wrapper, data)

    assert artifact.direction.shape == (wrapper.hidden_size,)
    assert artifact.metadata.architecture == wrapper.architecture
    assert artifact.metadata.tokenizer["id"] == wrapper.model_id
    assert artifact.metadata.tokenizer["revision"] == wrapper.revision
    assert "torch" in artifact.metadata.provenance["environment"]["dependencies"]
    artifact.save(tmp_path)
    assert torch.equal(load_artifact(tmp_path).direction, artifact.direction)

    cross_layer = Intervention(
        artifact=artifact,
        site=resid_post(1),
        operator=Add(),
        positions=GeneratedTokens(),
        strength=Constant(1),
        phase="decode",
    )
    explanation = wrapper.compile(cross_layer).explain(format="dict")
    assert "cross-site" in explanation["interventions"][0]["warnings"][0]


def test_revision_mismatch_fails_by_default():
    wrapper = _wrapper()
    control = Intervention(
        artifact=_artifact(wrapper, revision="other-revision"),
        operator=Add(),
        positions=AllTokens(),
        strength=Constant(1),
    )

    with torch.no_grad():
        try:
            wrapper.compile(control)
        except ArtifactCompatibilityError as error:
            assert "model revision differs" in str(error)
        else:
            raise AssertionError("revision mismatch must fail closed")


def test_capture_fails_closed_inside_active_steering_session():
    wrapper = _wrapper()
    artifact = _artifact(wrapper)
    control = Intervention(
        artifact=artifact,
        operator=Add(),
        positions=AllTokens(),
        strength=Constant(5),
    )
    request = CaptureRequest(
        inputs=["kind words"],
        site=resid_post(0),
        positions=LastNonPaddingToken(),
    )

    with wrapper.steer(control), pytest.raises(HookLifecycleError, match="cache key"):
        wrapper.capture(request)


def test_hf_capture_supplies_offsets_for_text_span_selector():
    wrapper = _wrapper()
    result = wrapper.capture(
        CaptureRequest(
            inputs=["kind words", "cruel words"],
            site=resid_post(0),
            positions=TextSpan("words"),
        )
    )

    assert result.activations.shape == (2, 1, wrapper.hidden_size)
    assert result.attention_mask.tolist() == [[True], [True]]


def test_structured_chat_generate_auto_templates_and_avoids_duplicate_special_tokens():
    wrapper, tokenizer = _chat_wrapper()
    messages = [{"role": "user", "content": "kind words"}]

    result = wrapper.generate(messages, max_new_tokens=1, do_sample=False)

    assert result.token_ids.ndim == 2
    assert tokenizer.template_calls == [
        {
            "messages": ({"role": "user", "content": "kind words"},),
            "tokenize": False,
            "add_generation_prompt": True,
            "kwargs": {},
        }
    ]
    assert tokenizer.encode_calls[-1]["texts"] in (
        "<user>kind words\n<assistant>",
        ["<user>kind words\n<assistant>"],
    )
    assert tokenizer.encode_calls[-1]["kwargs"]["add_special_tokens"] is False


def test_raw_prompt_can_explicitly_use_chat_template_with_system_prompt():
    wrapper, tokenizer = _chat_wrapper()

    wrapper.generate(
        "kind words",
        apply_chat_template=True,
        system_prompt="Answer helpfully.",
        max_new_tokens=1,
        do_sample=False,
    )

    assert tokenizer.template_calls == [
        {
            "messages": (
                {"role": "system", "content": "Answer helpfully."},
                {"role": "user", "content": "kind words"},
            ),
            "tokenize": False,
            "add_generation_prompt": True,
            "kwargs": {},
        }
    ]
    assert tokenizer.encode_calls[-1]["kwargs"]["add_special_tokens"] is False


def test_structured_chat_generate_requires_a_chat_template():
    wrapper = _wrapper()

    with pytest.raises(MissingOptionalDependencyError, match="apply_chat_template"):
        wrapper.generate(
            [{"role": "user", "content": "kind words"}],
            max_new_tokens=1,
            do_sample=False,
        )


def test_capture_templates_single_and_batched_chat_inputs():
    wrapper, tokenizer = _chat_wrapper()
    single = wrapper.capture(
        CaptureRequest(
            inputs=[{"role": "user", "content": "kind words"}],
            site=resid_post(0),
            positions=LastNonPaddingToken(),
            system_prompt="Answer helpfully.",
        )
    )

    assert single.activations.shape == (1, 1, wrapper.hidden_size)
    assert tokenizer.template_calls == [
        {
            "messages": (
                {"role": "system", "content": "Answer helpfully."},
                {"role": "user", "content": "kind words"},
            ),
            "tokenize": False,
            "add_generation_prompt": False,
            "kwargs": {},
        }
    ]
    assert tokenizer.encode_calls[-1]["kwargs"]["add_special_tokens"] is False

    tokenizer.template_calls.clear()
    batched = wrapper.capture(
        CaptureRequest(
            inputs=[
                [{"role": "user", "content": "kind words"}],
                [{"role": "user", "content": "cruel words"}],
            ],
            site=resid_post(0),
            positions=LastNonPaddingToken(),
        )
    )

    assert batched.activations.shape == (2, 1, wrapper.hidden_size)
    assert [call["messages"] for call in tokenizer.template_calls] == [
        ({"role": "user", "content": "kind words"},),
        ({"role": "user", "content": "cruel words"},),
    ]
    assert all(
        call["add_generation_prompt"] is False for call in tokenizer.template_calls
    )


def test_chat_learner_forwards_rendering_configuration_and_template_metadata():
    wrapper, tokenizer = _chat_wrapper()
    data = ContrastivePairs.from_records(
        [
            {
                "positive_messages": [{"role": "user", "content": "kind words"}],
                "negative_messages": [{"role": "user", "content": "cruel words"}],
            }
        ]
    )

    artifact = DiffMean(
        site=resid_post(0),
        positions=LastNonPaddingToken(),
        apply_chat_template=True,
        add_generation_prompt=True,
        system_prompt="Answer helpfully.",
    ).fit(wrapper, data)

    assert data.is_chat
    assert len(tokenizer.template_calls) == 2
    assert all(
        call["messages"][0] == {"role": "system", "content": "Answer helpfully."}
        and call["add_generation_prompt"] is True
        for call in tokenizer.template_calls
    )
    rendering = artifact.metadata.config["rendering"]
    assert rendering["apply_chat_template"] is True
    assert rendering["add_generation_prompt"] is True
    assert rendering["system_prompt"] == "Answer helpfully."
    assert "chat_template_sha256" in artifact.metadata.tokenizer


def test_messages_keyword_renders_each_chat_in_a_padded_batch():
    wrapper, tokenizer = _chat_wrapper()
    conversations = [
        [{"role": "user", "content": "kind words"}],
        [{"role": "user", "content": "help people"}],
    ]

    result = wrapper.generate(
        messages=conversations,
        max_new_tokens=1,
        do_sample=False,
    )

    assert result.token_ids.shape[0] == 2
    assert [call["messages"] for call in tokenizer.template_calls] == [
        ({"role": "user", "content": "kind words"},),
        ({"role": "user", "content": "help people"},),
    ]
    assert all(
        call["add_generation_prompt"] is True for call in tokenizer.template_calls
    )
    assert tokenizer.encode_calls[-1]["texts"] == [
        "<user>kind words\n<assistant>",
        "<user>help people\n<assistant>",
    ]
    assert tokenizer.encode_calls[-1]["padding"] is True


def test_chat_template_kwargs_reach_renderer_and_are_recorded_in_provenance():
    wrapper, tokenizer = _chat_wrapper()
    messages = [{"role": "user", "content": "kind words"}]
    template_kwargs = {"template_mode": "unit-test"}
    request = CaptureRequest(
        inputs=[messages],
        site=resid_post(0),
        positions=LastNonPaddingToken(),
        chat_template_kwargs=template_kwargs,
    )
    changed_request = CaptureRequest(
        inputs=[messages],
        site=resid_post(0),
        positions=LastNonPaddingToken(),
        chat_template_kwargs={"template_mode": "different"},
    )

    assert request.fingerprint != changed_request.fingerprint
    wrapper.capture(request)
    assert tokenizer.template_calls[-1]["kwargs"] == template_kwargs

    tokenizer.template_calls.clear()
    data = ContrastivePairs.from_records(
        [
            {
                "positive_messages": messages,
                "negative_messages": [{"role": "user", "content": "cruel words"}],
            }
        ]
    )
    artifact = DiffMean(
        site=resid_post(0),
        positions=LastNonPaddingToken(),
        chat_template_kwargs=template_kwargs,
    ).fit(wrapper, data)

    assert [call["kwargs"] for call in tokenizer.template_calls] == [
        template_kwargs,
        template_kwargs,
    ]
    serialized = artifact.metadata.to_dict()
    rendering = serialized["config"]["rendering"]
    assert rendering["chat_template_kwargs"] == canonicalize(template_kwargs)
    provenance = serialized["provenance"]["capture"]["rendering"]
    assert provenance["chat_template_kwargs"] == canonicalize(template_kwargs)


def test_structured_messages_cannot_disable_chat_templating():
    wrapper, _ = _chat_wrapper()
    messages = [{"role": "user", "content": "kind words"}]

    with pytest.raises(TypeError, match="structured chat messages require"):
        wrapper.generate(
            messages,
            apply_chat_template=False,
            max_new_tokens=1,
            do_sample=False,
        )
    with pytest.raises(TypeError, match="structured chat messages require"):
        wrapper.capture(
            CaptureRequest(
                inputs=[messages],
                site=resid_post(0),
                positions=LastNonPaddingToken(),
                apply_chat_template=False,
            )
        )


def test_explicit_tokenizer_special_tokens_override_chat_safe_default():
    wrapper, tokenizer = _chat_wrapper()

    wrapper.generate(
        [{"role": "user", "content": "kind words"}],
        tokenizer_kwargs={"add_special_tokens": True},
        max_new_tokens=1,
        do_sample=False,
    )

    assert tokenizer.encode_calls[-1]["kwargs"]["add_special_tokens"] is True
