from types import SimpleNamespace

import pytest
import torch
from torch import nn

from repsteer.core import PositionResolutionError, ProcessorCompatibilityError
from repsteer.models.hf.adapters import ArchitectureAdapter
from repsteer.models.hf.model import HFSteerableModel


class _Tokenizer:
    def batch_decode(self, sequences, **kwargs):
        del kwargs
        return [" ".join(str(value) for value in row.tolist()) for row in sequences]


class _Processor:
    name_or_path = "tiny/vlm"
    _commit_hash = "r1"

    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.calls = []

    def __call__(self, *, text, images, **kwargs):
        self.calls.append((text, images, kwargs))
        return {
            "input_ids": torch.tensor([[1, 9, 9, 2]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "pixel_values": torch.as_tensor(images).reshape(1, 1, 2, 2).float(),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }


class _TokenProcessor(_Processor):
    def __init__(self, image_token):
        super().__init__()
        self.image_token = image_token

    def __call__(self, *, text, images, **kwargs):
        self.calls.append((text, images, kwargs))
        return {
            "input_ids": torch.tensor([[1, 9, 2]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }


class _ChatTokenProcessor(_TokenProcessor):
    def __init__(self, image_token):
        super().__init__(image_token)
        self.template_calls = []

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        self.template_calls.append(
            {
                "messages": tuple(dict(message) for message in messages),
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        content = "|".join(str(message["content"]) for message in messages)
        suffix = "<assistant>" if add_generation_prompt else ""
        return f"<user>{content}{self.image_token}{suffix}"


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            _name_or_path="tiny/vlm",
            _commit_hash="r1",
            hidden_size=4,
            architectures=["TinyVLM"],
        )
        self.seen = None

    def generate(self, **kwargs):
        self.seen = kwargs
        return kwargs["input_ids"]


class _Adapter(ArchitectureAdapter):
    architecture_name = "tiny_vlm"

    def supports(self, model):
        return isinstance(model, _Model)

    def resolve(self, model, site):
        raise NotImplementedError

    def hidden_size(self, model, site=None):
        return 4

    def build_modality_map(self, batch, *, model=None):
        if "image_grid_thw" not in batch:
            return None
        return {
            "grid": tuple(torch.as_tensor(batch["image_grid_thw"]).shape),
            "model": type(model).__name__,
        }


class _NamedVLMAdapter(_Adapter):
    def __init__(self, architecture_name):
        self.architecture_name = architecture_name


def _vlm_wrapper(architecture_name, image_token):
    processor = _TokenProcessor(image_token)
    wrapper = HFSteerableModel(
        _Model(),
        processor=processor,
        adapter=_NamedVLMAdapter(architecture_name),
        model_id="tiny/vlm",
        revision="r1",
    )
    return wrapper, processor


def test_high_level_image_generation_uses_processor_and_builds_modality_map():
    raw = _Model()
    processor = _Processor()
    wrapper = HFSteerableModel(
        raw,
        processor=processor,
        adapter=_Adapter(),
        model_id="tiny/vlm",
        revision="r1",
    )
    image = torch.arange(4)

    prepared = wrapper._prepare_generation(
        (),
        {
            "prompt": "describe",
            "image": image,
            "processor_kwargs": {"padding": False},
            "max_new_tokens": 1,
        },
    )
    assert prepared[-1] == {"grid": (1, 3), "model": "_Model"}

    result = wrapper.generate(
        image=image,
        prompt="describe",
        processor_kwargs={"padding": False},
        max_new_tokens=1,
    )

    assert processor.calls[-1][0] == "describe"
    assert processor.calls[-1][2]["return_tensors"] == "pt"
    assert raw.seen is not None and "pixel_values" in raw.seen
    assert result.text == "1 9 9 2"


@pytest.mark.parametrize(
    ("architecture", "image_token", "injected"),
    [
        (
            "qwen2_5_vl",
            "<|image_pad|>",
            "<|vision_start|><|image_pad|><|vision_end|>\ndescribe",
        ),
        ("internvl", "<IMG_CONTEXT>", "<IMG_CONTEXT>\ndescribe"),
    ],
)
def test_builtin_vlm_raw_prompt_injects_exactly_one_processor_placeholder(
    architecture, image_token, injected
):
    wrapper, processor = _vlm_wrapper(architecture, image_token)
    image = torch.zeros(3, 2, 2)

    wrapper._encode_multimodal("describe", image)
    assert processor.calls[-1][0] == injected

    explicit = f"prefix {image_token} suffix"
    wrapper._encode_multimodal(explicit, image)
    assert processor.calls[-1][0] == explicit
    assert processor.calls[-1][0].count(image_token) == 1


@pytest.mark.parametrize(
    ("texts", "images"),
    [
        ("describe both", [torch.zeros(3, 2, 2), torch.ones(3, 2, 2)]),
        (
            ["describe first", "describe second"],
            [torch.zeros(3, 2, 2), torch.ones(3, 2, 2)],
        ),
    ],
)
def test_builtin_vlm_ambiguous_image_prompt_mapping_fails_closed(texts, images):
    wrapper, _ = _vlm_wrapper("qwen2_5_vl", "<|image_pad|>")

    with pytest.raises(PositionResolutionError, match="only unambiguous"):
        wrapper._encode_multimodal(texts, images)


def test_builtin_vlm_explicit_placeholders_validate_image_cardinality():
    token = "<|image_pad|>"
    wrapper, processor = _vlm_wrapper("qwen2_5_vl", token)
    images = [torch.zeros(3, 2, 2), torch.ones(3, 2, 2)]
    explicit = f"{token} first, {token} second"

    wrapper._encode_multimodal(explicit, images)
    assert processor.calls[-1][0] == explicit

    with pytest.raises(PositionResolutionError, match="placeholder count"):
        wrapper._encode_multimodal(token, images)

    with pytest.raises(PositionResolutionError, match="per-prompt"):
        wrapper._encode_multimodal(
            [f"{token}{token} first", "second"],
            [[images[0]], [images[1]]],
        )


def test_instruction_vlm_messages_prefer_processor_template_before_processing():
    token = "<|image_pad|>"
    processor = _ChatTokenProcessor(token)
    raw = _Model()
    wrapper = HFSteerableModel(
        raw,
        processor=processor,
        adapter=_NamedVLMAdapter("qwen2_5_vl"),
        model_id="tiny/vlm",
        revision="r1",
    )

    wrapper.generate(
        messages=[{"role": "user", "content": "describe this image"}],
        image=torch.zeros(3, 2, 2),
        max_new_tokens=1,
    )

    assert processor.template_calls == [
        {
            "messages": ({"role": "user", "content": "describe this image"},),
            "tokenize": False,
            "add_generation_prompt": True,
            "kwargs": {},
        }
    ]
    text, _images, options = processor.calls[-1]
    assert text == f"<user>describe this image{token}<assistant>"
    assert options["add_special_tokens"] is False


def test_processor_revision_mismatch_fails_closed():
    processor = _Processor()
    processor._commit_hash = "processor-r2"

    with pytest.raises(
        ProcessorCompatibilityError, match="processor/model revision mismatch"
    ):
        HFSteerableModel(
            _Model(),
            processor=processor,
            adapter=_Adapter(),
            model_id="tiny/vlm",
            revision="r1",
        )
