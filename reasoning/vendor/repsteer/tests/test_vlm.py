from types import SimpleNamespace

import pytest
import torch
from torch import nn

from repsteer import sites
from repsteer.artifacts import ArtifactMetadata, DirectionArtifact
from repsteer.core import (
    Intervention,
    PlanCompilationError,
    PositionResolutionError,
    Site,
    SteeringPlan,
    StepContext,
)
from repsteer.models import from_model
from repsteer.models.hf.adapters import InternVLAdapter, Qwen2_5_VLAdapter, get_adapter
from repsteer.operators import Add
from repsteer.positions import (
    ImagePatches,
    ImageTokens,
    ObjectPatches,
    position_selector_from_dict,
)
from repsteer.schedules import Constant


class _Attention(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 0.1, {"cache": True}


class _DecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = _Attention()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.mlp.weight)

    def forward(self, hidden_states):
        attention = self.self_attn(hidden_states)[0]
        middle = self.post_attention_layernorm(hidden_states + attention)
        return middle + self.mlp(middle)


class _VisionBlock(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 0.25


class _QwenMerger(nn.Module):
    def __init__(self, vision_size, text_size, merge_size):
        super().__init__()
        self.projection = nn.Linear(vision_size, text_size, bias=False)
        self.merge_size = merge_size

    def forward(self, hidden_states):
        merged = hidden_states.reshape(
            -1, self.merge_size**2, hidden_states.shape[-1]
        ).mean(dim=1)
        return self.projection(merged)


class _QwenVisual(nn.Module):
    def __init__(self, vision_size, text_size, merge_size):
        super().__init__()
        self.blocks = nn.ModuleList([_VisionBlock(), _VisionBlock()])
        self.merger = _QwenMerger(vision_size, text_size, merge_size)

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.merger(hidden_states)


class _LanguageModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [_DecoderLayer(hidden_size), _DecoderLayer(hidden_size)]
        )

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class FakeQwen2_5_VLForConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        vision_size, text_size, merge_size = 4, 8, 2
        self.config = SimpleNamespace(
            model_type="qwen2_5_vl",
            image_token_id=99,
            vision_start_token_id=97,
            vision_end_token_id=98,
            text_config=SimpleNamespace(hidden_size=text_size),
            vision_config=SimpleNamespace(
                hidden_size=vision_size,
                out_hidden_size=text_size,
                spatial_merge_size=merge_size,
                patch_size=1,
                window_size=4,
            ),
        )
        self.model = nn.Module()
        self.model.visual = _QwenVisual(vision_size, text_size, merge_size)
        self.model.language_model = _LanguageModel(text_size)
        self.token_embeddings = nn.Embedding(128, text_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        *,
        language_hidden=None,
        vision_hidden=None,
    ):
        del attention_mask, image_grid_thw
        if input_ids is not None:
            language_hidden = self.token_embeddings(input_ids)
        if pixel_values is not None:
            vision_hidden = pixel_values
        if language_hidden is None or vision_hidden is None:
            raise ValueError("fake Qwen model requires language and vision inputs")
        image_features = self.model.visual(vision_hidden)
        language_hidden = language_hidden.clone()
        if input_ids is None:
            language_hidden[:, : image_features.shape[0]] = image_features
        else:
            image_positions = input_ids == self.config.image_token_id
            language_hidden[image_positions] = image_features
        return self.model.language_model(language_hidden)


class _InternProjector(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.projection = nn.Linear(input_size, output_size, bias=False)

    def forward(self, hidden_states):
        return self.projection(hidden_states)


class FakeInternVLForConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        vision_size, text_size = 4, 8
        self.config = SimpleNamespace(
            model_type="internvl",
            image_token_id=99,
            image_seq_length=4,
            downsample_ratio=0.5,
            vision_feature_select_strategy="default",
            text_config=SimpleNamespace(hidden_size=text_size),
            vision_config=SimpleNamespace(
                hidden_size=vision_size,
                patch_size=(2, 2),
            ),
        )
        self.model = nn.Module()
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.encoder = nn.Module()
        self.model.vision_tower.encoder.layer = nn.ModuleList(
            [_VisionBlock(), _VisionBlock()]
        )
        self.model.multi_modal_projector = _InternProjector(vision_size * 4, text_size)
        self.model.language_model = _LanguageModel(text_size)

    def forward(self, language_hidden, vision_hidden):
        for layer in self.model.vision_tower.encoder.layer:
            vision_hidden = layer(vision_hidden)
        patches = vision_hidden[:, 1:, :].reshape(vision_hidden.shape[0], 4, 16)
        image_features = self.model.multi_modal_projector(patches)
        language_hidden = language_hidden.clone()
        language_hidden[:, : image_features.shape[1]] = image_features[0]
        return self.model.language_model(language_hidden)


@pytest.mark.parametrize(
    ("model_factory", "adapter_type", "vision_shape", "projector_shapes"),
    [
        (
            FakeQwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLAdapter,
            (16, 4),
            ((16, 4), (4, 8)),
        ),
        (
            FakeInternVLForConditionalGeneration,
            InternVLAdapter,
            (1, 17, 4),
            ((1, 4, 16), (1, 4, 8)),
        ),
    ],
)
def test_vlm_architecture_contract_without_transformers(
    model_factory, adapter_type, vision_shape, projector_shapes
):
    model = model_factory()
    adapter = get_adapter(model)
    assert isinstance(adapter, adapter_type)
    language_sites = {
        "language.resid_pre": sites.resid_pre(0),
        "language.attn_out": sites.attn_out(0),
        "language.resid_mid": sites.resid_mid(0),
        "language.mlp_out": sites.mlp_out(0),
        "language.resid_post": sites.resid_post(0),
    }
    declared = {
        **{name: adapter.resolve(model, site) for name, site in language_sites.items()},
        "vision": adapter.resolve(model, sites.vision_resid(0)),
        "projector_in": adapter.resolve(model, sites.projector_in()),
        "projector_out": adapter.resolve(model, sites.projector_out()),
    }
    assert set(adapter.capabilities.residual_sites) == {
        *language_sites,
        "vision.vision_resid",
        "projector.projector_in",
        "projector.projector_out",
    }
    assert all(declared[name].hidden_dim == 8 for name in language_sites)
    assert declared["vision"].hidden_dim == 4
    assert declared["projector_out"].hidden_dim == 8
    assert declared["projector_in"].hook_kind == "forward_pre"
    assert declared["projector_out"].hook_kind == "forward"
    expected_projector_input = 4 if adapter_type is Qwen2_5_VLAdapter else 16
    assert declared["projector_in"].hidden_dim == expected_projector_input

    seen = {}
    handles = []
    for name, resolved in declared.items():
        if resolved.hook_kind == "forward_pre":

            def pre_hook(_module, inputs, *, key=name, target=resolved):
                activation = target.read(inputs)
                seen[key] = tuple(activation.shape)
                return target.rebuild(inputs, activation.clone())

            handles.append(resolved.module.register_forward_pre_hook(pre_hook))
        else:

            def forward_hook(_module, _inputs, output, *, key=name, target=resolved):
                activation = target.read(output)
                seen[key] = tuple(activation.shape)
                return target.rebuild(output, activation.clone())

            handles.append(resolved.module.register_forward_hook(forward_hook))
    try:
        output = model(
            language_hidden=torch.zeros(1, 6, 8),
            vision_hidden=torch.zeros(vision_shape),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == (1, 6, 8)
    assert all(seen[name] == (1, 6, 8) for name in language_sites)
    assert seen["vision"] == vision_shape
    assert seen["projector_in"] == projector_shapes[0]
    assert seen["projector_out"] == projector_shapes[1]


def test_vlm_site_helpers_and_invalid_cross_stream_components():
    model = FakeQwen2_5_VLForConditionalGeneration()
    adapter = get_adapter(model)

    assert sites.vision_resid(1) == Site("vision", "vision_resid", 1)
    assert sites.projector_in() == Site("projector", "projector_in", io="input")
    assert sites.projector_out() == Site("projector", "projector_out")
    with pytest.raises(Exception, match="vision stream requires"):
        adapter.resolve(model, Site("vision", "resid_post", 0))


class _VisionSequenceGate:
    evaluate_at = sites.vision_resid(0)

    def evaluate(self, context):
        return torch.ones(context.resolved_batch_size(), dtype=torch.bool)

    def to_dict(self):
        return {
            "type": "vision_sequence",
            "evaluate_at": self.evaluate_at.to_dict(),
        }


def test_compiler_rejects_non_language_sequence_gate_sites():
    wrapper = from_model(
        FakeQwen2_5_VLForConditionalGeneration().eval(),
        model_id="fake/qwen2.5-vl",
        revision="test-revision",
    )
    language_site = sites.resid_post(0)
    artifact = DirectionArtifact(
        ArtifactMetadata(
            model_id=wrapper.model_id,
            model_revision=wrapper.revision,
            architecture=wrapper.architecture,
            site=language_site,
            method="unit",
        ),
        torch.ones(8),
    )
    control = Intervention(
        artifact=artifact,
        operator=Add(),
        positions=ImageTokens(),
        strength=Constant(1),
        gate=_VisionSequenceGate(),
    )

    with pytest.raises(PlanCompilationError, match="only on the language stream"):
        wrapper.compile(control)


def test_qwen_dynamic_multi_image_modality_map_is_verifiable():
    model = FakeQwen2_5_VLForConditionalGeneration()
    ids = torch.tensor(
        [
            [97, 99, 99, 99, 99, 98, 7, 0, 0, 0, 0, 0, 0, 0, 0],
            [97, 99, 99, 99, 99, 99, 99, 99, 99, 98, 7, 99, 99, 98, 8],
        ]
    )
    batch = {
        "input_ids": ids,
        "attention_mask": ids != 0,
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 8], [1, 2, 4]]),
    }

    modality_map = get_adapter(model).build_modality_map(batch, model=model)

    assert modality_map is not None
    assert modality_map.image_patch_grids == [(4, 4), (4, 8), (2, 4)]
    assert modality_map.image_token_counts == (4, 8, 2)
    assert modality_map.image_batch_indices == (0, 1, 1)
    assert modality_map.image_indices == (0, 0, 1)
    assert modality_map.patch_to_token[0].reshape(4, 4).tolist() == [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ]
    # A 2x4 merged grid spans two vision windows, so projector order differs
    # from row-major language-token order.
    assert modality_map.token_to_projector[1].tolist() == [
        0,
        1,
        4,
        5,
        2,
        3,
        6,
        7,
    ]
    assert modality_map.text_token_mask.sum(dim=-1).tolist() == [1, 2]

    context = StepContext(
        phase="prefill",
        prompt_lengths=torch.tensor([7, 15]),
        attention_mask=ids != 0,
        token_ids=ids,
        modality_map=modality_map,
        metadata={"site_stream": "language", "site_component": "resid_post"},
    )
    selected = ImageTokens(image_index=1).select(torch.zeros(2, 15, 8), context)
    assert selected[0].sum() == 0
    assert selected[1].nonzero().flatten().tolist() == [11, 12]


def _qwen_single_image_map():
    model = FakeQwen2_5_VLForConditionalGeneration()
    ids = torch.tensor([[97, *([99] * 8), 98, 7]])
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "image_grid_thw": torch.tensor([[1, 4, 8]]),
    }
    return get_adapter(model).build_modality_map(batch, model=model)


def _context(modality_map, stream, component, sequence):
    return StepContext(
        phase="prefill",
        prompt_lengths=torch.tensor([modality_map.sequence_length]),
        batch_size=1,
        sequence_length=sequence,
        modality_map=modality_map,
        metadata={"site_stream": stream, "site_component": component},
    )


def test_qwen_patch_selector_tracks_window_merge_across_all_three_streams():
    modality_map = _qwen_single_image_map()
    spatial = torch.zeros(4, 8, dtype=torch.bool)
    spatial[0, 4] = True  # language token 2, projector-window position 4
    spatial[2, 0] = True  # language token 4, projector-window position 2
    selector = ImagePatches(spatial)

    language = selector.select(
        torch.zeros(1, 11, 8),
        _context(modality_map, "language", "resid_post", 11),
    )
    vision = selector.select(
        torch.zeros(32, 4),
        _context(modality_map, "vision", "vision_resid", 32),
    )
    projector_in = selector.select(
        torch.zeros(32, 4),
        _context(modality_map, "projector", "projector_in", 32),
    )
    projector_out = selector.select(
        torch.zeros(8, 8),
        _context(modality_map, "projector", "projector_out", 8),
    )

    assert language.nonzero(as_tuple=False).tolist() == [[0, 3], [0, 5]]
    assert vision.nonzero(as_tuple=False).tolist() == [[0, 8], [0, 16]]
    assert torch.equal(projector_in, vision)
    assert projector_out.nonzero(as_tuple=False).tolist() == [[0, 2], [0, 4]]


def test_internvl_pixel_shuffle_and_cls_offset_mapping():
    model = FakeInternVLForConditionalGeneration()
    ids = torch.tensor([[99, 99, 99, 99, 7, 99, 99, 99, 99]])
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "pixel_values": torch.zeros(2, 3, 8, 8),
    }
    modality_map = get_adapter(model).build_modality_map(batch, model=model)

    assert modality_map.image_indices == (0, 1)
    assert modality_map.image_patch_grids == [(4, 4), (4, 4)]
    assert modality_map.patch_to_token[0].reshape(4, 4).tolist() == [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ]
    assert modality_map.patch_to_vision[0].tolist() == list(range(1, 17))

    spatial = torch.zeros(4, 4, dtype=torch.bool)
    spatial[0, 0] = True
    selector = ImagePatches(spatial, image_index=1)
    language = selector.select(
        torch.zeros(1, 9, 8),
        _context(modality_map, "language", "resid_post", 9),
    )
    vision = selector.select(
        torch.zeros(2, 17, 4),
        _context(modality_map, "vision", "vision_resid", 17),
    )
    projector = selector.select(
        torch.zeros(2, 4, 8),
        _context(modality_map, "projector", "projector_out", 4),
    )

    assert language.nonzero(as_tuple=False).tolist() == [[0, 5]]
    assert vision.nonzero(as_tuple=False).tolist() == [[1, 1]]
    assert projector.nonzero(as_tuple=False).tolist() == [[1, 0]]


def test_internvl_contiguous_tiles_share_one_source_image_index():
    model = FakeInternVLForConditionalGeneration()
    ids = torch.tensor([[99, 99, 99, 99, 99, 99, 99, 99, 7]])
    modality_map = get_adapter(model).build_modality_map(
        {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.zeros(2, 3, 8, 8),
        },
        model=model,
    )

    assert modality_map.image_indices == (0, 0)
    context = _context(modality_map, "language", "resid_post", 9)
    selected = ImageTokens(image_index=0).select(torch.zeros(1, 9, 8), context)
    assert selected.nonzero(as_tuple=False).tolist() == [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
    ]


def test_internvl_tiled_object_boxes_fail_closed_but_patch_masks_remain_local():
    model = FakeInternVLForConditionalGeneration()
    ids = torch.tensor([[99, 99, 99, 99, 99, 99, 99, 99, 7]])
    modality_map = get_adapter(model).build_modality_map(
        {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.zeros(2, 3, 8, 8),
        },
        model=model,
    )
    context = _context(modality_map, "vision", "vision_resid", 17)

    with pytest.raises(PositionResolutionError, match="no tile crop transforms"):
        ObjectPatches([[0.0, 0.0, 0.5, 0.5]]).select(
            torch.zeros(2, 17, 4),
            context,
        )

    tile_local = torch.zeros(4, 4, dtype=torch.bool)
    tile_local[0, 0] = True
    selected = ImagePatches(tile_local).select(torch.zeros(2, 17, 4), context)
    assert selected.nonzero(as_tuple=False).tolist() == [[0, 1], [1, 1]]


def test_object_patch_and_selector_serialization_contract():
    modality_map = _qwen_single_image_map()
    selector = ObjectPatches([[0.0, 0.0, 0.5, 0.5]], image_index=0)
    restored = position_selector_from_dict(selector.to_dict())
    context = _context(modality_map, "language", "resid_post", 11)

    mask = restored.select(torch.zeros(1, 11, 8), context)

    assert mask.nonzero(as_tuple=False).tolist() == [[0, 1], [0, 2]]
    image = ImageTokens(image_index=None)
    assert position_selector_from_dict(image.to_dict()) == image


def test_vision_and_language_interventions_share_one_steering_plan():
    raw = FakeQwen2_5_VLForConditionalGeneration().eval()
    wrapper = from_model(
        raw,
        model_id="fake/qwen2.5-vl",
        revision="test-revision",
    )
    vision_site = sites.vision_resid(0)
    language_site = sites.resid_post(0)

    def artifact(site, hidden_size):
        return DirectionArtifact(
            ArtifactMetadata(
                model_id=wrapper.model_id,
                model_revision=wrapper.revision,
                architecture=wrapper.architecture,
                site=site,
                hidden_size=hidden_size,
                method="unit",
            ),
            torch.ones(hidden_size),
        )

    spatial = torch.zeros(4, 4, dtype=torch.bool)
    spatial[:2, :2] = True
    plan = SteeringPlan(
        [
            Intervention(
                artifact=artifact(vision_site, 4),
                site=vision_site,
                operator=Add(),
                positions=ImagePatches(spatial),
                strength=Constant(0.5),
                phase="both",
            ),
            Intervention(
                artifact=artifact(language_site, 8),
                site=language_site,
                operator=Add(),
                positions=ImageTokens(),
                strength=Constant(0.25),
                phase="both",
            ),
        ]
    )
    batch = {
        "input_ids": torch.tensor([[97, 99, 99, 99, 99, 98, 7]]),
        "attention_mask": torch.ones(1, 7, dtype=torch.long),
        "pixel_values": torch.zeros(16, 4),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }

    input_ids = batch["input_ids"]
    model_kwargs = {key: value for key, value in batch.items() if key != "input_ids"}
    baseline = wrapper(input_ids, **model_kwargs)
    compiled = wrapper.compile(plan)
    with wrapper.steer(compiled):
        steered = wrapper(input_ids, **model_kwargs)

    assert not torch.equal(steered, baseline)
    assert "model.visual.blocks.0" in compiled.explain()
    assert "model.language_model.layers.0" in compiled.explain()
    assert wrapper.hook_manager.hook_count == 0


def test_positional_vlm_input_ids_are_available_to_generation_modality_mapping():
    raw = FakeQwen2_5_VLForConditionalGeneration().eval()
    wrapper = from_model(raw)
    input_ids = torch.tensor([[97, 99, 99, 99, 99, 98, 7]])

    prepared = wrapper._prepare_generation(
        (input_ids,),
        {
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.zeros(16, 4),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        },
    )

    assert prepared[0][0] is input_ids
    assert prepared[-1].image_token_counts == (4,)


def test_modality_map_rejects_processor_model_token_count_mismatch():
    qwen = FakeQwen2_5_VLForConditionalGeneration()
    with pytest.raises(PositionResolutionError, match="image-token mismatch"):
        get_adapter(qwen).build_modality_map(
            {
                "input_ids": torch.tensor([[99, 99, 99]]),
                "image_grid_thw": torch.tensor([[1, 4, 4]]),
            },
            model=qwen,
        )

    internvl = FakeInternVLForConditionalGeneration()
    internvl.config.image_seq_length = 5
    with pytest.raises(PositionResolutionError, match="sequence mismatch"):
        get_adapter(internvl).build_modality_map(
            {
                "input_ids": torch.tensor([[99, 99, 99, 99]]),
                "pixel_values": torch.zeros(1, 3, 8, 8),
            },
            model=internvl,
        )
