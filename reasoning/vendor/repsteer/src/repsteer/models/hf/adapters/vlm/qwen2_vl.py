"""Qwen2.5-VL architecture adapter and dynamic-resolution modality mapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from repsteer.core.errors import PositionResolutionError

from .base import VisionLanguageAdapter
from .modality_map import (
    ModalityMap,
    split_image_token_occurrences,
    text_and_special_masks,
)


def _window_permutation(
    output_height: int, output_width: int, window_size: int
) -> Tensor:
    """Match Qwen's window-major permutation of spatial-merge groups."""

    if window_size <= 0:
        raise PositionResolutionError(
            "Qwen2.5-VL vision window size must resolve to a positive value"
        )
    values = [
        row * output_width + col
        for window_row in range(0, output_height, window_size)
        for window_col in range(0, output_width, window_size)
        for row in range(window_row, min(window_row + window_size, output_height))
        for col in range(window_col, min(window_col + window_size, output_width))
    ]
    return torch.tensor(values, dtype=torch.long)


def _qwen_patch_mapping(
    height: int,
    width: int,
    *,
    merge_size: int,
    merger_window_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map row-major spatial patches to language, vision, and projector axes."""

    output_height = height // merge_size
    output_width = width // merge_size
    window_index = _window_permutation(output_height, output_width, merger_window_size)
    token_count = output_height * output_width
    inverse_window = torch.empty(token_count, dtype=torch.long)
    inverse_window[window_index] = torch.arange(token_count, dtype=torch.long)

    rows = torch.arange(height, dtype=torch.long)[:, None].expand(height, width)
    columns = torch.arange(width, dtype=torch.long)[None, :].expand(height, width)
    patch_to_token = (
        (rows // merge_size) * output_width + columns // merge_size
    ).reshape(-1)
    within_group = ((rows % merge_size) * merge_size + columns % merge_size).reshape(-1)
    patch_to_vision = inverse_window[patch_to_token] * (merge_size**2) + within_group
    token_to_projector = inverse_window
    return patch_to_token, patch_to_vision, token_to_projector


class Qwen2_5_VLAdapter(VisionLanguageAdapter):
    """Semantic adapter for Hugging Face Qwen2.5-VL implementations."""

    architecture_name = "qwen2_5_vl"
    model_types = frozenset({"qwen2_5_vl", "qwen2_5_vl_moe"})
    class_prefixes = ("Qwen2_5_VL", "Qwen2_5VL")
    language_layer_paths = (
        ("model.language_model.layers", ("model", "language_model", "layers")),
        ("language_model.layers", ("language_model", "layers")),
        (
            "model.model.language_model.layers",
            ("model", "model", "language_model", "layers"),
        ),
    )
    vision_layer_paths = (
        ("model.visual.blocks", ("model", "visual", "blocks")),
        ("visual.blocks", ("visual", "blocks")),
        ("model.model.visual.blocks", ("model", "model", "visual", "blocks")),
    )
    projector_paths = (
        ("model.visual.merger", ("model", "visual", "merger")),
        ("visual.merger", ("visual", "merger")),
        ("model.model.visual.merger", ("model", "model", "visual", "merger")),
    )

    def _projector_output_size(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        size = getattr(vision_config, "out_hidden_size", None)
        if size is not None:
            return int(size)
        return super()._projector_output_size(model)

    def build_modality_map(
        self,
        batch: Mapping[str, Tensor],
        *,
        model: nn.Module | None = None,
    ) -> ModalityMap | None:
        grid_value = batch.get("image_grid_thw")
        if grid_value is None:
            return None
        if model is None:
            raise PositionResolutionError(
                "Qwen2.5-VL modality mapping requires the target model config"
            )
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise PositionResolutionError(
                "Qwen2.5-VL modality mapping requires processor input_ids"
            )
        config: Any = getattr(model, "config", None)
        vision_config: Any = getattr(config, "vision_config", None)
        merge_size = int(getattr(vision_config, "spatial_merge_size", 0))
        patch_size = int(getattr(vision_config, "patch_size", 0))
        window_size = int(getattr(vision_config, "window_size", 0))
        image_token_id = getattr(config, "image_token_id", None)
        if merge_size <= 0 or patch_size <= 0 or window_size <= 0:
            raise PositionResolutionError(
                "Qwen2.5-VL config must define positive spatial_merge_size, "
                "patch_size, and window_size"
            )
        if image_token_id is None:
            raise PositionResolutionError("Qwen2.5-VL config.image_token_id is absent")
        merger_window_size = window_size // merge_size // patch_size
        grid = torch.as_tensor(grid_value, dtype=torch.long)
        if grid.ndim == 1:
            grid = grid.unsqueeze(0)
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise PositionResolutionError(
                "image_grid_thw must have shape [num_images, 3]"
            )

        grids: list[tuple[int, int]] = []
        patch_to_token: list[Tensor] = []
        patch_to_vision: list[Tensor] = []
        token_to_projector: list[Tensor] = []
        token_counts: list[int] = []
        for occurrence, raw_grid in enumerate(grid.tolist()):
            temporal, height, width = map(int, raw_grid)
            if temporal != 1:
                raise PositionResolutionError(
                    "repsteer maps static Qwen2.5-VL images only; "
                    f"image_grid_thw[{occurrence}] has temporal size {temporal}"
                )
            if height <= 0 or width <= 0 or height % merge_size or width % merge_size:
                raise PositionResolutionError(
                    f"image_grid_thw[{occurrence}]={raw_grid!r} is incompatible "
                    f"with spatial_merge_size={merge_size}"
                )
            token_count = (height // merge_size) * (width // merge_size)
            language, vision, projector = _qwen_patch_mapping(
                height,
                width,
                merge_size=merge_size,
                merger_window_size=merger_window_size,
            )
            grids.append((height, width))
            patch_to_token.append(language)
            patch_to_vision.append(vision)
            token_to_projector.append(projector)
            token_counts.append(token_count)

        image_masks, batch_indices, image_indices = split_image_token_occurrences(
            input_ids,
            image_token_id=int(image_token_id),
            token_counts=token_counts,
        )
        special_ids = {
            "<image>": int(image_token_id),
            "<|image_pad|>": int(image_token_id),
        }
        for name, attribute in (
            ("<|vision_start|>", "vision_start_token_id"),
            ("<|vision_end|>", "vision_end_token_id"),
        ):
            token_id = getattr(config, attribute, None)
            if token_id is not None:
                special_ids[name] = int(token_id)
        text_mask, special = text_and_special_masks(
            input_ids,
            attention_mask=batch.get("attention_mask"),
            special_token_ids=special_ids,
        )
        return ModalityMap(
            text_token_mask=text_mask,
            image_token_masks=image_masks,
            image_patch_grids=grids,
            patch_to_token=patch_to_token,
            special_token_indices=special,
            image_batch_indices=batch_indices,
            image_indices=image_indices,
            patch_to_vision=patch_to_vision,
            token_to_projector=token_to_projector,
            vision_layout="flattened",
            projector_layout="flattened",
            architecture_name=self.architecture_name,
        )


# A spelling without the tokenizer-style underscore is convenient in user code.
Qwen25VLAdapter = Qwen2_5_VLAdapter
Qwen2VLAdapter = Qwen2_5_VLAdapter


__all__ = ["Qwen25VLAdapter", "Qwen2VLAdapter", "Qwen2_5_VLAdapter"]
