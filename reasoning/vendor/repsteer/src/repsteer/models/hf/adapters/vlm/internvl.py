"""InternVL architecture adapter and pixel-shuffle modality mapping."""

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


def _pair(value: Any, *, name: str) -> tuple[int, int]:
    if isinstance(value, tuple | list):
        if len(value) != 2:
            raise PositionResolutionError(f"{name} must contain two dimensions")
        return int(value[0]), int(value[1])
    size = int(value)
    return size, size


def _pixel_shuffle_mapping(height: int, width: int, *, downsample_group: int) -> Tensor:
    output_width = width // downsample_group
    rows = torch.arange(height, dtype=torch.long)[:, None].expand(height, width)
    columns = torch.arange(width, dtype=torch.long)[None, :].expand(height, width)
    return (
        (rows // downsample_group) * output_width + columns // downsample_group
    ).reshape(-1)


class InternVLAdapter(VisionLanguageAdapter):
    """Semantic adapter for native HF and common remote-code InternVL layouts."""

    architecture_name = "internvl"
    model_types = frozenset({"internvl", "internvl_chat"})
    class_prefixes = ("InternVL",)
    language_layer_paths = (
        ("model.language_model.layers", ("model", "language_model", "layers")),
        (
            "model.language_model.model.layers",
            ("model", "language_model", "model", "layers"),
        ),
        ("language_model.layers", ("language_model", "layers")),
        (
            "language_model.model.layers",
            ("language_model", "model", "layers"),
        ),
    )
    vision_layer_paths = (
        (
            "model.vision_tower.encoder.layer",
            ("model", "vision_tower", "encoder", "layer"),
        ),
        (
            "vision_tower.encoder.layer",
            ("vision_tower", "encoder", "layer"),
        ),
        (
            "model.vision_model.encoder.layers",
            ("model", "vision_model", "encoder", "layers"),
        ),
        (
            "vision_model.encoder.layers",
            ("vision_model", "encoder", "layers"),
        ),
        (
            "vision_model.encoder.layer",
            ("vision_model", "encoder", "layer"),
        ),
    )
    projector_paths = (
        (
            "model.multi_modal_projector",
            ("model", "multi_modal_projector"),
        ),
        ("multi_modal_projector", ("multi_modal_projector",)),
        ("model.mlp1", ("model", "mlp1")),
        ("mlp1", ("mlp1",)),
    )

    def _downsample_group(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        ratio = float(getattr(config, "downsample_ratio", 0.0))
        if not 0 < ratio <= 1:
            raise PositionResolutionError(
                "InternVL config.downsample_ratio must lie in (0, 1]"
            )
        group = round(1.0 / ratio)
        if group <= 0 or abs(ratio * group - 1.0) > 1e-6:
            raise PositionResolutionError(
                "InternVL support requires downsample_ratio to be the reciprocal "
                "of a positive integer"
            )
        return int(group)

    def _projector_input_size(self, model: nn.Module) -> int:
        return self._vision_hidden_size(model) * self._downsample_group(model) ** 2

    def build_modality_map(
        self,
        batch: Mapping[str, Tensor],
        *,
        model: nn.Module | None = None,
    ) -> ModalityMap | None:
        pixel_value = batch.get("pixel_values")
        if pixel_value is None:
            return None
        if model is None:
            raise PositionResolutionError(
                "InternVL modality mapping requires the target model config"
            )
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise PositionResolutionError(
                "InternVL modality mapping requires processor input_ids"
            )
        pixels = torch.as_tensor(pixel_value)
        if pixels.ndim == 5:
            pixels = pixels.flatten(0, 1)
        if pixels.ndim != 4:
            raise PositionResolutionError(
                "InternVL pixel_values must have shape [num_images, C, H, W] "
                "or [batch, num_images, C, H, W]"
            )
        config: Any = getattr(model, "config", None)
        vision_config: Any = getattr(config, "vision_config", None)
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is None:
            raise PositionResolutionError("InternVL config.image_token_id is absent")
        strategy = str(
            batch.get(
                "vision_feature_select_strategy",
                getattr(config, "vision_feature_select_strategy", "default"),
            )
        )
        if strategy != "default":
            raise PositionResolutionError(
                "InternVL modality mapping requires "
                "vision_feature_select_strategy='default'"
            )
        patch_height, patch_width = _pair(
            getattr(vision_config, "patch_size", 0),
            name="vision_config.patch_size",
        )
        if patch_height <= 0 or patch_width <= 0:
            raise PositionResolutionError(
                "InternVL vision_config.patch_size must be positive"
            )
        image_height, image_width = int(pixels.shape[-2]), int(pixels.shape[-1])
        if image_height % patch_height or image_width % patch_width:
            raise PositionResolutionError(
                f"InternVL pixel size {image_height}x{image_width} is not divisible "
                f"by patch size {patch_height}x{patch_width}"
            )
        grid_height = image_height // patch_height
        grid_width = image_width // patch_width
        # The native HF implementation reshapes the sequence through sqrt(N).
        if grid_height != grid_width:
            raise PositionResolutionError(
                "InternVL native pixel shuffle requires a square patch grid"
            )
        group = self._downsample_group(model)
        if grid_height % group or grid_width % group:
            raise PositionResolutionError(
                f"InternVL patch grid {grid_height}x{grid_width} is incompatible "
                f"with downsample_ratio={getattr(config, 'downsample_ratio', None)}"
            )
        projected_count = (grid_height // group) * (grid_width // group)
        configured_count = getattr(config, "image_seq_length", None)
        if configured_count is not None and int(configured_count) != projected_count:
            raise PositionResolutionError(
                "InternVL processor/model image sequence mismatch: pixel shuffle "
                f"produces {projected_count} tokens but config.image_seq_length="
                f"{configured_count}"
            )
        num_images = int(pixels.shape[0])
        token_counts = [projected_count] * num_images
        image_masks, batch_indices, image_indices = split_image_token_occurrences(
            input_ids,
            image_token_id=int(image_token_id),
            token_counts=token_counts,
            group_contiguous=True,
        )
        patch_mapping = _pixel_shuffle_mapping(
            grid_height,
            grid_width,
            downsample_group=group,
        )
        grids = [(grid_height, grid_width)] * num_images
        patch_to_token = [patch_mapping.clone() for _ in range(num_images)]
        # Vision encoder layer outputs retain the CLS token at position zero.
        patch_to_vision = [
            torch.arange(1, grid_height * grid_width + 1, dtype=torch.long)
            for _ in range(num_images)
        ]
        token_to_projector = [
            torch.arange(projected_count, dtype=torch.long) for _ in range(num_images)
        ]
        special_ids = {
            "<image>": int(image_token_id),
            "<IMG_CONTEXT>": int(image_token_id),
        }
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
            vision_layout="batched",
            projector_layout="batched",
            architecture_name=self.architecture_name,
        )


__all__ = ["InternVLAdapter"]
