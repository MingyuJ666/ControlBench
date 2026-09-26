"""Image-token and spatial-patch selectors backed by a validated ModalityMap."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from repsteer.core import PositionResolutionError, StepContext
from repsteer.models.hf.adapters.vlm.modality_map import ModalityMap

from .base import PositionSelector, empty_mask


def _modality_map(context: StepContext) -> ModalityMap:
    value = context.modality_map
    if value is None:
        raise PositionResolutionError(
            "image positions require StepContext.modality_map; use a supported "
            "VLM processor and architecture adapter"
        )
    if not isinstance(value, ModalityMap):
        raise PositionResolutionError(
            f"unsupported modality map type {type(value).__name__}; expected "
            "repsteer.models.hf.adapters.ModalityMap"
        )
    return value


def _site_hint(context: StepContext) -> tuple[str | None, str | None]:
    return (
        context.metadata.get("site_stream"),
        context.metadata.get("site_component"),
    )


def _repeat_language_mask(mask: Tensor, activation: Tensor) -> Tensor:
    batch, sequence = (
        (int(activation.shape[0]), int(activation.shape[-2]))
        if activation.ndim >= 3
        else (1, int(activation.shape[-2]))
    )
    if sequence != mask.shape[1]:
        raise PositionResolutionError(
            "language activation sequence does not align with the processor "
            f"modality map: {sequence} != {mask.shape[1]}"
        )
    if batch == mask.shape[0]:
        return mask.to(device=activation.device)
    if batch % int(mask.shape[0]) != 0:
        raise PositionResolutionError(
            f"activation batch {batch} is not an expansion of modality-map "
            f"batch {mask.shape[0]}"
        )
    factor = batch // int(mask.shape[0])
    return mask.repeat_interleave(factor, dim=0).to(device=activation.device)


def _language_token_mask(
    modality_map: ModalityMap,
    occurrences: tuple[int, ...],
    *,
    selected_tokens: dict[int, Tensor] | None = None,
) -> Tensor:
    result = torch.zeros_like(modality_map.text_token_mask, dtype=torch.bool)
    for occurrence in occurrences:
        image_mask = modality_map.image_token_masks[occurrence]
        if selected_tokens is None:
            result |= image_mask.to(device=result.device)
            continue
        positions = torch.nonzero(
            image_mask[modality_map.image_batch_indices[occurrence]],
            as_tuple=False,
        ).flatten()
        local = selected_tokens[occurrence].to(
            device=positions.device, dtype=torch.bool
        )
        if local.numel() != positions.numel():
            raise PositionResolutionError(
                f"image occurrence {occurrence} has {positions.numel()} language "
                f"tokens but patch mapping selected from {local.numel()} entries"
            )
        row = modality_map.image_batch_indices[occurrence]
        result[row, positions[local]] = True
    return result


def _flattened_mask(
    activation: Tensor,
    modality_map: ModalityMap,
    occurrences: tuple[int, ...],
    selected_positions: dict[int, Tensor],
    *,
    lengths: tuple[int, ...],
) -> Tensor:
    if activation.ndim != 2:
        raise PositionResolutionError(
            "flattened VLM layout expects a rank-2 [sequence, hidden] tensor"
        )
    sequence = int(activation.shape[-2])
    expected = sum(lengths)
    if sequence != expected:
        raise PositionResolutionError(
            f"flattened VLM activation length {sequence} does not match modality "
            f"layout length {expected}"
        )
    result = torch.zeros((1, sequence), dtype=torch.bool, device=activation.device)
    offset = 0
    selected_set = set(occurrences)
    for occurrence, length in enumerate(lengths):
        if occurrence in selected_set:
            positions = selected_positions[occurrence].to(
                device=activation.device, dtype=torch.long
            )
            if positions.numel() and (
                bool((positions < 0).any()) or bool((positions >= length).any())
            ):
                raise PositionResolutionError(
                    f"image occurrence {occurrence} mapping exceeds its local "
                    f"activation length {length}"
                )
            result[0, positions + offset] = True
        offset += length
    return result


def _batched_mask(
    activation: Tensor,
    modality_map: ModalityMap,
    occurrences: tuple[int, ...],
    selected_positions: dict[int, Tensor],
    *,
    lengths: tuple[int, ...],
) -> Tensor:
    if activation.ndim < 3:
        raise PositionResolutionError("batched VLM layout expects a rank-3 tensor")
    batch, sequence = int(activation.shape[0]), int(activation.shape[-2])
    if batch != modality_map.num_images:
        raise PositionResolutionError(
            f"batched VLM activation has {batch} image rows, modality map has "
            f"{modality_map.num_images}"
        )
    if any(length != sequence for length in lengths):
        raise PositionResolutionError(
            "batched VLM activation cannot represent image occurrences with "
            "different sequence lengths"
        )
    result = torch.zeros((batch, sequence), dtype=torch.bool, device=activation.device)
    for occurrence in occurrences:
        positions = selected_positions[occurrence].to(
            device=activation.device, dtype=torch.long
        )
        if positions.numel() and (
            bool((positions < 0).any()) or bool((positions >= sequence).any())
        ):
            raise PositionResolutionError(
                f"image occurrence {occurrence} mapping exceeds activation length "
                f"{sequence}"
            )
        result[occurrence, positions] = True
    return result


def _patch_tokens(
    modality_map: ModalityMap, patch_masks: dict[int, Tensor]
) -> dict[int, Tensor]:
    selected: dict[int, Tensor] = {}
    for occurrence, patch_mask in patch_masks.items():
        mapping = modality_map.patch_to_token[occurrence]
        token_count = modality_map.image_token_counts[occurrence]
        token_mask = torch.zeros(token_count, dtype=torch.bool, device=mapping.device)
        chosen = mapping[patch_mask.to(device=mapping.device, dtype=torch.bool)]
        if chosen.numel():
            token_mask[chosen] = True
        selected[occurrence] = token_mask
    return selected


def _vision_positions(
    modality_map: ModalityMap, patch_masks: dict[int, Tensor]
) -> dict[int, Tensor]:
    return {
        occurrence: modality_map.patch_to_vision[occurrence][
            patch_mask.to(
                device=modality_map.patch_to_vision[occurrence].device,
                dtype=torch.bool,
            )
        ]
        for occurrence, patch_mask in patch_masks.items()
    }


def _projector_positions(
    modality_map: ModalityMap, selected_tokens: dict[int, Tensor]
) -> dict[int, Tensor]:
    return {
        occurrence: modality_map.token_to_projector[occurrence][
            token_mask.to(
                device=modality_map.token_to_projector[occurrence].device,
                dtype=torch.bool,
            )
        ]
        for occurrence, token_mask in selected_tokens.items()
    }


@dataclass(frozen=True, slots=True)
class ImageTokens(PositionSelector):
    """Select the image placeholder tokens for a per-sample image ordinal."""

    image_index: int | None = 0

    def __post_init__(self) -> None:
        if self.image_index is not None and self.image_index < 0:
            raise ValueError("ImageTokens.image_index cannot be negative")

    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        if context.phase == "decode":
            return empty_mask(activation, context)
        modality_map = _modality_map(context)
        stream, _ = _site_hint(context)
        if stream not in (None, "language"):
            raise PositionResolutionError(
                "ImageTokens selects fused language placeholder tokens; use "
                "ImagePatches or ObjectPatches for vision/projector sites"
            )
        occurrences = modality_map.occurrences(self.image_index)
        mask = _language_token_mask(modality_map, occurrences)
        return _repeat_language_mask(mask, activation)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "image_tokens", "image_index": self.image_index}


class _SpatialSelector(PositionSelector):
    image_index: int | None

    def _mask_for_grid(self, grid: tuple[int, int], occurrence: int) -> Tensor:
        raise NotImplementedError

    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        if context.phase == "decode":
            return empty_mask(activation, context)
        modality_map = _modality_map(context)
        occurrences = modality_map.occurrences(self.image_index)
        patch_masks = {
            occurrence: self._mask_for_grid(
                modality_map.image_patch_grids[occurrence], occurrence
            ).reshape(-1)
            for occurrence in occurrences
        }
        selected_tokens = _patch_tokens(modality_map, patch_masks)
        stream, component = _site_hint(context)

        if stream == "language":
            language = _language_token_mask(
                modality_map, occurrences, selected_tokens=selected_tokens
            )
            return _repeat_language_mask(language, activation)
        if stream == "vision" or (
            stream == "projector"
            and component == "projector_in"
            and modality_map.architecture_name == "qwen2_5_vl"
        ):
            return self._select_vision(
                activation, modality_map, occurrences, patch_masks
            )
        if stream == "projector":
            return self._select_projector(
                activation, modality_map, occurrences, selected_tokens
            )

        # Direct selector calls do not carry resolved-site metadata.  Infer the
        # target conservatively from exact public layout dimensions.
        if (
            activation.ndim >= 3
            and int(activation.shape[-2]) == modality_map.sequence_length
            and int(activation.shape[0]) % modality_map.batch_size == 0
        ):
            language = _language_token_mask(
                modality_map, occurrences, selected_tokens=selected_tokens
            )
            return _repeat_language_mask(language, activation)
        vision_lengths = tuple(
            int(mapping.max().item()) + 1 for mapping in modality_map.patch_to_vision
        )
        projector_lengths = tuple(
            int(mapping.max().item()) + 1 for mapping in modality_map.token_to_projector
        )
        if modality_map.vision_layout == "flattened" and activation.ndim == 2:
            if int(activation.shape[-2]) == sum(vision_lengths):
                return self._select_vision(
                    activation, modality_map, occurrences, patch_masks
                )
            if int(activation.shape[-2]) == sum(projector_lengths):
                return self._select_projector(
                    activation, modality_map, occurrences, selected_tokens
                )
        if modality_map.vision_layout == "batched" and activation.ndim >= 3:
            if all(length == int(activation.shape[-2]) for length in vision_lengths):
                return self._select_vision(
                    activation, modality_map, occurrences, patch_masks
                )
            if all(length == int(activation.shape[-2]) for length in projector_lengths):
                return self._select_projector(
                    activation, modality_map, occurrences, selected_tokens
                )
        raise PositionResolutionError(
            "activation shape does not match language, vision, or projector "
            "layout in the modality map"
        )

    @staticmethod
    def _select_vision(
        activation: Tensor,
        modality_map: ModalityMap,
        occurrences: tuple[int, ...],
        patch_masks: dict[int, Tensor],
    ) -> Tensor:
        positions = _vision_positions(modality_map, patch_masks)
        lengths = tuple(
            int(mapping.max().item()) + 1 for mapping in modality_map.patch_to_vision
        )
        if modality_map.vision_layout == "flattened":
            return _flattened_mask(
                activation,
                modality_map,
                occurrences,
                positions,
                lengths=lengths,
            )
        return _batched_mask(
            activation,
            modality_map,
            occurrences,
            positions,
            lengths=lengths,
        )

    @staticmethod
    def _select_projector(
        activation: Tensor,
        modality_map: ModalityMap,
        occurrences: tuple[int, ...],
        selected_tokens: dict[int, Tensor],
    ) -> Tensor:
        positions = _projector_positions(modality_map, selected_tokens)
        lengths = tuple(
            int(mapping.max().item()) + 1 for mapping in modality_map.token_to_projector
        )
        if modality_map.projector_layout == "flattened":
            return _flattened_mask(
                activation,
                modality_map,
                occurrences,
                positions,
                lengths=lengths,
            )
        return _batched_mask(
            activation,
            modality_map,
            occurrences,
            positions,
            lengths=lengths,
        )


@dataclass(frozen=True, slots=True)
class ImagePatches(_SpatialSelector):
    """Select a boolean spatial mask on an image's pre-merge patch grid."""

    mask: Any
    image_index: int | None = 0

    def __post_init__(self) -> None:
        value = torch.as_tensor(self.mask)
        if value.ndim != 2:
            raise ValueError("ImagePatches.mask must be a 2-D spatial mask")
        if self.image_index is not None and self.image_index < 0:
            raise ValueError("ImagePatches.image_index cannot be negative")

    def _mask_for_grid(self, grid: tuple[int, int], occurrence: int) -> Tensor:
        del occurrence
        value = torch.as_tensor(self.mask, dtype=torch.bool)
        if tuple(value.shape) != grid:
            raise PositionResolutionError(
                f"ImagePatches mask shape {tuple(value.shape)} does not match "
                f"the processor patch grid {grid}"
            )
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "image_patches",
            "mask": torch.as_tensor(self.mask, dtype=torch.bool).tolist(),
            "image_index": self.image_index,
        }


@dataclass(frozen=True, slots=True, init=False)
class ObjectPatches(_SpatialSelector):
    """Select patches whose centers lie in one or more normalized XYXY boxes."""

    boxes: tuple[tuple[float, float, float, float], ...]
    image_index: int | None

    def __init__(
        self,
        boxes: Iterable[Iterable[float]],
        *,
        image_index: int | None = 0,
    ) -> None:
        normalized = tuple(tuple(float(value) for value in box) for box in boxes)
        for box in normalized:
            if len(box) != 4:
                raise ValueError("ObjectPatches boxes must be XYXY quadruples")
            left, top, right, bottom = box
            if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
                raise ValueError(
                    "ObjectPatches boxes must satisfy 0 <= x1 < x2 <= 1 and "
                    "0 <= y1 < y2 <= 1"
                )
        if not normalized:
            raise ValueError("ObjectPatches requires at least one box")
        if image_index is not None and image_index < 0:
            raise ValueError("ObjectPatches.image_index cannot be negative")
        object.__setattr__(self, "boxes", normalized)
        object.__setattr__(self, "image_index", image_index)

    def select(self, activation: Tensor, context: StepContext) -> Tensor:
        if context.phase != "decode":
            modality_map = _modality_map(context)
            occurrences = modality_map.occurrences(self.image_index)
            source_keys = tuple(
                (
                    modality_map.image_batch_indices[occurrence],
                    modality_map.image_indices[occurrence],
                )
                for occurrence in occurrences
            )
            if len(source_keys) != len(set(source_keys)):
                raise PositionResolutionError(
                    "ObjectPatches cannot map original-image boxes for a source "
                    "image expanded into multiple tiles because the modality map "
                    "has no tile crop transforms; use ImagePatches with an explicit "
                    "tile-local mask"
                )
        return _SpatialSelector.select(self, activation, context)

    def _mask_for_grid(self, grid: tuple[int, int], occurrence: int) -> Tensor:
        del occurrence
        height, width = grid
        y = (torch.arange(height, dtype=torch.float32) + 0.5) / height
        x = (torch.arange(width, dtype=torch.float32) + 0.5) / width
        centers_y, centers_x = torch.meshgrid(y, x, indexing="ij")
        mask = torch.zeros(grid, dtype=torch.bool)
        for left, top, right, bottom in self.boxes:
            mask |= (
                (centers_x >= left)
                & (centers_x < right)
                & (centers_y >= top)
                & (centers_y < bottom)
            )
        return mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "object_patches",
            "boxes": [list(box) for box in self.boxes],
            "image_index": self.image_index,
        }


__all__ = ["ImagePatches", "ImageTokens", "ObjectPatches"]
