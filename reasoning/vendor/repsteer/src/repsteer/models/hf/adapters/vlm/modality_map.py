"""Validated processor-to-model modality alignment for vision-language models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from repsteer.core.errors import PositionResolutionError

SequenceLayout = Literal["flattened", "batched"]


def _as_bool_mask(value: Tensor, shape: tuple[int, int], *, name: str) -> Tensor:
    mask = torch.as_tensor(value)
    if mask.ndim == 1 and shape[0] == 1:
        mask = mask.unsqueeze(0)
    if tuple(mask.shape) != shape:
        raise PositionResolutionError(
            f"{name} has shape {tuple(mask.shape)}, expected {shape}"
        )
    return mask.to(dtype=torch.bool)


@dataclass(slots=True)
class ModalityMap:
    """A verifiable mapping between processor tokens and model-side image patches.

    The five leading fields are the RFC contract.  Each list entry describes one
    image occurrence in processor order.  ``patch_to_token[i]`` is indexed in
    spatial row-major patch order and contains *local* image-token ordinals; the
    actual language positions live in ``image_token_masks[i]``.

    The remaining fields make architecture-specific tensor layouts explicit.
    Qwen2.5-VL flattens all images into one vision sequence, while InternVL keeps
    one image per batch row and prefixes its vision sequence with a CLS token.
    """

    text_token_mask: Tensor
    image_token_masks: list[Tensor]
    image_patch_grids: list[tuple[int, int]]
    patch_to_token: list[Tensor]
    special_token_indices: dict[str, Tensor]
    image_batch_indices: tuple[int, ...] = ()
    image_indices: tuple[int, ...] = ()
    patch_to_vision: list[Tensor] = field(default_factory=list)
    token_to_projector: list[Tensor] = field(default_factory=list)
    vision_layout: SequenceLayout = "flattened"
    projector_layout: SequenceLayout = "flattened"
    architecture_name: str = "unknown"

    def __post_init__(self) -> None:
        text = torch.as_tensor(self.text_token_mask)
        if text.ndim != 2:
            raise PositionResolutionError(
                "ModalityMap.text_token_mask must have shape [batch, sequence]"
            )
        self.text_token_mask = text.to(dtype=torch.bool)
        shape = (int(text.shape[0]), int(text.shape[1]))
        count = len(self.image_token_masks)
        if len(self.image_patch_grids) != count or len(self.patch_to_token) != count:
            raise PositionResolutionError(
                "ModalityMap image masks, grids, and patch mappings must have "
                "the same number of entries"
            )

        self.image_token_masks = [
            _as_bool_mask(mask, shape, name=f"image_token_masks[{index}]")
            for index, mask in enumerate(self.image_token_masks)
        ]
        normalized_grids: list[tuple[int, int]] = []
        normalized_patch_to_token: list[Tensor] = []
        token_counts: list[int] = []
        for index, (grid, mapping, token_mask) in enumerate(
            zip(
                self.image_patch_grids,
                self.patch_to_token,
                self.image_token_masks,
                strict=True,
            )
        ):
            if len(grid) != 2:
                raise PositionResolutionError(
                    f"image_patch_grids[{index}] must be a (height, width) pair"
                )
            height, width = int(grid[0]), int(grid[1])
            if height <= 0 or width <= 0:
                raise PositionResolutionError(
                    f"image_patch_grids[{index}] must contain positive dimensions"
                )
            normalized_grids.append((height, width))
            values = torch.as_tensor(mapping, dtype=torch.long).reshape(-1)
            if values.numel() != height * width:
                raise PositionResolutionError(
                    f"patch_to_token[{index}] has {values.numel()} entries for "
                    f"a {height}x{width} patch grid"
                )
            if bool((values < 0).any()):
                raise PositionResolutionError(
                    f"patch_to_token[{index}] contains a negative token ordinal"
                )
            token_count = int(token_mask.sum().item())
            token_counts.append(token_count)
            unique = torch.unique(values, sorted=True)
            expected = torch.arange(token_count, dtype=torch.long, device=unique.device)
            if token_count == 0 or not torch.equal(unique, expected):
                raise PositionResolutionError(
                    f"patch_to_token[{index}] must cover every local image-token "
                    f"ordinal exactly in [0, {token_count - 1}]"
                )
            normalized_patch_to_token.append(values)
        self.image_patch_grids = normalized_grids
        self.patch_to_token = normalized_patch_to_token

        if not self.image_batch_indices:
            batch_indices: list[int] = []
            for index, mask in enumerate(self.image_token_masks):
                rows = torch.nonzero(mask.any(dim=-1), as_tuple=False).flatten()
                if rows.numel() != 1:
                    raise PositionResolutionError(
                        f"image_token_masks[{index}] must belong to exactly one "
                        "language batch row"
                    )
                batch_indices.append(int(rows[0].item()))
            self.image_batch_indices = tuple(batch_indices)
        if len(self.image_batch_indices) != count:
            raise PositionResolutionError(
                "image_batch_indices must contain one value per image"
            )
        if any(index < 0 or index >= shape[0] for index in self.image_batch_indices):
            raise PositionResolutionError("image_batch_indices contains an invalid row")

        if not self.image_indices:
            per_row: dict[int, int] = {}
            ordinals: list[int] = []
            for batch_index in self.image_batch_indices:
                ordinal = per_row.get(batch_index, 0)
                ordinals.append(ordinal)
                per_row[batch_index] = ordinal + 1
            self.image_indices = tuple(ordinals)
        if len(self.image_indices) != count or any(
            index < 0 for index in self.image_indices
        ):
            raise PositionResolutionError(
                "image_indices must contain one non-negative value per image"
            )

        if not self.patch_to_vision:
            self.patch_to_vision = [
                torch.arange(height * width, dtype=torch.long)
                for height, width in self.image_patch_grids
            ]
        if len(self.patch_to_vision) != count:
            raise PositionResolutionError(
                "patch_to_vision must contain one mapping per image"
            )
        normalized_patch_to_vision: list[Tensor] = []
        for index, (grid, mapping) in enumerate(
            zip(self.image_patch_grids, self.patch_to_vision, strict=True)
        ):
            values = torch.as_tensor(mapping, dtype=torch.long).reshape(-1)
            if values.numel() != grid[0] * grid[1] or bool((values < 0).any()):
                raise PositionResolutionError(
                    f"patch_to_vision[{index}] must contain one non-negative "
                    "position per spatial patch"
                )
            if torch.unique(values).numel() != values.numel():
                raise PositionResolutionError(
                    f"patch_to_vision[{index}] must be one-to-one"
                )
            normalized_patch_to_vision.append(values)
        self.patch_to_vision = normalized_patch_to_vision

        if not self.token_to_projector:
            self.token_to_projector = [
                torch.arange(token_count, dtype=torch.long)
                for token_count in token_counts
            ]
        if len(self.token_to_projector) != count:
            raise PositionResolutionError(
                "token_to_projector must contain one mapping per image"
            )
        normalized_token_to_projector: list[Tensor] = []
        for index, (token_count, mapping) in enumerate(
            zip(token_counts, self.token_to_projector, strict=True)
        ):
            values = torch.as_tensor(mapping, dtype=torch.long).reshape(-1)
            if (
                values.numel() != token_count
                or bool((values < 0).any())
                or torch.unique(values).numel() != values.numel()
            ):
                raise PositionResolutionError(
                    f"token_to_projector[{index}] must be a one-to-one mapping "
                    f"for {token_count} image tokens"
                )
            normalized_token_to_projector.append(values)
        self.token_to_projector = normalized_token_to_projector

        if self.vision_layout not in ("flattened", "batched"):
            raise PositionResolutionError(
                "vision_layout must be 'flattened' or 'batched'"
            )
        if self.projector_layout not in ("flattened", "batched"):
            raise PositionResolutionError(
                "projector_layout must be 'flattened' or 'batched'"
            )
        self.special_token_indices = {
            str(name): torch.as_tensor(value)
            for name, value in self.special_token_indices.items()
        }

    @property
    def batch_size(self) -> int:
        return int(self.text_token_mask.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.text_token_mask.shape[1])

    @property
    def num_images(self) -> int:
        return len(self.image_token_masks)

    @property
    def image_token_counts(self) -> tuple[int, ...]:
        return tuple(int(mask.sum().item()) for mask in self.image_token_masks)

    def occurrences(self, image_index: int | None) -> tuple[int, ...]:
        """Return occurrence indices for a per-sample image ordinal."""

        if image_index is None:
            return tuple(range(self.num_images))
        if image_index < 0:
            raise PositionResolutionError("image_index cannot be negative")
        matches = tuple(
            occurrence
            for occurrence, ordinal in enumerate(self.image_indices)
            if ordinal == image_index
        )
        if not matches:
            raise PositionResolutionError(
                f"image_index={image_index} is absent from the modality map"
            )
        return matches


def split_image_token_occurrences(
    input_ids: Tensor,
    *,
    image_token_id: int,
    token_counts: list[int],
    group_contiguous: bool = False,
) -> tuple[list[Tensor], tuple[int, ...], tuple[int, ...]]:
    """Partition placeholder positions according to processor image order.

    ``group_contiguous`` is used by tiled InternVL processors: several vision
    batch rows can belong to one source image and are represented by one long,
    contiguous placeholder run.  Those tile occurrences receive the same
    per-sample ``image_index`` while retaining separate patch mappings.
    """

    ids = torch.as_tensor(input_ids)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2:
        raise PositionResolutionError("input_ids must have shape [batch, sequence]")
    coordinates = torch.nonzero(ids == int(image_token_id), as_tuple=False)
    required = sum(token_counts)
    if int(coordinates.shape[0]) != required:
        raise PositionResolutionError(
            "processor/model image-token mismatch: modality metadata requires "
            f"{required} image tokens, input_ids contains {coordinates.shape[0]}"
        )

    masks: list[Tensor] = []
    batch_indices: list[int] = []
    image_indices: list[int] = []
    next_row_ordinal: dict[int, int] = {}
    previous_end: dict[int, int] = {}
    previous_ordinal: dict[int, int] = {}
    offset = 0
    for occurrence, token_count in enumerate(token_counts):
        if token_count <= 0:
            raise PositionResolutionError(
                f"image occurrence {occurrence} has no projected tokens"
            )
        selected = coordinates[offset : offset + token_count]
        offset += token_count
        rows = torch.unique(selected[:, 0])
        if rows.numel() != 1:
            raise PositionResolutionError(
                f"image occurrence {occurrence} crosses language batch rows"
            )
        row = int(rows[0].item())
        positions = selected[:, 1]
        if positions.numel() > 1 and not bool(
            torch.all(positions[1:] == positions[:-1] + 1)
        ):
            raise PositionResolutionError(
                f"image occurrence {occurrence} placeholder tokens are not contiguous"
            )
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[row, positions] = True
        masks.append(mask)
        batch_indices.append(row)
        continues_source_image = (
            group_contiguous
            and row in previous_end
            and int(positions[0].item()) == previous_end[row] + 1
        )
        if continues_source_image:
            ordinal = previous_ordinal[row]
        else:
            ordinal = next_row_ordinal.get(row, 0)
            next_row_ordinal[row] = ordinal + 1
        image_indices.append(ordinal)
        previous_end[row] = int(positions[-1].item())
        previous_ordinal[row] = ordinal
    return masks, tuple(batch_indices), tuple(image_indices)


def text_and_special_masks(
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None,
    special_token_ids: dict[str, int],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Build aligned text and named-special masks without tokenizer imports."""

    ids = torch.as_tensor(input_ids)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2:
        raise PositionResolutionError("input_ids must have shape [batch, sequence]")
    if attention_mask is None:
        text = torch.ones_like(ids, dtype=torch.bool)
    else:
        attention = torch.as_tensor(attention_mask, device=ids.device)
        if attention.ndim == 1 and ids.shape[0] == 1:
            attention = attention.unsqueeze(0)
        if tuple(attention.shape) != tuple(ids.shape):
            raise PositionResolutionError(
                "attention_mask must have the same shape as input_ids when "
                "building a modality map"
            )
        text = attention.to(dtype=torch.bool)
    special: dict[str, Tensor] = {}
    for name, token_id in special_token_ids.items():
        mask = ids == int(token_id)
        special[name] = mask
        text &= ~mask
    return text, special


__all__ = [
    "ModalityMap",
    "SequenceLayout",
    "split_image_token_occurrences",
    "text_and_special_masks",
]
