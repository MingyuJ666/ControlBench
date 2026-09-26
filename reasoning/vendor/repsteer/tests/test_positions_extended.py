"""CPU-only boundary tests for text and special-token position selectors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import pytest
import torch

from repsteer.core import PositionResolutionError, StepContext
from repsteer.positions import SpecialToken, TextSpan


def _activation(batch: int = 1, sequence: int = 1) -> torch.Tensor:
    return torch.zeros(batch, sequence, 2)


def _context(
    batch: int,
    sequence: int,
    **updates: Any,
) -> StepContext:
    values: dict[str, Any] = {
        "phase": "forward",
        "prompt_lengths": torch.full((batch,), sequence),
        "attention_mask": torch.ones(batch, sequence, dtype=torch.bool),
    }
    values.update(updates)
    return StepContext(**values)


@pytest.mark.parametrize(
    ("occurrence", "expected"),
    [
        ("first", [[True, True, True, False, False, False, False]]),
        ("last", [[False, False, False, False, True, True, True]]),
        ("all", [[True, True, True, False, True, True, True]]),
    ],
)
def test_text_span_offsets_honor_first_last_and_all(
    occurrence: Literal["first", "last", "all"], expected: list[list[bool]]
):
    context = _context(
        1,
        7,
        texts=["red fox red fox"],
        token_offsets=torch.tensor(
            [[[0, 3], [3, 4], [4, 7], [7, 8], [8, 11], [11, 12], [12, 15]]]
        ),
    )

    actual = TextSpan("red fox", occurrence=occurrence).select(
        _activation(1, 7), context
    )

    assert actual.tolist() == expected


def test_text_span_supports_casefolded_metadata_and_suffix_aligned_offsets():
    metadata_context = _context(
        1,
        3,
        metadata={
            "texts": ["MiXeD Case"],
            "offset_mapping": torch.tensor([[[0, 5], [5, 6], [6, 10]]]),
        },
    )
    suffix_context = _context(
        1,
        2,
        texts=["a b c"],
        token_offsets=torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]]),
    )

    assert TextSpan("mixed", case_sensitive=False).select(
        _activation(1, 3), metadata_context
    ).tolist() == [[True, False, False]]
    assert TextSpan("c").select(_activation(1, 2), suffix_context).tolist() == [
        [False, True]
    ]


def test_text_span_falls_back_to_token_strings_and_trims_full_history():
    context = _context(
        1,
        5,
        texts=["fox fox"],
        token_strings=[["history", "f", "ox", " ", "f", "ox"]],
    )

    assert TextSpan("fox", occurrence="all").select(
        _activation(1, 5), context
    ).tolist() == [[True, True, False, True, True]]


@pytest.mark.parametrize(
    "context",
    [
        _context(1, 3),
        _context(
            1,
            3,
            texts=["abc"],
            token_offsets=torch.zeros(1, 3),
        ),
        _context(
            1,
            3,
            texts=["abc"],
            token_offsets=torch.zeros(1, 2, 2),
        ),
        _context(
            1,
            3,
            texts=["abc"],
            token_strings=[["a", "b"]],
        ),
        _context(
            2,
            3,
            texts=["only one row"],
            token_offsets=torch.zeros(2, 3, 2),
        ),
    ],
)
def test_text_span_reports_missing_or_misaligned_context(context: StepContext):
    activation = _activation(
        int(context.prompt_lengths.numel()),
        3,
    )

    with pytest.raises(PositionResolutionError):
        TextSpan("a").select(activation, context)


def test_special_token_prefers_token_ids_and_supports_metadata_token_id():
    suffix_context = _context(
        2,
        2,
        token_ids=torch.tensor([[7, 42, 8, 42], [42, 9, 42, 10]]),
    )
    metadata_context = _context(
        1,
        3,
        token_ids=torch.tensor([[5, 6, 5]]),
        metadata={"special_token_ids": {"image": 5}},
    )

    assert SpecialToken("image", token_id=42).select(
        _activation(2, 2), suffix_context
    ).tolist() == [[False, True], [True, False]]
    assert SpecialToken("image").select(
        _activation(1, 3), metadata_context
    ).tolist() == [[True, False, True]]


def test_special_token_supports_boolean_and_index_modality_maps():
    boolean_map = SimpleNamespace(
        special_token_indices={
            "image": torch.tensor(
                [[False, True, False, False], [True, False, True, False]]
            )
        }
    )
    per_row_indices = {"special_token_indices": {"image": torch.tensor([1, 3])}}
    matrix_indices = {
        "special_token_indices": {"image": torch.tensor([[0, 3], [1, 2]])}
    }

    assert SpecialToken("image").select(
        _activation(2, 4), _context(2, 4, modality_map=boolean_map)
    ).tolist() == [[False, True, False, False], [True, False, True, False]]
    assert SpecialToken("image").select(
        _activation(2, 4), _context(2, 4, modality_map=per_row_indices)
    ).tolist() == [[False, True, False, False], [False, False, False, True]]
    assert SpecialToken("image").select(
        _activation(2, 4), _context(2, 4, modality_map=matrix_indices)
    ).tolist() == [[True, False, False, True], [False, True, True, False]]


def test_special_token_falls_back_to_suffix_aligned_token_strings():
    context = _context(
        1,
        2,
        token_strings=[["<pad>", "ordinary", "<image>", "ordinary"]],
    )

    assert SpecialToken("<image>").select(_activation(1, 2), context).tolist() == [
        [True, False]
    ]


@pytest.mark.parametrize(
    ("context", "sequence"),
    [
        (_context(1, 2), 2),
        (_context(2, 2, token_ids=torch.tensor([1, 2])), 2),
        (_context(1, 3, token_ids=torch.tensor([[1, 2]])), 3),
        (_context(1, 2, token_strings=[["<image>"]]), 2),
    ],
)
def test_special_token_reports_unresolvable_or_misaligned_context(
    context: StepContext, sequence: int
):
    activation = _activation(int(context.prompt_lengths.numel()), sequence)

    with pytest.raises(PositionResolutionError):
        SpecialToken("<image>", token_id=1).select(activation, context)


def test_text_and_special_token_reject_empty_names():
    with pytest.raises(ValueError, match="TextSpan.text cannot be empty"):
        TextSpan("")
    with pytest.raises(ValueError, match="SpecialToken.token cannot be empty"):
        SpecialToken("")
