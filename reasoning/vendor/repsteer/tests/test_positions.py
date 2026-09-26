import pytest
import torch

from repsteer.core import PositionResolutionError, StepContext
from repsteer.positions import (
    AllTokens,
    GeneratedTokens,
    LastNonPaddingToken,
    LastPromptToken,
    PromptTokens,
    TextSpan,
    TokenIndices,
)


@pytest.mark.parametrize(
    ("attention_mask", "expected_last"),
    [
        ([[1, 1, 0, 0], [1, 1, 1, 0]], [[0, 1, 0, 0], [0, 0, 1, 0]]),
        ([[0, 0, 1, 1], [0, 1, 1, 1]], [[0, 0, 0, 1], [0, 0, 0, 1]]),
    ],
)
def test_padding_aware_prompt_selectors(attention_mask, expected_last):
    activation = torch.zeros(2, 4, 3)
    context = StepContext(
        phase="prefill",
        prompt_lengths=torch.tensor([2, 3]),
        attention_mask=torch.tensor(attention_mask),
    )

    assert torch.equal(
        LastNonPaddingToken().select(activation, context),
        torch.tensor(expected_last).bool(),
    )
    assert torch.equal(
        LastPromptToken().select(activation, context),
        torch.tensor(expected_last).bool(),
    )
    assert torch.equal(
        PromptTokens().select(activation, context), torch.tensor(attention_mask).bool()
    )


def test_prefill_decode_selector_contract():
    activation = torch.zeros(2, 1, 3)
    decode = StepContext(
        phase="decode",
        prompt_lengths=torch.tensor([2, 3]),
        generation_step=0,
        attention_mask=torch.ones(2, 4),
    )

    assert not PromptTokens().select(activation, decode).any()
    assert not LastPromptToken().select(activation, decode).any()
    assert GeneratedTokens().select(activation, decode).all()
    assert AllTokens().select(activation, decode).all()

    prefill = decode.with_updates(phase="prefill", attention_mask=torch.ones(2, 1))
    assert not GeneratedTokens().select(activation, prefill).any()


def test_token_indices_and_text_span_are_explicit():
    activation = torch.zeros(1, 4, 2)
    context = StepContext(
        phase="forward",
        prompt_lengths=torch.tensor([4]),
        attention_mask=torch.ones(1, 4),
        texts=["kind fox"],
        token_offsets=torch.tensor([[[0, 4], [4, 5], [5, 8], [0, 0]]]),
    )

    assert TokenIndices([0, -1]).select(activation, context).tolist() == [
        [True, False, False, True]
    ]
    assert TextSpan("fox").select(activation, context).tolist() == [
        [False, False, True, False]
    ]
    with pytest.raises(PositionResolutionError, match="out of range"):
        TokenIndices([9], strict=True).select(activation, context)
