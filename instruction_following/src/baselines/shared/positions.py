"""Which token positions an intervention touches.

Papers disagree, and the disagreement is rarely stated: contrastive activation
addition adds to every position after the user's prompt, Tan et al. add at the
last prompt position only, and other work adds everywhere. The choice is not a
property of any one method — it is a property of the harness — so it lives
here, one setting shared by every baseline and recorded with every run.

``from_last_prompt_token`` is the default. It is the smallest set of
positions that can reach every generated token, including the first: an
autoregressive model writes its first token from the final prompt position's
hidden state, so leaving that position alone leaves the first token unsteered.
Measured on this benchmark, that one token decides the lowercase task
outright — 385 of 405 outputs at one configuration were already lowercase
everywhere except their first letter, and all of them failed.

Everything before that position is left as the model read it. The prompt
carries the question and the chat template's control tokens, and moving those
asks the model to answer something it can no longer read the same way.

The prompt's length is learned from the first forward pass inside the context
rather than passed in, so the rule holds whether or not the cache is on. With
a cache, later passes carry one new token; without one, they carry the whole
sequence again. Both cases reduce to the same question — is this position at
or past where the prompt ended.
"""

import torch

ALL = "all"
FROM_LAST_PROMPT_TOKEN = "from_last_prompt_token"
GENERATED_ONLY = "generated_only"
LAST_PROMPT_TOKEN = "last_prompt_token"
MODES = (ALL, FROM_LAST_PROMPT_TOKEN, GENERATED_ONLY, LAST_PROMPT_TOKEN)

DEFAULT = FROM_LAST_PROMPT_TOKEN


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"unknown position mode {mode!r}; expected {MODES}")
    return mode


class PositionMask:
    """Decides, per forward pass, which positions an addition applies to.

    One instance belongs to one intervention context. It counts the positions
    it has seen so it can tell the prompt from what follows it.
    """

    def __init__(self, mode: str = DEFAULT):
        self.mode = check_mode(mode)
        self.prompt_length: int | None = None
        self.seen = 0

    def apply(self, hidden: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Return ``hidden`` with ``delta`` added at the selected positions.

        ``hidden`` is ``[batch, seq, hidden]`` and ``delta`` broadcasts over
        the last dimension.
        """
        length = hidden.shape[1]
        if self.prompt_length is None:
            self.prompt_length = length
        # A single-token pass is a cached decoding step and continues where
        # the last one stopped; any longer pass carries the sequence from its
        # start, whether it is the prefill or a run with no cache.
        start = self.seen if length == 1 else 0
        self.seen = max(self.seen, start + length)

        if self.mode == ALL:
            return hidden + delta

        positions = torch.arange(start, start + length, device=hidden.device)
        if self.mode == FROM_LAST_PROMPT_TOKEN:
            selected = positions >= self.prompt_length - 1
        elif self.mode == GENERATED_ONLY:
            selected = positions >= self.prompt_length
        else:                                    # LAST_PROMPT_TOKEN
            selected = positions == self.prompt_length - 1

        if not bool(selected.any()):
            return hidden
        mask = selected.reshape(1, length, *([1] * (hidden.ndim - 2)))
        return hidden + mask.to(hidden.dtype) * delta
