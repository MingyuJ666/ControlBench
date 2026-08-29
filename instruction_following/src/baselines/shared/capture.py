"""Shared activation capture from contrastive extraction pairs.

Every direction-learning method reads the same thing from the model: the
hidden state at the final rendered input token, at every hidden-state index,
for the positive and negative member of each extraction pair. Only what a
method computes from those states differs — a mean difference, a probe, a
feature score — so the capture itself lives here rather than inside any one
method's producer.

Index 0 is the embedding output; index k >= 1 is the output of decoder block
k - 1, matching the injection convention in ``baselines.shared.residual_add``.
The top index is read from the last block directly rather than from the
model's returned hidden states, whose last entry is post-final-norm and so
would be a vector in one space injected into another.
"""

import torch

from kernel.generate import render_chat
from kernel.splits import extraction_pairs
from kernel.tasks import TASKS


def validate_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(
            f"Unknown task {task!r}; expected one of {tuple(TASKS)}"
        )


def validate_pairs(pairs: list[dict], task: str) -> None:
    """Reject pair sets no direction can be learned from."""
    if len(pairs) < 2:
        raise ValueError(
            f"Extraction for {task!r} requires at least two pairs; "
            f"got {len(pairs)}"
        )
    for index, pair in enumerate(pairs):
        for field in ("positive", "negative"):
            value = pair.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Extraction pair {index} has invalid {field!r} text"
                )
        if pair["positive"] == pair["negative"]:
            raise ValueError(
                f"Extraction pair {index} has identical positive and negative "
                "text"
            )


def validate_states(states: torch.Tensor, model, expected_rows: int,
                    label: str) -> None:
    if not isinstance(states, torch.Tensor) or states.ndim != 3:
        shape = getattr(states, "shape", None)
        raise ValueError(
            f"{label} states must have shape [batch, layers+1, hidden]; "
            f"got {shape}"
        )

    expected_layers = int(model.config.num_hidden_layers) + 1
    if states.shape[0] != expected_rows or states.shape[1] != expected_layers:
        raise ValueError(
            f"{label} states have shape {tuple(states.shape)}; expected "
            f"[{expected_rows}, {expected_layers}, hidden]"
        )

    expected_hidden = getattr(model.config, "hidden_size", None)
    if expected_hidden is not None and states.shape[2] != int(expected_hidden):
        raise ValueError(
            f"{label} states have hidden size {states.shape[2]}; expected "
            f"{expected_hidden}"
        )
    if not torch.isfinite(states).all():
        raise ValueError(f"{label} states contain NaN or Inf")


def has_contrast(positive: torch.Tensor, negative: torch.Tensor) -> bool:
    """Whether two activation sets differ at all at one site.

    Hidden-state index 0 is the embedding lookup for the final input token,
    which is the same chat-template token in both prompts, so the rows there
    are identical and nothing separates them. Individual attention heads can
    be identical for the same reason. A site with no contrast is not a
    candidate for steering — it just must not stop the others from being
    fitted, which is what an exception in the middle of a producer does.
    """
    return not torch.equal(positive, negative)


@torch.no_grad()
def last_token_states(model, tok, texts: list[str]) -> torch.Tensor:
    """Return float32 CPU states at the final rendered input token.

    The returned tensor has shape ``[len(texts), n_layers+1, hidden]``. The
    caller controls batching, so no call retains more than one batch of model
    states on CPU.
    """
    if not texts:
        raise ValueError("last_token_states requires at least one text")

    rendered = [render_chat(tok, text) for text in texts]
    encoded = tok(
        rendered,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)
    # The top hidden state transformers returns is last_hidden_state, which
    # for this architecture is the final RMSNorm applied to the last decoder
    # block's output — a different space from every other index, and from
    # where a direction is injected. Read that block's own output instead, so
    # index n_layers means what indices 1..n_layers-1 mean.
    final_block = {}

    def keep_final_block(_module, _args, output):
        final_block["states"] = (
            output[0] if isinstance(output, tuple) else output
        )

    handle = model.model.layers[-1].register_forward_hook(keep_final_block)
    try:
        hidden_states = model(
            # Only the hidden states are read. Without this the lm_head runs
            # over every position, building a [batch, seq, vocab] tensor that
            # is thrown away — about 0.6 GB a batch on this model.
            **encoded,
            output_hidden_states=True,
            logits_to_keep=1,
        ).hidden_states
    finally:
        handle.remove()
    if hidden_states is None:
        raise ValueError("Model did not return hidden states")
    if "states" not in final_block:
        raise ValueError("Final decoder block produced no output to capture")

    layers = list(hidden_states[:-1]) + [final_block["states"]]
    states = torch.stack(
        [layer[:, -1, :] for layer in layers], dim=1
    ).float().cpu()
    validate_states(states, model, len(texts), "last-token")
    return states


def pair_state_batches(model, tok, pairs: list[dict], batch: int):
    """Yield ``(start, positive_states, negative_states)`` batch by batch.

    Positives and negatives keep pair order, so a consumer can rely on row i
    of both tensors belonging to the same pair — which is what lets a probe
    keep a pair together in one fold.
    """
    if batch < 1:
        raise ValueError(f"batch must be positive; got {batch}")
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        positive = last_token_states(
            model, tok, [pair["positive"] for pair in chunk]
        )
        negative = last_token_states(
            model, tok, [pair["negative"] for pair in chunk]
        )
        validate_states(positive, model, len(chunk), "positive")
        validate_states(negative, model, len(chunk), "negative")
        if positive.shape != negative.shape:
            raise ValueError(
                "Positive and negative state shapes differ: "
                f"{tuple(positive.shape)} != {tuple(negative.shape)}"
            )
        yield start, positive, negative


def task_pairs(task: str) -> list[dict]:
    """Validated extraction pairs for one task."""
    validate_task(task)
    pairs = extraction_pairs(task)
    validate_pairs(pairs, task)
    return pairs
