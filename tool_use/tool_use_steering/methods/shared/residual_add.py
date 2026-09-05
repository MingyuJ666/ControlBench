"""Residual-stream addition: the one injection mechanism vector methods share.

Methods differ in where their direction COMES FROM (mean difference, probe
weights, a dictionary feature); most of them apply it the same way -- add
alpha * v to one decoder layer's output stream. Implementing that once is
what keeps the comparison honest: no method gets a subtly different
injection. The one switch is WHICH positions are pushed, and each method's
paper sets it:

  positions="all"   every position of every forward pass, prompt and
                    generation alike -- Arditi et al.'s activation addition
                    (dim, probe, sae).
  positions="last"  only the last position of every forward pass: on the
                    prompt pass the final prompt token, on each decode step
                    the token being generated. Together that is exactly
                    "from the end of the instruction onward", the rule in
                    Rimsky et al.'s official CAA code (caa).

Indexing convention (inherited from the extraction archives): hidden_idx
refers to rows of the [n_layers+1, hidden] stack captured with
output_hidden_states, where row 0 is the embeddings and row L is the raw
output of decoder layer L-1. We therefore hook model.model.layers[hidden_idx
- 1]. Verified quirk: output_hidden_states snapshots are recorded BEFORE
external hooks apply, so never use them to check the injection at the hooked
row -- probe the next layer instead.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def load_vector(path: str, position: str, hidden_idx: int) -> torch.Tensor:
    """One [hidden] direction out of an extraction archive."""
    return torch.load(path)["vectors"][position][hidden_idx]


@contextmanager
def steering(model, hidden_idx: int, vector: torch.Tensor, alpha: float,
             positions: str = "all"):
    """alpha * vector added to one layer's output for the block's duration.

    Broadcasts over the batch. positions="all" pushes every [batch, seq]
    entry; positions="last" pushes column -1 only, which under the loop's
    left padding is every row's real last token. The hook is always removed,
    even on error -- no run may leak into the next.
    """
    if positions not in ("all", "last"):
        raise ValueError(f"positions must be 'all' or 'last'; got {positions!r}")
    n_layers = len(model.model.layers)
    if not 1 <= hidden_idx <= n_layers - 1:
        raise ValueError(
            f"hidden_idx must be in 1..{n_layers - 1} (row 0 is the embedding "
            f"and would silently hook layers[-1]); got {hidden_idx}")
    v = vector.to(device=model.device, dtype=model.dtype)

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        # Sharded models put later layers on other cards; the push must live
        # where the hooked activation lives, not on the first card.
        nonlocal v
        if v.device != h.device:
            v = v.to(h.device)
        if positions == "all":
            h = h + alpha * v
        else:
            h = h.clone()
            h[:, -1, :] += alpha * v
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = model.model.layers[hidden_idx - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
