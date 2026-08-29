"""Residual-stream addition, shared by every method that adds a direction.

Layer numbering is the benchmark's hidden-state index: index 0 is the
embedding output and index k >= 1 is the output of decoder block k - 1, so
adding at layer k installs a hook on ``model.model.layers[k - 1]``. Methods
that read directions from external checkpoints with their own numbering must
convert before calling this.

Which positions the addition reaches is the harness's decision, shared by
every baseline: see ``baselines.shared.positions``.
"""

from contextlib import contextmanager

import torch

from baselines.shared.positions import DEFAULT, PositionMask, check_mode


def check_layer(model, layer: int) -> int:
    """Validate a hidden-state layer index against the loaded model."""
    n_layers = len(model.model.layers)
    if not isinstance(layer, int) or isinstance(layer, bool):
        raise ValueError(f"layer must be an integer in 1..{n_layers}")
    if not 1 <= layer <= n_layers:
        raise ValueError(f"layer must be in 1..{n_layers}; got {layer}")
    return n_layers


@contextmanager
def add_direction(model, delta: torch.Tensor, layer: int,
                  positions: str = DEFAULT):
    """Add a fixed vector at ``layer`` for the duration of the context.

    ``positions`` selects which tokens are touched; see
    ``baselines.shared.positions``. The default leaves the prompt as the
    model read it and steers only what the model writes.
    """
    check_layer(model, layer)
    check_mode(positions)
    expected = (int(model.config.hidden_size),)
    if not isinstance(delta, torch.Tensor) or tuple(delta.shape) != expected:
        shape = getattr(delta, "shape", None)
        raise ValueError(f"delta must have shape {expected}; got {shape}")

    module = model.model.layers[layer - 1]
    delta = delta.to(dtype=model.dtype, device=model.device)

    mask = PositionMask(positions)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = mask.apply(hidden, delta)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
