"""Read and write individual attention heads, before their outputs are mixed.

Qwen3 attention concatenates its per-head outputs and passes the result to
``o_proj``. That concatenation is the last moment the heads are separable: a
tensor of shape ``[batch, seq, n_heads * head_dim]`` whose slice
``[..., h*head_dim : (h+1)*head_dim]`` is head ``h`` alone. Both the capture
and the intervention therefore attach to ``o_proj``'s input rather than to the
attention module's output, which is already the projected sum.

Qwen3-8B uses grouped-query attention (32 query heads over 8 key/value heads).
Grouping is about which keys and values a query head reads; each query head
still writes its own slice here, so per-head slicing is well defined.

Nothing here decides which heads matter. It only makes one head addressable,
so a mistake in the arithmetic is a mistake about a head rather than a silent
shift of the whole layer.
"""

from contextlib import contextmanager

import torch

from baselines.shared.positions import DEFAULT, PositionMask, check_mode


def head_layout(model) -> tuple[int, int, int]:
    """Return ``(n_layers, n_heads, head_dim)`` as the attention writes them."""
    config = model.config
    n_heads = int(config.num_attention_heads)
    head_dim = int(
        getattr(config, "head_dim", None) or config.hidden_size // n_heads
    )
    return len(model.model.layers), n_heads, head_dim


def _o_proj(model, layer: int):
    """The projection whose input still has heads side by side."""
    return model.model.layers[layer].self_attn.o_proj


def check_head(model, layer: int, head: int) -> None:
    n_layers, n_heads, _ = head_layout(model)
    if not 0 <= layer < n_layers:
        raise ValueError(f"layer must be in 0..{n_layers - 1}; got {layer}")
    if not 0 <= head < n_heads:
        raise ValueError(f"head must be in 0..{n_heads - 1}; got {head}")


def split_heads(packed: torch.Tensor, n_heads: int, head_dim: int
                ) -> torch.Tensor:
    """View ``[..., n_heads * head_dim]`` as ``[..., n_heads, head_dim]``."""
    expected = n_heads * head_dim
    if packed.shape[-1] != expected:
        raise ValueError(
            f"expected a final dimension of {expected}; got {packed.shape[-1]}"
        )
    return packed.reshape(*packed.shape[:-1], n_heads, head_dim)


@torch.no_grad()
def capture_last_token(model, encoded: dict) -> torch.Tensor:
    """Return per-head outputs at the final input token.

    Shape is ``[batch, n_layers, n_heads, head_dim]``, float32 on CPU. Inputs
    are left-padded by the caller, so the final position is the real last
    prompt token for every row.
    """
    n_layers, n_heads, head_dim = head_layout(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(_module, inputs):
            packed = inputs[0]
            captured[layer] = split_heads(
                packed[:, -1, :], n_heads, head_dim
            ).detach().float().cpu()
        return hook

    handles = [
        _o_proj(model, layer).register_forward_pre_hook(make_hook(layer))
        for layer in range(n_layers)
    ]
    try:
        model(**encoded, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()

    if len(captured) != n_layers:
        raise RuntimeError(
            f"captured {len(captured)} of {n_layers} layers; a hook did not fire"
        )
    return torch.stack([captured[layer] for layer in range(n_layers)], dim=1)


@contextmanager
def add_to_heads(model, additions: dict[tuple[int, int], torch.Tensor],
                 positions: str = DEFAULT):
    """Add a vector to each named head for the duration of the context.

    ``additions`` maps ``(layer, head)`` to a ``[head_dim]`` tensor. The
    addition lands in that head's slice of ``o_proj``'s input, so what reaches
    the residual stream is the projection of the change — which is why dose
    has to be measured after ``o_proj`` rather than assumed from these norms.
    """
    check_mode(positions)
    _, n_heads, head_dim = head_layout(model)
    by_layer: dict[int, torch.Tensor] = {}
    for (layer, head), vector in additions.items():
        check_head(model, layer, head)
        if tuple(vector.shape) != (head_dim,):
            raise ValueError(
                f"addition for layer {layer} head {head} must have shape "
                f"({head_dim},); got {tuple(vector.shape)}"
            )
        packed = by_layer.setdefault(
            layer, torch.zeros(n_heads * head_dim, dtype=torch.float32)
        )
        packed[head * head_dim:(head + 1) * head_dim] += vector.float()

    handles = []
    for layer, packed in by_layer.items():
        delta = packed.to(dtype=model.dtype, device=model.device)

        def make_hook(delta=delta):
            mask = PositionMask(positions)

            def hook(_module, inputs):
                return (mask.apply(inputs[0], delta), *inputs[1:])
            return hook

        handles.append(
            _o_proj(model, layer).register_forward_pre_hook(make_hook())
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
