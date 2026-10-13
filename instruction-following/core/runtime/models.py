"""Load one model on one GPU and wrap it for repsteer."""

from contextlib import contextmanager
from inspect import signature

import torch

import repsteer as rs


def load_model(model_id: str, device: str = "cuda:0", dtype=torch.bfloat16, revision: str | None = None):
    """Return (wrapper, model, tokenizer).

    The tokenizer pads on the left so a batch's last real token sits at the
    same index for every row; repsteer's capture and generate both use the
    tokenizer's own padding side. The wrapper carries the exact model revision
    so artifacts record which weights they were learned from.
    """
    wrapper = rs.models.from_pretrained(
        model_id, dtype=dtype, device_map=device, multimodal=False, revision=revision,
        tokenizer_kwargs={"padding_side": "left"},
    )
    model, tokenizer = wrapper.model, wrapper.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return wrapper, model, tokenizer


@contextmanager
def capture_forward(model):
    """Skip unused decoder outputs while learning from hidden activations.

    Capture consumes the hidden states, so it needs neither a generation KV
    cache nor vocabulary logits for every input token. The temporary hook
    leaves the model's normal generation behavior and config unchanged.
    """
    parameters = signature(model.forward).parameters
    options = {key: value for key, value in
               (("use_cache", False), ("logits_to_keep", 1)) if key in parameters}

    def configure(module, args, kwargs):
        return args, {**kwargs, **options}

    hook = model.register_forward_pre_hook(configure, with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()
