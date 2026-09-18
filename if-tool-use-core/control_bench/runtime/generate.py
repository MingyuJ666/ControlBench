"""One generate call carries one plan.

Going through the repsteer wrapper is what makes prefill/decode positions
work; a raw model.generate inside steer() would see neither phase.
"""

import torch

from .rendering import Rendering

GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None)


def generate(wrapper, inputs, plan=None, *, max_new_tokens: int, **kwargs):
    """Pass encoded inputs, rendered strings or chat messages through repsteer."""
    options = {**GREEDY, "max_new_tokens": max_new_tokens,
               "pad_token_id": wrapper.tokenizer.pad_token_id,
               "return_dict_in_generate": True, **kwargs}
    if plan is None:
        return wrapper.generate(inputs, **options).raw
    with wrapper.steer(plan):
        return wrapper.generate(inputs, **options).raw


@torch.no_grad()
def generate_texts(wrapper, texts: list[str], plan=None, *, batch_size: int,
                   max_new_tokens: int, progress=None, messages: list | None = None,
                   rendering: Rendering | None = None) -> list[dict]:
    """Rendered texts -> [{"output", "truncated", "n_new_tokens"}] in input order.

    Batches are formed over length-sorted texts to cut padding waste and the
    results are put back in the caller's order, so padding never decides
    which condition a prompt lands in. Every batch is one generate call under
    the same plan. With ``messages``, repsteer also applies the chat template;
    ``texts`` determine batch ordering and the prompt length for slicing completions.
    Otherwise repsteer tokenizes the already-rendered strings as supplied.
    """
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    tok = wrapper.tokenizer
    eos = getattr(wrapper.model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tok.eos_token_id
    eos_ids = [] if eos is None else list(eos) if isinstance(eos, (list, tuple)) else [eos]
    if messages is not None and len(messages) != len(texts):
        raise ValueError("messages and texts must contain the same number of inputs")
    if messages is not None and (rendering is None or not rendering.apply_chat_template):
        raise ValueError("chat messages require an explicit chat Rendering")
    options = (rendering if messages is not None else Rendering.strings()).generation_kwargs()
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    results: list[dict | None] = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch_texts = [texts[i] for i in idx]
        # GenerationResult includes the prompt but does not expose its length.
        # Count it on CPU for slicing; repsteer owns the actual padded inputs.
        lengths = tok(batch_texts, add_special_tokens=False,
                      return_attention_mask=False)["input_ids"]
        prompt_len = max(map(len, lengths))
        inputs = [messages[i] for i in idx] if messages is not None else batch_texts
        raw = generate(wrapper, inputs, plan, max_new_tokens=max_new_tokens, **options)
        new_ids = raw.sequences[:, prompt_len:]
        for row, i in zip(new_ids, idx):
            ends = torch.nonzero(torch.isin(row, torch.tensor(eos_ids, device=row.device, dtype=row.dtype))).flatten()
            ended = bool(ends.numel())
            n_new = int(ends[0]) + 1 if ended else int(row.numel())
            keep = row[:n_new]
            results[i] = {
                "output": tok.decode(keep, skip_special_tokens=True),
                "truncated": not ended and n_new >= max_new_tokens,
                "n_new_tokens": n_new,
            }
        if progress:
            progress(min(start + batch_size, len(order)), len(order))
    return results  # type: ignore[return-value]
