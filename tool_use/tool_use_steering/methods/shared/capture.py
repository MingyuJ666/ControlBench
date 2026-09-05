"""Activation capture: what extraction recipes read from the model.

Both functions return a [n_layers+1, hidden] float32 stack (row 0 =
embeddings, row L = raw output of decoder layer L-1), which is the indexing
convention residual_add.py injects back under.

Read points and their rationale:

  prompt_last    the last prompt token, before the model commits to any
                 output. Only informative across DIFFERENT prompts: within a
                 same-prompt pair these states are identical by causality,
                 and their difference is exactly zero.
  response       states over a candidate response teacher-forced after the
                 prompt; resp_mean averages them, resp_first is the first
                 response token. This is where same-prompt pairs DO differ.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def prompt_last(model, tokenizer, prompt_text: str) -> torch.Tensor:
    ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([ids], device=model.device)
    out = model(input_ids=input_ids, output_hidden_states=True)
    return torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])


@torch.no_grad()
def response(model, tokenizer, prompt_text: str,
             response_text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (resp_mean, resp_first), both [n_layers+1, hidden]."""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + tokenizer(response_text,
                                      add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([full_ids], device=model.device)
    out = model(input_ids=input_ids, output_hidden_states=True)
    p = len(prompt_ids)
    resp_mean = torch.stack(
        [h[0, p:, :].float().mean(dim=0).cpu() for h in out.hidden_states])
    resp_first = torch.stack(
        [h[0, p, :].float().cpu() for h in out.hidden_states])
    return resp_mean, resp_first
