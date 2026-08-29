"""Shared target-model generation for reference and steered conditions.

All conditions use the same chat rendering, greedy decoding, batching and
truncation accounting. Qwen3 thinking is disabled in the chat template, and
any complete or truncated ``<think>`` block that survives is removed.
"""
import re
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The cap must remain identical across all conditions compared for one model.
#
# 3072 is about 1.5x the longest answer the reference model writes when
# nothing constrains it: over 405 dev prompts, unsteered Qwen3-8B produced a
# maximum of 2010 new tokens, p99 1857, and nothing reached a 4096 ceiling.
# The margin is for a more verbose model, so that a benchmark run elsewhere
# measures instruction following rather than the budget. It is deliberately
# not larger: only steering that has broken the model's ability to stop runs
# to the cap, and giving those configurations more room buys no information
# while tripling what they cost to generate.
MAX_NEW_TOKENS = 3072
THINK_BLOCK = re.compile(r"<think>.*?(?:</think>\s*|$)", re.S)


def strip_think_blocks(text: str) -> tuple[str, bool]:
    """Remove complete or truncated Qwen think blocks.

    Returns the cleaned text and whether at least one block was removed.
    """
    cleaned, count = THINK_BLOCK.subn("", text)
    return cleaned.strip(), count > 0


def render_chat(tok, prompt: str) -> str:
    """THE input format. Every text the model ever sees — generation and
    vector extraction alike — goes through this one function, so the two can
    never drift apart. (A vector extracted under one rendering and injected
    under another is silently off-policy: no error, just weaker steering.)
    Thinking is disabled for Qwen3; other tokenizers lack the argument."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


@dataclass
class GenConfig:
    model: str = "Qwen/Qwen3-8B"
    max_new_tokens: int = MAX_NEW_TOKENS
    batch_size: int = 16
    dtype: torch.dtype = torch.bfloat16


class Generator:
    def __init__(self, cfg: GenConfig | None = None):
        self.cfg = cfg or GenConfig()
        cfg = self.cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model, dtype=cfg.dtype, device_map="cuda").eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.stripped_think = 0        # outputs that still carried a think block


    @torch.no_grad()
    def generate(self, prompts: list[str], intervention=None,
                 progress: bool = True) -> list[dict]:
        """prompts -> [{"output", "truncated", "n_new_tokens"}], input order.

        `intervention` is a zero-arg callable returning a context manager that
        installs forward hooks (see steering/inject.py). None -> plain
        generation, which is exactly how unsteered and prompted references are
        produced.

        Batches are formed over length-sorted prompts to cut padding waste,
        then restored to the caller's order — padding differences must never
        leak into which condition a prompt belongs to.
        """
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        results: list[dict | None] = [None] * len(prompts)
        cfg = self.cfg

        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            enc = self.tok([render_chat(self.tok, prompts[i]) for i in idx],
                           return_tensors="pt", padding=True,
                           padding_side="left").to(self.model.device)
            ctx = intervention() if intervention is not None else nullcontext()
            with ctx:
                out = self.model.generate(
                    **enc, max_new_tokens=cfg.max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=self.tok.pad_token_id)

            new_ids = out[:, enc["input_ids"].shape[1]:]
            for row, i in zip(new_ids, idx):
                keep = row[row != self.tok.pad_token_id]
                n_new = int(keep.numel())
                text = self.tok.decode(keep, skip_special_tokens=True)
                text, stripped = strip_think_blocks(text)
                if stripped:
                    self.stripped_think += 1
                results[i] = {"output": text,
                              "truncated": n_new >= cfg.max_new_tokens,
                              "n_new_tokens": n_new}
            if progress:
                print(f"\r  generated {min(start + cfg.batch_size, len(order))}"
                      f"/{len(order)}", end="", flush=True)
        if progress:
            print()
        return results  # type: ignore[return-value]
