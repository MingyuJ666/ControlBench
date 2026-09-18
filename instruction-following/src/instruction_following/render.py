"""How a prompt becomes model text, and how a completion is cleaned."""

import re

THINK_BLOCK = re.compile(r"<think>.*?(?:</think>\s*|$)", re.S)


def render_chat(tokenizer, prompt: str) -> str:
    """One user message through the tokenizer's chat template, generation prompt on,
    thinking disabled where the template has that switch (Qwen3)."""
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def strip_think_blocks(text: str) -> str:
    """Drop a complete or unfinished <think> block if the model emitted one anyway."""
    return THINK_BLOCK.sub("", text).strip()
