"""How a domain's inputs become model text. Two ways, nothing else.

chat():    inputs are chat messages; repsteer applies the tokenizer's chat
           template. This is the default for domains without special formats.
strings(): inputs are strings the domain has already rendered (its own
           template, whatever it needs); repsteer tokenizes
           them without adding special tokens, so a model that prepends BOS in
           its template does not get a second one.

Both produce keyword arguments for repsteer learning/capture and generation.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Rendering:
    apply_chat_template: bool
    add_generation_prompt: bool = False
    chat_template_kwargs: dict[str, Any] | None = None
    special_tokens: bool | None = None

    @classmethod
    def chat(cls, add_generation_prompt: bool = True,
             **chat_template_kwargs: Any) -> "Rendering":
        return cls(apply_chat_template=True,
                   add_generation_prompt=add_generation_prompt,
                   chat_template_kwargs=chat_template_kwargs or None)

    @classmethod
    def strings(cls) -> "Rendering":
        return cls(apply_chat_template=False, special_tokens=False)

    def learner_kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {"apply_chat_template": self.apply_chat_template}
        if self.apply_chat_template:
            out["add_generation_prompt"] = self.add_generation_prompt
            out["chat_template_kwargs"] = self.chat_template_kwargs
        else:
            out["special_tokens"] = self.special_tokens
        return out

    def generation_kwargs(self) -> dict[str, Any]:
        """The same rendering choices in repsteer's generation API."""
        if self.apply_chat_template:
            return {"apply_chat_template": True,
                    "add_generation_prompt": self.add_generation_prompt,
                    "chat_template_kwargs": self.chat_template_kwargs}
        return {"apply_chat_template": False,
                "tokenizer_kwargs": {"add_special_tokens": self.special_tokens}}
