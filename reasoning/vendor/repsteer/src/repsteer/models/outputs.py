from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class GenerationResult:
    """A small, backend-independent view of a generation result.

    ``raw`` intentionally retains the object returned by Transformers (a tensor
    or a ``Generate*Output``).  ``token_ids`` always points at its sequences.
    For a one-item batch ``text`` is a string; for larger batches it is a list.
    """

    text: str | list[str] | None
    token_ids: Tensor
    raw: Any

    @property
    def sequences(self) -> Tensor:
        return self.token_ids

    def __str__(self) -> str:
        if isinstance(self.text, str):
            return self.text
        return str(self.text)
