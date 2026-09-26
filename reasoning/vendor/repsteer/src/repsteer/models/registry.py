"""Public model construction helpers.

Architecture adapter registration itself lives under ``models.hf.adapters``;
this module keeps the ergonomic ``repsteer.models.from_pretrained`` surface.
"""

from .hf.adapters import get_adapter, register_adapter, registered_adapters
from .hf.model import HFSteerableModel, from_model, from_pretrained

__all__ = [
    "HFSteerableModel",
    "from_model",
    "from_pretrained",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
