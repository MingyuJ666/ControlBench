"""Domain plugins and the registry that finds them.

A domain is a module. It is looked up by name, so nothing in the shared layer
and nothing in the command line tools names a particular domain:

    from core.domains import get
    dom = get("instruction_following")

A domain exposes, at minimum:

    NAME, TARGETS
    pairs(target) -> repsteer.data.ContrastivePairs
    rendering() -> core.Rendering
    render(tokenizer, item) -> str
    items(target, split) -> list[dict]          each item carries item_id
    EVAL_SPLIT, SCORING_VERSION
    judging_complete(model, experiment, method, target)
    prepare_judging(...), judge(dirs), metrics(...)

and optionally clean(text). Domains can expose messages(item) to let repsteer
render the generation inputs; these must match render() under rendering().
An optional INPUTS path tells selfcheck where to find shared input files instead
of using data/<domain>/inputs.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from ..paths import REPO_ROOT

# Keep the public core.domains imports while each domain lives in its
# own project directory. Installed wheels already contain the normal subpackages.
for _project in ("instruction-following", "tool-use"):
    _source = REPO_ROOT / _project / "src"
    if _source.is_dir():
        __path__.append(str(_source))

__all__ = ["get", "names"]


def names() -> list[str]:
    """Every domain module or package that ships in this directory."""
    return sorted(m.name for m in pkgutil.iter_modules(__path__))


def get(name: str) -> ModuleType:
    """Import a domain by name, or say which names exist."""
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{name}":
            raise
        raise KeyError(f"unknown domain {name!r}; available: {', '.join(names())}") from None
