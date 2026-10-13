"""Input locations and experiment output paths.

inputs:    <domain-project>/data/inputs/
           or <STEERING_DATA>/<domain>/inputs/ when overridden
outputs:   --out when supplied, otherwise <DATA_ROOT>/<domain>/outputs/<PIPELINE_ID>/{artifacts,runs,cache}/
"""

from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CORE_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = CORE_ROOT.parent
DATA_ROOT = Path(os.environ["STEERING_DATA"]) if os.environ.get("STEERING_DATA") else REPO_ROOT / "data"

PIPELINE_ID = "repsteer-native-v2"
_OUTPUT_ROOT = ContextVar("output_root", default=None)


def configured_output_root() -> Path | None:
    """The explicit output directory for the current CLI invocation."""
    return _OUTPUT_ROOT.get()


@contextmanager
def output_directory(directory):
    """Scope an output override without changing input locations or later calls."""
    root = Path(directory).expanduser().resolve() if directory is not None else None
    token = _OUTPUT_ROOT.set(root)
    try:
        yield root
    finally:
        _OUTPUT_ROOT.reset(token)


def input_root(domain: str) -> Path:
    """Use shipped domain inputs, or the original layout on an external data disk."""
    if os.environ.get("STEERING_DATA"):
        return DATA_ROOT / domain / "inputs"
    project = {"instruction_following": "instruction-following", "tool_use": "tool-use"}[domain]
    return REPO_ROOT / project / "data" / "inputs"


def output_root(domain: str) -> Path:
    """Outputs for the native repsteer pipeline."""
    return configured_output_root() or DATA_ROOT / domain / "outputs" / PIPELINE_ID


def model_slug(model_id: str) -> str:
    """Hugging Face id after the slash, lowercased: Qwen/Qwen3-8B -> qwen3-8b."""
    return model_id.split("/")[-1].lower()


def site_slug(site) -> str:
    """repsteer Site -> path segment: resid_post(20) -> resid_post_20."""
    unit = getattr(site, "unit", None)
    base = f"{site.component}_{site.layer}"
    return base if unit is None else f"{base}_h{unit}"


def artifact_dir(domain: str, model: str, method: str, target: str, site,
                 *, training: str | None = None) -> Path:
    slug = site if isinstance(site, str) else site_slug(site)
    root = output_root(domain) / "artifacts" / model_slug(model) / method / target
    return (root / training if training is not None else root) / slug


def run_dir(domain: str, model: str, experiment: str, method: str, target: str,
            config: str) -> Path:
    return output_root(domain) / "runs" / model_slug(model) / experiment / method / target / config
