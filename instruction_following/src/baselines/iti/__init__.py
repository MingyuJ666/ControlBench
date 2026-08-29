"""ITI: steer a few attention heads rather than the whole residual stream.

Where the other baselines add one direction to everything the layer writes,
this one asks which individual heads carry the instruction and edits only
those. If a small set of heads suffices, the signal is localised; if it takes
many, it is spread out — and that is a different claim about the model than
any residual-stream method can make.

Two axes, and they interact: adding K heads at a fixed strength adds K times
as much, so choosing K at one strength would compare doses rather than K. A
joint grid over (heads_k, alpha) is what every baseline searches, and this is
the method that shows why: neither axis can be fixed to choose the other.

Strength is in standard deviations of each head's own projection, so one alpha
means the same thing at a head that varies a lot and one that barely varies.
"""

from pathlib import Path

import torch

from baselines.iti.heads import add_to_heads, head_layout
from baselines.shared.base import SearchPlan, joint_plan
from kernel.config import BASELINES, slug

KEY = "iti"
METHOD_VERSION = 1
ARTIFACT_METHOD = "iti_head_probes"
EXTRACTION_POSITION = "last_prompt_token"

STRUCTURAL_AXIS = "heads_k"
STRENGTH_AXIS = "alpha"
# Three counts by three strengths fills the shared budget of nine exactly,
# and brackets the published ITI setting (K=48 heads at 15 sigma) rather than
# assuming it transfers to a different model and a different behaviour.
HEAD_COUNTS = [16, 48, 128]
ALPHAS = [5.0, 15.0, 30.0]
DEFAULT_BUDGET = 9



def check_available() -> str | None:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return "missing scikit-learn"
    return None


def artifact_path(model_slug: str, task: str) -> Path:
    return BASELINES / model_slug / KEY / f"{task}.pt"


def load_artifact(model_slug: str, task: str) -> dict:
    """Load an ITI artifact after checking its identity and shapes."""
    path = artifact_path(model_slug, task)
    record = torch.load(path, weights_only=True)
    if not isinstance(record, dict):
        raise ValueError(f"ITI artifact must be a dictionary: {path}")
    if record.get("method") != ARTIFACT_METHOD:
        raise ValueError(f"Artifact at {path} is not {ARTIFACT_METHOD}")
    if record.get("task") != task:
        raise ValueError(
            f"Artifact task {record.get('task')!r} does not match {task!r}"
        )
    artifact_model = record.get("model")
    if not isinstance(artifact_model, str) or slug(artifact_model) != model_slug:
        raise ValueError(
            f"Artifact model {artifact_model!r} does not match {model_slug!r}"
        )
    if record.get("extraction_position") != EXTRACTION_POSITION:
        raise ValueError(
            f"Artifact was captured at "
            f"{record.get('extraction_position')!r}, not "
            f"{EXTRACTION_POSITION!r}; produce it again: {path}"
        )
    direction = record.get("direction")
    sigma = record.get("sigma")
    if not isinstance(direction, torch.Tensor) or direction.ndim != 3:
        raise ValueError(
            f"Directions must have shape [layers, heads, head_dim]: {path}"
        )
    if not isinstance(sigma, torch.Tensor) or sigma.shape != direction.shape[:2]:
        raise ValueError(f"sigma must have one value per head: {path}")
    ranking = record.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != direction.shape[0] * direction.shape[1]:
        raise ValueError(
            f"ranking must order every head exactly once: {path}"
        )
    blank = {(row[0], row[1]) for row in record.get("heads_without_direction", [])}
    norms = torch.linalg.vector_norm(direction.float(), dim=-1)
    for layer in range(norms.shape[0]):
        for head in range(norms.shape[1]):
            norm = float(norms[layer, head])
            if (layer, head) in blank:
                if norm != 0.0:
                    raise ValueError(
                        f"Head ({layer}, {head}) was recorded as having no "
                        f"contrast but holds a direction: {path}"
                    )
            elif abs(norm - 1.0) > 1e-3:
                raise ValueError(
                    f"Head ({layer}, {head}) direction is not a unit "
                    f"vector: {path}"
                )
    return record


def config_name(config: dict) -> str:
    return f"K{config['heads_k']}_a{config['alpha']}"


def search_plan(context: dict, overrides: dict | None = None,
                budget: int | None = None) -> SearchPlan:
    """One joint stage over (heads_k, alpha), filling the shared budget.

    Fixing one axis to choose the other is confounded here in a way it is not
    elsewhere: more heads at the same alpha is a larger intervention, not a
    better one.
    """
    overrides = overrides or {}
    budget = budget or DEFAULT_BUDGET
    counts = overrides.get("heads_k") or list(HEAD_COUNTS)
    alphas = overrides.get("alpha") or list(ALPHAS)
    n_heads_total = context.get("n_heads_total")

    if len(set(counts)) != len(counts):
        raise ValueError(f"head counts contain duplicates: {counts}")
    if any(count < 1 for count in counts):
        raise ValueError(f"head counts must be positive: {counts}")
    if n_heads_total is not None and max(counts) > n_heads_total:
        raise ValueError(
            f"head counts exceed the model's {n_heads_total} heads: {counts}"
        )
    if len(set(alphas)) != len(alphas):
        raise ValueError(f"alphas contain duplicates: {alphas}")

    # No truncation: a grid cut to fit would leave the largest head counts
    # untested while still looking like a full sweep. SearchPlan rejects an
    # over-budget plan, so an external grid has to be chosen to fit.
    return joint_plan(
        STRUCTURAL_AXIS, list(counts), STRENGTH_AXIS, list(alphas), budget,
        notes="adding heads changes the dose, so K and alpha cannot be "
              "chosen one after the other",
    )


def selected_heads(artifact: dict, heads_k: int) -> list[tuple[int, int]]:
    """The top-K heads, ranked globally rather than within each layer.

    The layer in each pair is a zero-based index: it addresses both a row of
    the direction tensor and ``model.model.layers``, which are the same
    numbering. Reporting uses the shared one-based hidden-state convention
    instead, and ``dose_metadata`` converts once, at that boundary.
    """
    ranking = artifact["ranking"]
    blank = {(row[0], row[1]) for row in artifact.get("heads_without_direction", [])}
    usable = [
        (int(layer), int(head)) for layer, head, _ in ranking
        if (int(layer), int(head)) not in blank
    ]
    if heads_k > len(usable):
        raise ValueError(
            f"heads_k={heads_k} exceeds the {len(usable)} heads that separate "
            f"anything ({len(ranking) - len(usable)} had no contrast)"
        )
    return usable[:heads_k]


def additions(artifact: dict, config: dict) -> dict[tuple[int, int], torch.Tensor]:
    """What this configuration adds to each selected head."""
    alpha = config["alpha"]
    return {
        (layer, head): (
            alpha * float(artifact["sigma"][layer, head])
            * artifact["direction"][layer, head]
        )
        for layer, head in selected_heads(artifact, config["heads_k"])
    }


def make_intervention(model, artifact: dict, config: dict):
    """Return a zero-argument callable producing the intervention context."""
    n_layers, n_heads, head_dim = head_layout(model)
    expected = (n_layers, n_heads, head_dim)
    shape = tuple(artifact["direction"].shape)
    if shape != expected:
        raise ValueError(f"directions must have shape {expected}; got {shape}")
    edits = additions(artifact, config)
    return lambda: add_to_heads(model, edits)


def dose_metadata(artifact: dict, config: dict) -> dict:
    """Per-head strength facts.

    The residual-stream effect is the ``o_proj`` image of these additions, so
    the comparable number needs a forward pass and is filled in by the
    analysis utility rather than guessed here.

    Head layers are reported one-based, matching the hidden-state index every
    other baseline reports: a table that puts this method's layer beside
    another's has to mean the same site by the same name, and this method's
    internal index is one lower.
    """
    edits = additions(artifact, config)
    norms = [float(torch.linalg.vector_norm(vector)) for vector in edits.values()]
    return {
        "method": KEY,
        "raw_strength": float(config["alpha"]),
        "n_heads": len(edits),
        "head_delta_norms": norms,
        "summed_head_norm": float(sum(norms)),
        "site": {"component": "attn_head_pre_o_proj",
                 "layer_convention": "one-based hidden-state index",
                 "heads": [[layer + 1, head] for layer, head in edits]},
    }
