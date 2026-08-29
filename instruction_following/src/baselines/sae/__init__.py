"""SAE: steer along the decoder direction of one selected feature.

The other baselines estimate a direction from the activations themselves.
This one takes a dictionary someone else trained — Qwen-Scope's sparse
autoencoder for this model family — and asks which of its 65,536 features
carries the instruction. That is a different claim: not that a direction
exists, but that the model's own learned feature basis already contains one.

Two conditions gate it, both measured rather than assumed. The checkpoint was
trained on Qwen3-8B-Base and applied here to the post-trained model, so only
layers whose reconstruction passes the pre-registered gate are searchable.
And a feature's separation is confirmed on extraction pairs it was not chosen
on, because the best of 65,536 features will separate something.

Strength multiplies the feature's measured maximum activation, so alpha=1
pushes the feature about as far as the extraction data ever pushed it.
"""

import json
from pathlib import Path

import torch

from baselines.shared import unit_direction
from baselines.shared.base import SearchPlan, joint_plan
from kernel.config import BASELINES

KEY = "sae"
METHOD_VERSION = 1
ARTIFACT_METHOD = "sae_decoder_feature"
SCALE_KEY = "max_act"

# The shared shape: three sites by three strengths. Strength here is in units
# of the feature's measured maximum activation, and it starts at 1.0 rather
# than below it — a pilot at 1.0 moved the model without reaching the
# instruction on any admitted layer, so the useful range is upward.
ALPHAS = [1.0, 2.0, 4.0]
MAX_SEARCHED_LAYERS = 3
DEFAULT_BUDGET = 9



def check_available() -> str | None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return "missing huggingface_hub"
    return None


def artifact_path(model_slug: str, task: str) -> Path:
    return unit_direction.artifact_path(model_slug, KEY, task)


def gate_report_path(model_slug: str, tag: str = "chat") -> Path:
    """One file per gate run.

    The runs are independent measurements of different models, and the
    runner's own instructions have them running at the same time on
    different cards. Sharing one file made that a read-modify-write race:
    both processes read the same report, both wrote it back whole, and the
    one that finished second silently erased the other's results.
    """
    return BASELINES / model_slug / KEY / f"gate.{tag}.json"


def passing_layers(model_slug: str, tag: str = "chat") -> list[int]:
    """Layers the reconstruction gate admitted, or an explanation of why not.

    Reading this from disk rather than from a constant is deliberate: which
    layers are admissible is a measurement about this checkpoint on this
    model, and steering somewhere the gate rejected would report a number the
    protocol says we do not have.
    """
    from baselines.sae.gate import MEASURE

    path = gate_report_path(model_slug, tag)
    if not path.exists():
        raise SystemExit(
            f"no reconstruction gate at {path}; run the gate before selecting "
            "features (baselines/sae/gate_runner.py)"
        )
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("measure") != MEASURE:
        raise SystemExit(
            f"{path} was measured as {run.get('measure')!r}, not {MEASURE!r}; "
            "re-run the gate with the current measure"
        )
    layers = run.get("passing_layers") or []
    if not layers:
        raise SystemExit(
            f"the gate admitted no layer for this model; the sae baseline "
            f"reports that rather than a score (see {path})"
        )
    return sorted(int(layer) for layer in layers)


def load_artifact(model_slug: str, task: str) -> dict:
    path = artifact_path(model_slug, task)
    return unit_direction.validate_artifact(
        torch.load(path, weights_only=True), path, ARTIFACT_METHOD, task,
        model_slug, scale_key=SCALE_KEY,
    )


def config_name(config: dict) -> str:
    return unit_direction.config_name(config)


def search_plan(context: dict, overrides: dict | None = None,
                budget: int | None = None) -> SearchPlan:
    """One joint grid over the admitted layers and the strengths.

    Admission takes two votes, and both are binding. The gate says the
    dictionary reconstructs this layer; the artifact says a feature there
    survived the confirmation half. A layer the gate let through but the
    confirmation rejected holds a zero direction, so steering it adds
    nothing — searching it would score an unsteered generation as a
    configuration, and an unsteered generation is coherent enough to win.
    """
    overrides = overrides or {}
    admitted = passing_layers(context["model_slug"])
    artifact = context.get("artifact")
    if artifact is not None:
        empty = set(int(index) for index in
                    artifact.get("indices_without_direction", []))
        admitted = [layer for layer in admitted if layer not in empty]
        if not admitted:
            raise SystemExit(
                "no gate-admitted layer kept a confirmed feature for this "
                "task; the sae baseline reports that rather than a score"
            )
    requested = overrides.get("layer")
    if requested is None:
        layers = admitted
    else:
        # A participant may narrow the search but not widen it past the gate,
        # which would make the gate advisory and the reported layers ones the
        # protocol says we have no reconstruction for.
        rejected = [layer for layer in requested if layer not in admitted]
        if rejected:
            raise SystemExit(
                f"layers {rejected} did not pass the reconstruction gate; "
                f"admitted layers are {admitted}"
            )
        layers = list(requested)
    alphas = overrides.get("alpha") or list(ALPHAS)
    if requested is None and len(layers) > MAX_SEARCHED_LAYERS:
        # More layers may reconstruct well than the shared grid has room for.
        # Keep the best-reconstructing ones rather than the lowest-numbered:
        # the gate already ordered them by how well the dictionary describes
        # this model there.
        layers = sorted(layers)[-MAX_SEARCHED_LAYERS:]

    invalid = [
        layer for layer in layers if not 1 <= layer <= context["n_layers"]
    ]
    if invalid:
        raise ValueError(
            f"layers must be between 1 and {context['n_layers']}: {invalid}"
        )
    return joint_plan(
        "layer", list(layers), "alpha", list(alphas),
        budget or DEFAULT_BUDGET,
        notes="layers restricted to those the reconstruction gate admitted; "
              "alpha is in units of the feature's measured maximum activation",
    )


def delta(artifact: dict, config: dict) -> torch.Tensor:
    return unit_direction.delta(artifact, config, scale_key=SCALE_KEY)


def make_intervention(model, artifact: dict, config: dict):
    return unit_direction.make_intervention(
        model, artifact, config, scale_key=SCALE_KEY
    )


def dose_metadata(artifact: dict, config: dict) -> dict:
    dose = unit_direction.dose_metadata(
        artifact, config, KEY, scale_key=SCALE_KEY
    )
    layer = config["layer"]
    dose["projection_sigma"] = None          # the scale here is not a sigma
    dose["max_act"] = float(artifact[SCALE_KEY][layer])
    dose["feature"] = int(artifact["feature"][layer])
    return dose
