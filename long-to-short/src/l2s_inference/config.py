"""Protocol configuration for the long-to-short evaluation."""

import os
from pathlib import Path

from .generation import MODEL_IDS
from .settings import DEFAULT_BATCH_SIZE, MAX_NEW_TOKENS, eval_sets_for, token_limit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = Path(os.environ.get("L2S_RUN_ROOT", str(ROOT / "runs")))
DEFAULT_PAIRS = ROOT / "data" / "l2s"
LAYERS = {
    "Qwen3-8B": [9, 18, 27],
    "Gemma-2-9B": [11, 21, 32],
    "Llama-3.1-8B": [8, 16, 24],
}
SAE_LAYERS = {"Gemma-2-9B": [9, 20, 31], "Qwen3-8B": [], "Llama-3.1-8B": []}
STRENGTHS = [0.5, 1.0, 1.5, 2.0, 2.5]
STRENGTH_GRID_FOR_ITI = [1, 5, 10, 15, 20]
RESIDUAL_METHODS = ["CAA", "PCA", "LAT", "Linear-Probe"]
TRACKS = ("all", "non-iti", "iti")


def run_root_for(track):
    if track not in TRACKS:
        raise ValueError(f"Unknown execution track: {track}")
    if "L2S_RUN_ROOT" in os.environ or track == "all":
        return DEFAULT_RUN_ROOT
    return DEFAULT_RUN_ROOT.with_name(f"{DEFAULT_RUN_ROOT.name}-{track}")


ITI_PROTOCOL = dict(
    candidate_layers="all",
    top_k=48,
    selection_metric="held_out_accuracy",
    split="pair_id grouped holdout",
    validation_fraction=0.2,
    seed=0,
    direction="mass_mean",
    std_correction=1,
    calibration="all 64 pairs, including internal validation",
    head_surface="self_attn.o_proj.input",
    pooling="reasoning-span mean; final-answer line excluded",
    tie_break="accuracy descending, layer ascending, head ascending",
    operator="repsteer.recipes.iti / ITIAdd",
    fit_implementation="src/l2s_inference/iti.py; repsteer unchanged",
)


def artifact_key(method, layer):
    return "ITI/global" if method == "ITI" else f"{method}/layer_{layer}"


SAE_RELEASE = "gemma-scope-9b-it-res-canonical"
SAE_REPO = "google/gemma-scope-9b-it-res"
SAE_PATHS = {
    9: "layer_9/width_16k/average_l0_88",
    20: "layer_20/width_16k/average_l0_91",
    31: "layer_31/width_16k/average_l0_76",
}


def methods_for(model_name, track="all"):
    if track not in TRACKS:
        raise ValueError(f"Unknown execution track: {track}")
    if track == "iti":
        return ["ITI"]
    return (
        RESIDUAL_METHODS
        + (["ITI"] if track == "all" else [])
        + (["SAE"] if SAE_LAYERS[model_name] else [])
    )


def learned_cells(model_name, track="all"):
    methods = methods_for(model_name, track)
    if "ITI" in methods:
        for strength in STRENGTH_GRID_FOR_ITI:
            yield "ITI", None, strength
    for method in RESIDUAL_METHODS:
        if method not in methods:
            continue
        for layer in LAYERS[model_name]:
            for strength in STRENGTHS:
                yield method, layer, strength
    for layer in SAE_LAYERS[model_name] if "SAE" in methods else []:
        for strength in STRENGTHS:
            yield "SAE", layer, strength


def protocol(
    model_name,
    track="all",
    eval_sets=None,
    max_new_tokens=MAX_NEW_TOKENS,
    *,
    zero_check=False,
):
    methods = methods_for(model_name, track)
    return dict(
        model=MODEL_IDS[model_name],
        track=track,
        zero_check=zero_check,
        eval_sets=eval_sets_for(eval_sets),
        layers=LAYERS[model_name] if track != "iti" else [],
        sae_layers=SAE_LAYERS[model_name] if "SAE" in methods else [],
        strengths=(STRENGTH_GRID_FOR_ITI if track == "iti" else STRENGTHS),
        methods=methods,
        blocked_methods={},
        iti=ITI_PROTOCOL if "ITI" in methods else None,
        seed=0,
        dtype="bfloat16",
        decoding=dict(
            do_sample=False,
            max_new_tokens=token_limit(max_new_tokens),
            batch_size=DEFAULT_BATCH_SIZE,
            padding_side="left",
            batching="fixed input-order groups; replay incomplete batches on resume",
            stop="model EOS",
        ),
        qwen_thinking=False,
        residual_normalization="l2",
        sae_normalization="native decoder",
        capture="teacher-forced reasoning span, mean pooled; final answer excluded",
        intervention=dict(operator="Add", phase="decode", positions="GeneratedTokens"),
        gemma_system_baseline="density instruction prepended to user turn; no system role in native template",
        design_constraints=[
            "SAE only on Gemma",
            "ITI all-layer global Top-48, grouped holdout; do not edit repsteer",
            "Require exactly 64 frozen pairs before GPU execution",
        ],
    )
