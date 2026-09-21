"""Explicit experiment settings; machine paths are supplied at the CLI."""

import json
from pathlib import Path

MODELS = {
    "qwen3_8b": ("Qwen/Qwen3-8B", "qwen3", 36),
    "gemma2_9b": ("google/gemma-2-9b-it", "gemma2", 42),
    "llama31_8b": ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama", 32),
}
METHODS = ("no_steering", "prompted", "caa", "pca", "lat", "linear_probe", "sae", "iti")
PINNED = {
    "torch": "2.5.1",
    "transformers": "5.5.4",
    "repsteer": "0.2.4",
    "tokenizers": "0.22.2",
    "numpy": "2.1.2",
    "safetensors": "0.8.0",
    "accelerate": "1.13.0",
    "huggingface_hub": "1.31.0",
}
DEFAULTS = {
    "layer_grid": {
        "qwen3_8b": [9, 18, 27],
        "gemma2_9b": [11, 21, 32],
        "llama31_8b": [8, 16, 24],
    },
    "non_iti_alphas": [0.5, 1.0, 1.5, 2.0, 2.5],
    "iti_alphas": [1, 5, 10, 15, 20],
    "iti_top_heads": 48,
    "dtype": "bfloat16",
    "layer_convention": "zero_based_decoder_block_output",
    "seed": 42,
    "sae_layers": [9, 20, 31],
    "sae_alphas": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "sae_fvu_policy": "diagnostic_only",
    "context_overrides": {
        "gemma2_9b": {"max_position_embeddings": 9216},
    },
}


def load_config(path, model):
    requested = json.loads(Path(path).read_text())
    if set(requested) != {"methods"}:
        raise ValueError("Configuration files declare methods only")
    spec = {**DEFAULTS, **requested}
    if model not in MODELS:
        raise ValueError("Unknown model")
    methods = spec["methods"]
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("Unknown or duplicate method")
    if "sae" in methods and model != "gemma2_9b":
        raise ValueError("SAE is supported only for Gemma instruction checkpoints")
    if spec["dtype"] != "bfloat16" or spec["iti_top_heads"] != 48:
        raise ValueError("This protocol requires BF16 and global top48 ITI")
    if spec["layer_convention"] != "zero_based_decoder_block_output":
        raise ValueError("Unsupported layer convention")
    layers = spec["layer_grid"][model]
    if (
        not layers
        or layers != sorted(set(layers))
        or any(type(x) is not int or not 0 <= x < MODELS[model][2] for x in layers)
    ):
        raise ValueError("Invalid residual layer grid")
    for key in ("non_iti_alphas", "iti_alphas", "sae_alphas"):
        values = spec[key]
        if (
            not values
            or values != sorted(set(values))
            or any(
                type(x) not in (int, float) or not 0 < x < float("inf") for x in values
            )
        ):
            raise ValueError(f"Invalid strengths: {key}")
    if spec["sae_layers"] != [9, 20, 31] or spec["sae_fvu_policy"] != "diagnostic_only":
        raise ValueError("Use the three instruction-model Gemma Scope layers")
    if type(spec["seed"]) is not int:
        raise ValueError("Seed must be an integer")
    override = spec["context_overrides"].get(model)
    if override and (
        override["max_position_embeddings"] != 9216 or model != "gemma2_9b"
    ):
        raise ValueError("Unsupported context override")
    return spec


def configs(a, method, artifact):
    if method not in a.spec["methods"]:
        raise ValueError("Method outside configured scope")
    if method in ("no_steering", "prompted"):
        return [
            dict(method=method, layer=-1, alpha=0.0, delta_norm=0.0, config_id=method)
        ]
    if method == "iti":
        selected = artifact["selected"]
        if (
            len(selected) != 48
            or len({(r["layer"], r["head"]) for r in selected}) != 48
        ):
            raise ValueError("ITI needs 48 distinct globally selected heads")
        norm = sum(r["sigma"] ** 2 for r in selected) ** 0.5
        return [
            dict(
                method=method,
                layer=-1,
                alpha=x,
                delta_norm=x * norm,
                config_id=f"iti-a{x:g}",
            )
            for x in a.spec["iti_alphas"]
        ]
    expected = (
        a.spec["sae_layers"] if method == "sae" else a.spec["layer_grid"][a.model]
    )
    if sorted(artifact["layers"]) != expected:
        raise ValueError("Artifact layers differ from configured grid")
    alphas = a.spec["sae_alphas"] if method == "sae" else a.spec["non_iti_alphas"]
    return [
        dict(
            method=method,
            layer=layer,
            alpha=x,
            delta_norm=x * float(e["vector"].norm()),
            config_id=f"{method}-b{layer}-a{x:g}",
        )
        for layer, e in sorted(artifact["layers"].items())
        for x in alphas
    ]
