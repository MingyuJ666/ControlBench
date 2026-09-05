"""CAST calibration: pick the condition layer and threshold.

The official find_best_condition_point (malleable_model.py @52be602) grid-
searches condition layers and thresholds for the (layer, threshold,
comparator) that best separates positive from negative prompts by F1 of
"condition fired". Reproduced here over TRAIN pairs only -- the same
material the vectors came from; dev items are never touched. The chosen
point is written back into the cast artifact.

    python -m tool_use_steering.methods.cast.calibrate --axis clarify
"""

from __future__ import annotations

import argparse

import torch

from ...eval.prompt import render
from ...paths import ARTIFACTS
from ..dim.extract import piles


@torch.no_grad()
def pooled_layer_inputs(model, tokenizer, text: str) -> torch.Tensor:
    """[n_decoder_layers, hidden]: each layer's INPUT, mean over prompt tokens
    (the official "mean" comparison mode)."""
    captured = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def make(i):
            def hook(module, args, kwargs=None):
                captured[i] = args[0][0].float().mean(dim=0).cpu()
                return None
            return hook
        handles.append(layer.register_forward_pre_hook(make(i)))
    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        model(input_ids=torch.tensor([ids], device=model.device))
    finally:
        for h in handles:
            h.remove()
    return torch.stack([captured[i] for i in range(len(model.model.layers))])


def similarities(pooled: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """cos(h, tanh(P h)) for each row of pooled, official formula."""
    v = direction / direction.norm()
    proj = torch.tanh(torch.outer(pooled @ v, v))          # P h = (h.v̂) v̂
    return torch.nn.functional.cosine_similarity(pooled, proj, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=["clarify", "abstain", "call"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=700)
    args = ap.parse_args()

    positive, negative = piles(args.axis, args.n)
    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    artifact = f"{ARTIFACTS}/cast/{args.axis}_{model_short}_n{len(positive)}.pt"
    data = torch.load(artifact)
    condition = data["vectors"]["condition"]               # [rows, hidden]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    pools = {"pos": [], "neg": []}
    for tag, group in (("pos", positive), ("neg", negative)):
        for i, pair in enumerate(group):
            text = render(tokenizer, pair.messages, pair.tools)
            pools[tag].append(pooled_layer_inputs(model, tokenizer, text))
            if (i + 1) % 200 == 0:
                print(f"  {tag}: {i + 1}/{len(group)}", flush=True)
    pos = torch.stack(pools["pos"])                        # [n, layers, hidden]
    neg = torch.stack(pools["neg"])

    best = None
    n_layers = pos.shape[1]
    for layer in range(n_layers):
        direction = condition[layer + 1]                   # brick row = id + 1
        sim_p = similarities(pos[:, layer, :], direction)
        sim_n = similarities(neg[:, layer, :], direction)
        for threshold in torch.arange(0.0, 1.0, 0.01):
            for comparator in ("larger", "smaller"):
                if comparator == "larger":                 # official inversion
                    fire_p, fire_n = sim_p < threshold, sim_n < threshold
                else:
                    fire_p, fire_n = sim_p > threshold, sim_n > threshold
                tp = fire_p.sum().item()
                fp = fire_n.sum().item()
                fn = (~fire_p).sum().item()
                f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
                if best is None or f1 > best["f1"]:
                    best = {"condition_layer": layer,
                            "threshold": round(float(threshold), 2),
                            "comparator": comparator, "f1": round(f1, 4)}
        if (layer + 1) % 6 == 0:
            print(f"  layer {layer + 1}/{n_layers}, best so far {best}", flush=True)

    print("calibrated:", best)
    data["calibration"] = best | {"calibrated_on": "train pairs (in-sample), "
                                  "official find_best_condition_point recipe"}
    torch.save(data, artifact)
    print(f"written back -> {artifact}")


if __name__ == "__main__":
    main()
