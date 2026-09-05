"""SAE extraction: find the dictionary feature that separates the piles.

The public SAE suite is matched to the subject model by suites.sae_for --
one module per release under suites/, loud on unknown models. Recipe:
encode both situation piles' last-prompt-token residuals at the chosen
layer, score every feature by its mean activation difference (positive
pile minus negative), and keep the top features; the steering direction
for a feature is its decoder column, unit-normalised so dose stays in raw
norm units under the shared residual injection.

One caveat is measured rather than assumed: the suites were trained on the
BASE models and our subjects are the instruct models. The artifact records
each kept feature's firing rate on both piles, and the printed L0 should
sit near the suite's training sparsity -- a feature that never fires on
instruct activations transferred badly, and the numbers say so.

    python -m tool_use_steering extract --method sae --axis clarify \
        --model meta-llama/Llama-3.1-8B-Instruct --sae-layer 15
"""

from __future__ import annotations

import argparse
import os
import subprocess

import torch

from ...eval.prompt import render
from ...paths import ARTIFACTS
from ..shared.capture import prompt_last
from ..dim.extract import piles
from .suites import sae_for

TOP_FEATURES = 16   # kept in the artifact; the method uses config["feature_rank"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=["clarify", "abstain", "call"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--sae-layer", type=int, default=24,
                    help="residual layer whose SAE to use (the injection "
                         "row is sae_layer + 1)")
    args = ap.parse_args()

    sae = sae_for(args.model, args.sae_layer)

    positive, negative = piles(args.axis, args.n)
    print(f"piles: {len(positive)} vs {len(negative)}, "
          f"SAE layer {args.sae_layer} from {sae['repo']}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    row = args.sae_layer + 1        # brick row R = output of decoder layer R-1
    residuals = {}
    for tag, group in (("pos", positive), ("neg", negative)):
        acts = []
        for i, pair in enumerate(group):
            text = render(tokenizer, pair.messages, pair.tools)
            acts.append(prompt_last(model, tokenizer, text)[row])
            if (i + 1) % 200 == 0:
                print(f"  {tag}: {i + 1}/{len(group)}", flush=True)
        residuals[tag] = torch.stack(acts)

    feats_pos = sae["encode"](residuals["pos"])
    feats_neg = sae["encode"](residuals["neg"])
    print(f"L0: pos {float((feats_pos > 0).sum(-1).float().mean()):.1f}, "
          f"neg {float((feats_neg > 0).sum(-1).float().mean()):.1f}", flush=True)
    diff = feats_pos.mean(0) - feats_neg.mean(0)
    top = torch.topk(diff, TOP_FEATURES).indices

    features = []
    for j in top.tolist():
        features.append({
            "feature": j,
            "score": round(float(diff[j]), 4),
            "fire_rate_pos": round(float((feats_pos[:, j] > 0).float().mean()), 4),
            "fire_rate_neg": round(float((feats_neg[:, j] > 0).float().mean()), 4),
        })
    print("top features:", features[:4], flush=True)

    out_dir = os.path.join(ARTIFACTS, "sae")
    os.makedirs(out_dir, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out = os.path.join(out_dir,
                       f"{args.axis}_{model_short}_L{args.sae_layer}_n{len(positive)}.pt")
    commit = subprocess.run(
        ["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    directions = torch.stack([sae["W_dec"][:, f["feature"]] for f in features])
    torch.save({
        "directions": directions / directions.norm(dim=-1, keepdim=True),
        "features": features,
        "sae_layer": args.sae_layer,
        "sae_repo": sae["repo"],
        "method": "sae",
        "axis": args.axis,
        "model": args.model,
        "piles": {"positive": len(positive), "negative": len(negative)},
        "code_commit": commit or None,
    }, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
