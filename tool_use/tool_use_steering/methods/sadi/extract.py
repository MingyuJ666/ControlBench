"""SADI extraction: find the heads whose activity separates the behaviours.

Faithful to Wang et al. (ICLR 2025) and the official repo
(weixuan-wang123/SADI, steering.py), verified line by line:

- Same-prompt response pairs are teacher-forced (prompt + positive reply,
  prompt + negative reply); head outputs are read at the LAST token.
- Each (layer, head) is scored by the MEAN over head_dim of its activation;
  the per-pair difference (positive minus negative) is averaged over pairs
  into one signed [layers, heads] matrix.
- The top-K most positive entries are the heads to AMPLIFY to push toward
  the axis behaviour ("heads_pos"); the K most negative push the other way
  ("heads_neg"). Both are stored; how many to use is the method's
  hyperparameter (official sweeps 1..10).

    python -m tool_use_steering extract --method sadi --axis clarify

Writes artifacts/sadi/<axis>_<model>_n<pairs>.pt:
    {"score": [layers, heads], "heads_pos": [[layer, head]...K],
     "heads_neg": [...], meta...}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import torch

from ...eval.prompt import render
from ...paths import ARTIFACTS, PAIRS
from ...schema import read_pairs
from ..caa.extract import to_tag_format
from ..iti.extract import head_out_last_token

TOP_K = 128  # stored pool, official size; the used count is a method config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=["clarify", "abstain", "call"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=700)
    args = ap.parse_args()

    all_pairs = read_pairs(PAIRS)
    if args.axis == "call":
        half = args.n // 2
        pairs = ([p for p in all_pairs if p.positive_action == "tool_call"
                  and p.negative_action == "clarify"][:half]
                 + [p for p in all_pairs if p.positive_action == "tool_call"
                    and p.negative_action == "abstain"][:half])
    else:
        pairs = [p for p in all_pairs
                 if p.positive_action == args.axis][:args.n]
    print(f"{len(pairs)} pairs for axis {args.axis}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    def side_text(pair, positive_side: bool) -> str:
        action = pair.positive_action if positive_side else pair.negative_action
        body = pair.positive if positive_side else pair.negative
        if action == "tool_call":
            body = to_tag_format(body)
        return render(tokenizer, pair.messages, pair.tools) + body

    total = 0
    for i, pair in enumerate(pairs):
        pos = head_out_last_token(model, tokenizer, side_text(pair, True))
        neg = head_out_last_token(model, tokenizer, side_text(pair, False))
        total = total + (pos.mean(dim=-1) - neg.mean(dim=-1))  # [layers, heads]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pairs)}", flush=True)
    score = total / len(pairs)

    flat = score.flatten()
    n_heads = score.shape[1]
    top = torch.topk(flat, TOP_K).indices
    bottom = torch.topk(-flat, TOP_K).indices
    heads_pos = [[int(i // n_heads), int(i % n_heads)] for i in top]
    heads_neg = [[int(i // n_heads), int(i % n_heads)] for i in bottom]
    print("top head scores:", [round(float(flat[i]), 4) for i in top[:5]],
          "| layers:", sorted({h[0] for h in heads_pos[:10]}), flush=True)

    out_dir = os.path.join(ARTIFACTS, "sadi")
    os.makedirs(out_dir, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out = os.path.join(out_dir, f"{args.axis}_{model_short}_n{len(pairs)}.pt")
    commit = subprocess.run(
        ["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    torch.save({
        "score": score,
        "heads_pos": heads_pos,
        "heads_neg": heads_neg,
        "method": "sadi",
        "axis": args.axis,
        "model": args.model,
        "recipe": "signed head-mean diff @ last token, teacher-forced pairs; "
                  "weixuan-wang123/SADI steering.py",
        "n_pairs": len(pairs),
        "code_commit": commit or None,
    }, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
