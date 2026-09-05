"""CAST extraction: condition and behavior directions via pairwise PCA.

Faithful to IBM's official implementation (github.com/IBM/activation-steering
@52be602, activation_steering/steering_vector.py), verified line by line:

- Direction per layer = first PCA component of PAIRWISE-CENTERED activations:
  each (positive, negative) pair is centred on its own midpoint, positives and
  negatives are stacked, PCA(1) is fit on the stack ("pca_pairwise").
- Sign fix: project the raw activations onto the component; if positives
  project smaller than negatives on average, flip.
- Read point: the LAST token's hidden state (their accumulate_last_x_tokens=1
  default). Their layer id L reads hidden_states[L+1]; we store directions
  under our brick row R = L+1 and record the mapping, so cast/method.py can
  inject at the input of decoder layer L (our hook at layers[L-1] output).

Two artifacts per axis:
  condition  from the SITUATION piles (same pairs as dim, index-paired):
             "is this a should-X situation?"
  behavior   from same-prompt response pairs (as caa): the direction of the
             X-behaviour, read at the last token of prompt+response.

    python -m tool_use_steering extract --method cast --axis clarify
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
from ..shared.capture import prompt_last
from ..dim.extract import piles
from ..caa.extract import to_tag_format


def pairwise_pca_direction(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Official pca_pairwise + sign fix, in torch. pos/neg: [n, hidden]."""
    center = (pos + neg) / 2
    train = torch.cat([pos - center, neg - center])          # [2n, hidden]
    train = train - train.mean(0)                            # PCA centering
    # first right-singular vector == first PCA component
    _, _, v = torch.linalg.svd(train, full_matrices=False)
    direction = v[0]
    raw = torch.cat([pos, neg])
    proj = raw @ direction
    n = pos.shape[0]
    if (proj[:n] < proj[n:]).float().mean() > (proj[:n] > proj[n:]).float().mean():
        direction = -direction
    return direction


def collect_last_token(model, tokenizer, texts: list[str], tag: str) -> torch.Tensor:
    acts = []
    for i, text in enumerate(texts):
        acts.append(prompt_last(model, tokenizer, text))     # [L+1, hidden]
        if (i + 1) % 200 == 0:
            print(f"  {tag}: {i + 1}/{len(texts)}", flush=True)
    return torch.stack(acts)                                 # [n, L+1, hidden]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=["clarify", "abstain", "call"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=700)
    args = ap.parse_args()

    positive, negative = piles(args.axis, args.n)
    print(f"piles: {len(positive)} vs {len(negative)}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    # --- condition: situation piles, prompt last token --------------------
    pos_prompts = [render(tokenizer, p.messages, p.tools) for p in positive]
    neg_prompts = [render(tokenizer, p.messages, p.tools) for p in negative]
    pos_acts = collect_last_token(model, tokenizer, pos_prompts, "cond+")
    neg_acts = collect_last_token(model, tokenizer, neg_prompts, "cond-")
    n_rows = pos_acts.shape[1]
    condition = torch.stack([
        pairwise_pca_direction(pos_acts[:, r, :], neg_acts[:, r, :])
        for r in range(n_rows)])

    # --- behavior: same-prompt response pairs, last token of full text ----
    def side_text(pair, positive_side: bool) -> str:
        action = pair.positive_action if positive_side else pair.negative_action
        body = pair.positive if positive_side else pair.negative
        if action == "tool_call":
            body = to_tag_format(body)
        return render(tokenizer, pair.messages, pair.tools) + body

    behave_pairs = [p for p in read_pairs(PAIRS) if p.positive_action == args.axis]
    if args.axis == "call":   # pooled, mirroring caa's call axis
        half = args.n // 2
        allp = read_pairs(PAIRS)
        behave_pairs = ([p for p in allp if p.positive_action == "tool_call"
                         and p.negative_action == "clarify"][:half]
                        + [p for p in allp if p.positive_action == "tool_call"
                           and p.negative_action == "abstain"][:half])
    else:
        behave_pairs = behave_pairs[:args.n]
    pos_b = collect_last_token(model, tokenizer,
                               [side_text(p, True) for p in behave_pairs], "behav+")
    neg_b = collect_last_token(model, tokenizer,
                               [side_text(p, False) for p in behave_pairs], "behav-")
    behavior = torch.stack([
        pairwise_pca_direction(pos_b[:, r, :], neg_b[:, r, :])
        for r in range(n_rows)])

    out_dir = os.path.join(ARTIFACTS, "cast")
    os.makedirs(out_dir, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out = os.path.join(out_dir, f"{args.axis}_{model_short}_n{len(positive)}.pt")
    commit = subprocess.run(
        ["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    torch.save({
        # brick row R holds the direction read from hidden_states[R]
        # (= official layer id R-1); unit norm by SVD construction.
        "vectors": {"condition": condition, "behavior": behavior},
        "method": "cast",
        "axis": args.axis,
        "model": args.model,
        "recipe": "pca_pairwise@last_token, IBM/activation-steering@52be602",
        "piles": {"positive": len(positive), "negative": len(negative),
                  "behavior_pairs": len(behave_pairs)},
        "code_commit": commit or None,
    }, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
