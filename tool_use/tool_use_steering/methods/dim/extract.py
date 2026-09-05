"""DiM (difference in means, Arditi et al. style): contrast two PROMPT piles.

The direction is  mean(states | prompts that call for X)
                - mean(states | prompts that call for a tool call)
read at the last prompt token, per layer. No responses are involved: the two
piles differ in their situations, so the direction captures "this situation
warrants X" rather than the wording of any reply.

Pile selection reproduces the previous codebase exactly. The negative pile is
not every should-call pair but the should-call pairs from the SAME axis --
those whose rejected side is a clarification for the clarify axis (732), an
abstention for the abstain axis (948) -- so both piles live in the same
neighbourhood of situations and the contrast isolates the decision.

    python -m tool_use_steering.methods.dim.extract --axis clarify
    python -m tool_use_steering.methods.dim.extract --axis abstain

Writes artifacts/dim/<axis>_<model>_n<pile>.pt:
    {"vectors": {"prompt_last": [n_layers+1, hidden]},
     "axis", "model", "revision"?, "piles", "code_commit"}
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


def piles(axis: str, n: int | None = None):
    """The two prompt piles; `n` caps BOTH at the same size (first n in
    pair_id order, deterministic), so a user can extract from balanced piles
    of any budget without touching the data.

    The per-axis axes contrast within one flavour of restraint. `call` is the
    pooled axis -- the canonical "act rather than hold back" direction: its
    positive pile mixes both flavours of should-call prompts half and half,
    its negative pile mixes should-clarify and should-abstain half and half,
    so neither flavour dominates either side. The per-axis directions overlap
    only partially (cos ~ 0.5 at the production layer), and this pooled
    direction is their shared component.
    """
    pairs = read_pairs(PAIRS)
    if axis == "call":
        half = (n or 700) // 2
        vs_clarify = [p for p in pairs if p.positive_action == "tool_call"
                      and p.negative_action == "clarify"][:half]
        vs_abstain = [p for p in pairs if p.positive_action == "tool_call"
                      and p.negative_action == "abstain"][:half]
        clarify = [p for p in pairs if p.positive_action == "clarify"][:half]
        abstain = [p for p in pairs if p.positive_action == "abstain"][:half]
        return vs_clarify + vs_abstain, clarify + abstain
    positive = [p for p in pairs if p.positive_action == axis]
    negative = [p for p in pairs if p.positive_action == "tool_call"
                and p.negative_action == axis]
    if n:
        positive, negative = positive[:n], negative[:n]
    return positive, negative


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=["clarify", "abstain", "call"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=700,
                    help="pairs per pile (default 700: balanced piles, fits "
                         "every axis; pass 0 for the full unbalanced piles)")
    args = ap.parse_args()

    positive, negative = piles(args.axis, args.n)
    print(f"piles: {len(positive)} should_{args.axis} "
          f"vs {len(negative)} should_call({args.axis}-rejecting)", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    means = {}
    for name, group in (("positive", positive), ("negative", negative)):
        total = 0
        for i, pair in enumerate(group):
            text = render(tokenizer, pair.messages, pair.tools)
            total = total + prompt_last(model, tokenizer, text)
            if (i + 1) % 100 == 0:
                print(f"  {name}: {i + 1}/{len(group)}", flush=True)
        means[name] = total / len(group)

    vector = means["positive"] - means["negative"]
    norms = vector.norm(dim=-1)
    print(f"prompt_last: shape {tuple(vector.shape)}, layer norms "
          f"{norms.min():.1f} .. {norms.max():.1f}")

    out_dir = os.path.join(ARTIFACTS, "dim")  # artifacts/<method>/: the path says who made it
    os.makedirs(out_dir, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out = os.path.join(out_dir, f"{args.axis}_{model_short}_n{len(positive)}.pt")
    commit = subprocess.run(
        ["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    torch.save({
        "vectors": {"prompt_last": vector},
        "method": "dim",
        "axis": ("call_minus_restraint" if args.axis == "call"
                 else f"{args.axis}_minus_toolcall"),
        "model": args.model,
        "piles": {"positive": len(positive), "negative": len(negative)},
        "code_commit": commit or None,
    }, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
