"""CAA (contrastive activation addition, Rimsky et al. 2024): contrast two
RESPONSES to one prompt.

Where DiM contrasts situations, CAA contrasts behaviours: the same prompt is
continued with the pair's positive response and with its negative response,
and the direction is the mean over pairs of (positive minus negative) states.
Only same-axis piles are used (should_clarify pairs for the clarify axis):
within a pair the prompt is identical, so prompt-position states cancel
exactly and the difference isolates the reply.

Read point, from the paper and the official code (nrimsky/CAA
generate_vectors.py reads index -2 of "... [/INST] (A)", the answer letter):
the state at the single token where the reply commits to its choice, read
after the model has taken that token in. Our replies are free text rather
than A/B letters, so that token is the FIRST response token -- "<tool_call>"
against the first word of a question or a refusal -- which is `resp_first`
in shared/capture.py. Nothing is averaged over the rest of the reply.

Tool-call responses are stored model-neutrally as a JSON call list; they are
wrapped into the subject model's own tag format here, at read time, so the
teacher-forced text is exactly what the model would emit.

    python -m tool_use_steering.methods.caa.extract --axis clarify
    python -m tool_use_steering.methods.caa.extract --axis abstain

Writes artifacts/caa/<axis>_<model>_n<pairs>.pt:
    {"vectors": {"resp_first": [n_layers+1, hidden]}, meta...}
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
from ..shared.capture import response


def to_tag_format(call_list_json: str) -> str:
    """Neutral JSON call list -> the hermes tag format Qwen emits."""
    calls = json.loads(call_list_json)
    return "\n".join(
        "<tool_call>\n" + json.dumps(c, ensure_ascii=False) + "\n</tool_call>"
        for c in calls)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=["clarify", "abstain", "call"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=700,
                    help="pairs to use (default 700; pass 0 for all)")
    args = ap.parse_args()

    # The call axis pools should-call pairs half and half by what their
    # rejected side is (asking / declining), one vector per axis like the
    # others. Each pair still contrasts within itself -- the correct call
    # against that pair's own wrong reply -- so pooling mixes situations,
    # never responses across pairs. The third flavour of not-calling
    # (answering from memory) has no audited pairs and is future work.
    all_pairs = read_pairs(PAIRS)
    if args.axis == "call":
        half = (args.n or 700) // 2
        pairs = ([p for p in all_pairs if p.positive_action == "tool_call"
                  and p.negative_action == "clarify"][:half]
                 + [p for p in all_pairs if p.positive_action == "tool_call"
                    and p.negative_action == "abstain"][:half])
    else:
        pairs = [p for p in all_pairs if p.positive_action == args.axis]
        if args.n:
            pairs = pairs[:args.n]
    print(f"{len(pairs)} pairs for axis {args.axis}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    total = 0
    for i, pair in enumerate(pairs):
        prompt = render(tokenizer, pair.messages, pair.tools)
        # whichever side is the tool call gets the model's tag format
        positive = (to_tag_format(pair.positive)
                    if pair.positive_action == "tool_call" else pair.positive)
        negative = (to_tag_format(pair.negative)
                    if pair.negative_action == "tool_call" else pair.negative)
        _, pos_first = response(model, tokenizer, prompt, positive)
        _, neg_first = response(model, tokenizer, prompt, negative)
        total = total + (pos_first - neg_first)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pairs)}", flush=True)

    vector = total / len(pairs)
    norms = vector.norm(dim=-1)
    print(f"resp_first: shape {tuple(vector.shape)}, layer norms "
          f"{norms.min():.1f} .. {norms.max():.1f}")

    out_dir = os.path.join(ARTIFACTS, "caa")
    os.makedirs(out_dir, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out = os.path.join(out_dir, f"{args.axis}_{model_short}_n{len(pairs)}.pt")
    commit = subprocess.run(
        ["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    torch.save({
        "vectors": {"resp_first": vector},
        "method": "caa",
        "axis": ("call_minus_restraint" if args.axis == "call"
                 else f"{args.axis}_minus_toolcall"),
        "model": args.model,
        "n_pairs": len(pairs),
        "code_commit": commit or None,
    }, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
