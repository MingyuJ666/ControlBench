"""Probe: a logistic-regression direction, the discriminative counterpart.

DiM asks "where do the two piles' means sit"; a probe asks "which direction
best SEPARATES the piles". Same piles, same read point, same injection --
only the estimator changes, which is exactly what makes the comparison
informative: if the probe direction steers better than the mean difference,
separation carries more steerable signal than location, and vice versa.

Pile selection is imported from dim's extractor so the two methods see the
same pairs by construction. Training is plain logistic regression in torch
(fixed seed, L-BFGS, small ridge) -- no sklearn dependency, deterministic.

    python -m tool_use_steering extract --method probe --axis clarify

Writes artifacts/probe/<axis>_<model>_n<pile>.pt with the familiar
{"vectors": {"prompt_last": [n_layers+1, hidden]}} shape, plus each layer's
training accuracy -- a free read on where the decision is decodable, which
the layer scan can be compared against.
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


def fit_direction(X: torch.Tensor, y: torch.Tensor, seed: int = 0,
                  ridge: float = 1e-3, steps: int = 200,
                  return_bias: bool = False):
    """Logistic regression; returns the weight vector, plus the bias when
    asked -- anything that SCORES with the classifier must use both."""
    torch.manual_seed(seed)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = X @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y) + ridge * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    if return_bias:
        return w.detach(), b.detach()
    return w.detach()


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

    acts, labels = [], []
    for label, group in ((1.0, positive), (0.0, negative)):
        for i, pair in enumerate(group):
            text = render(tokenizer, pair.messages, pair.tools)
            acts.append(prompt_last(model, tokenizer, text))  # [L+1, hidden]
            labels.append(label)
            if (i + 1) % 200 == 0:
                print(f"  {'pos' if label else 'neg'}: {i + 1}/{len(group)}",
                      flush=True)
    X_all = torch.stack(acts)                    # [n, L+1, hidden]
    y = torch.tensor(labels)

    n_layers = X_all.shape[1]
    weights, accuracy = [], []
    for layer in range(n_layers):
        X = X_all[:, layer, :]
        # standardize per feature: keeps LBFGS well-conditioned; the direction
        # is mapped back to raw activation space afterwards
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
        w, b = fit_direction((X - mu) / sd, y, return_bias=True)
        w_raw = w / sd                            # undo the scaling
        # accuracy must use the full classifier -- the bias too
        acc = ((((X - mu) / sd) @ w + b > 0).float() == y).float().mean().item()
        weights.append(w_raw)
        accuracy.append(round(acc, 4))
    vector = torch.stack(weights)
    print("train accuracy by layer:",
          {L: accuracy[L] for L in range(0, n_layers, 6)}, flush=True)

    out_dir = os.path.join(ARTIFACTS, "probe")
    os.makedirs(out_dir, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out = os.path.join(out_dir, f"{args.axis}_{model_short}_n{len(positive)}.pt")
    commit = subprocess.run(
        ["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    torch.save({
        "vectors": {"prompt_last": vector},
        "method": "probe",
        "axis": args.axis,
        "model": args.model,
        "piles": {"positive": len(positive), "negative": len(negative)},
        "train_accuracy": accuracy,
        "code_commit": commit or None,
    }, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
