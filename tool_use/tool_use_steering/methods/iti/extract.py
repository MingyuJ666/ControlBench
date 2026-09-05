"""ITI extraction: probe every attention head, keep the discriminative few.

Faithful to Li et al. (NeurIPS 2023) and the official honest_llama repo
(@2c6b217, utils.py/interveners.py), verified against the code:

- Activations are the attention block's head outputs -- the INPUT of o_proj,
  reshaped to [heads, head_dim] -- at the last token of each example.
- A logistic probe is trained per (layer, head) on a train split; heads are
  ranked by validation accuracy and the top K kept.
- Each kept head's direction is the mass-mean shift (positive mean minus
  negative mean), unit-normalised -- the paper's better variant -- and its
  sigma is the std of ALL examples' activations projected on that direction,
  so the push is scaled to each head's own activation spread.

    python -m tool_use_steering extract --method iti --axis clarify

Writes artifacts/iti/<axis>_<model>_n<pile>_k<K>.pt:
    {"heads": [[layer, head], ...], "directions": [K, head_dim],
     "proj_std": [K], "val_accuracy": [K], meta...}
Layer ids here are decoder indices 0..n-1 (the module list), not brick rows.
"""

from __future__ import annotations

import argparse
import os
import subprocess

import torch

from ...eval.prompt import render
from ...paths import ARTIFACTS
from ..dim.extract import piles
from ..probe.extract import fit_direction


@torch.no_grad()
def head_out_last_token(model, tokenizer, text: str) -> torch.Tensor:
    """[n_layers, heads, head_dim] at the last token: o_proj inputs."""
    captured = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def make(i):
            def hook(module, args, kwargs=None):
                captured[i] = args[0][0, -1].detach().float().cpu()
                return None
            return hook
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make(i)))
    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        model(input_ids=torch.tensor([ids], device=model.device))
    finally:
        for h in handles:
            h.remove()
    n_heads = model.config.num_attention_heads
    return torch.stack([captured[i].reshape(n_heads, -1)
                        for i in range(len(model.model.layers))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=["clarify", "abstain", "call"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--k", type=int, default=48,
                    help="heads to intervene on (official default scale)")
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
            acts.append(head_out_last_token(model, tokenizer, text))
            labels.append(label)
            if (i + 1) % 200 == 0:
                print(f"  {'pos' if label else 'neg'}: {i + 1}/{len(group)}",
                      flush=True)
    X = torch.stack(acts)                      # [n, layers, heads, head_dim]
    y = torch.tensor(labels)

    # official 2-fold: each half validates a probe trained on the other,
    # scored with the FULL classifier (weights and bias), accuracies averaged
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(y), generator=g)
    half = len(y) // 2
    folds = [(perm[:half], perm[half:]), (perm[half:], perm[:half])]

    n_layers, n_heads = X.shape[1], X.shape[2]
    val_acc = torch.zeros(n_layers, n_heads)
    for layer in range(n_layers):
        for head in range(n_heads):
            feats = X[:, layer, head, :]
            accs = []
            for tr, va in folds:
                mu, sd = feats[tr].mean(0), feats[tr].std(0).clamp_min(1e-6)
                w, b = fit_direction((feats[tr] - mu) / sd, y[tr],
                                     return_bias=True)
                pred = ((feats[va] - mu) / sd) @ w + b > 0
                accs.append((pred.float() == y[va]).float().mean())
            val_acc[layer, head] = torch.stack(accs).mean()
        if (layer + 1) % 6 == 0:
            print(f"  probes: layer {layer + 1}/{n_layers}", flush=True)

    flat = val_acc.flatten()
    top = torch.topk(flat, args.k).indices
    heads = [(int(i // n_heads), int(i % n_heads)) for i in top]

    directions, sigmas, accs = [], [], []
    for layer, head in heads:
        feats = X[:, layer, head, :]
        direction = feats[y == 1].mean(0) - feats[y == 0].mean(0)  # mass mean
        direction = direction / direction.norm()
        proj = feats @ direction
        directions.append(direction)
        sigmas.append(proj.std().item())
        accs.append(val_acc[layer, head].item())
    print(f"top-{args.k} heads: val acc {min(accs):.3f} .. {max(accs):.3f}, "
          f"layers {sorted(set(l for l, _ in heads))}", flush=True)

    out_dir = os.path.join(ARTIFACTS, "iti")
    os.makedirs(out_dir, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out = os.path.join(out_dir,
                       f"{args.axis}_{model_short}_n{len(positive)}_k{args.k}.pt")
    commit = subprocess.run(
        ["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    torch.save({
        "heads": heads,
        "directions": torch.stack(directions),
        "proj_std": torch.tensor(sigmas),
        "val_accuracy": torch.tensor(accs),
        "method": "iti",
        "axis": args.axis,
        "model": args.model,
        "recipe": "mass-mean shift, top-k by probe val acc; honest_llama@2c6b217",
        "k": args.k,
        "piles": {"positive": len(positive), "negative": len(negative)},
        "code_commit": commit or None,
    }, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
