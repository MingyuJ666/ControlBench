"""Step 1: sample the small model's own (sparse) reasoning traces on GSM8K train.

These become x_neg for Dense-Rewriting. We keep the trace whether or not it got
the answer right -- DenseSteer contrasts reasoning *style*, not correctness.

Out: data/traces.json  [{"question","gold","trace"}]
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controlbench.data import load_gsm8k
from controlbench.models import SteerableModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen2.5-3b.yaml")
    ap.add_argument("--out", default="data/traces.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    problems = load_gsm8k(split="train", limit=cfg["n_calibration"], seed=0)
    print(f"Loaded {len(problems)} GSM8K train problems")

    model = SteerableModel(cfg["model_name"], device=cfg["device"])
    print(f"Loaded {cfg['model_name']} ({model.n_layers} layers, d={model.d_model})")

    traces = []
    for i, ex in enumerate(problems):
        trace = model.generate(ex["question"], max_new_tokens=cfg["max_new_tokens"])
        traces.append({"question": ex["question"], "gold": ex["gold"], "trace": trace})
        print(f"  trace {i + 1}/{len(problems)}", end="\r")
    print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(traces, open(args.out, "w"), indent=2)
    print(f"Wrote {len(traces)} traces -> {args.out}")


if __name__ == "__main__":
    main()
