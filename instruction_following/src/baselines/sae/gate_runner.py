"""Measure whether the Qwen-Scope SAE describes our model's activations.

The checkpoint was trained on Qwen3-8B-Base; the benchmark steers the
post-trained chat model. This runs the gate specified in RESTRUCTURE.md 8.4
over the candidate layers and writes its verdict, so the decision to include
or drop the sae baseline is made from measurements taken before any steering
was attempted.

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      src/baselines/sae/gate_runner.py
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
      src/baselines/sae/gate_runner.py \
      --model Qwen/Qwen3-8B-Base --tag base_reference

The base run is a diagnostic, not a gate: without it a weak number cannot be
told apart from a checkpoint that was mediocre to begin with.
"""

# Run directly as a script as well as imported: the package root
# is this file's ancestor, not its own directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.sae import checkpoint, gate, gate_report_path
from baselines.shared.capture import task_pairs
from kernel.config import BASELINES, TARGET_MODEL, slug
from kernel.generate import render_chat

CANDIDATE_LAYERS = [9, 15, 21, 27, 33]
MAX_LENGTH = 512


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument(
        "--tag", default="chat",
        help="name for this run inside the report (chat, base_reference)",
    )
    parser.add_argument(
        "--layers", default=",".join(str(layer) for layer in CANDIDATE_LAYERS),
        help="candidate hidden-state indices",
    )
    parser.add_argument("--prompts", type=int, default=gate.N_PROMPTS)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser


def _batches(model, tok, prompts: list[str], batch: int) -> list[dict]:
    """Tokenize once; the measurement iterates the same batches repeatedly."""
    encoded = []
    for start in range(0, len(prompts), batch):
        chunk = [render_chat(tok, text) for text in prompts[start:start + batch]]
        tokens = tok(
            chunk, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_LENGTH, padding_side="left",
        )
        encoded.append({key: value.to(model.device) for key, value in tokens.items()})
    return encoded


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    layers = [int(value) for value in args.layers.split(",")]
    model_slug = slug(TARGET_MODEL)
    report_path = gate_report_path(model_slug, args.tag)
    if report_path.exists() and not args.force:
        raise SystemExit(
            f"{report_path} already holds a {args.tag!r} run; pass --force"
        )

    # One task's pairs are enough: the gate asks whether the SAE describes
    # this model's activations at all, which is not task specific.
    prompts = [pair["positive"] for pair in task_pairs("no_comma")][:args.prompts]
    print(f"model={args.model}  layers={layers}  prompts={len(prompts)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    batches = _batches(model, tok, prompts, args.batch)

    results = []
    identity = {"repo": checkpoint.REPO_ID, "revision": None, "files": {}}
    for index in layers:
        print(f"  layer {index}: loading checkpoint layer "
              f"{checkpoint.checkpoint_layer(index)}", flush=True)
        loaded = checkpoint.load_layer(index)
        on_device = checkpoint.LoadedSae(
            **{
                field: (
                    value.to(model.device)
                    if isinstance(value, torch.Tensor) else value
                )
                for field, value in vars(loaded).items()
            }
        )
        identity["revision"] = loaded.revision
        identity["files"][f"layer{loaded.checkpoint_layer}.sae.pt"] = (
            loaded.file_sha256
        )
        del loaded
        result = gate.measure_layer(model, on_device, batches, index)
        results.append(result)
        print(
            f"    loss_recovered={result.loss_recovered:.3f}  "
            f"fvu={result.fvu:.3f}  cosine={result.cosine:.3f}  "
            f"{'PASS' if result.passed else 'FAIL'}",
            flush=True,
        )
        del on_device
        torch.cuda.empty_cache()

    report = {
        "measure": gate.MEASURE,
        "tag": args.tag,
        "model": args.model,
        "checkpoint": identity,
        "n_prompts": len(prompts),
        "max_length": MAX_LENGTH,
        "layers": [result.as_record() for result in results],
        "passing_layers": [
            result.hidden_state_index for result in results if result.passed
        ],
        "thresholds": {
            "loss_recovered_min": gate.LOSS_RECOVERED_MIN,
            "fvu_max": gate.FVU_MAX,
            "pre_registered": "RESTRUCTURE.md section 8.4, 2026-07-31",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)

    passing = report["passing_layers"]
    print(f"\npassing layers ({args.tag}): {passing or 'none'}")
    print(f"-> {report_path}")


if __name__ == "__main__":
    main()
