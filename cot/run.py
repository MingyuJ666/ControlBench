"""Usage: python cot/run.py {eval,tune} --help."""
import argparse
import hashlib
import json
from pathlib import Path

from data import DEFAULT_DATA, load, score, write_json
from model import Model, PROMPTS, SYSTEM
from steering import METHODS, fit, intervention


def evaluate(model, rows, style, budget, batch_size, path, control=None):
    results = []
    with path.open("x") as stream:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            generated = model.generate([r["problem"] for r in batch], style, budget, control)
            if len(generated) != len(batch):
                raise ValueError("Generation batch size mismatch")
            for row, result in zip(batch, generated):
                record = {**row, **result, **score(result["completion"], row["answer"])}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                results.append(record)
            print(f"{path.stem}: {len(results)}/{len(rows)}", flush=True)
    return {"count": len(results), "accuracy": sum(r["correct"] for r in results) / len(results),
            "truncated": sum(r["truncated"] for r in results),
            "mean_tokens": sum(r["tokens"] for r in results) / len(results)}


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["eval", "tune"])
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--method", choices=["normal", "cot", *METHODS], default="normal")
    parser.add_argument("--layers", type=int, nargs="+", default=[16])
    parser.add_argument("--strengths", type=float, nargs="+", default=[1.0])
    parser.add_argument("--selection", type=Path, help="selected.json from tune; eval only")
    parser.add_argument("--max-tokens", type=positive, default=2048)
    parser.add_argument("--batch-size", type=positive, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=positive, help="Smoke test subset; omit for full evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading a model")
    args = parser.parse_args()
    import math
    if any(layer < 0 for layer in args.layers) or any(not math.isfinite(a) for a in args.strengths):
        parser.error("Layers must be nonnegative and strengths finite")
    if args.command == "tune" and (args.method not in METHODS or args.selection):
        parser.error("tune requires a steering --method and cannot take --selection")
    if args.command == "eval" and not args.selection and (len(args.layers) != 1 or len(args.strengths) != 1):
        parser.error("Use tune to search multiple layers or strengths")
    suite, pairs = load(args.data)
    hashes = {name: hashlib.sha256((args.data / name).read_bytes()).hexdigest()
              for name in ("problems.jsonl", "pairs.jsonl", "split.json")}
    selected = None
    if args.selection:
        selected = json.loads(args.selection.read_text())
        for key, current in {"model": args.model, "revision": args.revision,
                             "data_hashes": hashes, "max_tokens": args.max_tokens,
                             "seed": args.seed}.items():
            if selected[key] != current:
                parser.error(f"Selection {key} differs from current run")
        args.method = selected["method"]
        if args.method not in METHODS:
            parser.error("Selection contains an unsupported method")
    rows = suite["dev" if args.command == "tune" else "eval"]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        parser.error("Empty evaluation split")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(data_hashes=hashes, system=SYSTEM, prompts=PROMPTS,
                  ids=[r["id"] for r in rows], training_ids=[p["id"] for p in pairs],
                  selection_details=selected)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", config)
    import torch
    torch.manual_seed(args.seed)
    model = Model(args.model, args.device, args.revision)
    if args.command == "tune":
        scores = []
        for layer in sorted(set(args.layers)):
            artifact = fit(model, pairs, args.method, layer)
            artifact.save(args.output / f"vector_L{layer}")
            for strength in sorted(set(args.strengths)):
                metrics = evaluate(model, rows, "normal", args.max_tokens, args.batch_size,
                                   args.output / f"L{layer}_a{strength:g}.jsonl",
                                   intervention(artifact, strength))
                scores.append(dict(layer=layer, strength=strength, **metrics))
        best = min(scores, key=lambda r: (-r["accuracy"], abs(r["strength"]), r["layer"], r["strength"]))
        write_json(args.output / "scores.json", scores)
        write_json(args.output / "selected.json", {
            **best, "method": args.method, "model": args.model, "revision": args.revision,
            "data_hashes": hashes, "max_tokens": args.max_tokens, "seed": args.seed,
            "dev_count": len(rows), "artifact": f"vector_L{best['layer']}",
        })
    else:
        control = None
        style = args.method if args.method in PROMPTS else "normal"
        if args.method in METHODS:
            if selected:
                import repsteer as rs
                artifact = rs.load_artifact(args.selection.resolve().parent / selected["artifact"])
                artifact.bind(model.wrapped)
                strength = selected["strength"]
            else:
                artifact = fit(model, pairs, args.method, args.layers[0])
                artifact.save(args.output / "vector")
                strength = args.strengths[0]
            control = intervention(artifact, strength)
        metrics = evaluate(model, rows, style, args.max_tokens, args.batch_size,
                           args.output / "predictions.jsonl", control)
        write_json(args.output / "summary.json", metrics)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
