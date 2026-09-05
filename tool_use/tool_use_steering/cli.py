"""The command line. One entry point, three verbs.

    python -m tool_use_steering extract --method dim --axis clarify
    python -m tool_use_steering eval --method dim --vector <pt> --layer 24 \
        --dose 0,24,48 --split dev [--tasks clarify,abstain] [--n 100] \
        [--tag my_run] [--steer-turns first]
    python -m tool_use_steering sweep --grid <data>/runs/sweep100/grid.json \
        --phase generate --shard 0/4          # one process per GPU, then
    python -m tool_use_steering sweep --grid <data>/runs/sweep100/grid.json \
        --phase judge

`eval` runs the single-turn track: generate at every dose (the subject model
loads once), then judge and score (the judge loads once, after the subject
model is freed). Every dose lands in

    runs/<tag>/<split>/dose<K>/{raw.jsonl,decisions.jsonl,summary.json}

plus a meta.json pinning argv, code commit, model, method config and pool
sizes -- enough to rerun the command years later. raw.jsonl is written before
judging starts, so a judge crash costs a re-score, never a re-generation.

`sweep` is `eval` over a grid. The grid file names the sweep, the split, the
per-task item cap, the batch size and a list of configurations (name,
method, config, doses, task; `vector` paths are relative to artifacts/).
Every configuration is examined on its own task's pool alone -- `task:
"all"` is the no-method baseline, which serves every task; cross-task
side-effects are the multi-turn track's job. Each configuration lands in
runs/<sweep>/<name>/<split>/dose<K>/ with the same files and meta as an
`eval` run. The grid file lives with the results it
describes, runs/<sweep>/grid.json, and is the record of what the sweep
contained. Generation and judging are separate phases because
the subject model and the judge do not share a GPU and generation
parallelises while judging does not. Both phases skip what is already on
disk: rerunning the same command resumes a crashed sweep, and a
configuration added to the grid is the only thing the next run evaluates.

A smoke run is `--n 5 --dose 0`: five items per task, one dose, a few
minutes end to end.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys

from .eval.calls import family_for
from .paths import RUNS


def _commit() -> str | None:
    out = subprocess.run(["git", "-C", os.path.dirname(__file__),
                          "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return out or None


def cmd_eval(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .methods import load as load_method
    from .tracks import single_turn as track

    config = json.loads(args.method_config) if args.method_config else {}
    for key, value in (("vector", args.vector), ("layer", args.layer),
                       ("axis", args.axis)):
        if value is not None:
            config[key] = value
    method = load_method(args.method, config)

    tasks = tuple(args.tasks.split(",")) if args.tasks else track.TASKS
    items = track.pool(args.split, limit=args.n, tasks=tasks)
    doses = [float(d) for d in args.dose.split(",")]
    tag = args.tag or f"{args.method}_{args.split}"
    print(f"{len(items)} items ({', '.join(tasks)}), split={args.split}, "
          f"doses={doses}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    method.prepare(model, tokenizer)

    run_dirs = {}
    for dose in doses:
        records = track.generate(model, tokenizer, items, method, dose,
                                 batch_size=args.batch,
                                 steer_turns=args.steer_turns)
        out = os.path.join(RUNS, tag, args.split, f"dose{dose:g}")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "raw.jsonl"), "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump({
                "argv": sys.argv,
                "code_commit": _commit(),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "host": socket.gethostname(),
                "model": args.model,
                "method": method.describe(),
                "dose": dose,
                "split": args.split,
                "tasks": list(tasks),
                "steer_turns": args.steer_turns,
                "n_items": len(items),
            }, f, indent=2)
        run_dirs[dose] = out
        print(f"dose {dose:g}: raw.jsonl written -> {out}", flush=True)

    del model
    torch.cuda.empty_cache()

    from .eval.judge import load_judge
    judge_model, judge_tokenizer = load_judge(device_map=args.judge_device)
    for dose in doses:
        out = run_dirs[dose]
        with open(os.path.join(out, "raw.jsonl")) as f:
            records = [json.loads(line) for line in f]
        rows, summary = track.score(records, items, judge_model, judge_tokenizer,
                                    family_for(args.model))
        with open(os.path.join(out, "decisions.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== dose {dose:g} ===", flush=True)
        for task, entry in summary.items():
            print(f"  {task:8s} decision {entry['decision']:.3f}  "
                  f"followup {entry['followup']:.3f}  (n={entry['n']})",
                  flush=True)


def cmd_eval_multi_turn(args) -> None:
    """The multi-turn track: generate and score in one pass per dose.

    Scoring is mechanical (backend state comparison), so unlike the
    single-turn track there is no judge phase; every dose's numbers are
    final as soon as its generation ends.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .eval.loop import run as run_loop
    from .methods import load as load_method
    from .tracks import multi_turn as mt

    config = json.loads(args.method_config) if args.method_config else {}
    for key, value in (("vector", args.vector), ("layer", args.layer),
                       ("axis", args.axis)):
        if value is not None:
            config[key] = value
    method = load_method(args.method, config)
    doses = [float(d) for d in args.dose.split(",")]
    categories = args.categories.split(",")
    tag = args.tag or f"mt_{args.method}_{args.split}"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    method.prepare(model, tokenizer)

    for dose in doses:
        for category in categories:
            items = mt.entries(category, limit=args.n, split=args.split)
            driver = mt.Driver(f"{tag}_{category}_d{dose:g}",
                               family_for(args.model))
            records = run_loop(model, tokenizer, items, method, dose,
                               driver.next_message, batch_size=args.batch,
                               max_new_tokens=256)
            rows, summary = mt.score(driver, items)
            out = os.path.join(RUNS, tag, args.split, category, f"dose{dose:g}")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "raw.jsonl"), "w") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with open(os.path.join(out, "decisions.jsonl"), "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with open(os.path.join(out, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            with open(os.path.join(out, "meta.json"), "w") as f:
                json.dump({"argv": sys.argv, "code_commit": _commit(),
                           "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                           "host": socket.gethostname(), "model": args.model,
                           "method": method.describe(), "dose": dose,
                           "split": args.split, "category": category,
                           "n_items": len(items)}, f, indent=2)
            print(f"{category} dose {dose:g}: "
                  + json.dumps(summary), flush=True)


def cmd_sweep(args) -> None:
    """A grid of configurations, evaluated the way `eval` evaluates one.

    `generate --shard i/n` takes every i-th configuration that still has a
    dose without raw.jsonl -- sharding over pending work, not over the grid,
    so a partially finished sweep still spreads evenly over the GPUs -- and
    loads the subject model once for all of them. `judge` loads the judge
    once and scores every raw.jsonl in the grid that has no summary.json.
    """
    import torch

    from .methods import load as load_method
    from .paths import ARTIFACTS
    from .tracks import single_turn as track

    grid = json.load(open(args.grid))
    sweep, split, limit = grid["name"], grid["split"], grid.get("n")
    steer_turns = grid.get("steer_turns", "all")

    def items_for(entry: dict) -> list:
        # The tasks are independent by design: every config sits on one axis
        # and is examined on that axis's pool alone (the grid's "task");
        # cross-task side-effects are the multi-turn track's job. "all" is
        # the no-method baseline, which serves every task.
        tasks = track.TASKS if entry["task"] == "all" else (entry["task"],)
        return track.pool(split, limit=limit, tasks=tasks)

    def run_dir(name: str, dose: float) -> str:
        return os.path.join(RUNS, sweep, name, split, f"dose{dose:g}")

    def pending(entry: dict) -> list[float]:
        return [float(d) for d in entry["doses"]
                if not os.path.exists(os.path.join(run_dir(entry["name"], d),
                                                   "raw.jsonl"))]

    if args.phase == "generate":
        shard, nshards = (int(x) for x in args.shard.split("/"))
        todo = [e for e in grid["configs"] if pending(e)]
        mine = [e for i, e in enumerate(todo) if i % nshards == shard]
        print(f"{sweep}: {len(todo)} configurations pending, "
              f"{len(mine)} in shard {shard}/{nshards}", flush=True)
        if not mine:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(grid["model"])
        model = AutoModelForCausalLM.from_pretrained(
            grid["model"], dtype=torch.bfloat16, device_map=args.device)
        model.eval()
        for entry in mine:
            items = items_for(entry)
            config = dict(entry.get("config", {}))
            if "vector" in config and not os.path.isabs(config["vector"]):
                config["vector"] = os.path.join(ARTIFACTS, config["vector"])
            method = load_method(entry["method"], config)
            method.prepare(model, tokenizer)
            for dose in pending(entry):
                out = run_dir(entry["name"], dose)
                os.makedirs(out, exist_ok=True)
                print(f"[{datetime.datetime.now():%H:%M}] {entry['name']} "
                      f"dose {dose:g}", flush=True)
                records = track.generate(model, tokenizer, items, method, dose,
                                         batch_size=grid["batch"],
                                         steer_turns=steer_turns)
                raw = os.path.join(out, "raw.jsonl")
                with open(raw + ".tmp", "w") as f:     # whole file or nothing
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                os.replace(raw + ".tmp", raw)
                with open(os.path.join(out, "meta.json"), "w") as f:
                    json.dump({
                        "argv": sys.argv,
                        "code_commit": _commit(),
                        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                        "host": socket.gethostname(),
                        "model": grid["model"],
                        "method": method.describe(),
                        "dose": dose,
                        "split": split,
                        "tasks": (list(track.TASKS) if entry["task"] == "all"
                                  else [entry["task"]]),
                        "steer_turns": steer_turns,
                        "n_items": len(items),
                        "sweep": sweep,
                        "config": entry["name"],
                    }, f, indent=2)
        return

    from .eval.judge import load_judge
    items = track.pool(split, limit=limit)  # superset for id lookup; each
    #                                         raw.jsonl names what it holds
    judge_model, judge_tokenizer = load_judge(device_map=args.judge_device)
    for entry in grid["configs"]:
        for dose in entry["doses"]:
            out = run_dir(entry["name"], dose)
            raw = os.path.join(out, "raw.jsonl")
            if not os.path.exists(raw) \
                    or os.path.exists(os.path.join(out, "summary.json")):
                continue
            with open(raw) as f:
                records = [json.loads(line) for line in f]
            rows, summary = track.score(records, items, judge_model,
                                        judge_tokenizer,
                                        family_for(grid["model"]))
            with open(os.path.join(out, "decisions.jsonl"), "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with open(os.path.join(out, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print(f"{entry['name']} dose {dose:g}: " + "  ".join(
                f"{task} {e['decision']:.2f}/{e['followup']:.2f}"
                for task, e in summary.items()), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="tool_use_steering")
    sub = ap.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract",
                        help="compute a method's steering artifact from the pairs")
    ex.add_argument("--method", required=True, choices=["dim", "caa", "probe", "cast", "iti", "sadi", "sae"])
    ex.add_argument("--axis", required=True,
                    choices=["clarify", "abstain", "call"])
    ex.add_argument("--model", default="Qwen/Qwen3-8B")
    ex.add_argument("--device", default="cuda:0")
    ex.add_argument("--n", type=int, default=700)

    ev = sub.add_parser("eval", help="run a track")
    ev.add_argument("--track", default="single_turn",
                    choices=["single_turn", "multi_turn"])
    ev.add_argument("--categories", default="base,miss_param,miss_func",
                    help="multi_turn only: comma list of BFCL categories")
    ev.add_argument("--method", required=True)
    ev.add_argument("--method-config", help="JSON dict for the method")
    ev.add_argument("--vector", help="sugar for config['vector']")
    ev.add_argument("--layer", type=int, help="sugar for config['layer']")
    ev.add_argument("--axis", help="sugar for config['axis'] (prompting)")
    ev.add_argument("--dose", default="0", help="comma list, e.g. 0,24,48")
    ev.add_argument("--split", default="dev", choices=["dev", "test"])
    ev.add_argument("--tasks", help="comma list; default all three")
    ev.add_argument("--n", type=int, help="items per task (default: full pool)")
    ev.add_argument("--tag", help="run directory name (default: method_split)")
    ev.add_argument("--model", default="Qwen/Qwen3-8B")
    ev.add_argument("--device", default="cuda:0")
    ev.add_argument("--judge-device", default="auto")
    ev.add_argument("--batch", type=int, default=8)
    ev.add_argument("--steer-turns", default="all", choices=["all", "first"],
                    help="single_turn only: apply the intervention on every "
                         "turn, or only while the model writes turn 1")

    sw = sub.add_parser("sweep", help="eval over a grid file (single_turn)")
    sw.add_argument("--grid", required=True,
                    help="the sweep's grid file, runs/<sweep>/grid.json")
    sw.add_argument("--phase", required=True, choices=["generate", "judge"])
    sw.add_argument("--shard", default="0/1",
                    help="generate only: i/n, this process takes every "
                         "n-th pending configuration")
    sw.add_argument("--device", default="cuda:0")
    sw.add_argument("--judge-device", default="auto")

    # `extract` forwards method-specific flags (--k, --sae-layer, ...) to the
    # method's own parser, which stays strict -- a typo still errors there.
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        args, extra = ap.parse_known_args()
    else:
        args, extra = ap.parse_args(), []
    if args.command == "extract":
        # The extract modules are runnable on their own; the subcommand just
        # routes to the right one with the same flags.
        import importlib
        module = importlib.import_module(
            f".methods.{args.method}.extract", package=__package__)
        sys.argv = [f"{args.method}/extract.py",
                    "--axis", args.axis, "--model", args.model,
                    "--device", args.device, "--n", str(args.n)] + extra
        module.main()
    elif args.command == "eval":
        if args.track == "multi_turn":
            cmd_eval_multi_turn(args)
        else:
            cmd_eval(args)
    elif args.command == "sweep":
        cmd_sweep(args)


if __name__ == "__main__":
    main()
