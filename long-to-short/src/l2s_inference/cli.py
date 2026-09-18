"""GPU workflow CLI. Pair crafting is always a separate process/stage."""

import argparse
import fcntl
import json
import os
from contextlib import ExitStack
from pathlib import Path

from l2s.runtime import load_environment

from .config import (
    DEFAULT_PAIRS,
    MODEL_IDS,
    STRENGTHS,
    STRENGTH_GRID_FOR_ITI,
    TRACKS,
    learned_cells,
    methods_for,
    protocol,
    run_root_for,
)
from .settings import (
    EVAL_SIZES,
    MAX_NEW_TOKENS,
    eval_sets_for,
    token_limit,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "plan",
            "fit",
            "fit-zero",
            "baselines",
            "zero-check",
            "grid",
            "judge",
            "report",
        ],
    )
    parser.add_argument("--model", choices=list(MODEL_IDS))
    parser.add_argument(
        "--track", choices=TRACKS, default=os.environ.get("L2S_TRACK", "all")
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Output root; default L2S_RUN_ROOT or PROJECT/runs",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_PAIRS,
        help="Frozen input pairs; default PROJECT/data/l2s, independent of --root",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--eval-sets",
        nargs="+",
        choices=list(EVAL_SIZES),
        help="Default math500; add amc12 to include the optional transfer set",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Default 3072; larger values are capped at 3072",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["CAA", "PCA", "LAT", "Linear-Probe", "SAE", "ITI"],
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        help="Grid shard layers; ITI remains a single global profile",
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        choices=sorted(set(STRENGTHS + STRENGTH_GRID_FOR_ITI)),
        help="Positive grid shard strengths",
    )
    parser.add_argument("--shard", help="Stable suffix for a resumable grid shard")
    parser.add_argument(
        "--zero-check",
        action="store_true",
        default=None,
        help="Opt in to a zero-dose grid gate (default: off)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Report only; clearly label unfinished grids",
    )
    parser.add_argument("--no-plots", action="store_true", help="Report only")
    args = parser.parse_args()
    if args.stage == "fit-zero":
        args.zero_check = True
    try:
        eval_sets = eval_sets_for(args.eval_sets)
        max_new_tokens = token_limit(
            args.max_new_tokens if args.max_new_tokens is not None else MAX_NEW_TOKENS
        )
    except ValueError as error:
        parser.error(str(error))
    if args.track not in TRACKS:
        parser.error("L2S_TRACK must be all, non-iti or iti")
    if args.stage not in ("judge", "report", "plan") and not args.model:
        parser.error("--model is required for this stage")
    if args.methods and args.stage != "grid":
        parser.error("--methods is only supported by grid")
    if (args.layers or args.strengths or args.shard) and args.stage != "grid":
        parser.error("--layers, --strengths and --shard are only supported by grid")
    if args.shard and any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in args.shard
    ):
        parser.error("--shard may contain only letters, numbers, underscore and hyphen")
    if args.methods and "SAE" in args.methods and args.model != "Gemma-2-9B":
        parser.error("SAE is only applicable to Gemma-2-9B")
    if args.methods and set(args.methods) - set(methods_for(args.model, args.track)):
        parser.error("--methods must belong to the selected --track")
    if args.track == "iti" and args.layers:
        parser.error("ITI uses a global profile; --layers is not applicable")
    if args.strengths:
        allowed_strengths = (
            STRENGTH_GRID_FOR_ITI if args.track == "iti" else STRENGTHS
        )
        if any(strength not in allowed_strengths for strength in args.strengths):
            parser.error("--strengths must belong to the selected --track")
    if (args.allow_partial or args.no_plots) and args.stage != "report":
        parser.error("--allow-partial and --no-plots are report options")
    root = (args.root or run_root_for(args.track)).resolve()
    names = [args.model] if args.model else list(MODEL_IDS)
    if args.stage == "plan":
        for name in names:
            cells = len(list(learned_cells(name, args.track))) + 2
            print(
                json.dumps(
                    dict(
                        model=name,
                        output_root=str(root),
                        pairs_directory=str(args.pairs.resolve()),
                        cells_per_dataset=cells,
                        total_generations=cells
                        * sum(EVAL_SIZES[name] for name in eval_sets),
                        baseline_generations=2
                        * sum(EVAL_SIZES[name] for name in eval_sets),
                        protocol=protocol(
                            name,
                            args.track,
                            eval_sets,
                            max_new_tokens,
                            zero_check=bool(args.zero_check),
                        ),
                    ),
                    indent=2,
                )
            )
        return
    load_environment(teacher=False)
    # Serialize each model's mutation commands; separate models may run on separate GPUs.
    # Grid shards for different methods/layers write disjoint JSONL files and
    # may run concurrently. Fit/baselines/report/judge remain serialized by
    # their stage lock; a full unsplit grid uses the ``all`` lock name.
    lock_name = "all" if not args.methods else "-".join(args.methods)
    if args.stage == "grid" and args.shard:
        # Each shard writes a distinct JSONL/COMPLETE marker.  Include its
        # validated suffix so layers can run concurrently instead of sharing
        # the method-wide lock.
        lock_name = f"{lock_name}-{args.shard}"
    lock_path = (
        root
        / "runs"
        / (args.model or args.stage)
        / ".locks"
        / f"{args.stage}-{lock_name}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        try:
            stages = (
                ("fit", "zero-check") if args.stage == "fit-zero" else (args.stage,)
            )
            for stage in stages:
                path = lock_path.with_name(f"{stage}-{lock_name}.lock")
                lock = stack.enter_context(path.open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another process is using this model/stage output directory")
        try:
            if args.stage == "report":
                from .report import build_report

                build_report(
                    root,
                    names,
                    args.allow_partial,
                    not args.no_plots,
                    track=args.track,
                    eval_sets=args.eval_sets,
                    max_new_tokens=args.max_new_tokens,
                )
                return
            if args.stage == "judge":
                from .judge import judge_pending

                judge_pending(root, args.device, names=names)
                return
            from .artifacts import load_pairs
            from .generation import load_subject

            pairs = load_pairs(args.pairs)
            run_config = protocol(
                args.model,
                args.track,
                eval_sets,
                max_new_tokens,
                zero_check=bool(args.zero_check),
            )
            if args.stage != "fit":
                from .scoring import load_eval

                items = load_eval(pairs, run_config["eval_sets"])
            model = load_subject(args.model, args.device)
            if args.stage == "fit":
                from .fit import fit_all

                fit_all(root, args.model, pairs, run_config, model)
            elif args.stage == "fit-zero":
                from .fit import fit_and_zero_check

                fit_and_zero_check(root, args.model, pairs, run_config, model, items)
            elif args.stage == "baselines":
                from .evaluate import baselines

                baselines(root, args.model, model, run_config, items)
            elif args.stage == "zero-check":
                from .evaluate import zero_check

                zero_check(root, args.model, model, run_config, items)
            else:
                from .evaluate import grid

                grid(
                    root,
                    args.model,
                    model,
                    run_config,
                    items,
                    args.methods,
                    args.layers,
                    args.strengths,
                    args.shard,
                )
        except (RuntimeError, FileNotFoundError) as error:
            parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
