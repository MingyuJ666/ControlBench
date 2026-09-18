"""Execute one stage on the merged full evaluation set."""

import argparse
import json
from control_bench.methods import BASELINES
from .options import METHOD_NAMES, add_run_options, make_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("learn", "generate", "judge", "score"),
                        help="learn steering artifacts, generate answers, run the blind judge, or summarize every configuration")
    parser.add_argument("--method", required=True, choices=METHOD_NAMES)
    parser.add_argument("--force", action="store_true", help="rerun this stage even when its outputs are complete")
    add_run_options(parser)
    args = parser.parse_args()
    if args.command == "learn" and args.method in BASELINES:
        parser.error(f"{args.method} has no learning step; start with generate")
    run, targets = make_run(parser, args, (args.method,))
    try:
        if args.dry_run:
            print(json.dumps(run.description((args.method,), targets), indent=2))
            return
        getattr(run, args.command)(args.method, targets, force=args.force)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
