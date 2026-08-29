"""Produce steering artifacts for any registered method.

Each method owns how its artifact is computed; this is the one command that
runs it, so adding a method does not add a new script to remember:

  CUDA_VISIBLE_DEVICES=2 .venv/bin/python src/produce.py --baseline probe

The target model is loaded once and reused across tasks. Artifacts are never
replaced without ``--force``, and that is checked before the weights load.
"""

# Run directly as a script as well as imported: the package root
# is this file's ancestor, not its own directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import baselines
from kernel.config import TARGET_MODEL, slug
from kernel.tasks import TASKS


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # No choices= on this positional: with nargs="*" some argparse versions
    # reject the empty "produce everything" invocation against the list.
    parser.add_argument(
        "tasks", nargs="*", metavar="TASK",
        help="tasks to produce (default: all five)",
    )
    parser.add_argument(
        "--baseline", default=baselines.DEFAULT_BASELINE, choices=baselines.keys()
    )
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--batch", type=_positive_int, default=32)
    parser.add_argument(
        "--force", action="store_true",
        help="replace artifacts that already exist",
    )
    return parser


def resolve_tasks(parser: argparse.ArgumentParser,
                  requested: list[str]) -> list[str]:
    tasks = requested or list(TASKS)
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        parser.error(
            f"unknown task(s): {', '.join(unknown)}; "
            f"choose from {', '.join(TASKS)}"
        )
    return tasks


def producer_for(baseline):
    """Return the method's producer entry point, by contract ``produce_task``."""
    module = __import__(
        f"{baseline.__name__}.producer", fromlist=["produce_task"]
    )
    produce = getattr(module, "produce_task", None)
    if produce is None:
        raise SystemExit(
            f"baseline {baseline.KEY!r} has no producer.produce_task"
        )
    return produce


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    tasks = resolve_tasks(parser, args.tasks)

    baseline = baselines.get(args.baseline)
    unavailable = baseline.check_available()
    if unavailable is not None:
        raise SystemExit(f"method {baseline.KEY!r} cannot run: {unavailable}")

    model_slug = slug(args.model)
    if not args.force:
        existing = [
            task for task in tasks
            if baseline.artifact_path(model_slug, task).exists()
        ]
        if existing:
            raise SystemExit(
                f"{baseline.KEY} artifacts already exist for "
                f"{', '.join(existing)}; pass --force to replace them"
            )

    produce = producer_for(baseline)
    print(f"method={baseline.KEY}  model={args.model}  tasks={tasks}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    ).eval()

    for task in tasks:
        produce(model, tok, model_slug, task, args.batch, args.force)


if __name__ == "__main__":
    main()
