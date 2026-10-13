"""Argument types and shared options for the public execution commands."""

import argparse
import math
from pathlib import Path

from core.methods import BASELINES, METHODS
from core.methods import iti

METHOD_NAMES = (*BASELINES, *METHODS)
DEFAULT_METHODS = ("unsteered", "caa", "lat")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def indices(value):
    try:
        values = tuple(int(x) for x in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("use comma-separated integer indices") from None
    if not values or min(values) < 0 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("indices must be distinct and nonnegative")
    return values


def strengths(value):
    try:
        values = tuple(float(x) for x in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("use comma-separated numeric strengths") from None
    if not values or not all(math.isfinite(x) for x in values):
        raise argparse.ArgumentTypeError("strengths must be finite numbers")
    if len({f"{x:g}" for x in values}) != len(values):
        raise argparse.ArgumentTypeError("strengths must have distinct candidate IDs (six significant digits)")
    return values


def method_names(value):
    values = tuple(value.split(","))
    if len(set(values)) != len(values) or any(x not in METHOD_NAMES for x in values):
        raise argparse.ArgumentTypeError(f"choose distinct methods from {', '.join(METHOD_NAMES)}")
    return values


def add_output_option(parser):
    parser.add_argument("--out", type=Path, metavar="DIR",
                        help="output root for artifacts, runs and cache; campaign logs default beneath DIR/logs")


def add_run_options(parser):
    add_output_option(parser)
    parser.add_argument("--domain", default="instruction_following", help="domain package (default: instruction_following)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Hugging Face model ID or local model path")
    parser.add_argument("--revision", help="Hugging Face model/tokenizer revision")
    parser.add_argument("--targets", default="all", help="comma-separated target names, or all")
    parser.add_argument("--layers", "--blocks", dest="blocks", metavar="LAYERS", type=indices, help="zero-based Transformer layer indices; required for layer steering methods; ITI selects across all layers")
    parser.add_argument("--strengths", "--grid", dest="grid", metavar="STRENGTHS", type=strengths, help="strengths; required for steering methods; signed values: --strengths=-1,1")
    parser.add_argument("--batch", type=positive_int, default=16, help="capture/generation batch size (default: 16)")
    parser.add_argument("--max-new", type=positive_int, default=3072, help="generation token limit (default: 3072)")
    parser.add_argument("--eval-split", choices=("full", "dev", "test"), default="full",
                        help="evaluation pool (default: full); IF dev/test use the original 405/609 split")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=positive_int, metavar="N", help="first N items per target within --eval-split")
    selection.add_argument("--sample", type=positive_int, metavar="N", help="randomly select N items per target within --eval-split, without replacement")
    parser.add_argument("--seed", type=int, help="evaluation sampling seed for --sample (default: 42)")
    parser.add_argument("--train-sample", type=positive_int, metavar="N", help="randomly select N extraction pairs per target without replacement; default: all pairs")
    parser.add_argument("--train-seed", type=int, help="training sampling seed for --train-sample (default: 42)")
    parser.add_argument("--iti-top-k", type=positive_int, metavar="K", help="ITI heads selected across all layers (default: 48)")
    parser.add_argument("--iti-seed", type=int, help="ITI probe train/validation split seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="print configuration and output counts without loading a model or launching jobs")


def make_run(parser, args, methods):
    from core import domains
    from core.runtime.experiment import Run

    if args.domain not in domains.names():
        parser.error(f"unknown domain {args.domain!r}; available: {', '.join(domains.names())}")
    domain = domains.get(args.domain)
    targets = tuple(domain.TARGETS) if args.targets == "all" else tuple(args.targets.split(","))
    if len(set(targets)) != len(targets) or set(targets) - set(domain.TARGETS):
        parser.error(f"choose distinct targets from: {', '.join(domain.TARGETS)}")
    unsupported = [m for m in methods
                   if (m == "prompted" and not callable(getattr(domain, "prompted_items", None)))
                   or (m not in BASELINES and not METHODS[m].supports(domain.NAME))]
    if unsupported:
        parser.error(f"{', '.join(unsupported)} is not supported for {domain.NAME}")
    layer_methods = [m for m in methods if m not in (*BASELINES, "iti")]
    if layer_methods and (args.blocks is None or args.grid is None):
        parser.error("layer steering methods require both --layers and --strengths; neither has a default")
    if "iti" in methods:
        if args.grid is None:
            parser.error("ITI requires --strengths (alpha values); it has no default")
        if not layer_methods and args.blocks is not None:
            parser.error("ITI selects heads across all layers; omit --layers when no layer method is requested")
        if args.train_sample is not None and args.train_sample < 2:
            parser.error("ITI requires --train-sample of at least 2 pairs")
    elif args.iti_top_k is not None or args.iti_seed is not None:
        parser.error("--iti-top-k and --iti-seed require iti in --method or --methods")
    if getattr(domain, "EVAL_SPLIT", None) != "full":
        parser.error(f"{args.domain} has no full evaluation workflow yet")
    if args.eval_split not in getattr(domain, "EVAL_SPLITS", (domain.EVAL_SPLIT,)):
        parser.error(f"{args.domain} does not support --eval-split {args.eval_split}")
    if args.seed is not None and args.sample is None:
        parser.error("--seed requires --sample")
    if args.train_seed is not None and args.train_sample is None:
        parser.error("--train-seed requires --train-sample")
    run = Run(domain, args.model, tuple(args.blocks or ()), tuple(args.grid or ()),
              batch=args.batch, max_new=args.max_new, limit=args.limit, revision=args.revision,
              eval_split=args.eval_split,
              sample=args.sample, seed=args.seed if args.seed is not None else 42,
              train_sample=args.train_sample, train_seed=args.train_seed if args.train_seed is not None else 42,
              iti_top_k=args.iti_top_k if args.iti_top_k is not None else iti.DEFAULT_TOP_K,
              iti_seed=args.iti_seed if args.iti_seed is not None else iti.DEFAULT_SEED)
    try:
        for method in methods:
            run.check_method(method)
        for target in targets:
            if args.sample is not None:
                run.items(target)
            if args.train_sample is not None:
                run.training_pairs(target)
    except ValueError as exc:
        parser.error(str(exc))
    return run, targets
