"""CAA dev-calibration and frozen-test entry point.

    python run.py status                      where every task stands
    python run.py calibrate --task all        sweep + judge + rank, on dev
    python run.py calibrate --task all --parallel 2 --gpus 0,1,2,3
    python run.py calibrate --task no_comma --grid layer=9,15,21,27,33
    python run.py test --task all             final evaluation, on test
    python run.py test --task no_comma --config layer=18,alpha=2.5
    python run.py all                         calibrate, then test

This command starts after data preparation and CAA vector extraction.  The
requested task vectors must already exist; reference-condition summaries are
optional.

Each stage is a separate subprocess. The target model and the judge are
different models, and swapping them inside one process invites CUDA memory
trouble and makes a late failure cost the whole run; separate processes mean a
stage can fail and be re-run alone.

Stages are skipped when their output already exists, so an interrupted run is
resumed by repeating the same command. `test` refuses to overwrite results
unless --force: the test split answers once.

GPU selection is the caller's (CUDA_VISIBLE_DEVICES); the judge needs about
54GB, so give it two 40GB cards or one 80GB.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import baselines  # noqa: E402
from kernel.config import (JUDGE_MODEL, TARGET_MODEL, artifact_path,  # noqa: E402
                    calibration_dir, env, slug, test_dir)
from kernel.generate import MAX_NEW_TOKENS
from kernel.tasks import TASKS  # noqa: E402

SRC = Path(__file__).resolve().parent / "src"
PY = sys.executable


def sh(script: str, *args, gpus: str | None = None) -> None:
    """Run a stage as a subprocess.

    GPUs are passed explicitly rather than through os.environ: run_pool drives
    several tasks from threads of one process, and a thread setting a
    process-wide CUDA_VISIBLE_DEVICES would silently change what its siblings
    see — two tasks landing on one pair of cards, out of memory, no error that
    points at the cause."""
    cmd = [PY, str(SRC / script), *map(str, args)]
    e = env()
    if gpus:
        e["CUDA_VISIBLE_DEVICES"] = gpus
    print(f"\n$ {'CUDA_VISIBLE_DEVICES=' + gpus + ' ' if gpus else ''}"
          f"{' '.join(cmd[1:])}", flush=True)
    r = subprocess.run(cmd, env=e, cwd=SRC)
    if r.returncode:
        raise SystemExit(f"{script} failed ({r.returncode})")


def judge(judging_dir: Path, gpus=None, batch=None) -> None:
    if not any(judging_dir.glob("judge_in_*.jsonl")):
        raise SystemExit(f"no judge batches in {judging_dir}")
    # judge_local owns completeness and provenance checks. It exits before
    # loading model weights when every compatible shard is already complete,
    # and otherwise resumes only the missing shards.
    sh(
        "kernel/judge_local.py", judging_dir, "--model", JUDGE_MODEL,
        *(["--batch", batch] if batch is not None else []),
        gpus=gpus,
    )


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def judging_complete(judging_dir: Path) -> bool:
    """Every input item needs a verdict, not just every shard a file.

    A shard whose responses would not parse is written out with the verdicts
    that did parse, and the runner then skips it because its file exists. With
    greedy decoding a rejudge reproduces the same unparseable text, so those
    items can never arrive — which makes an existence check report a search as
    judged when part of it has no decisions at all.
    """
    inputs = sorted(judging_dir.glob("judge_in_*.jsonl"))
    if not inputs:
        return False
    for path in inputs:
        name = path.name.replace("judge_in_", "")
        verdicts = _line_count(judging_dir / f"judge_out_{name}")
        unparsed = _line_count(judging_dir / f"judge_unparsed_{name}")
        if verdicts + unparsed != _line_count(path) or unparsed:
            return False
    return True


def _plan_axis_values(plan: dict, axis: str) -> list:
    """Every value a plan sweeps along one axis, in the order it declares them."""
    seen = []
    for stage in plan["stages"]:
        for config in stage["configs"]:
            if axis in config and config[axis] not in seen:
                seen.append(config[axis])
    return seen


def _check_resumable(search: dict, model: str, method: str, limit,
                     grid: dict) -> None:
    """Refuse to extend a dev search whose declared plan differs from this
    request: mixing two grids inside one search would make the reported
    budget and the selected configuration disagree with each other.

    The comparison is by axis name, so it holds for any baseline's plan —
    ITI's (heads_k, alpha) as much as a residual-stream (layer, alpha)."""
    if search["model"] != model:
        raise SystemExit(
            "--model differs from the existing dev search; pass --force "
            "to start a new search"
        )
    if search.get("method", method) != method:
        raise SystemExit(
            f"existing dev search used baseline {search.get('method')!r}; "
            f"pass --baseline {search.get('method')} or --force"
        )
    if limit != search["limit"]:
        raise SystemExit(
            "--limit differs from the existing dev search; pass --force "
            "to start a new search"
        )
    # The generation budget decides what the outputs are, so a search made
    # under a different one is a different measurement. Nothing else here
    # would notice: the model, the baseline and the grid all still match.
    recorded = search.get("generation")
    recorded_cap = (
        recorded.get("max_new_tokens") if isinstance(recorded, dict) else None
    )
    if recorded_cap != MAX_NEW_TOKENS:
        raise SystemExit(
            f"the existing dev search generated at {recorded_cap} new tokens, "
            f"this one at {MAX_NEW_TOKENS}; its outputs are not the same "
            "measurement, so pass --force to redo it"
        )
    plan = search["plan"]
    axes = plan["axes"]
    unknown = [axis for axis in grid if axis not in axes]
    if unknown:
        raise SystemExit(
            f"this dev search varies {axes}; a grid for {unknown} does not "
            "apply to it, so pass --force to start a different search"
        )
    source = search.get("grid_source", {})
    for axis in axes:
        stored = _plan_axis_values(plan, axis)
        requested = grid.get(axis)
        if requested is None:
            if source.get(axis) == "external":
                raise SystemExit(
                    f"existing search used an external {axis} grid; repeat it "
                    "or pass --force"
                )
            continue
        values = list(requested)
        if values != stored:
            raise SystemExit(
                f"the {axis} grid differs from the existing dev search "
                f"({values} against {stored}); pass --force"
            )


def split_gpus(gpus: str | None, per_worker: int) -> list[str | None]:
    """Cut the caller's GPUs into workers of a given size.

    Generation fits one card; the judge needs about 54GB and so needs two.
    Sizing each phase separately is the whole point: with one worker per pair
    of cards, the card that only the judge needs sits idle through generation,
    which is where most of the time goes.
    """
    if not gpus:
        return [None]
    ids = [value.strip() for value in gpus.split(",") if value.strip()]
    if len(ids) < per_worker:
        raise SystemExit(
            f"need at least {per_worker} GPUs for this phase; got {gpus}"
        )
    return [
        ",".join(ids[start:start + per_worker])
        for start in range(0, len(ids) - per_worker + 1, per_worker)
    ]


def _phase(name: str, jobs: list, groups: list) -> None:
    if not jobs:
        print(f"\n{name}: nothing to do")
        return
    print(f"\n{'=' * 70}\n{name}: {len(jobs)} job(s) over {len(groups)} "
          f"worker(s) {groups}\n{'=' * 70}", flush=True)
    if len(groups) > 1:
        run_pool(jobs, groups)
    else:
        for job in jobs:
            job(groups[0])


def calibrate_phased(tasks: list[str], model: str, method: str = "caa",
                     limit=None, grid=None,
                     force=False, gpus=None, target_batch=None,
                     judge_batch=None) -> None:
    """Run one baseline's dev search over several tasks, phase by phase.

    Every task generates before any task judges, so generation can spread
    across single cards and judging can pair them up. Tasks are independent,
    so nothing is lost by the reordering, and a phase skips a task whose
    output is already on disk.
    """
    generation = split_gpus(gpus, 1)
    judging = split_gpus(gpus, 2)
    grid = grid or {}
    directories = {
        task: calibration_dir(slug(model), method, task) for task in tasks
    }
    # Before generating anything, refuse a request that would silently extend
    # a search made under different terms — a pilot run with --limit is the
    # dangerous case, since its summary would be published as the dev result.
    if not force:
        for task in tasks:
            search_path = directories[task] / "search.json"
            if search_path.exists():
                _check_resumable(
                    json.loads(search_path.read_text(encoding="utf-8")),
                    model, method, limit, grid,
                )

    def generate_job(task: str):
        def job(worker=None):
            if (directories[task] / "search.json").exists() and not force:
                print(f"  {task}: search.json exists")
                return
            flags = [
                arg for axis, values in grid.items()
                for arg in ("--grid", f"{axis}={','.join(str(v) for v in values)}")
            ]
            sh("pipeline/calibrate.py", task, "--model", model,
               "--baseline", method, *flags,
               *(["--limit", limit] if limit is not None else []),
               *(["--batch", target_batch] if target_batch is not None else []),
               *(["--force"] if force else []),
               gpus=worker)
        job.__name__ = f"generate:{task}"
        return job

    def judge_job(task: str):
        def job(worker=None):
            judge(directories[task] / "judging" / "search", worker, judge_batch)
        job.__name__ = f"judge:{task}"
        return job

    # Two phases, because the plan declares one stage: every task generates
    # before any task judges, so generation spreads across single cards and
    # judging pairs them up. Tasks are independent, so nothing is lost by the
    # reordering, and a phase skips a task whose output is already on disk.
    _phase("generation", [generate_job(task) for task in tasks], generation)
    _phase("judging", [judge_job(task) for task in tasks], judging)

    for task in tasks:
        sh("pipeline/calibrate.py", task, "--model", model,
           "--baseline", method, "--report")


def produce(task: str, model: str, method: str = "caa", force=False,
            gpus=None, batch=None) -> None:
    """Make this method's artifact for one task, unless it already exists.

    Skipping rather than failing is what lets ``all`` run end to end for a
    method whose artifacts were produced earlier; ``--force`` still replaces
    them deliberately."""
    path = artifact_path(slug(model), method, task)
    if path.exists() and not force:
        print(f"  artifact exists: {path}")
        return
    sh("pipeline/produce.py", task, "--model", model, "--baseline", method,
       *(["--batch", batch] if batch is not None else []),
       *(["--force"] if force else []), gpus=gpus)


def test_generate(task: str, model: str, method: str = "caa", config=None,
                  reason="", force=False, gpus=None,
                  target_batch=None) -> None:
    manual = ["--config", config] if config else []
    d = test_dir(slug(model), method, task, test_subdir_name(method, config))
    if (d / "run.json").exists() and not force:
        print(f"  test generation complete: {d / 'run.json'}")
        return
    sh("pipeline/evaluate.py", task, "--model", model, "--baseline", method,
       *manual,
       *(["--reason", reason] if reason else []),
       *(["--batch", target_batch] if target_batch is not None else []),
       *(["--force"] if force else []), gpus=gpus)


def test_judge(task: str, model: str, method: str = "caa", config=None,
               gpus=None, judge_batch=None) -> None:
    manual = ["--config", config] if config else []
    d = test_dir(slug(model), method, task, test_subdir_name(method, config))
    judge(d / "judging", gpus, judge_batch)
    sh("pipeline/evaluate.py", task, "--model", model, "--baseline", method,
       *manual, "--report")


def test_subdir_name(method: str, config: str | None) -> str:
    """The directory one frozen configuration answers test in."""
    if not config:
        return "calibrated"
    baseline = baselines.get(method)
    from pipeline.evaluate import parse_config

    return f"manual_{baseline.config_name(parse_config(config))}"


def status(model: str, task: str = "all", method: str = "caa",
           config=None) -> None:
    if task != "all" and task not in TASKS:
        raise SystemExit(f"unknown task {task}; choose from {list(TASKS)}")
    if config and task == "all":
        raise SystemExit("status for a manual --config needs one --task")
    selected_tasks = list(TASKS) if task == "all" else [task]
    test_subdir = test_subdir_name(method, config)
    ms = slug(model)
    print(f"model {model}\nmethod {method}\ntest run {test_subdir}\n")
    for key, state in baselines.availability().items():
        if key == method or state != "available":
            print(f"  {key:6s} {state}")
    print()
    print(f"{'task':17s} {'vector':>7s} {'calib':>7s} {'judged':>7s} "
          f"{'chosen':>12s} {'test':>6s} {'judged':>7s} {'accuracy':>9s}")
    for task_name in selected_tasks:
        vec = artifact_path(ms, method, task_name).exists()
        cd = calibration_dir(ms, method, task_name)
        cal = (cd / "search.json").exists() or (cd / "summary.json").exists()
        cj = judging_complete(cd / "judging" / "search") if cd.exists() else False
        chosen = ""
        if (cd / "summary.json").exists():
            summary = json.loads((cd / "summary.json").read_text())
            chosen_config = summary.get("chosen") or {}
            axes = summary["axes"]
            if chosen_config:
                chosen = " ".join(
                    f"{axis}={chosen_config[axis]}"
                    for axis in axes if axis in chosen_config
                )
        td = test_dir(ms, method, task_name, test_subdir)
        test_complete = (td / "run.json").exists()
        test_judged = judging_complete(td / "judging") if td.exists() else False
        accuracy = ""
        if (td / "summary.json").exists():
            test_summary = json.loads((td / "summary.json").read_text())
            accuracy = f"{test_summary['accuracy']:.1%}"
        marker = lambda value: "  ok" if value else "   -"
        print(
            f"{task_name:17s} {marker(vec):>7s} {marker(cal):>7s} "
            f"{marker(cj):>7s} {chosen:>12s} {marker(test_complete):>6s} "
            f"{marker(test_judged):>7s} {accuracy:>9s}"
        )


def run_pool(jobs, gpu_groups):
    """Run jobs across GPU groups, one job per group, next job starts when a
    group frees up. The judge needs ~54GB, so a group is normally two 40GB
    cards."""
    import threading
    free = list(gpu_groups)
    cv = threading.Condition()
    errors = []

    def worker(job, gpus):
        try:
            job(gpus)
        except (Exception, SystemExit) as error:
            with cv:
                errors.append(f"{job.__name__}: {error}")
        finally:
            with cv:
                free.append(gpus)
                cv.notify()

    threads = []
    for job in jobs:
        with cv:
            while not free:
                cv.wait()
            gpus = free.pop()
        t = threading.Thread(target=worker, args=(job, gpus), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if errors:
        raise SystemExit("\n".join(errors))


def gpu_groups(gpus: str | None, parallel: int, n_jobs: int):
    """Validate and split caller-selected GPUs into equal worker groups."""
    if parallel < 1:
        raise SystemExit("--parallel must be at least 1")
    if parallel > n_jobs:
        raise SystemExit(
            f"--parallel {parallel} exceeds the {n_jobs} selected task(s)"
        )
    if not gpus:
        if parallel > 1:
            raise SystemExit("--parallel needs --gpus, e.g. --gpus 0,1,2,3")
        return [None]
    ids = [value.strip() for value in gpus.split(",")]
    if any(not value for value in ids):
        raise SystemExit("--gpus contains an empty GPU identifier")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"--gpus contains duplicate identifiers: {ids}")
    if len(ids) % parallel:
        raise SystemExit(
            f"{len(ids)} GPU(s) cannot be divided evenly across "
            f"--parallel {parallel}"
        )
    per_group = len(ids) // parallel
    return [
        ",".join(ids[start:start + per_group])
        for start in range(0, len(ids), per_group)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stage",
        choices=["status", "produce", "calibrate", "test", "all", "report"],
    )
    parser.add_argument("--task", default="all")
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument(
        "--baseline", default=baselines.DEFAULT_BASELINE, choices=baselines.keys(),
        help="steering method to run; each writes under its own path segment"
    )
    parser.add_argument(
        "--config",
        help="one frozen configuration for test, e.g. layer=21,alpha=1.0 or "
             "heads_k=48,alpha=15 — the axes are the baseline's own",
    )
    parser.add_argument("--reason", default="")
    parser.add_argument("--limit", type=int, help="dev calibration item cap")
    parser.add_argument(
        "--grid", action="append", metavar="AXIS=V1,V2",
        help="calibration grid for one of the baseline's axes, repeatable: "
             "layer=9,15,21,27,33 or heads_k=16,48,128 or alpha=0.5,1,2"
    )
    parser.add_argument(
        "--split", default="test", choices=("dev", "test"),
        help="which split report tabulates"
    )
    parser.add_argument(
        "--target-batch", type=int, help="target-model generation batch size"
    )
    parser.add_argument(
        "--judge-batch", type=int, help="judge-model batch size"
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="number of tasks to run at once (needs --gpus when greater than 1)"
    )
    parser.add_argument(
        "--gpus", help="GPUs available, e.g. 0,1,2,3; divided equally "
        "across parallel workers"
    )
    parser.add_argument("--force", action="store_true")
    return parser


def validate_stage_args(args) -> None:
    if args.stage == "calibrate" and (args.config is not None or args.reason):
        raise SystemExit(
            "--layer/--alpha/--reason are for a manual test run, not calibrate"
        )
    if args.stage == "test" and (
        args.grid is not None or args.limit is not None
    ):
        raise SystemExit("--grid/--limit are for calibrate, not test")
    if args.split != "test" and args.stage != "report":
        raise SystemExit("--split selects what report tabulates, nothing else")
    if args.parallel != 1 and args.stage == "calibrate":
        raise SystemExit(
            "calibrate divides --gpus by phase itself — generation one card "
            "per task, judging two — so --parallel would contradict it"
        )
    if args.stage == "all" and (
        args.config is not None or args.reason or args.limit is not None
    ):
        raise SystemExit(
            "all uses the dev-chosen test configuration; manual "
            "--layer/--alpha/--reason and pilot --limit are not allowed"
        )
    if args.stage == "status" and (
        args.grid is not None or args.limit is not None
        or args.reason or args.target_batch is not None
        or args.judge_batch is not None or args.force or args.gpus
        or args.parallel != 1
    ):
        raise SystemExit(
            "status accepts only --task, --model, and an optional manual "
            "--layer/--alpha"
        )
    for name, value in (
        ("--target-batch", args.target_batch),
        ("--judge-batch", args.judge_batch),
    ):
        if value is not None and value < 1:
            raise SystemExit(f"{name} must be at least 1")


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    validate_stage_args(args)
    if args.stage == "status":
        return status(args.model, args.task, args.baseline, args.config)
    if args.stage == "report":
        from pipeline.report import collect, render_markdown, write

        collected = collect(args.model, args.split)
        print(render_markdown(collected))
        for path in write(collected):
            print(f"-> {path}")
        return None
    tasks = list(TASKS) if args.task == "all" else [args.task]
    if args.task != "all" and args.task not in TASKS:
        raise SystemExit(
            f"unknown task {args.task}; choose from {list(TASKS)}"
        )
    if args.config is not None and len(tasks) > 1:
        raise SystemExit("--layer/--alpha apply to a single task")
    from pipeline.calibrate import parse_grid

    grid = parse_grid(args.grid)

    def produce_job(task):
        def job(worker=None):
            produce(
                task, args.model, args.baseline, args.force, worker,
                args.target_batch,
            )
        job.__name__ = f"produce:{task}"
        return job

    def test_generate_job(task):
        def job(worker=None):
            test_generate(
                task, args.model, args.baseline, args.config, args.reason,
                args.force, worker, args.target_batch,
            )
        job.__name__ = f"test-generate:{task}"
        return job

    def test_judge_job(task):
        def job(worker=None):
            test_judge(
                task, args.model, args.baseline, args.config, worker,
                args.judge_batch,
            )
        job.__name__ = f"test-judge:{task}"
        return job

    # Order matters: an artifact has to exist before it can be calibrated, and
    # a configuration has to be chosen before test can freeze it.
    if args.stage in ("produce", "all"):
        _phase(
            "produce", [produce_job(task) for task in tasks],
            gpu_groups(args.gpus, args.parallel, len(tasks)),
        )
    if args.stage in ("calibrate", "all"):
        calibrate_phased(
            tasks, args.model, args.baseline, args.limit, grid,
            args.force, args.gpus,
            args.target_batch, args.judge_batch,
        )
    if args.stage in ("test", "all"):
        # Same split as the dev search: generation fits one card, the judge
        # needs two, so every task generates before any task judges.
        _phase(
            "test-generation", [test_generate_job(task) for task in tasks],
            split_gpus(args.gpus, 1),
        )
        _phase(
            "test-judging", [test_judge_job(task) for task in tasks],
            split_gpus(args.gpus, 2),
        )

    print()
    status(
        args.model, args.task, args.baseline,
        args.config if args.stage == "test" else None,
    )


if __name__ == "__main__":
    main()
