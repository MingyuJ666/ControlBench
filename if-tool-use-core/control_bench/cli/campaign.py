"""Run learn -> generate -> judge -> score over every requested configuration."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import control_bench as cb
from control_bench.methods import BASELINES
from control_bench.paths import CORE_ROOT, REPO_ROOT
from control_bench.runtime.search import input_fingerprint, write_json
from .options import DEFAULT_METHODS, add_run_options, indices, make_run, method_names

MAX_RETRIES = 2


def log(message):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {message}", flush=True)


def _process_state(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return fields[0], fields[19]


def _alive(pid):
    try:
        os.kill(pid, 0)
        if Path("/proc/self/stat").exists():
            return _process_state(pid)[0] not in ("Z", "X")
        return True
    except OSError:
        return False


class Job:
    """A detached stage with a persisted request and process identity."""

    def __init__(self, name, argv, gpus, *, request, running_dir, log_dir, run_id, devices):
        self.name, self.gpus, self.request = name, list(gpus), request
        self.marker = running_dir / f"{name}.json"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices[g] for g in gpus),
               "PYTHONPATH": str(CORE_ROOT) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
        logfile = log_dir / f"{time.strftime('%m%d_%H%M%S')}_{name}.log"
        with logfile.open("a") as handle:
            self.proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=handle,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        self.pid = self.proc.pid
        start = _process_state(self.pid)[1] if Path("/proc/self/stat").exists() else None
        write_json(self.marker, {"pid": self.pid, "start": start, "gpus": self.gpus, "request": request,
                                 "run_id": run_id, "log": str(logfile)})
        log(f"start {name} on GPU {self.gpus} -> {logfile}")

    @classmethod
    def adopt(cls, marker, run_id):
        info = json.loads(marker.read_text())
        if not _alive(info["pid"]) or (info.get("start") is not None and _process_state(info["pid"])[1] != info["start"]):
            marker.unlink()
            return None
        if info["run_id"] != run_id:
            raise ValueError(f"{marker}: another configuration still has a running job; resume that command first")
        job = cls.__new__(cls)
        job.name, job.gpus, job.request = marker.stem, info["gpus"], info["request"]
        job.marker, job.pid, job.proc = marker, info["pid"], None
        log(f"adopt {job.name} (pid {job.pid})")
        return job

    def done(self):
        finished = self.proc.poll() is not None if self.proc is not None else not _alive(self.pid)
        if finished:
            self.marker.unlink(missing_ok=True)
        return finished

    @property
    def returncode(self):
        return self.proc.returncode if self.proc is not None else None


def child_command(run, stage, method, targets):
    argv = [sys.executable, "-m", "control_bench.cli.run", stage, "--domain", run.domain.NAME,
            "--model", run.model, "--method", method, "--targets", ",".join(targets),
            "--batch", str(run.batch), "--max-new", str(run.max_new)]
    if method not in BASELINES:
        argv += ["--strengths=" + ",".join(map(str, run.grid))]
        if method == "iti":
            argv += ["--iti-top-k", str(run.iti_top_k), "--iti-seed", str(run.iti_seed)]
        else:
            argv += ["--layers", ",".join(map(str, run.blocks))]
    if run.limit is not None:
        argv += ["--limit", str(run.limit)]
    if run.sample is not None:
        argv += ["--sample", str(run.sample), "--seed", str(run.seed)]
    if run.train_sample is not None:
        argv += ["--train-sample", str(run.train_sample), "--train-seed", str(run.train_seed)]
    if run.revision:
        argv += ["--revision", run.revision]
    return argv


def schedule(run, methods, targets, gpus, log_dir, poll):
    log_dir.mkdir(parents=True, exist_ok=True)
    running_dir = log_dir / "running"
    running_dir.mkdir(exist_ok=True)
    description = run.description(methods, targets)
    run_id = input_fingerprint({**description, "gpus": gpus,
                                "inputs": {t: input_fingerprint(run.items(t)) for t in targets}})
    running = [job for marker in sorted(running_dir.glob("*.json")) if (job := Job.adopt(marker, run_id))]
    write_json(log_dir / "run.json", description)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        device_list = visible.split(",")
        if max(gpus) >= len(device_list):
            raise ValueError("--gpus indexes the devices in CUDA_VISIBLE_DEVICES; a requested index is out of range")
        devices = {g: device_list[g] for g in gpus}
    else:
        devices = {g: str(g) for g in gpus}
    retries = {}
    failures = {}

    def failed(name, reason, request):
        retries[name] = retries.get(name, 0) + 1
        if retries[name] > MAX_RETRIES:
            message = f"{name} failed after {retries[name]} attempts: {reason}; logs: {log_dir}"
            method = request["method"]
            incomplete = [t for t in request["targets"] if not run.complete(request["stage"], method, [t])]
            for target in incomplete or request["targets"]:
                failures[method, target] = message
            retries.pop(name)
            log(message)
        else:
            log(f"retry {name} ({retries[name]}/{MAX_RETRIES}): {reason}")

    def launch(stage, method, task_targets, free):
        name = f"{stage}_{method}" + (f"_{task_targets[0]}" if stage == "generate" else "")
        request = {"stage": stage, "method": method, "targets": list(task_targets)}
        running.append(Job(name, child_command(run, stage, method, task_targets), [free.pop(0)], request=request,
                           running_dir=running_dir, log_dir=log_dir, run_id=run_id, devices=devices))

    while True:
        for job in [j for j in running if j.done()]:
            running.remove(job)
            request = job.request
            complete = run.complete(request["stage"], request["method"], request["targets"])
            if job.returncode not in (None, 0) or not complete:
                failed(job.name, f"exit={job.returncode}, outputs_complete={complete}", request)
            else:
                retries.pop(job.name, None)
                log(f"finished {job.name}")
        busy = {g for j in running for g in j.gpus}
        free = [g for g in gpus if g not in busy]

        def active(stage, method=None, target=None):
            return any(j.request["stage"] == stage and (method is None or j.request["method"] == method)
                       and (target is None or target in j.request["targets"]) for j in running)

        for method in methods:
            for target in targets:
                if (method, target) in failures:
                    continue
                if run.generated(method, target) and run.judged(method, target) and not run.scored(method, target):
                    key = f"score_{method}_{target}"
                    try:
                        run.score(method, [target])
                        if not run.scored(method, target):
                            raise RuntimeError("score command produced incomplete summaries")
                    except (OSError, ValueError, KeyError, RuntimeError) as exc:
                        with (log_dir / f"{key}.log").open("a") as handle:
                            handle.write(str(exc) + "\n")
                        failed(key, str(exc), {"stage": "score", "method": method, "targets": [target]})
                    else:
                        retries.pop(key, None)

        if free and not active("judge"):
            for method in methods:
                ready = [t for t in targets if (method, t) not in failures and not active("generate", method, t)
                         and run.generated(method, t) and not run.judged(method, t)]
                if ready:
                    launch("judge", method, ready, free)
                    break

        for method in methods:
            missing = [t for t in targets if (method, t) not in failures and not run.learned(method, t)]
            if free and missing and not active("learn", method):
                launch("learn", method, missing, free)

        for method in methods:
            if active("learn", method):
                continue
            for target in targets:
                if (method, target) in failures:
                    continue
                if free and not active("generate", method, target) and run.learned(method, target) and not run.generated(method, target):
                    launch("generate", method, [target], free)

        def stage(method, target):
            if (method, target) in failures:
                return "failed"
            if run.scored(method, target):
                return "done"
            if run.generated(method, target):
                return "judged" if run.judged(method, target) else "generated"
            return "learned" if run.learned(method, target) else "pending"

        state = {m: {t: stage(m, t) for t in targets} for m in methods}
        errors = list(dict.fromkeys(failures.values()))
        write_json(log_dir / "state.json", {"state": state, "running": [j.name for j in running], "failures": errors})
        if not running and all(value in ("done", "failed") for row in state.values() for value in row.values()):
            if errors:
                raise RuntimeError("Campaign finished with failed tasks:\n" + "\n".join(errors))
            log("campaign complete")
            return
        time.sleep(poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_options(parser)
    parser.add_argument("--methods", type=method_names, default=DEFAULT_METHODS,
                        help="comma-separated methods (default: unsteered,caa,lat)")
    parser.add_argument("--gpus", type=indices, required=True, help="GPU indices visible to this process, e.g. 0,1")
    parser.add_argument("--logs", type=Path, help="log directory; default: logs/<domain>/<pipeline>/<model>/<experiment>")
    parser.add_argument("--poll", type=int, default=60, help="scheduler poll interval in seconds (default: 60)")
    args = parser.parse_args()
    if args.poll < 0:
        parser.error("--poll must be nonnegative")
    run, targets = make_run(parser, args, args.methods)
    try:
        if args.dry_run:
            print(json.dumps(run.description(args.methods, targets), indent=2))
            return
        log_dir = args.logs or REPO_ROOT / "logs" / run.domain.NAME / cb.PIPELINE_ID / cb.model_slug(run.model) / run.experiment
        schedule(run, args.methods, targets, args.gpus, log_dir, args.poll)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Scheduler stopped; detached jobs continue. Rerun the same command to adopt them.\n")


if __name__ == "__main__":
    main()
