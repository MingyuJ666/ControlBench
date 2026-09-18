"""Check that an installation can run, without a GPU and without downloading.

Checks the shipped data, installed methods, output paths and judge environment.

    python -m control_bench.cli.selfcheck
    python -m control_bench.cli.selfcheck --domain instruction_following

Exit status is 0 when every check passes, 1 otherwise. The judge check reports
whether the second interpreter exists and has vLLM; it is a warning, not a
failure, because judging is a separate step with its own environment.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

PASS, FAIL, WARN = "  ok  ", " FAIL ", " warn "


class Report:
    def __init__(self):
        self.failed = 0

    def ok(self, name, detail=""):
        print(f"[{PASS}] {name}{'  ' + detail if detail else ''}")

    def bad(self, name, detail=""):
        self.failed += 1
        print(f"[{FAIL}] {name}{'  ' + detail if detail else ''}")

    def warn(self, name, detail=""):
        print(f"[{WARN}] {name}{'  ' + detail if detail else ''}")

    def check(self, name, fn):
        try:
            detail = fn()
        except Exception as exc:
            self.bad(name, f"{type(exc).__name__}: {exc}")
        else:
            self.ok(name, detail or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="instruction_following")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="only used to shape paths, not loaded")
    args = ap.parse_args()
    r = Report()

    import repsteer

    import control_bench as cb
    from control_bench import domains
    from control_bench.methods import METHODS

    r.ok("imports", f"repsteer {version('repsteer')}, control_bench from {Path(cb.__file__).parent}")

    names = domains.names()
    if args.domain not in names:
        r.bad("domain registry", f"{args.domain!r} not among {names}")
        sys.exit(1)
    r.ok("domain registry", f"{len(names)} domain(s): {', '.join(names)}")
    dom = domains.get(args.domain)

    inputs = getattr(dom, "INPUTS", cb.DATA_ROOT / dom.NAME / "inputs")
    manifest_path = inputs / "MANIFEST.json"

    def inputs_present():
        if not manifest_path.exists():
            raise FileNotFoundError(f"no MANIFEST.json under {inputs}")
        return f"{inputs}"

    r.check("inputs directory", inputs_present)

    def hashes():
        manifest = json.loads(manifest_path.read_text())
        files = manifest.get("files", manifest)
        checked = 0
        for name, record in files.items():
            if not isinstance(record, dict) or "sha256" not in record:
                continue
            path = inputs / name
            if not path.exists():
                raise FileNotFoundError(f"{name} is in the manifest but missing")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record["sha256"]:
                raise ValueError(f"{name} sha256 differs from the manifest")
            if "rows" in record:
                rows = sum(1 for _ in path.open())
                if rows != record["rows"]:
                    raise ValueError(f"{name} has {rows} rows, manifest says {record['rows']}")
            checked += 1
        if not checked:
            raise ValueError("the manifest lists no hashed files")
        return f"{checked} file(s) match their sha256 and row count"

    r.check("shipped inputs", hashes)

    def items():
        is_if = dom.NAME == "instruction_following"
        id_key = "prompt_id" if is_if else "item_id"
        ids = {t: [i[id_key] for i in dom.items(t, dom.EVAL_SPLIT)] for t in dom.TARGETS}
        for target, values in ids.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{target}: duplicate evaluation {id_key}")
        if not is_if:
            counts = ", ".join(f"{t}: {len(values)}" for t, values in ids.items())
            return f"{sum(map(len, ids.values()))} full evaluation items ({counts})"

        from control_bench.domains.instruction_following.data import load_prompts
        if len({tuple(v) for v in ids.values()}) != 1:
            raise ValueError("targets do not share the full evaluation set")
        first = ids[dom.TARGETS[0]]
        extraction = {r["prompt_id"] for r in load_prompts("extraction")}
        if extraction.intersection(first):
            raise ValueError("extraction and evaluation prompts overlap")
        return (f"{len(first)} full evaluation items shared by {len(dom.TARGETS)} targets"
                "; no overlap with extraction")

    r.check("evaluation items", items)

    def pairs():
        p = dom.pairs(dom.TARGETS[0])
        n = len(p.positive) if hasattr(p, "positive") else len(p)
        return f"{n} contrastive pairs for {dom.TARGETS[0]}"

    r.check("contrastive pairs", pairs)

    r.ok("methods", f"native learners: {', '.join(m for m in METHODS if METHODS[m].supports(dom.NAME))}")

    def writable():
        probe = cb.output_root(dom.NAME) / "cache" / ".selfcheck"
        probe.mkdir(parents=True, exist_ok=True)
        (probe / "probe").write_text("ok")
        (probe / "probe").unlink()
        probe.rmdir()
        return f"{cb.output_root(dom.NAME)} accepts new directories"

    r.check("data root writable", writable)

    judge = dom.judge_python() if hasattr(dom, "judge_python") else None
    if not judge or not Path(judge).exists():
        r.warn("judge interpreter", f"not found at {judge}; set JUDGE_PYTHON before running judge")
    else:
        probe = subprocess.run([judge, "-c", "import vllm; print(vllm.__version__)"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            r.ok("judge interpreter", f"{judge} has vLLM {probe.stdout.strip().splitlines()[-1]}")
        else:
            r.warn("judge interpreter", f"{judge} exists but cannot import vLLM")

    print()
    if r.failed:
        print(f"{r.failed} check(s) failed.")
        sys.exit(1)
    print("All checks passed. A run still needs a GPU and the model weights.")


if __name__ == "__main__":
    main()
