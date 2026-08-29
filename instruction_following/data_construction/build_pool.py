"""Merge all screening verdicts under ONE rubric into the final carrier pool.

Provenance of the two screening rounds (see DATA.md):
  batches 00-15  screened twice: verdict_*.jsonl (loose rubric, VOID) and
                 strict_*.jsonl (strict rubric, AUTHORITATIVE)
  batches 16-28  screened once:  verdict_*.jsonl, already the strict rubric

The loose round is void because a blind independent audit of its output found
25/100 contaminated (mostly artifact_reply: code, document genres, generated
lists). Its verdicts are kept on disk only as rubric-sensitivity evidence.

Emits:
  final_pool.jsonl        every prompt kept under the strict rubric
  final_stats.json        drop taxonomy, per-task availability, rubric delta
  rubric_delta.jsonl      batches 00-15 prompts whose verdict flipped, with
                          both reasons — the paper-appendix evidence that
                          screening-rubric wording materially moves the dataset

Run:
  .venv/bin/python instruction_following/data_construction/build_pool.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from kernel.config import DATA_CONSTRUCTION  # noqa: E402

SCREEN = DATA_CONSTRUCTION / "screening"
RESCREENED = range(0, 16)    # strict_*.jsonl authoritative
STRICT_FIRST = range(16, 29)  # verdict_*.jsonl already strict
REASONS = {"ok", "format_req", "not_self_contained", "too_thin",
           "meta_assistant", "roleplay", "sensitive", "broken"}

def load(path):
    return {v["prompt_id"]: v for v in
            (json.loads(l) for l in open(path, encoding="utf-8"))}


def main() -> None:
    """Rebuild the final pool from the screening verdicts.

    Guarded so that importing this module does not rewrite the pool:
    it is importable from the repo root, and every line below writes
    into the real data root.
    """
    pool = {}
    for line in open(DATA_CONSTRUCTION / "oasst_pool_filtered.jsonl", encoding="utf-8"):
        r = json.loads(line)
        pool[r["prompt_id"]] = r




    problems, strict, loose = [], {}, {}
    for i in list(RESCREENED) + list(STRICT_FIRST):
        batch_ids = [json.loads(l)["prompt_id"]
                     for l in open(SCREEN / f"batch_{i:02d}.jsonl", encoding="utf-8")]
        src = SCREEN / (f"strict_{i:02d}.jsonl" if i in RESCREENED
                        else f"verdict_{i:02d}.jsonl")
        if not src.exists():
            problems.append(f"batch {i:02d}: {src.name} missing")
            continue
        vs = load(src)
        if set(vs) != set(batch_ids):
            problems.append(f"batch {i:02d}: id set mismatch ({len(vs)} vs {len(batch_ids)})")
        if bad := {v.get("reason") for v in vs.values()} - REASONS:
            problems.append(f"batch {i:02d}: unknown reasons {bad}")
        strict.update(vs)
        if i in RESCREENED:                       # loose verdicts on the same batch
            loose.update(load(SCREEN / f"verdict_{i:02d}.jsonl"))

    print("integrity:", "all 29 batches OK" if not problems else problems)

    kept = [dict(pool[pid], screen_reason="ok")
            for pid, v in strict.items() if v.get("keep") and pid in pool]
    with open(DATA_CONSTRUCTION / "final_pool.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(kept, key=lambda r: r["prompt_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Rubric sensitivity: same prompts, loose rubric vs strict
    flips = []
    for pid, lv in loose.items():
        sv = strict.get(pid)
        if sv and bool(lv.get("keep")) != bool(sv.get("keep")):
            flips.append({"prompt_id": pid, "text": pool[pid]["text"],
                          "loose": lv.get("reason"), "strict": sv.get("reason"),
                          "direction": "kept→dropped" if lv.get("keep") else "dropped→kept"})
    with open(DATA_CONSTRUCTION / "rubric_delta.jsonl", "w", encoding="utf-8") as f:
        for r in flips:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tax = Counter(v["reason"] for v in strict.values() if not v.get("keep"))
    n = len(strict)
    per_task = {f"task{i}": sum(r["task_flags"][f"task{i}"] for r in kept)
                for i in range(1, 6)}
    # Shared pool = intersection where every task flag is True
    shared = sum(all(r["task_flags"].values()) for r in kept)

    stats = {"n_screened": n, "n_kept": len(kept), "keep_rate": len(kept) / n,
             "drop_taxonomy": dict(tax), "per_task_available": per_task,
             "shared_pool_all_tasks": shared,
             "rubric_flips_in_batches_00_15": Counter(f["direction"] for f in flips)}
    (DATA_CONSTRUCTION / "final_stats.json").write_text(json.dumps(stats, indent=2, default=dict))

    print(f"\nstrict-rubric screening: {n} prompts -> kept {len(kept)} ({len(kept)/n:.1%})\n")
    for reason, c in tax.most_common():
        print(f"  {reason:20s} {c:4d}  ({c/n:5.2%})")
    print(f"\nper-task availability:")
    for t, c in per_task.items():
        print(f"  {t}: {c:4d}")
    print(f"  shared pool (all five tasks): {shared}")
    d = Counter(f["direction"] for f in flips)
    print(f"\nrubric sensitivity (batches 00-15): {len(flips)} / 2400 verdicts flipped = {len(flips)/2400:.1%}")
    print(f"  kept->dropped {d['kept→dropped']}   dropped->kept {d['dropped→kept']}")
    print(f"\n-> final_pool.jsonl / final_stats.json / rubric_delta.jsonl")



if __name__ == "__main__":
    main()
