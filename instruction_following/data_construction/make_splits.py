"""Cut the carrier pool into extraction / dev / test (DESIGN.md §2.3).

One shared 50/20/30 split serving all five tasks, drawn from the prompts that
pass every task's S5 filter. Shared carriers mean the difficulty ladder varies
only the instruction — the point of the main figure.

Each split is stored as ONE list of prompts; task-specific views (pairs,
prompted evaluation inputs) are derived at load time by src/splits.py. See that
module for why the derivations are not written to disk.

Determinism: one seeded shuffle, with the pool hash and seed recorded in
splits_manifest.json so the exact split can be rebuilt.

Run:
  .venv/bin/python instruction_following/data_construction/make_splits.py
"""
import hashlib
import json
import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from kernel.config import DATA_CONSTRUCTION, SPLITS  # noqa: E402

OUT = SPLITS
SEED = 20260720
FRACTIONS = {"extraction": 0.50, "dev": 0.20, "test": 0.30}
SCHEMA_VERSION = 1
PROMPT_FIELDS = ("prompt_id", "source", "text")


def _sha256(path: Path) -> str:
    """Return the byte-level identity of one released split file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(pool_sha256: str, parts: dict[str, list[dict]],
                   output_root: Path) -> dict:
    """Build portable split identity without code paths or task copies."""
    sizes = {split: len(items) for split, items in parts.items()}
    files = {
        f"{split}/prompts.jsonl": {
            "rows": sizes[split],
            "sha256": _sha256(output_root / split / "prompts.jsonl"),
        }
        for split in FRACTIONS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_pool_sha256": pool_sha256,
        "shared_pool_size": sum(sizes.values()),
        "seed": SEED,
        "fractions": FRACTIONS,
        "sizes": sizes,
        "prompt_fields": list(PROMPT_FIELDS),
        "files": files,
    }


def main():
    pool_path = DATA_CONSTRUCTION / "final_pool.jsonl"
    raw = pool_path.read_bytes()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]

    shared = [r for r in rows if all(r["task_flags"].values())]
    shared.sort(key=lambda r: r["prompt_id"])       # canonical order, then seeded shuffle
    random.Random(SEED).shuffle(shared)

    n = len(shared)
    n_ext = int(n * FRACTIONS["extraction"])
    n_dev = int(n * FRACTIONS["dev"])
    parts = {"extraction": shared[:n_ext],
             "dev": shared[n_ext:n_ext + n_dev],
             "test": shared[n_ext + n_dev:]}

    # Three fields only: a join key back to final_pool.jsonl and the screening
    # verdicts, the carrier text itself, and provenance. n_words is derivable
    # from the text, and task_flags would be constant-true here because the
    # shared pool is the intersection of every task's filter — a field that
    # carries no information invites the reader to think a filter is happening.
    for split, items in parts.items():
        (OUT / split).mkdir(parents=True, exist_ok=True)
        with open(OUT / split / "prompts.jsonl", "w", encoding="utf-8") as f:
            for r in items:
                payload = {field: r[field] for field in PROMPT_FIELDS}
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    ids = {k: {r["prompt_id"] for r in v} for k, v in parts.items()}
    overlaps = {f"{a}&{b}": len(ids[a] & ids[b]) for a, b in
                [("extraction", "dev"), ("extraction", "test"), ("dev", "test")]}
    assert all(v == 0 for v in overlaps.values()), overlaps

    manifest = build_manifest(hashlib.sha256(raw).hexdigest(), parts, OUT)
    (OUT / "splits_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"shared pool {n} prompts (usable by all five tasks), seed {SEED}")
    for k, v in parts.items():
        print(f"  {k:11s} {len(v):5d}  ({len(v)/n:.1%})  -> {k}/prompts.jsonl")
    print(f"\ndisjointness check: {overlaps} OK")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
