"""Rebuild the pinned BFCL split without model or GPU dependencies."""

import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def main():
    module = runpy.run_path(str(ROOT / "src/tool_use/multi_turn/splits.py"))
    source = ROOT / "data/inputs/multi_turn.jsonl"
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    record = module["build_split"](rows, hashlib.sha256(raw).hexdigest())
    destination = source.with_name("multi_turn_split.json")
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"{destination}: {len(record['dev'])} dev / {len(record['test'])} test")


if __name__ == "__main__":
    main()
