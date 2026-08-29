"""Fetch the carrier-prompt source: OASST1 + OASST2 (DESIGN.md §2.1).

We take only first-turn English user messages — `role == "prompter"`,
`parent_id is None`, `lang == "en"`. OASST stores a conversation tree one
message per row, so a root prompter message is a self-contained user request;
deeper prompter messages are follow-ups that depend on what the assistant just
said and cannot serve as standalone carriers.

Both corpora are fetched because DESIGN.md §2.5A lists the source as
"OASST1/2". They overlap heavily — OASST2 is effectively a superset of OASST1,
and deduplication downstream in filters.py removes ~39% of the merged rows.
Fetching both anyway is cheap and makes the overlap a measured fact rather than
an assumption.

Revisions are resolved at fetch time and printed; re-running with a printed
commit as `revision` reproduces the pool byte-for-byte. Provenance goes in
DATA.md.

Run:
  .venv/bin/python instruction_following/data_construction/fetch_oasst.py
"""
import hashlib
import json
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from kernel.config import RAW  # noqa: E402

REPOS = ["OpenAssistant/oasst1", "OpenAssistant/oasst2"]
api = HfApi()


def fetch(repo: str):
    name = repo.split("/")[-1]
    rev = api.repo_info(repo, repo_type="dataset").sha
    print(f"{repo} @ {rev}")

    ds = load_dataset(repo, split="train", revision=rev)
    rows = [{"source": name, "text": ex["text"],
             "meta": {"message_id": ex["message_id"]}}
            for ex in ds
            if ex["role"] == "prompter" and ex["parent_id"] is None
            and ex["lang"] == "en"]

    out_dir = RAW / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "pool.jsonl"
    with open(dest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f"  {len(rows)} first-turn English prompts -> {dest}")
    print(f"  sha256: {sha}")


if __name__ == "__main__":
    for repo in REPOS:
        fetch(repo)
