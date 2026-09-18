"""Copy the existing approved splits byte-for-byte; never construct a new split."""

import argparse
import shutil
import filecmp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safety_steering.data import audit, write_new

LEGACY_RESULTS = Path("VRAP_0309/agentx2/agentx/VRAP/results")
LEGACY_CANONICAL = LEGACY_RESULTS / "0907/safety_task_splits_20260915T141731Z"
LEGACY_OUTER = LEGACY_RESULTS / "0915-train-dev-test"


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    before = audit(source, canonical=LEGACY_CANONICAL, outer=LEGACY_OUTER)
    if destination.exists():
        raise FileExistsError(
            "Use a new destination; existing data will not be overwritten"
        )
    destination.mkdir(parents=True)
    copied = []
    for origin, name in ((LEGACY_CANONICAL, "canonical"), (LEGACY_OUTER, "outer")):
        for path in sorted((source / origin).rglob("*.json")):
            relative = Path(name) / path.relative_to(source / origin)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if not filecmp.cmp(path, target, shallow=False):
                raise ValueError("Data copy differs")
            copied.append(str(relative))
    after = audit(destination)
    if before["models"] != after["models"]:
        raise ValueError("Migrated split summary differs")
    write_new(
        destination / "IMPORT.json",
        dict(
            operation="byte_identical_copy_no_resplit",
            files=copied,
            source_audit=before,
            destination_audit=after,
        ),
    )
    return after


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.source_root, a.out)
    print(result["status"])


if __name__ == "__main__":
    main()
