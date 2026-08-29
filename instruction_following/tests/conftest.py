"""Guard: a unit test must never create anything in the real data root.

Tests use a fake model named ``model`` and redirect the path helpers to
``tmp_path``. When a redirect is missed, the test still passes — it just
writes its scratch output next to real results, where a stray ``model``
directory once appeared beside ``qwen3-8b``. Nothing fails, so the leak is
found by eye or not at all.

Reads are left alone: several tests legitimately load the released splits and
reference summaries. Only new entries are a defect, and they are always a new
model-slug directory, so watching that one level costs nothing per test.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kernel.config import (BASELINES, CALIBRATION,  # noqa: E402
                    REFERENCE_CONDITIONS, TEST)

WATCHED = (BASELINES, CALIBRATION, TEST, REFERENCE_CONDITIONS)


def _model_directories() -> set[Path]:
    return {
        child
        for root in WATCHED if root.exists()
        for child in root.iterdir()
    }


@pytest.fixture(autouse=True)
def real_data_root_is_not_written_to():
    before = _model_directories()
    yield
    created = sorted(_model_directories() - before)
    if not created:
        return
    # Residue from the test that just ran; an empty directory it made is
    # unambiguous, so remove it. Anything with contents is left in place to
    # be looked at rather than silently deleted.
    removed, kept = [], []
    for path in created:
        if path.is_dir() and not any(path.rglob("*")):
            path.rmdir()
            removed.append(path)
        else:
            kept.append(path)
    pytest.fail(
        "test wrote into the real data root: "
        + ", ".join(str(path) for path in created)
        + f"\nremoved {len(removed)} empty, kept {len(kept)} with contents"
        + "\npatch the path helper (calibration_dir / test_dir / "
        "artifact_path) onto tmp_path instead"
    )
