"""The questions themselves: the split that fixes them, the task definitions
that pair each with an instruction, and the paths they are read from.
"""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kernel import splits as splits_module  # noqa: E402
from kernel.tasks import TASKS, make_prompted_prompt  # noqa: E402
import hashlib
from data_construction import make_splits
from kernel.tasks import TASKS, make_pair, make_prompted_prompt  # noqa: E402
from kernel.verifiers import VERIFIERS  # noqa: E402
from kernel.tasks import TASKS, make_pair  # noqa: E402
from kernel.verifiers import score  # noqa: E402
EXPECTED_TASKS = {
    "lowercase",
    "fixed_ending",
    "question_ending",
    "one_sentence",
    "no_comma",
}
import os
import subprocess
SRC = Path(__file__).resolve().parents[1] / "src"


def _write_split(root: Path, split: str, rows: list[dict]) -> None:
    path = root / split / "prompts.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_prompts_preserves_rows_and_order(tmp_path, monkeypatch):
    rows = [
        {"prompt_id": "p2", "source": "unit", "text": "Second prompt"},
        {"prompt_id": "p1", "source": "unit", "text": "First prompt"},
    ]
    _write_split(tmp_path, "dev", rows)
    monkeypatch.setattr(splits_module, "SPLITS", tmp_path)

    assert splits_module._load_prompts("dev") == rows


def test_load_prompts_rejects_duplicate_prompt_ids(tmp_path, monkeypatch):
    rows = [
        {"prompt_id": "same", "source": "unit", "text": "First"},
        {"prompt_id": "same", "source": "unit", "text": "Second"},
    ]
    _write_split(tmp_path, "dev", rows)
    monkeypatch.setattr(splits_module, "SPLITS", tmp_path)

    with pytest.raises(ValueError, match="Duplicate prompt_id 'same'.*line 2"):
        splits_module._load_prompts("dev")


@pytest.mark.parametrize("split", ["extraction", "deev", "test_ood"])
def test_eval_items_rejects_non_evaluation_splits(split):
    with pytest.raises(ValueError, match="Evaluation split must be"):
        splits_module.eval_items("no_comma", split)


def test_extraction_pairs_build_stable_normalized_pair(tmp_path, monkeypatch):
    _write_split(tmp_path, "extraction", [
        {"prompt_id": "p7", "source": "unit", "text": "  Explain rain.  "},
    ])
    monkeypatch.setattr(splits_module, "SPLITS", tmp_path)

    assert splits_module.extraction_pairs("no_comma") == [{
        "task": "no_comma",
        "pair_id": "no_comma_ext_p7",
        "prompt_id": "p7",
        "base_prompt": "Explain rain.",
        "positive": "Explain rain. Do not use any commas in your response.",
        "negative": "Explain rain.",
    }]


@pytest.mark.parametrize("split", ["dev", "test"])
def test_eval_items_build_stable_normalized_item(tmp_path, monkeypatch, split):
    _write_split(tmp_path, split, [
        {"prompt_id": "p9", "source": "unit", "text": "  Explain snow.  "},
    ])
    monkeypatch.setattr(splits_module, "SPLITS", tmp_path)

    assert splits_module.eval_items("one_sentence", split) == [{
        "task": "one_sentence",
        "item_id": f"one_sentence_{split}_p9",
        "prompt_id": "p9",
        "base_prompt": "Explain snow.",
        "prompted_prompt": make_prompted_prompt("Explain snow.", "one_sentence"),
        "verifier": TASKS["one_sentence"]["verifier"],
    }]


def _write_split_splits_smoke(root: Path, split: str) -> None:
    path = root / split / "prompts.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"prompt_id": "a", "source": "smoke", "text": "Explain the sky."},
        {"prompt_id": "b", "source": "smoke", "text": "Explain the ocean."},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_all_published_splits_flow_to_every_task_view(tmp_path, monkeypatch):
    for split in ("extraction", "dev", "test"):
        _write_split_splits_smoke(tmp_path, split)
    monkeypatch.setattr(splits_module, "SPLITS", tmp_path)

    for task in TASKS:
        pairs = splits_module.extraction_pairs(task)
        assert len(pairs) == 2
        assert len({pair["pair_id"] for pair in pairs}) == 2
        assert all(pair["base_prompt"] == pair["negative"] for pair in pairs)

        for split in ("dev", "test"):
            items = splits_module.eval_items(task, split)
            assert len(items) == 2
            assert len({item["item_id"] for item in items}) == 2
            assert all(
                item["prompted_prompt"]
                == make_prompted_prompt(item["base_prompt"], task)
                for item in items
            )


def test_build_manifest_contains_only_stable_data_identity(tmp_path):
    parts = {
        "extraction": [{"prompt_id": "p0"}],
        "dev": [{"prompt_id": "p1"}],
        "test": [{"prompt_id": "p2"}, {"prompt_id": "p3"}],
    }
    expected_hashes = {}
    for split in make_splits.FRACTIONS:
        path = tmp_path / split / "prompts.jsonl"
        path.parent.mkdir(parents=True)
        payload = f"{split}\n".encode()
        path.write_bytes(payload)
        expected_hashes[f"{split}/prompts.jsonl"] = hashlib.sha256(
            payload
        ).hexdigest()

    manifest = make_splits.build_manifest("pool-hash", parts, tmp_path)

    assert set(manifest) == {
        "schema_version",
        "source_pool_sha256",
        "shared_pool_size",
        "seed",
        "fractions",
        "sizes",
        "prompt_fields",
        "files",
    }
    assert manifest["schema_version"] == 1
    assert manifest["source_pool_sha256"] == "pool-hash"
    assert manifest["shared_pool_size"] == 4
    assert manifest["seed"] == 20260720
    assert manifest["fractions"] == make_splits.FRACTIONS
    assert manifest["sizes"] == {"extraction": 1, "dev": 1, "test": 2}
    assert manifest["prompt_fields"] == ["prompt_id", "source", "text"]
    assert {
        name: record["sha256"] for name, record in manifest["files"].items()
    } == expected_hashes
    assert {
        name: record["rows"] for name, record in manifest["files"].items()
    } == {"extraction/prompts.jsonl": 1,
          "dev/prompts.jsonl": 1,
          "test/prompts.jsonl": 2}

    text = json.dumps(manifest)
    assert "/common/" not in text
    assert "tasks" not in manifest
    assert "note" not in manifest


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_make_splits_writes_data_and_portable_manifest(tmp_path, monkeypatch):
    interim = tmp_path / "interim"
    output = tmp_path / "splits"
    interim.mkdir()
    rows = [
        {
            "prompt_id": f"p{index}",
            "source": "fake",
            "text": f"carrier {index}",
            "task_flags": {"task_a": True, "task_b": True},
            "ignored": "not released",
        }
        for index in range(10)
    ]
    source = interim / "final_pool.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(make_splits, "DATA_CONSTRUCTION", interim)
    monkeypatch.setattr(make_splits, "OUT", output)

    make_splits.main()

    written = {
        split: _read_jsonl(output / split / "prompts.jsonl")
        for split in make_splits.FRACTIONS
    }
    assert {split: len(items) for split, items in written.items()} == {
        "extraction": 5,
        "dev": 2,
        "test": 3,
    }
    actual_rows = [row for items in written.values() for row in items]
    assert {row["prompt_id"] for row in actual_rows} == {
        row["prompt_id"] for row in rows
    }
    assert all(set(row) == {"prompt_id", "source", "text"}
               for row in actual_rows)

    manifest = json.loads((output / "splits_manifest.json").read_text())
    assert manifest["source_pool_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert manifest["sizes"] == {"extraction": 5, "dev": 2, "test": 3}
    for name, identity in manifest["files"].items():
        path = output / name
        assert identity == {
            "rows": len(_read_jsonl(path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def test_task_schema_is_complete_and_stable():
    assert set(TASKS) == EXPECTED_TASKS
    for definition in TASKS.values():
        assert set(definition) == {"instruction", "verifier"}
        assert isinstance(definition["instruction"], str)
        assert definition["instruction"].strip()


def test_every_task_names_a_registered_verifier():
    for definition in TASKS.values():
        assert definition["verifier"] in VERIFIERS
        assert callable(VERIFIERS[definition["verifier"]])


@pytest.mark.parametrize("task", sorted(EXPECTED_TASKS))
def test_make_prompted_prompt_appends_the_canonical_instruction(task):
    result = make_prompted_prompt("  Why is the sky blue?  ", task)

    assert result == f"Why is the sky blue? {TASKS[task]['instruction']}"


@pytest.mark.parametrize("task", sorted(EXPECTED_TASKS))
def test_make_pair_returns_prompted_then_unprompted(task):
    prompted, unprompted = make_pair("  Why is the sky blue?  ", task)

    assert unprompted == "Why is the sky blue?"
    assert prompted == make_prompted_prompt(unprompted, task)


def test_unknown_task_is_rejected():
    with pytest.raises(KeyError):
        make_prompted_prompt("Why is the sky blue?", "not_a_task")


def test_all_tasks_flow_from_prompt_pair_to_rule_scorer():
    for task, definition in TASKS.items():
        prompted, unprompted = make_pair("Explain why the sky is blue.", task)

        assert unprompted == "Explain why the sky is blue."
        assert prompted.endswith(definition["instruction"])

        result = score("sample response", definition["verifier"])
        assert set(result) == {"rule_pass", "truncated"}
        assert isinstance(result["rule_pass"], bool)
        assert result["truncated"] is False


def test_reference_conditions_is_derived_from_interim():
    sys.path.insert(0, str(SRC))
    from kernel import config

    assert config.REFERENCE_CONDITIONS == config.RUNS / "reference_conditions"


def test_child_environment_uses_resolved_project_paths():
    sys.path.insert(0, str(SRC))
    from kernel import config

    child = config.env()
    assert child["IF_DATA_ROOT"] == str(config.DATA_ROOT)
    assert child["HF_HOME"] == config.HF_HOME


def test_data_root_override_moves_outputs_but_not_released_splits(tmp_path):
    """Outputs follow the data root; the released splits are versioned with
    the code and must not move with it, or two machines pointing at two data
    roots could silently evaluate on two different question sets."""
    code = """
import json
from kernel import config
print(json.dumps({
    "data_root": str(config.DATA_ROOT),
    "splits": str(config.SPLITS),
    "references": str(config.REFERENCE_CONDITIONS),
}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env["IF_DATA_ROOT"] = str(tmp_path / "elsewhere")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = json.loads(result.stdout)

    assert paths["data_root"] == str((tmp_path / "elsewhere").resolve())
    assert paths["splits"] == str(SRC.parent / "splits")
    assert paths["references"] == str(
        (tmp_path / "elsewhere").resolve() / "runs" / "reference_conditions"
    )
