"""The pipeline as a user meets it: the search plan, the calibration run, the
report, and a scope check that no module reads a global nothing defines.
"""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as pipeline  # noqa: E402
from kernel.generate import MAX_NEW_TOKENS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import calibrate  # noqa: E402
from contextlib import contextmanager
from types import SimpleNamespace
from kernel import generate  # noqa: E402
from baselines.shared.base import (SearchPlan, SearchStage,  # noqa: E402
                                   joint_plan)
from pipeline import report  # noqa: E402
TASKS = ["lowercase", "fixed_ending", "question_ending", "one_sentence",
         "no_comma"]
import builtins
import importlib
import symtable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
MODULES = sorted(
    str(path.relative_to(SRC).with_suffix("")).replace("/", ".")
    for path in SRC.rglob("*.py")
    if "__pycache__" not in path.parts
)


def test_judge_delegates_completeness_even_when_an_output_exists(
        tmp_path, monkeypatch):
    (tmp_path / "judge_in_0.jsonl").write_text("{}\n")
    (tmp_path / "judge_out_0.jsonl").write_text("{}\n")
    calls = []
    monkeypatch.setattr(
        pipeline, "sh", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    pipeline.judge(tmp_path, gpus="0,1", batch=7)

    assert calls == [(
        ("kernel/judge_local.py", tmp_path, "--model", pipeline.JUDGE_MODEL,
         "--batch", 7),
        {"gpus": "0,1"},
    )]


def _shard(directory, index, n_in, n_out, n_unparsed=0):
    (directory / f"judge_in_{index}.jsonl").write_text("x\n" * n_in)
    (directory / f"judge_out_{index}.jsonl").write_text("v\n" * n_out)
    if n_unparsed:
        (directory / f"judge_unparsed_{index}.jsonl").write_text(
            "u\n" * n_unparsed
        )


def test_judging_complete_requires_a_verdict_for_every_item(tmp_path):
    (tmp_path / "judge_in_0.jsonl").write_text("x\n" * 3)
    assert pipeline.judging_complete(tmp_path) is False    # nothing judged yet
    _shard(tmp_path, 0, n_in=3, n_out=3)
    assert pipeline.judging_complete(tmp_path) is True
    _shard(tmp_path, 1, n_in=3, n_out=3)
    assert pipeline.judging_complete(tmp_path) is True


def test_a_shard_with_unparsed_responses_is_not_complete(tmp_path):
    """Its output file exists, so the runner skips it; a rejudge is greedy and
    reproduces the same unparseable text, so those items never get a verdict.
    Reporting the stage as judged would publish a search that is missing
    decisions."""
    _shard(tmp_path, 0, n_in=3, n_out=2, n_unparsed=1)
    assert pipeline.judging_complete(tmp_path) is False


def test_calibrate_forwards_an_axis_named_grid_and_judges_every_stage(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "calibration_dir",
        lambda model_slug, method, task: tmp_path / model_slug / method / task,
    )
    monkeypatch.setattr(pipeline, "slug", lambda _: "model")
    calls = []
    monkeypatch.setattr(
        pipeline, "sh",
        lambda *args, **kwargs: calls.append(("sh", args, kwargs)),
    )
    monkeypatch.setattr(
        pipeline, "judge",
        lambda path, gpus=None, batch=None:
        calls.append(("judge", path, gpus, batch)),
    )


    pipeline.calibrate_phased(
        ["no_comma"], "org/model", limit=6,
        grid={"layer": [7, 19, 31], "alpha": [0.5, 2, 4]},
        gpus="0,1", target_batch=3, judge_batch=4,
    )

    judging = tmp_path / "model" / "caa" / "no_comma" / "judging"
    assert [call[0] for call in calls] == ["sh", "judge", "sh"]
    assert [call[1] for call in calls if call[0] == "sh"] == [
        (
            "pipeline/calibrate.py", "no_comma", "--model", "org/model",
            "--baseline", "caa",
            "--grid", "layer=7,19,31", "--grid", "alpha=0.5,2,4",
            "--limit", 6, "--batch", 3,
        ),
        (
            "pipeline/calibrate.py", "no_comma", "--model", "org/model",
            "--baseline", "caa", "--report",
        ),
    ]
    # generation takes one card per task, judging pairs them: that split is
    # the whole point of running the phases separately.
    assert calls[0][2].get("gpus") in {"0", "1"}   # generation takes one card
    assert [(call[1], call[2]) for call in calls if call[0] == "judge"] == [
        (judging / "search", "0,1"),
    ]


def test_calibrate_rejects_a_changed_external_grid_without_force(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "calibration_dir",
        lambda model_slug, method, task: tmp_path / model_slug / method / task,
    )
    monkeypatch.setattr(pipeline, "slug", lambda _: "model")
    out_dir = tmp_path / "model" / "caa" / "no_comma"
    out_dir.mkdir(parents=True)
    (out_dir / "search.json").write_text(json.dumps({
        "model": "org/model", "method": "caa", "limit": None,
        "generation": {"max_new_tokens": MAX_NEW_TOKENS},
        "planned_n_configs": 3,
        "grid_source": {"layer": "external", "alpha": "external"},
        "plan": {
            "budget": 9, "n_configs": 3, "axes": ["layer", "alpha"],
            "metadata": {"strength_axis": "alpha"},
            "stages": [
                {"name": "stage1", "carry": ["layer"], "configs": [
                    {"layer": 9, "alpha": 2.0}, {"layer": 18, "alpha": 2.0}]},
                {"name": "stage2", "carry": [], "configs": [
                    {"layer": 9, "alpha": 1.0}]},
            ],
        },
    }))

    with pytest.raises(SystemExit, match="layer grid differs"):
        pipeline.calibrate_phased(
            ["no_comma"], "org/model",
            grid={"layer": [9, 27], "alpha": [1, 2]},
        )


def test_test_stage_forwards_one_frozen_config_and_uses_run_marker(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "test_dir",
        lambda model_slug, method, task, config:
            tmp_path / model_slug / method / task / config,
    )
    monkeypatch.setattr(pipeline, "slug", lambda _: "model")
    calls = []
    monkeypatch.setattr(
        pipeline, "sh",
        lambda *args, **kwargs: calls.append(("sh", args, kwargs)),
    )
    monkeypatch.setattr(
        pipeline, "judge",
        lambda path, gpus=None, batch=None:
        calls.append(("judge", path, gpus, batch)),
    )

    pipeline.test_generate(
        "no_comma", "org/model", config="layer=18,alpha=2.0",
        reason="selected on public dev", gpus="0", target_batch=3,
    )
    pipeline.test_judge(
        "no_comma", "org/model", config="layer=18,alpha=2.0",
        gpus="0,1", judge_batch=4,
    )

    out_dir = tmp_path / "model" / "caa" / "no_comma" / "manual_L18_a2.0"
    assert calls == [
        ("sh", (
            "pipeline/evaluate.py", "no_comma", "--model", "org/model",
            "--baseline", "caa",
            "--config", "layer=18,alpha=2.0",
            "--reason", "selected on public dev",
            "--batch", 3,
        ), {"gpus": "0"}),
        ("judge", out_dir / "judging", "0,1", 4),
        ("sh", (
            "pipeline/evaluate.py", "no_comma", "--model", "org/model",
            "--baseline", "caa",
            "--config", "layer=18,alpha=2.0", "--report",
        ), {}),
    ]

    calls.clear()
    out_dir.mkdir(parents=True)
    (out_dir / "run.json").write_text("{}")
    pipeline.test_generate("no_comma", "org/model", config="layer=18,alpha=2.0")
    assert calls == []


def test_main_rejects_a_test_limit_before_running_any_stage(monkeypatch):
    with pytest.raises(SystemExit, match="for calibrate, not test"):
        pipeline.main(["test", "--task", "no_comma", "--limit", "5"])


def test_calibrate_rejects_pilot_when_full_dev_is_requested(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "calibration_dir",
        lambda model_slug, method, task: tmp_path / model_slug / method / task,
    )
    monkeypatch.setattr(pipeline, "slug", lambda _: "model")
    out_dir = tmp_path / "model" / "caa" / "no_comma"
    out_dir.mkdir(parents=True)
    (out_dir / "search.json").write_text(json.dumps({
        "model": "org/model", "method": "caa", "limit": 10,
        "generation": {"max_new_tokens": MAX_NEW_TOKENS},
        "planned_n_configs": 9,
        "grid_source": {"layer": "default", "alpha": "default"},
        "plan": {
            "budget": 9, "n_configs": 9, "axes": ["layer", "alpha"],
            "metadata": {"strength_axis": "alpha"},
            "stages": [
                {"name": "stage1", "carry": ["layer"], "configs": [
                    {"layer": layer, "alpha": 2.0}
                    for layer in (9, 15, 21, 27, 33)]},
                {"name": "stage2", "carry": [], "configs": [
                    {"layer": 9, "alpha": alpha}
                    for alpha in (0.5, 1.0, 3.0, 4.0)]},
            ],
        },
    }))

    # A pilot run capped at 10 items must not be silently extended into the
    # full dev search, or its summary is published as the dev result.
    with pytest.raises(SystemExit, match="--limit differs"):
        pipeline.calibrate_phased(["no_comma"], "org/model")


def test_calibrate_rejects_omitted_external_grid_on_resume(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "calibration_dir",
        lambda model_slug, method, task: tmp_path / model_slug / method / task,
    )
    monkeypatch.setattr(pipeline, "slug", lambda _: "model")
    out_dir = tmp_path / "model" / "caa" / "no_comma"
    out_dir.mkdir(parents=True)
    (out_dir / "search.json").write_text(json.dumps({
        "model": "org/model", "method": "caa", "limit": None,
        "generation": {"max_new_tokens": MAX_NEW_TOKENS},
        "planned_n_configs": 5,
        "grid_source": {"layer": "external", "alpha": "external"},
        "plan": {
            "budget": 9, "n_configs": 5, "axes": ["layer", "alpha"],
            "metadata": {"strength_axis": "alpha"},
            "stages": [
                {"name": "stage1", "carry": ["layer"], "configs": [
                    {"layer": layer, "alpha": 2.0} for layer in (7, 19, 31)]},
                {"name": "stage2", "carry": [], "configs": [
                    {"layer": 7, "alpha": alpha} for alpha in (0.5, 4.0)]},
            ],
        },
    }))

    with pytest.raises(SystemExit, match="external layer grid"):
        pipeline.calibrate_phased(["no_comma"], "org/model")


def test_gpu_groups_preserve_serial_selection_and_split_evenly():
    assert pipeline.gpu_groups(None, 1, 5) == [None]
    assert pipeline.gpu_groups("2,3", 1, 1) == ["2,3"]
    assert pipeline.gpu_groups("0,1,2,3", 2, 5) == ["0,1", "2,3"]


@pytest.mark.parametrize(("gpus", "parallel", "jobs", "message"), [
    ("0,1", 0, 2, "at least 1"),
    ("0,1", 3, 2, "exceeds"),
    (None, 2, 5, "needs --gpus"),
    ("0,1,2", 2, 5, "divided evenly"),
    ("0,0", 1, 1, "duplicate"),
])
def test_gpu_groups_reject_ambiguous_allocations(
        gpus, parallel, jobs, message):
    with pytest.raises(SystemExit, match=message):
        pipeline.gpu_groups(gpus, parallel, jobs)


def test_run_pool_propagates_regular_python_errors():
    def broken(_gpus):
        raise ValueError("bad manifest")

    broken.__name__ = "broken-task"
    with pytest.raises(SystemExit, match="broken-task: bad manifest"):
        pipeline.run_pool([broken], ["0"])


def test_main_gives_generation_single_cards_and_judging_the_pair(monkeypatch):
    """The judge does not fit on one 40GB card; generation does, so the
    test stage hands each generation job a single GPU and the judging
    phase the full pair."""
    calls = []
    monkeypatch.setattr(
        pipeline, "test_generate",
        lambda *args: calls.append(("generate", args)),
    )
    monkeypatch.setattr(
        pipeline, "test_judge", lambda *args: calls.append(("judge", args))
    )
    monkeypatch.setattr(
        pipeline, "status", lambda *args: calls.append(("status", args))
    )

    pipeline.main([
        "test", "--task", "no_comma", "--config", "layer=18,alpha=2.0",
        "--gpus", "2,3", "--target-batch", "3", "--judge-batch", "4",
    ])

    generate_call = calls[0][1]
    assert calls[0][0] == "generate"
    assert generate_call[0] == "no_comma"
    assert generate_call[2:6] == ("caa", "layer=18,alpha=2.0", "", False)
    assert generate_call[6] in ("2", "3")
    assert generate_call[7] == 3
    judge_call = calls[1][1]
    assert calls[1][0] == "judge"
    assert judge_call[0] == "no_comma"
    assert judge_call[2:4] == ("caa", "layer=18,alpha=2.0")
    assert judge_call[4] == "2,3"
    assert judge_call[5] == 4
    assert calls[2][0] == "status"
    assert calls[2][1][1:] == ("no_comma", "caa", "layer=18,alpha=2.0")


@pytest.mark.parametrize(("argv", "message"), [
    (["calibrate", "--task", "no_comma", "--config", "layer=18,alpha=2.0"],
     "not calibrate"),
    (["test", "--task", "no_comma", "--grid", "layer=9,18"], "not test"),
    (["calibrate", "--parallel", "2", "--gpus", "0,1"], "divides --gpus"),
    (["test", "--task", "no_comma", "--split", "dev"], "--split selects"),
    (["all", "--config", "layer=18,alpha=2.0"], "dev-chosen"),
    (["status", "--gpus", "0"], "status accepts only"),
])
def test_main_rejects_stage_specific_arguments(argv, message):
    with pytest.raises(SystemExit, match=message):
        pipeline.main(argv)


def test_status_reads_a_manual_test_directory(tmp_path, monkeypatch, capsys):
    baseline_root = tmp_path / "baselines"
    calibration_root = tmp_path / "calibration"
    test_root = tmp_path / "test"
    monkeypatch.setattr(
        pipeline, "artifact_path",
        lambda model, method, task: baseline_root / model / method / f"{task}.pt",
    )
    monkeypatch.setattr(
        pipeline, "calibration_dir",
        lambda model_slug, method, task: calibration_root / model_slug / method / task,
    )
    monkeypatch.setattr(
        pipeline, "test_dir",
        lambda model_slug, method, task, config:
            test_root / model_slug / method / task / config,
    )
    monkeypatch.setattr(pipeline, "slug", lambda _: "model")
    vector = baseline_root / "model" / "caa" / "no_comma.pt"
    vector.parent.mkdir(parents=True)
    vector.write_bytes(b"vector")

    out_dir = test_root / "model" / "caa" / "no_comma" / "manual_L18_a2.0"
    judging_dir = out_dir / "judging"
    judging_dir.mkdir(parents=True)
    (out_dir / "run.json").write_text("{}")
    (out_dir / "summary.json").write_text(json.dumps({"accuracy": 0.75}))
    (judging_dir / "judge_in_0.jsonl").write_text("{}\n")
    (judging_dir / "judge_out_0.jsonl").write_text("{}\n")

    pipeline.status("org/model", "no_comma", "caa", "layer=18,alpha=2.0")

    output = capsys.readouterr().out
    assert "test run manual_L18_a2.0" in output
    assert "no_comma" in output
    assert "75.0%" in output
    row = next(
        line for line in output.splitlines() if line.startswith("no_comma")
    )
    assert row.split()[1] == "ok"


def test_status_manual_config_requires_one_task():
    with pytest.raises(SystemExit, match="needs one --task"):
        pipeline.status("org/model", config="layer=18,alpha=2.0")


def test_a_search_generated_under_a_different_budget_is_not_resumed(
        tmp_path, monkeypatch):
    """The model, the baseline and the grid all still match, so nothing else
    in the guard notices — but the outputs were produced under a different
    token budget, which makes them a different measurement."""
    monkeypatch.setattr(
        pipeline, "calibration_dir",
        lambda model_slug, method, task: tmp_path / model_slug / method / task,
    )
    monkeypatch.setattr(pipeline, "slug", lambda _: "model")
    out_dir = tmp_path / "model" / "caa" / "no_comma"
    out_dir.mkdir(parents=True)
    (out_dir / "search.json").write_text(json.dumps({
        "model": "org/model", "method": "caa", "limit": None,
        "generation": {"max_new_tokens": 1024},
        "plan": {
            "axes": ["layer", "alpha"],
            "metadata": {"strength_axis": "alpha"},
            "stages": [{"name": "stage1", "carry": ["layer"], "configs": [
                {"layer": 9, "alpha": 2.0}]}],
        },
    }))

    with pytest.raises(SystemExit, match="generated at 1024 new tokens"):
        pipeline.calibrate_phased(["no_comma"], "org/model")


def test_a_search_that_recorded_no_budget_is_not_resumed(tmp_path, monkeypatch):
    """An older run wrote its generation facts in a different shape. It is
    not readable as the same measurement, so it is not resumable either."""
    monkeypatch.setattr(
        pipeline, "calibration_dir",
        lambda model_slug, method, task: tmp_path / model_slug / method / task,
    )
    monkeypatch.setattr(pipeline, "slug", lambda _: "model")
    out_dir = tmp_path / "model" / "caa" / "no_comma"
    out_dir.mkdir(parents=True)
    (out_dir / "search.json").write_text(json.dumps({
        "model": "org/model", "method": "caa", "limit": None,
        "generation": [{"config": "L9_a2.0"}],          # the old list shape
        "plan": {"axes": ["layer"], "metadata": {}, "stages": []},
    }))

    with pytest.raises(SystemExit, match="generated at None new tokens"):
        pipeline.calibrate_phased(["no_comma"], "org/model")


def test_fake_all_runs_calibration_judging_and_frozen_test(
        tmp_path, monkeypatch):
    calibration_root = tmp_path / "calibration"
    test_root = tmp_path / "test"
    baseline_root = tmp_path / "baselines"
    monkeypatch.setattr(
        pipeline, "calibration_dir",
        lambda model_slug, method, task: calibration_root / model_slug / method / task,
    )
    monkeypatch.setattr(
        pipeline, "test_dir",
        lambda model_slug, method, task, config:
            test_root / model_slug / method / task / config,
    )
    monkeypatch.setattr(
        pipeline, "artifact_path",
        lambda model, method, task: baseline_root / model / method / f"{task}.pt",
    )
    monkeypatch.setattr(pipeline, "slug", lambda _: "model")
    calls = []

    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def write_judge_input(path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "judge_in_0.jsonl").write_text("{}\n")

    def fake_sh(script, *args, gpus=None):
        calls.append((script, args, gpus))
        if script == "pipeline/calibrate.py":
            task = args[0]
            method = args[args.index("--baseline") + 1]
            out_dir = calibration_root / "model" / method / task
            if "--report" in args:
                write_json(out_dir / "summary.json", {
                    "axes": ["layer", "alpha"],
                    "chosen": {"layer": 18, "alpha": 2.0},
                })
            else:
                write_json(out_dir / "search.json", {
                    "model": "org/model", "method": method, "limit": None,
                    "planned_n_configs": 4,
                    "generation": {"max_new_tokens": MAX_NEW_TOKENS},
                    "grid_source": {"layer": "default", "alpha": "default"},
                    "plan": {
                        "budget": 9, "n_configs": 4, "axes": ["layer", "alpha"],
                        "metadata": {"strength_axis": "alpha"},
                        "stages": [
                            {"name": "search", "carry": [], "configs": [
                                {"layer": layer, "alpha": alpha}
                                for layer in (9, 18) for alpha in (1.0, 2.0)]},
                        ],
                    },
                })
                write_judge_input(out_dir / "judging" / "search")
        elif script == "kernel/judge_local.py":
            judging_dir = Path(args[0])
            for input_path in judging_dir.glob("judge_in_*.jsonl"):
                output_name = input_path.name.replace("judge_in_", "judge_out_")
                (judging_dir / output_name).write_text("{}\n")
        elif script == "pipeline/produce.py":
            method = args[args.index("--baseline") + 1]
            path = baseline_root / "model" / method / f"{args[0]}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"artifact")
        elif script == "pipeline/evaluate.py":
            task = args[0]
            method = args[args.index("--baseline") + 1]
            out_dir = test_root / "model" / method / task / "calibrated"
            if "--report" in args:
                write_json(out_dir / "summary.json", {"accuracy": 0.8})
            else:
                write_json(out_dir / "run.json", {})
                write_judge_input(out_dir / "judging")

    monkeypatch.setattr(pipeline, "sh", fake_sh)

    pipeline.main([
        "all", "--task", "no_comma", "--model", "org/model",
        "--gpus", "0,1", "--target-batch", "2", "--judge-batch", "3",
    ])

    assert [script for script, _, _ in calls] == [
        "pipeline/produce.py",
        "pipeline/calibrate.py", "kernel/judge_local.py",
        "pipeline/calibrate.py", "pipeline/evaluate.py",
        "kernel/judge_local.py", "pipeline/evaluate.py",
    ]
    target_calls = [args for script, args, _ in calls
                    if script in {"pipeline/calibrate.py", "pipeline/evaluate.py"}
                    and "--report" not in args]
    judge_calls = [args for script, args, _ in calls
                   if script == "kernel/judge_local.py"]
    # the filter used bare file names while fake_sh records "pipeline/…",
    # so this list was empty and the assertion below always passed
    # the single dev search and the frozen test run; the two --report passes
    # generate nothing and are filtered out above
    assert len(target_calls) == 2
    assert all(("--batch", 2) == args[-2:] for args in target_calls)
    assert all(("--batch", 3) == args[-2:] for args in judge_calls)
    # Generation fits one card; only the judge needs a pair. Handing every
    # phase two cards is what left one idle through the longest phase.
    generation = [gpus for script, args, gpus in calls
                  if script == "pipeline/calibrate.py" and "--report" not in args]
    judging = [gpus for script, args, gpus in calls
               if script == "kernel/judge_local.py"]
    assert generation and all(gpus in {"0", "1"} for gpus in generation), generation
    assert judging and all(gpus == "0,1" for gpus in judging), judging
    assert (calibration_root / "model" / "caa" / "no_comma" / "summary.json").exists()
    assert (test_root / "model" / "caa" / "no_comma" / "calibrated"
            / "summary.json").exists()


def test_cli_grids_reach_the_baseline_as_search_plan_overrides():
    """Grid validation lives with the baseline that owns those axes, and a
    grid names the axis it varies, so a baseline with different axes gets
    the grid rather than silently ignoring it."""
    parser = calibrate.build_parser()
    args = parser.parse_args([
        "no_comma", "--grid", "layer=7,19", "--grid", "alpha=0.5,3.0",
    ])
    assert calibrate._search_overrides(args) == {
        "layer": [7, 19], "alpha": [0.5, 3.0]
    }


def test_a_grid_can_name_an_axis_no_residual_stream_baseline_has():
    """ITI searches heads_k, not layer. The old per-axis flags could not say
    so, and the grid was dropped without a word."""
    args = calibrate.build_parser().parse_args(
        ["no_comma", "--grid", "heads_k=16,48,128"]
    )
    assert calibrate._search_overrides(args) == {"heads_k": [16, 48, 128]}


def test_a_malformed_grid_is_refused_rather_than_half_read():
    for bad in ("layer", "=1,2", "layer="):
        with pytest.raises(ValueError, match="axis=v1,v2"):
            calibrate.parse_grid([bad])
    with pytest.raises(ValueError, match="twice"):
        calibrate.parse_grid(["layer=1", "layer=2"])


def test_cli_without_grids_leaves_the_method_defaults_alone():
    args = calibrate.build_parser().parse_args(["no_comma"])
    assert calibrate._search_overrides(args) == {}


def test_cli_parses_external_grids_exactly(monkeypatch):
    captured = []
    monkeypatch.setattr(calibrate, "generate", captured.append)

    calibrate.main([
        "no_comma", "--grid", "layer=7,19,31", "--grid", "alpha=0.25,1.5,4",
        "--limit", "6", "--batch", "3",
    ])

    args = captured[0]
    assert calibrate._search_overrides(args) == {
        "layer": [7, 19, 31], "alpha": [0.25, 1.5, 4.0]
    }
    assert args.limit == 6
    assert args.batch == 3


@contextmanager
def _noop_intervention():
    yield


def _fake_method():
    return SimpleNamespace(
        KEY="fake",
        METHOD_VERSION=1,
        config_name=lambda config: f"L{config['layer']}_a{config['alpha']}",
        load_artifact=lambda model_slug, task: {"fake": True},
        make_intervention=lambda model, artifact, config: _noop_intervention,
        dose_metadata=lambda artifact, config: {
            "method": "fake",
            "raw_strength": float(config["alpha"]),
            "effective_delta_norm": float(config["alpha"]),
            "site": {"component": "resid_post", "layer": config.get("layer")},
        },
        search_plan=lambda context, overrides=None, budget=None: joint_plan(
            "layer", overrides["layer"], "alpha", overrides["alpha"],
            budget or 9,
        ),
    )


def _args():
    return SimpleNamespace(
        model="org/model", task="no_comma", baseline="fake", batch=2, limit=None,
        grid=["layer=9,18", "alpha=1.0,2.0,3.0"],
        budget=None, force=False,
    )


def _write_stage_verdicts(judging_dir: Path) -> None:
    for input_path in judging_dir.glob("judge_in_*.jsonl"):
        output_path = judging_dir / input_path.name.replace("judge_in_", "judge_out_")
        with input_path.open(encoding="utf-8") as source, output_path.open(
            "w", encoding="utf-8"
        ) as destination:
            for line in source:
                uid = json.loads(line)["uid"]
                destination.write(json.dumps({
                    "uid": uid,
                    "coherent": "|L18_" in uid,
                    "answers_question": True,
                }) + "\n")


def _joint_method():
    """A baseline whose plan is one stage, as ITI's (heads_k, alpha) is."""
    return SimpleNamespace(
        KEY="joint",
        METHOD_VERSION=1,
        config_name=lambda config: f"K{config['heads_k']}_a{config['alpha']}",
        load_artifact=lambda model_slug, task: {"fake": True},
        make_intervention=lambda model, artifact, config: _noop_intervention,
        dose_metadata=lambda artifact, config: {
            "method": "fake",
            "raw_strength": float(config["alpha"]),
            "effective_delta_norm": float(config["alpha"]),
            "site": {"component": "resid_post", "layer": config.get("layer")},
        },
        search_plan=lambda context, overrides=None, budget=None: SearchPlan(
            stages=[SearchStage(
                name="search",
                configs=[{"heads_k": k, "alpha": a}
                         for k in (8, 16) for a in (5.0, 15.0)],
                carry=(),
            )],
            budget=budget or 9,
            metadata={"shape": "joint", "structural_axis": "heads_k",
                      "strength_axis": "alpha"},
        ),
    )


def test_a_joint_plan_generates_and_judges_in_one_pass(tmp_path, monkeypatch):
    """A single-stage plan must reach a report, not run off the end of the
    stage list looking for one that was never declared."""
    class FakeGenerator:
        n_layers = 36
        model = object()

        def __init__(self, config):
            self.cfg = config

        def generate(self, prompts, intervention=None, progress=False):
            return [
                {"output": "Blue light scatters more strongly",
                 "truncated": False, "n_new_tokens": 5}
                for _ in prompts
            ]

    items = [
        {"item_id": f"dev_{index}", "prompt_id": f"prompt_{index}",
         "base_prompt": f"Question {index}?"}
        for index in range(2)
    ]
    method = _joint_method()
    monkeypatch.setattr(
        calibrate, "calibration_dir",
        lambda model_slug, key, task: tmp_path / model_slug / key / task,
    )
    monkeypatch.setattr(calibrate.baselines, "get", lambda key: method)
    monkeypatch.setattr(calibrate, "slug", lambda _: "model")
    monkeypatch.setattr(calibrate, "calibration_items", lambda task: items)
    monkeypatch.setattr(
        calibrate, "load_reference_summary", lambda *a, **k: None
    )
    monkeypatch.setattr(generate, "Generator", FakeGenerator)

    args = SimpleNamespace(
        model="org/model", task="no_comma", baseline="joint", batch=2,
        limit=None, grid=None,
        budget=None, force=False,
    )
    calibrate.generate(args)

    out_dir = tmp_path / "model" / "joint" / "no_comma"
    search = json.loads((out_dir / "search.json").read_text())
    assert search["axes"] == ["heads_k", "alpha"]
    assert search["config_names"] == [
        "K8_a5.0", "K8_a15.0", "K16_a5.0", "K16_a15.0"
    ]
    assert [stage["name"] for stage in search["plan"]["stages"]] == ["search"]

    _write_stage_verdicts(out_dir / "judging" / "search")

    calibrate.report(args)
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["chosen"]["config"] in search["config_names"]
    assert set(summary["axes"]) == {"heads_k", "alpha"}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    """Point every path the report reads at a temporary tree."""
    monkeypatch.setattr(
        report, "test_dir",
        lambda model_slug, baseline, task, config:
            tmp_path / "test" / model_slug / baseline / task / config,
    )
    monkeypatch.setattr(
        report, "calibration_dir",
        lambda model_slug, baseline, task:
            tmp_path / "calibration" / model_slug / baseline / task,
    )
    monkeypatch.setattr(
        report, "load_reference_summary",
        lambda model_slug, split, task, condition, required=True: None,
    )
    monkeypatch.setattr(report, "slug", lambda _: "model")
    monkeypatch.setattr(report, "TASKS", TASKS)
    return tmp_path


def _test_summary(tmp_path, baseline, task, **overrides):
    value = {
        "accuracy": 0.5, "ci95": [0.45, 0.55], "n_scored": 609,
        "config_values": {"layer": 21, "alpha": 1.0},
        "code_version": "abc123", "intervention_positions": "after_prompt",
    }
    value.update(overrides)
    _write(
        tmp_path / "test" / "model" / baseline / task / "calibrated" / "summary.json",
        value,
    )


def test_a_missing_result_is_a_gap_not_a_zero(rooted):
    _test_summary(rooted, "caa", "lowercase")

    collected = report.collect("org/model", "test", ["caa"])

    assert collected["table"]["caa"]["lowercase"]["accuracy"] == 0.5
    assert collected["table"]["caa"]["no_comma"] is None
    assert ("caa", "no_comma") in collected["gaps"]

    markdown = report.render_markdown(collected)
    assert "4 cells not run" in markdown
    assert report.GAP in markdown


def test_a_row_is_averaged_only_when_every_task_ran(rooted):
    for task in TASKS:
        _test_summary(rooted, "caa", task, accuracy=0.4)
    _test_summary(rooted, "probe", "lowercase", accuracy=0.9)

    markdown = report.render_markdown(
        report.collect("org/model", "test", ["caa", "probe"])
    )

    caa_row = next(l for l in markdown.splitlines() if l.startswith("| caa |"))
    probe_row = next(l for l in markdown.splitlines() if l.startswith("| probe |"))
    assert caa_row.rstrip().endswith("| 40.0% |")
    assert probe_row.rstrip().endswith(f"| {report.GAP} |")


def test_cells_from_different_code_versions_are_called_out(rooted):
    _test_summary(rooted, "caa", "lowercase", code_version="abc123")
    _test_summary(rooted, "caa", "no_comma", code_version="def456-dirty")

    markdown = report.render_markdown(report.collect("org/model", "test", ["caa"]))

    assert "cells come from 2 code versions" in markdown
    assert "def456-dirty" in markdown


def test_cells_from_different_position_rules_are_called_out(rooted):
    _test_summary(rooted, "caa", "lowercase", intervention_positions="after_prompt")
    _test_summary(rooted, "caa", "no_comma", intervention_positions="all")

    markdown = report.render_markdown(report.collect("org/model", "test", ["caa"]))

    assert "2 intervention-position rules" in markdown


def test_one_consistent_provenance_is_stated_plainly(rooted):
    for task in TASKS:
        _test_summary(rooted, "caa", task)

    markdown = report.render_markdown(report.collect("org/model", "test", ["caa"]))

    assert "code version: abc123" in markdown
    assert "intervention positions: after_prompt" in markdown
    assert "every cell present" in markdown


def test_references_are_labelled_as_rule_pass_not_accuracy(rooted, monkeypatch):
    monkeypatch.setattr(
        report, "load_reference_summary",
        lambda model_slug, split, task, condition, required=True: (
            {"rule_pass_rate": 0.99 if condition == "prompted" else 0.0}
        ),
    )
    for task in TASKS:
        _test_summary(rooted, "caa", task)

    markdown = report.render_markdown(report.collect("org/model", "test", ["caa"]))

    assert "Rule-pass rates, not benchmark accuracy" in markdown
    assert "| prompted |" in markdown


def test_the_dev_table_reads_the_chosen_configuration(rooted):
    _write(
        rooted / "calibration" / "model" / "iti" / "lowercase" / "summary.json",
        {
            "axes": ["heads_k", "alpha"],
            "chosen": {"heads_k": 48, "alpha": 15.0, "accuracy": 0.3, "n": 405},
            "intervention_positions": "after_prompt",
        },
    )

    collected = report.collect("org/model", "dev", ["iti"])
    cell = collected["table"]["iti"]["lowercase"]

    assert cell["accuracy"] == 0.3
    assert cell["config"] == {"heads_k": 48, "alpha": 15.0}
    assert "heads_k=48 alpha=15.0" in report.render_markdown(collected)


def test_written_files_carry_the_table_and_the_raw_cells(rooted):
    for task in TASKS:
        _test_summary(rooted, "caa", task)
    collected = report.collect("org/model", "test", ["caa"])

    written = report.write(collected, root=rooted / "results")

    names = sorted(path.name for path in written)
    assert names == [
        "test_leaderboard.csv", "test_leaderboard.json", "test_leaderboard.md"
    ]
    rows = (rooted / "results" / "model" / "test_leaderboard.csv").read_text()
    assert "caa,lowercase,0.5,0.45,0.55,609,layer=21;alpha=1.0,abc123" in rows


def _global_reads(table: symtable.SymbolTable) -> set[str]:
    """Names read as globals anywhere below this table, never assigned there."""
    names = set()
    for child in table.get_children():
        for symbol in child.get_symbols():
            if symbol.is_global() and not symbol.is_assigned():
                names.add(symbol.get_name())
        names |= _global_reads(child)
    return names


@pytest.mark.parametrize("name", MODULES)
def test_module_resolves_every_global_name_it_reads(name):
    path = SRC / (name.replace(".", "/") + ".py")
    source = path.read_text(encoding="utf-8")
    reads = _global_reads(symtable.symtable(source, str(path), "exec"))
    module = importlib.import_module(name)

    missing = sorted(
        read for read in reads
        if not hasattr(module, read) and not hasattr(builtins, read)
    )
    assert missing == [], f"{name} reads undefined name(s): {missing}"


def test_the_check_catches_a_dropped_import(tmp_path):
    """The bug this file exists for: a helper called inside a function whose
    import is gone. Without this, the test above could silently stop working
    and nobody would know."""
    source = "def main():\n    return gate_report_path('x')\n"
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")

    reads = _global_reads(symtable.symtable(source, str(path), "exec"))

    assert "gate_report_path" in reads
