"""Everything that turns a generation into a verdict: the judge prompt, its
blinding, the parser that survives a malformed reply, and the scoring that
counts a rule and a judgement together.
"""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kernel.judge import (  # noqa: E402
    JUDGE_INSTRUCTIONS, load_verdicts, merge, prepare,
    write_scored_records,
)
from kernel.judge import load_verdicts, merge, prepare  # noqa: E402
from kernel.judge_local import blind_label, build_prompt, parse, run  # noqa: E402
import torch
from kernel import judge_local  # noqa: E402
UID = "no_comma|dev|L18_a2.0|no_comma_dev_x1"
LABEL = blind_label(UID)
from types import SimpleNamespace
from pipeline import calibrate  # noqa: E402
from pipeline import evaluate  # noqa: E402
from kernel.scoring import (  # noqa: E402
    build_records, prepare_rule_passers, summarize_merged, summarize_rules,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_prepare_keeps_full_questions_and_replaces_stale_artifacts(tmp_path):
    long_question = "why " * 90
    records = [
        {"uid": "a", "question": long_question, "output": "A short answer."},
        {"uid": "b", "question": "Second question", "output": "Two words"},
        {"uid": "c", "question": "Third question", "output": "Final answer"},
    ]
    (tmp_path / "judge_in_9.jsonl").write_text("stale", encoding="utf-8")
    (tmp_path / "judge_out_9.jsonl").write_text("stale", encoding="utf-8")
    (tmp_path / "judge_unparsed_9.jsonl").write_text("stale", encoding="utf-8")
    (tmp_path / "judge_run.json").write_text("{}", encoding="utf-8")

    paths = prepare(records, tmp_path, n_batches=2, seed=7)

    assert [path.name for path in paths] == ["judge_in_0.jsonl", "judge_in_1.jsonl"]
    assert not (tmp_path / "judge_in_9.jsonl").exists()
    assert not (tmp_path / "judge_out_9.jsonl").exists()
    assert not (tmp_path / "judge_unparsed_9.jsonl").exists()
    assert not (tmp_path / "judge_run.json").exists()
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    by_uid = {row["uid"]: row for row in rows}
    assert set(by_uid) == {"a", "b", "c"}
    assert by_uid["a"]["question"] == long_question
    assert by_uid["a"]["answer"] == "A short answer."
    assert by_uid["a"]["n_words"] == 3
    assert (tmp_path / "judge_instructions.txt").read_text(encoding="utf-8") == \
        JUDGE_INSTRUCTIONS


def test_prepare_is_deterministic_for_a_seed(tmp_path):
    records = [
        {"uid": str(i), "question": f"q{i}", "output": f"a{i}"}
        for i in range(8)
    ]
    first = prepare(records, tmp_path / "first", n_batches=3, seed=11)
    second = prepare(records, tmp_path / "second", n_batches=3, seed=11)
    assert [path.read_text() for path in first] == [path.read_text() for path in second]


def test_load_verdicts_preserves_real_json_booleans(tmp_path):
    _write_jsonl(tmp_path / "judge_out_0.jsonl", [
        {"uid": "a", "coherent": False, "answers_question": True,
         "note": "incomplete but real"},
        {"uid": "b", "coherent": True, "answers_question": False,
         "raw_response": "model response"},
    ])
    verdicts = load_verdicts(tmp_path)
    assert verdicts["a"] == {
        "coherent": False,
        "answers_question": True,
        "note": "incomplete but real",
    }
    assert verdicts["b"]["answers_question"] is False
    assert verdicts["b"]["note"] == "model response"


@pytest.mark.parametrize("field,value", [
    ("coherent", "false"),
    ("answers_question", 1),
    ("answers_question", None),
])
def test_load_verdicts_rejects_non_boolean_decisions(tmp_path, field, value):
    row = {"uid": "a", "coherent": True, "answers_question": True}
    row[field] = value
    _write_jsonl(tmp_path / "judge_out_0.jsonl", [row])
    with pytest.raises(ValueError, match=f"field {field!r} must be a JSON boolean"):
        load_verdicts(tmp_path)


def test_load_verdicts_rejects_duplicate_uids(tmp_path):
    row = {"uid": "a", "coherent": True, "answers_question": True}
    _write_jsonl(tmp_path / "judge_out_0.jsonl", [row])
    _write_jsonl(tmp_path / "judge_out_1.jsonl", [row])
    with pytest.raises(ValueError, match="Duplicate judge verdict uid 'a'"):
        load_verdicts(tmp_path)


GOOD = "a clearly written answer with no repeated filler at all"


def test_merge_requires_rule_coherence_and_an_answer():
    records = [
        {"uid": "a", "truncated": False, "rule_pass": True, "output": GOOD},
        {"uid": "b", "truncated": False, "rule_pass": True, "output": GOOD},
        {"uid": "c", "truncated": False, "rule_pass": True, "output": GOOD},
        {"uid": "d", "truncated": False, "rule_pass": False, "output": GOOD},
        {"uid": "e", "rule_pass": False, "output": GOOD, "truncated": True},
    ]
    verdicts = {
        "a": {"coherent": True, "answers_question": True},
        "b": {"coherent": False, "answers_question": True},
        "c": {"coherent": True, "answers_question": False},
    }

    summary = merge(records, verdicts)

    assert [row["success"] for row in records] == [True, False, False, False, False]
    assert summary["n_scored"] == 5
    assert summary["n_rule_pass"] == 3
    assert summary["n_rule_fail"] == 2       # one rule failure, one truncation
    assert summary["n_judged"] == 3
    assert summary["n_unjudged"] == 0
    assert summary["accuracy"] == 0.2
    assert summary["rule_only_rate"] == 0.6
    assert summary["judge_coverage"] == 1.0
    assert summary["n_degenerate_blocked"] == 0


def test_merge_denies_success_to_a_judge_approved_repetition_collapse():
    collapse = "basics " * 40
    records = [
        {"uid": "a", "truncated": False, "rule_pass": True, "output": collapse},
        {"uid": "b", "truncated": False, "rule_pass": True, "output": GOOD},
    ]
    verdicts = {
        "a": {"coherent": True, "answers_question": True},
        "b": {"coherent": True, "answers_question": True},
    }

    summary = merge(records, verdicts)

    assert [row["success"] for row in records] == [False, True]
    assert [row["degenerate"] for row in records] == [True, False]
    assert summary["n_degenerate_blocked"] == 1
    assert summary["n_rule_pass"] == 2
    assert summary["accuracy"] == 0.5


def test_merge_does_not_publish_partial_accuracy():
    records = [
        {"uid": "a", "truncated": False, "rule_pass": True, "output": GOOD},
        {"uid": "b", "truncated": False, "rule_pass": True, "output": GOOD},
        {"uid": "c", "truncated": False, "rule_pass": False, "output": GOOD},
    ]
    summary = merge(records, {
        "a": {"coherent": True, "answers_question": True},
    })
    assert [row["success"] for row in records] == [True, None, False]
    assert summary["n_unjudged"] == 1
    assert summary["judge_coverage"] == 0.5
    assert summary["accuracy"] is None


def test_write_scored_records_saves_merged_rows(tmp_path):
    records = [
        {"uid": "a", "truncated": False, "rule_pass": True, "success": True},
        {"uid": "b", "truncated": False, "rule_pass": False, "success": False},
    ]
    path = write_scored_records(records, tmp_path / "scored_outputs.jsonl")
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert saved == records


def test_prepare_load_and_merge_end_to_end(tmp_path):
    records = [
        {"uid": "good", "question": "Q1", "output": "A1", "rule_pass": True},
        {"uid": "nonanswer", "question": "Q2", "output": "A2",
         "rule_pass": True},
        {"uid": "rulefail", "question": "Q3", "output": "A3",
         "rule_pass": False},
    ]
    in_paths = prepare(records, tmp_path, n_batches=2, seed=4)

    answers_question = {"good": True, "nonanswer": False, "rulefail": True}
    for in_path in in_paths:
        out_path = in_path.with_name(in_path.name.replace("judge_in_", "judge_out_"))
        items = [json.loads(line) for line in in_path.read_text().splitlines()]
        with out_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps({
                    "uid": item["uid"],
                    "coherent": True,
                    "answers_question": answers_question[item["uid"]],
                }) + "\n")

    summary = merge(records, load_verdicts(tmp_path))
    assert [record["success"] for record in records] == [True, False, False]
    assert summary["accuracy"] == 1 / 3
    assert summary["n_unjudged"] == 0


def test_the_prompt_shows_the_judge_nothing_about_the_steering():
    """Our uid names the configuration that produced the answer, and the
    judge's verdict is what selects that configuration. It gets a hash."""
    prompt = build_prompt({
        "uid": UID, "question": "Full question", "answer": "Full answer",
        "n_words": 2, "layer": 18, "alpha": 2.0,
    })

    assert "Full question" in prompt
    assert "Full answer" in prompt
    assert LABEL in prompt
    for leak in (UID, "L18_a2.0", "18", "2.0", "layer", "alpha", "no_comma"):
        assert leak not in prompt


def test_parse_accepts_an_embedded_strict_json_verdict():
    text = (
        "```json\n"
        f'{{"id": "{LABEL}", "coherent": false, '
        '"answers_question": true, "note": "partial answer"}'
        "\n```"
    )
    verdict = parse(text, UID)
    # the real uid is attached here, having never passed through the model
    assert verdict["uid"] == UID
    assert verdict["coherent"] is False
    assert verdict["answers_question"] is True
    assert verdict["note"] == "partial answer"
    assert verdict["raw_response"] == text


def test_parse_skips_an_example_before_the_matching_label():
    text = (
        '{"id": "not-this-one", "coherent": true, "answers_question": true}\n'
        f'{{"id": "{LABEL}", "coherent": true, "answers_question": false}}'
    )
    assert parse(text, UID)["answers_question"] is False


@pytest.mark.parametrize("text", [
    f'{{"id": "{LABEL}", "coherent": "false", "answers_question": true}}',
    f'{{"id": "{LABEL}", "incoherent": true, "answers_question": true}}',
    f'{{"id": "{LABEL}", "coherent": true}}',
    'coherent=true answers_question=true',
])
def test_parse_rejects_ambiguous_or_invalid_responses(text):
    assert parse(text, UID) is None


def test_a_mistyped_label_does_not_discard_a_decided_verdict():
    """The judge copies a twelve-character hash and sometimes inserts a
    character while deciding the item correctly. One prompt carries one item
    and responses pair to items by position, so the echo is a check rather
    than the join; rejecting on it threw away a sound decision that greedy
    decoding reproduces identically, which is to say permanently."""
    text = (f'{{"id": "{LABEL}x", "coherent": false, '
            '"answers_question": false, "note": "Repetitive."}')

    verdict = parse(text, UID)

    assert verdict["coherent"] is False
    assert verdict["answers_question"] is False
    assert verdict["uid"] == UID
    assert verdict["id_mismatch"] is True


def test_two_mismatched_verdicts_stay_ambiguous():
    """One decision can be attributed to the one item; two cannot."""
    text = ('{"id": "aaa", "coherent": true, "answers_question": true}\n'
            '{"id": "bbb", "coherent": false, "answers_question": false}')
    assert parse(text, UID) is None


def test_parse_salvages_a_verdict_truncated_inside_the_note():
    text = (f'```json\n{{"id": "{LABEL}", "coherent": false, '
            '"answers_question": true, "note": "starts well but devol')
    verdict = parse(text, UID)
    assert verdict["coherent"] is False
    assert verdict["answers_question"] is True
    assert verdict["note"] == ""
    assert verdict["raw_response"] == text.strip()


def test_parse_salvage_reads_fields_only_after_the_matching_label():
    text = (
        '{"id": "not-this-one", "coherent": true, "answers_question": true}\n'
        f'{{"id": "{LABEL}", "coherent": true, "answers_question": false, '
        '"note": "cut mid-sente')
    assert parse(text, UID)["answers_question"] is False


def test_parse_salvage_still_requires_both_decisions():
    assert parse(f'{{"id": "{LABEL}", "coherent": true, "note": "cut', UID) is None


def test_the_label_is_stable_and_distinct_per_item():
    assert blind_label(UID) == blind_label(UID)
    assert blind_label(UID) != blind_label(UID.replace("x1", "x2"))


@pytest.mark.parametrize("paths,batch,message", [
    ([], 8, "no judge_in"),
    ([Path("judge_in_0.jsonl")], 0, "batch must be positive"),
])
def test_run_rejects_bad_inputs_before_loading_a_model(paths, batch, message):
    with pytest.raises(ValueError, match=message):
        run(paths, batch=batch)


class FakeConfig:
    _commit_hash = "revision-1"


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, texts, **kwargs):
        codes = []
        for text in texts:
            codes.append(1 if judge_local.blind_label("good") in text else 2)
        return FakeBatch(input_ids=torch.tensor([[10, code] for code in codes]))

    def decode(self, row, **kwargs):
        if int(row[-1]) == 1:
            return json.dumps({
                "id": judge_local.blind_label("good"), "coherent": True,
                "answers_question": True,
            })
        return "not valid JSON"


class FakeModel:
    loads = 0
    generate_calls = 0
    device = torch.device("cpu")
    hf_device_map = {"model": "cpu"}

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        type(self).generate_calls += 1
        return torch.cat([input_ids, input_ids[:, -1:]], dim=1)


def _write_input(path, uid):
    path.write_text(json.dumps({
        "uid": uid, "question": f"question for {uid}",
        "answer": f"answer for {uid}", "n_words": 3,
    }) + "\n")


def test_fake_model_writes_provenance_and_resumes_missing_shards(
        tmp_path, monkeypatch):
    FakeModel.loads = 0
    FakeModel.generate_calls = 0
    monkeypatch.setattr(
        judge_local.AutoConfig, "from_pretrained", lambda *args, **kwargs: FakeConfig()
    )
    monkeypatch.setattr(
        judge_local.AutoTokenizer, "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    def load_model(*args, **kwargs):
        FakeModel.loads += 1
        return FakeModel()

    monkeypatch.setattr(
        judge_local.AutoModelForCausalLM, "from_pretrained", load_model
    )
    in_paths = [tmp_path / "judge_in_0.jsonl", tmp_path / "judge_in_1.jsonl"]
    _write_input(in_paths[0], "good")
    _write_input(in_paths[1], "bad")

    judge_local.run(in_paths, model_name="fake/judge", batch=1)

    manifest = json.loads((tmp_path / "judge_run.json").read_text())
    assert manifest["model"] == "fake/judge"
    assert manifest["model_revision"] == "revision-1"
    assert manifest["input_shards"] == ["judge_in_0.jsonl", "judge_in_1.jsonl"]
    assert len(manifest["prompt_sha256"]) == 64
    good_output = (tmp_path / "judge_out_0.jsonl").read_text()
    assert json.loads(good_output)["uid"] == "good"
    assert (tmp_path / "judge_out_1.jsonl").read_text() == ""
    unparsed = json.loads((tmp_path / "judge_unparsed_1.jsonl").read_text())
    assert unparsed == {"uid": "bad", "raw_response": "not valid JSON"}
    assert FakeModel.generate_calls == 2

    (tmp_path / "judge_out_1.jsonl").unlink()
    (tmp_path / "judge_unparsed_1.jsonl").unlink()
    judge_local.run(in_paths, model_name="fake/judge", batch=1)
    assert (tmp_path / "judge_out_0.jsonl").read_text() == good_output
    assert (tmp_path / "judge_out_1.jsonl").exists()
    assert FakeModel.generate_calls == 3

    monkeypatch.setattr(
        judge_local.AutoConfig, "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("config must not load for a completed run")
        ),
    )
    judge_local.run(in_paths, model_name="fake/judge", batch=1)
    assert FakeModel.loads == 2


def _write_jsonl_judge_consumers(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _records() -> list[dict]:
    return [
        {"uid": "a", "config": "L1_a1.0", "layer": 1, "alpha": 1.0,
         "truncated": False, "rule_pass": True,
         "output": "A genuine answer."},
        {"uid": "b", "config": "L1_a1.0", "layer": 1, "alpha": 1.0,
         "truncated": False, "rule_pass": True,
         "output": "I cannot answer."},
        {"uid": "c", "config": "L1_a1.0", "layer": 1, "alpha": 1.0,
         "truncated": False, "rule_pass": False,
         "output": "Rule failure."},
        {"uid": "d", "config": "L1_a1.0", "layer": 1, "alpha": 1.0,
         "rule_pass": False, "truncated": True,
         "output": "An answer that ran into the cap and never finished"},
    ]


def _verdicts() -> list[dict]:
    return [
        {"uid": "a", "coherent": True, "answers_question": True},
        {"uid": "b", "coherent": True, "answers_question": False},
    ]


def test_calibration_report_uses_shared_success_and_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(
        calibrate, "calibration_dir",
        lambda model_slug, method_key, task: tmp_path / model_slug / method_key / task,
    )
    monkeypatch.setattr(calibrate, "slug", lambda _: "model")
    out_dir = tmp_path / "model" / "caa" / "no_comma"
    judging = out_dir / "judging" / "search"
    judging.mkdir(parents=True)
    (out_dir / "search.json").write_text(json.dumps({
        "model": "org/model", "task": "no_comma", "method": "caa",
        "n_items": 4, "config_names": ["L1_a1.0"], "references": {},
        "axes": ["layer", "alpha"],
        "plan": {"stages": [{"name": "search"}]},
    }))
    _write_jsonl_judge_consumers(out_dir / "L1_a1.0.jsonl", _records())
    _write_jsonl_judge_consumers(judging / "judge_out_0.jsonl", _verdicts())

    calibrate.report(
        SimpleNamespace(model="org/model", task="no_comma", baseline="caa")
    )

    summary = json.loads((out_dir / "summary.json").read_text())
    row = summary["configs"][0]
    # every rate is over all four items: the truncated one is a failure, not
    # a hole in the denominator
    assert row["accuracy"] == 1 / 4
    assert row["rule_rate"] == 2 / 4
    assert row["not_answering_rate"] == 1 / 4
    assert row["n"] == 4
    assert row["n_total"] == 4
    assert row["n_truncated"] == 1
    assert summary["scored_outputs"] == "scored_outputs.jsonl"
    scored = [
        json.loads(line)
        for line in (out_dir / "scored_outputs.jsonl").read_text().splitlines()
    ]
    # the truncated record is a decided failure now, not an absent decision
    assert [record["success"] for record in scored] == [True, False, False, False]
    assert scored[1]["answers_question"] is False


def test_calibration_report_refuses_an_unjudged_rule_passer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        calibrate, "calibration_dir",
        lambda model_slug, method_key, task: tmp_path / model_slug / method_key / task,
    )
    monkeypatch.setattr(calibrate, "slug", lambda _: "model")
    out_dir = tmp_path / "model" / "caa" / "no_comma"
    judging = out_dir / "judging" / "search"
    judging.mkdir(parents=True)
    (out_dir / "search.json").write_text(json.dumps({
        "n_items": 4, "config_names": ["L1_a1.0"], "references": {},
        "axes": ["layer", "alpha"],
        "plan": {"stages": [{"name": "search"}]},
    }))
    _write_jsonl_judge_consumers(out_dir / "L1_a1.0.jsonl", _records())
    _write_jsonl_judge_consumers(judging / "judge_out_0.jsonl", _verdicts()[:1])

    with pytest.raises(SystemExit, match="1 rule-pass outputs have no judge verdict"):
        calibrate.report(
        SimpleNamespace(model="org/model", task="no_comma", baseline="caa")
    )


def test_evaluation_report_uses_shared_success_and_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evaluate, "test_dir",
        lambda model_slug, method_key, task, config:
            tmp_path / model_slug / method_key / task / config,
    )
    monkeypatch.setattr(evaluate, "slug", lambda _: "model")
    out_dir = tmp_path / "model" / "caa" / "no_comma" / "manual_L1_a1.0"
    judging = out_dir / "judging"
    judging.mkdir(parents=True)
    records = _records()
    _write_jsonl_judge_consumers(out_dir / "outputs.jsonl", records)
    (out_dir / "run.json").write_text(json.dumps({
        "model": "org/model", "task": "no_comma", "method": "caa",
        "split": "test", "config": "manual_L1_a1.0",
        "config_values": {"layer": 1, "alpha": 1.0}, "config_source": "manual", "n_items": 4,
    }))
    _write_jsonl_judge_consumers(judging / "judge_out_0.jsonl", _verdicts())

    args = SimpleNamespace(model="org/model", task="no_comma",
                           config="layer=1,alpha=1.0", baseline="caa")
    evaluate.report(args)

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["accuracy"] == 1 / 4
    assert summary["n_scored"] == 4
    assert summary["n_truncated"] == 1
    assert summary["not_answering"] == 1
    assert summary["n_unjudged"] == 0
    assert summary["scored_outputs"] == "scored_outputs.jsonl"
    scored = [
        json.loads(line)
        for line in (out_dir / "scored_outputs.jsonl").read_text().splitlines()
    ]
    # the truncated record is a decided failure now, not an absent decision
    assert [record["success"] for record in scored] == [True, False, False, False]


def test_evaluation_report_refuses_an_unjudged_rule_passer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evaluate, "test_dir",
        lambda model_slug, method_key, task, config:
            tmp_path / model_slug / method_key / task / config,
    )
    monkeypatch.setattr(evaluate, "slug", lambda _: "model")
    out_dir = tmp_path / "model" / "caa" / "no_comma" / "manual_L1_a1.0"
    judging = out_dir / "judging"
    judging.mkdir(parents=True)
    _write_jsonl_judge_consumers(out_dir / "outputs.jsonl", _records())
    (out_dir / "run.json").write_text(json.dumps({
        "model": "org/model", "task": "no_comma", "method": "caa",
        "split": "test", "config": "manual_L1_a1.0",
        "config_values": {"layer": 1, "alpha": 1.0}, "config_source": "manual", "n_items": 4,
    }))
    _write_jsonl_judge_consumers(judging / "judge_out_0.jsonl", _verdicts()[:1])

    args = SimpleNamespace(model="org/model", task="no_comma",
                           config="layer=1,alpha=1.0", baseline="caa")
    with pytest.raises(SystemExit, match="1 rule-pass outputs have no judge verdict"):
        evaluate.report(args)


def _item(index: int) -> dict:
    return {
        "item_id": f"item_{index}",
        "prompt_id": f"prompt_{index}",
        "base_prompt": f"Question {index}?",
    }


def test_build_records_uses_one_schema_for_any_split():
    records = build_records(
        "no_comma", "dev", [_item(0)],
        [{"output": "A direct answer", "truncated": False, "n_new_tokens": 3}],
        "L18_a2.0", {"method": "caa", "config": "L18_a2.0"},
    )
    assert records == [{
        "method": "caa", "config": "L18_a2.0",
        "output": "A direct answer", "truncated": False, "n_new_tokens": 3,
        "rule_pass": True,
        "uid": "no_comma|dev|L18_a2.0|item_0",
        "item_id": "item_0", "prompt_id": "prompt_0",
        "task": "no_comma", "split": "dev", "question": "Question 0?",
    }]


def test_build_records_refuses_generation_alignment_loss():
    with pytest.raises(ValueError, match=r"zip\(\) argument 2"):
        build_records("no_comma", "test", [_item(0)], [], "run")


def test_every_configuration_is_scored_over_the_whole_item_set():
    """A denominator that moved with the configuration let steering which had
    broken the model be graded on the items it happened not to break."""
    summary = summarize_rules([
        {"rule_pass": True, "truncated": False},
        {"rule_pass": False, "truncated": False},
        {"rule_pass": False, "truncated": True},
    ])
    assert summary == {
        "n_total": 3, "n_rule_scored": 3, "n_rule_pass": 1, "n_truncated": 1,
        "rule_pass_rate": 1 / 3, "truncation_rate": 1 / 3,
    }


def test_prepare_sends_only_rule_passers_to_judge(tmp_path):
    records = [
        {"uid": "pass", "question": "Q?", "output": "answer", "truncated": False, "rule_pass": True},
        {"uid": "fail", "question": "Q?", "output": "bad", "truncated": False, "rule_pass": False},
        {"uid": "truncated", "question": "Q?", "output": "cut",
         "rule_pass": False},
    ]
    assert prepare_rule_passers(records, tmp_path, n_batches=2) == 1
    rows = [
        json.loads(line)
        for path in tmp_path.glob("judge_in_*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [row["uid"] for row in rows] == ["pass"]


def test_merged_summary_uses_full_success_and_shared_denominator():
    records = [
        {"truncated": False, "rule_pass": True, "success": True, "coherent": True,
         "answers_question": True, "output": "answer"},
        {"truncated": False, "rule_pass": True, "success": False, "coherent": True,
         "answers_question": False, "output": "refusal"},
        {"truncated": False, "rule_pass": False, "success": False, "coherent": None,
         "answers_question": None, "output": "rule failure"},
        {"rule_pass": False, "success": False, "coherent": None,
         "answers_question": None, "output": "cut", "truncated": True},
    ]
    summary = summarize_merged(records)
    assert summary["accuracy"] == 1 / 4
    assert summary["rule_pass_rate"] == 2 / 4
    assert summary["n_not_answering"] == 1
