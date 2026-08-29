"""Generation and what is measured from it: the decode budget, the evaluation
loop, the unsteered and prompted controls, and the residual scale that makes
one method's dose comparable to another's.
"""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kernel.generate import (  # noqa: E402
    MAX_NEW_TOKENS,
    GenConfig,
    render_chat,
    strip_think_blocks,
)
from contextlib import contextmanager
import torch
from kernel import generate as generate_module  # noqa: E402
from kernel.generate import GenConfig, Generator  # noqa: E402
import json
from baselines import caa, iti  # noqa: E402
from pipeline import evaluate  # noqa: E402
from types import SimpleNamespace
from kernel import generate  # noqa: E402
from kernel.reference_conditions import (  # noqa: E402
    condition_dir,
    load_reference_records,
    load_reference_summary,
    make_record,
    outputs_path,
    run_path,
    summarize_records,
    summary_path,
    validate_alignment,
)
from kernel import reference_conditions as rc  # noqa: E402
from kernel.tasks import TASKS, make_prompted_prompt  # noqa: E402
from kernel import residual_scale  # noqa: E402


class ThinkingTokenizer:
    def __init__(self):
        self.call = None

    def apply_chat_template(self, messages, **kwargs):
        self.call = (messages, kwargs)
        return "rendered"


class NonThinkingTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return f"fallback:{messages[0]['content']}"


class BrokenTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        raise TypeError("template rendering failed")


def test_render_chat_disables_thinking_when_supported():
    tok = ThinkingTokenizer()

    assert render_chat(tok, "Why is the sky blue?") == "rendered"
    messages, kwargs = tok.call
    assert messages == [{"role": "user", "content": "Why is the sky blue?"}]
    assert kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_render_chat_falls_back_only_for_unsupported_thinking_argument():
    assert render_chat(NonThinkingTokenizer(), "Hello") == "fallback:Hello"


def test_render_chat_does_not_hide_unrelated_type_errors():
    with pytest.raises(TypeError, match="template rendering failed"):
        render_chat(BrokenTokenizer(), "Hello")


@pytest.mark.parametrize(("text", "expected", "removed"), [
    ("plain answer", "plain answer", False),
    ("<think>hidden</think>\nanswer", "answer", True),
    ("prefix <think>unfinished reasoning", "prefix", True),
    ("<think>one</think>first <think>two</think>second", "first second", True),
    ("  <think></think>  answer  ", "answer", True),
])
def test_strip_think_blocks(text, expected, removed):
    assert strip_think_blocks(text) == (expected, removed)


def test_generation_defaults_keep_the_experiment_token_cap():
    cfg = GenConfig()
    # Set from the reference model's measured length distribution: unsteered
    # Qwen3-8B's longest dev answer was 2010 new tokens, so the budget carries
    # roughly half again as much room for a more verbose model.
    assert cfg.max_new_tokens == MAX_NEW_TOKENS == 3072


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    pad_token = "<eos>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self):
        self.batches = []

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, texts, **kwargs):
        assert kwargs == {
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left",
        }
        self.batches.append(list(texts))
        prompt_ids = {
            "short": 1,
            "the longest prompt": 2,
            "medium prompt": 3,
        }
        ids = torch.tensor([[0, prompt_ids[text]] for text in texts])
        return FakeBatch({
            "input_ids": ids,
            "attention_mask": (ids != 0).long(),
        })

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is True
        outputs = {
            (11,): "<think>hidden</think> short answer",
            (21, 22): "long answer",
            (31,): "medium answer",
        }
        return outputs[tuple(token_ids.tolist())]


class FakeModel:
    device = torch.device("cpu")
    dtype = torch.float32

    class config:
        num_hidden_layers = 2

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        completions = {
            1: [11, 0],
            2: [21, 22],
            3: [31, 0],
        }
        new_ids = torch.tensor([
            completions[int(row[-1])] for row in input_ids
        ])
        return torch.cat([input_ids, new_ids], dim=1)


def test_generator_batches_intervenes_decodes_and_restores_order(monkeypatch):
    tok = FakeTokenizer()
    model = FakeModel()
    monkeypatch.setattr(
        generate_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model: tok),
    )
    monkeypatch.setattr(
        generate_module.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda _model, **_kwargs: model),
    )

    events = []

    @contextmanager
    def intervention():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    gen = Generator(GenConfig(max_new_tokens=2, batch_size=2))
    results = gen.generate(
        ["short", "the longest prompt", "medium prompt"],
        intervention=intervention,
        progress=False,
    )

    assert [row["output"] for row in results] == [
        "short answer",
        "long answer",
        "medium answer",
    ]
    assert [row["n_new_tokens"] for row in results] == [1, 2, 1]
    assert [row["truncated"] for row in results] == [False, True, False]
    assert tok.batches == [["the longest prompt", "medium prompt"], ["short"]]
    assert events == ["enter", "exit", "enter", "exit"]
    assert gen.stripped_think == 1
    assert all(call["do_sample"] is False for call in model.calls)


def test_an_external_config_names_its_axes():
    """Axes differ by baseline, so a frozen config is named, not positional."""
    assert evaluate.parse_config("layer=18,alpha=2.5") == {
        "layer": 18, "alpha": 2.5
    }
    assert evaluate.parse_config("heads_k=48,alpha=15") == {
        "heads_k": 48, "alpha": 15
    }
    assert evaluate.parse_config(None) is None
    with pytest.raises(ValueError, match="axis=value"):
        evaluate.parse_config("18")


def test_a_manual_config_is_carried_through_under_the_baseline_s_name():
    assert evaluate.resolve_config(
        "no_comma", "org/model", {"layer": 18, "alpha": 2.5}, caa
    ) == ({"layer": 18, "alpha": 2.5}, "manual", "manual_L18_a2.5")

    assert evaluate.resolve_config(
        "no_comma", "org/model", {"heads_k": 48, "alpha": 15.0}, iti
    ) == ({"heads_k": 48, "alpha": 15.0}, "manual", "manual_K48_a15.0")


def test_calibrated_config_validates_artifact_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evaluate, "calibration_dir",
        lambda model_slug, method_key, task: tmp_path / model_slug / method_key / task,
    )
    monkeypatch.setattr(evaluate, "slug", lambda _: "model")
    path = tmp_path / "model" / "caa" / "no_comma" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "model": "org/model", "task": "no_comma", "method": "caa",
        "split": "dev", "axes": ["layer", "alpha"],
        "chosen": {"layer": 18, "alpha": 2.0},
    }))

    assert evaluate.resolve_config("no_comma", "org/model", None, caa) == (
        {"layer": 18, "alpha": 2.0}, "calibration", "calibrated"
    )

    data = json.loads(path.read_text())
    data["task"] = "lowercase"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="task='lowercase'.*expected 'no_comma'"):
        evaluate.resolve_config("no_comma", "org/model", None, caa)


def test_bootstrap_is_reproducible_and_rejects_zero_draws():
    assert evaluate.bootstrap([True, True], n=20) == (1.0, 1.0)
    assert evaluate.bootstrap([True, False], n=50, seed=7) == \
        evaluate.bootstrap([True, False], n=50, seed=7)
    with pytest.raises(ValueError, match="n must be positive"):
        evaluate.bootstrap([True], n=0)


def test_cli_passes_external_config_and_has_no_test_limit(monkeypatch):
    captured = []
    monkeypatch.setattr(evaluate, "generate", captured.append)
    evaluate.main([
        "no_comma", "--model", "org/model", "--config", "layer=18,alpha=2.5",
        "--reason", "selected on public dev", "--batch", "4",
    ])
    args = captured[0]
    assert (args.config, args.reason, args.batch) == (
        "layer=18,alpha=2.5", "selected on public dev", 4
    )

    with pytest.raises(SystemExit):
        evaluate.build_parser().parse_args(["no_comma", "--limit", "10"])


def _args(**overrides):
    values = {
        "model": "org/model", "task": "no_comma", "baseline": "caa",
        "config": "layer=18,alpha=2.0", "reason": "selected on public dev",
        "batch": 2, "force": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@contextmanager
def _noop_intervention():
    yield


def _fake_method():
    return SimpleNamespace(
        KEY="caa",
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
    )


def test_fake_generation_to_final_test_score(tmp_path, monkeypatch):
    class FakeGenerator:
        n_layers = 36

        def __init__(self, config):
            self.cfg = config
            self.model = SimpleNamespace(
                config=SimpleNamespace(_commit_hash="fake-revision")
            )
            self.stripped_think = 0

        def generate(self, prompts, intervention=None, progress=True):
            assert len(prompts) == 2
            return [
                {"output": "A direct answer", "truncated": False,
                 "n_new_tokens": 3},
                {"output": "An answer, with a comma", "truncated": False,
                 "n_new_tokens": 6},
            ]

    items = [
        {"item_id": f"test_{index}", "prompt_id": f"prompt_{index}",
         "base_prompt": f"Question {index}?"}
        for index in range(2)
    ]
    monkeypatch.setattr(
        evaluate, "test_dir",
        lambda model_slug, method_key, task, config:
            tmp_path / model_slug / method_key / task / config,
    )
    monkeypatch.setattr(evaluate, "slug", lambda _: "model")
    monkeypatch.setattr(evaluate, "eval_items", lambda task, split: items)
    monkeypatch.setattr(
        evaluate.baselines, "get", lambda key: _fake_method(),
    )
    monkeypatch.setattr(evaluate, "load_reference_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(generate, "Generator", FakeGenerator)

    evaluate.generate(_args())

    out_dir = tmp_path / "model" / "caa" / "no_comma" / "manual_L18_a2.0"
    run = json.loads((out_dir / "run.json").read_text())
    assert run["n_items"] == 2
    assert run["rules"]["rule_pass_rate"] == 0.5
    assert run["n_rule_pass_sent_to_judge"] == 1
    assert run["model_revision"] == "fake-revision"

    records = [
        json.loads(line)
        for line in (out_dir / "outputs.jsonl").read_text().splitlines()
    ]
    assert {row["split"] for row in records} == {"test"}
    assert [row["rule_pass"] for row in records] == [True, False]
    judge_inputs = [
        json.loads(line)
        for path in (out_dir / "judging").glob("judge_in_*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [row["uid"] for row in judge_inputs] == [records[0]["uid"]]

    (out_dir / "judging" / "judge_out_0.jsonl").write_text(json.dumps({
        "uid": records[0]["uid"], "coherent": True,
        "answers_question": True,
    }) + "\n")
    evaluate.report(_args())

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["accuracy"] == 0.5
    assert summary["rule_pass_rate"] == 0.5
    assert summary["n_judged"] == 1
    assert summary["n_unjudged"] == 0
    assert summary["scored_outputs"] == "scored_outputs.jsonl"


def _item(task="no_comma", base="Why is the sky blue?"):
    return {
        "task": task,
        "item_id": f"{task}_dev_p1",
        "prompt_id": "p1",
        "base_prompt": base,
        "prompted_prompt": f"{base} Do not use commas.",
        "verifier": "verify_no_comma",
    }


def test_paths_follow_model_split_task_condition_layout(tmp_path):
    directory = condition_dir(
        "qwen3-8b", "dev", "no_comma", "prompted", tmp_path
    )
    assert directory == tmp_path / "qwen3-8b/dev/no_comma/prompted"
    assert outputs_path(
        "qwen3-8b", "dev", "no_comma", "prompted", tmp_path
    ) == directory / "outputs.jsonl"
    assert summary_path(
        "qwen3-8b", "dev", "no_comma", "prompted", tmp_path
    ) == directory / "summary.json"
    assert run_path("qwen3-8b", "dev", tmp_path) == (
        tmp_path / "qwen3-8b/dev/run.json"
    )

    with pytest.raises(ValueError, match="Unknown reference condition"):
        condition_dir("qwen3-8b", "dev", "no_comma", "steered", tmp_path)


def test_a_reference_shares_the_denominator_every_condition_uses():
    """The references sit in the same column as the baselines, so a rate over
    a different subset would not be the same measurement."""
    records = [
        {"rule_pass": True, "truncated": False},
        {"rule_pass": False, "truncated": False},
        {"rule_pass": False, "truncated": True},
    ]
    assert summarize_records(records) == {
        "n_total": 3,
        "n_rule_scored": 3,
        "n_rule_pass": 1,
        "n_truncated": 1,
        "rule_pass_rate": 1 / 3,
        "truncation_rate": 1 / 3,
    }


def test_alignment_checks_both_prompt_id_and_base_text():
    aligned = {"lowercase": [_item("lowercase")],
               "no_comma": [_item("no_comma")]}
    validate_alignment(aligned)

    mismatched = {"lowercase": [_item("lowercase")],
                  "no_comma": [_item("no_comma", "Different question")]}
    with pytest.raises(ValueError, match="carrier alignment differs"):
        validate_alignment(mismatched)


def test_make_record_has_exact_schema_and_scores_rule():
    generated = {"output": "the sky scatters blue light",
                 "truncated": False, "n_new_tokens": 6}
    record = make_record(_item(), "prompted", generated, "dev")

    assert set(record) == {
        "uid", "item_id", "prompt_id", "task", "split", "condition",
        "question", "input_prompt", "output", "truncated",
        "n_new_tokens", "verifier", "rule_pass",
    }
    assert record["uid"] == "no_comma_dev_p1_prompted"
    assert record["question"] == "Why is the sky blue?"
    assert record["input_prompt"].endswith("Do not use commas.")
    assert record["rule_pass"] is True


def test_optional_loaders_distinguish_missing_from_required(tmp_path):
    args = ("qwen3-8b", "dev", "no_comma", "unsteered", tmp_path)
    assert load_reference_records(*args) is None
    assert load_reference_summary(*args) is None
    with pytest.raises(FileNotFoundError):
        load_reference_records(*args, required=True)
    with pytest.raises(FileNotFoundError):
        load_reference_summary(*args, required=True)

    output = outputs_path(*args)
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({"uid": "x"}) + "\n", encoding="utf-8")
    summary_path(*args).write_text(
        json.dumps({"n_total": 1}), encoding="utf-8"
    )
    assert load_reference_records(*args) == [{"uid": "x"}]
    # the per-task summary alone is not enough: the run that writes it can die
    # before reaching the other tasks, and run.json is what says it did not
    assert load_reference_summary(*args) is None
    run_path("qwen3-8b", "dev", tmp_path).write_text("{}", encoding="utf-8")
    assert load_reference_summary(*args) == {"n_total": 1}


def _fake_eval_items(task, split):
    return [
        {
            "task": task,
            "item_id": f"{task}_{split}_p{index}",
            "prompt_id": f"p{index}",
            "base_prompt": base,
            "prompted_prompt": make_prompted_prompt(base, task),
            "verifier": TASKS[task]["verifier"],
        }
        for index, base in enumerate(("Explain blue skies", "Name a planet"), 1)
    ]


class FakeGenerator:
    def __init__(self):
        self.cfg = SimpleNamespace(
            model="example/Fake-Model",
            max_new_tokens=32,
            batch_size=10,
            dtype="bfloat16",
        )
        self.stripped_think = 0
        self.calls = []

    def generate(self, prompts, progress=True):
        self.calls.append((list(prompts), progress))
        return [
            {
                "output": f"answer without forbidden punctuation {index}",
                "truncated": False,
                "n_new_tokens": 5,
            }
            for index, _ in enumerate(prompts)
        ]


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_all_references_use_two_calls_and_write_self_contained_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rc, "eval_items", _fake_eval_items)
    generator = FakeGenerator()

    run = rc.generate_reference_conditions(
        generator,
        "example/Fake-Model",
        "fake-model",
        "dev",
        root=tmp_path,
        progress=False,
    )

    assert [len(prompts) for prompts, _ in generator.calls] == [2, 10]
    assert all(progress is False for _, progress in generator.calls)
    assert run["n_prompts"] == 2
    assert run["n_unique_generations"] == 12
    assert run["generation_calls"] == 2

    unsteered_by_prompt = {}
    for task in TASKS:
        for condition in rc.CONDITIONS:
            output_path = rc.outputs_path(
                "fake-model", "dev", task, condition, tmp_path
            )
            summary_path = rc.summary_path(
                "fake-model", "dev", task, condition, tmp_path
            )
            assert output_path.is_file()
            assert summary_path.is_file()
            # No judging directory: reference conditions are never judged,
            # and an empty one made an unjudged condition look staged for it.
            assert not (output_path.parent / "judging").exists()
            records = _read_jsonl(output_path)
            assert len(records) == 2
            assert [row["uid"] for row in records] == [
                f"{task}_dev_p1_{condition}",
                f"{task}_dev_p2_{condition}",
            ]
            assert json.loads(summary_path.read_text())["n_total"] == 2
            if condition == "unsteered":
                for row in records:
                    previous = unsteered_by_prompt.setdefault(
                        row["prompt_id"], row["output"]
                    )
                    assert row["output"] == previous

    assert rc.run_path("fake-model", "dev", tmp_path).is_file()

    calls_before = len(generator.calls)
    with pytest.raises(FileExistsError, match="already exist"):
        rc.generate_reference_conditions(
            generator,
            "example/Fake-Model",
            "fake-model",
            "dev",
            root=tmp_path,
        )
    assert len(generator.calls) == calls_before


class _Block(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class _Model:
    device = torch.device("cpu")

    def __init__(self, values):
        self.values = values
        self.config = SimpleNamespace(
            num_hidden_layers=len(values) - 1, hidden_size=4,
            _name_or_path="example/fake-model",
        )
        self.model = SimpleNamespace(
            layers=[_Block() for _ in range(len(values) - 1)]
        )

    def __call__(self, input_ids, output_hidden_states, logits_to_keep=0):
        rows = input_ids.shape[0]
        states = [
            torch.full((rows, 2, 4), float(value)) for value in self.values
        ]
        self.model.layers[-1](states[-1])
        return SimpleNamespace(hidden_states=tuple(states))


class _Tokenizer:
    def __call__(self, texts, **kwargs):
        class _Encoded(dict):
            def to(self, _device):
                return self
        return _Encoded(input_ids=torch.zeros(len(texts), 2, dtype=torch.long))

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


def test_measure_returns_one_rms_per_hidden_state_index(monkeypatch):
    monkeypatch.setattr(
        "baselines.shared.capture.render_chat", lambda tok, text: text
    )
    model = _Model([0.0, 3.0, 4.0])

    values = residual_scale.measure(model, _Tokenizer(), ["a", "b", "c"], batch=2)

    # every element of an index is the same constant, so its RMS is that value
    assert values == pytest.approx([0.0, 3.0, 4.0])


def test_ratio_divides_the_dose_by_the_stream_it_is_added_to():
    dose = {"effective_delta_norm": 6.0, "site": {"layer": 2}}
    assert residual_scale.ratio(dose, [0.0, 3.0, 4.0]) == pytest.approx(1.5)


@pytest.mark.parametrize("dose,scale", [
    ({}, [1.0, 2.0]),                                   # nothing recorded
    ({"effective_delta_norm": 1.0}, [1.0, 2.0]),        # no site
    ({"effective_delta_norm": 1.0, "site": {}}, [1.0]),  # no layer
    ({"effective_delta_norm": 1.0, "site": {"layer": 9}}, [1.0]),  # out of range
    ({"effective_delta_norm": 1.0, "site": {"layer": 0}}, [0.0]),  # zero stream
    ({"effective_delta_norm": 1.0, "site": {"layer": 0}}, None),   # unmeasured
])
def test_ratio_reports_nothing_rather_than_guessing(dose, scale):
    assert residual_scale.ratio(dose, scale) is None


def test_a_head_dose_has_no_single_layer_and_is_left_unrated():
    """ITI edits many heads across layers, so there is no one site to divide
    by; the ratio says so instead of inventing one."""
    dose = {"effective_delta_norm": 5.0,
            "site": {"component": "attn_head_pre_o_proj", "heads": [[2, 1]]}}
    assert residual_scale.ratio(dose, [1.0, 1.0, 1.0]) is None
