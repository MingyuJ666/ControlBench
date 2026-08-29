"""The CAA baseline end to end: extracting the difference in means, and adding
it back at the layer it was taken from. The fake-model tests check the
arithmetic; the ones needing weights are opt-in.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import produce  # noqa: E402
from baselines.shared import capture  # noqa: E402
from baselines import caa  # noqa: E402
from kernel.config import slug  # noqa: E402
from kernel.tasks import TASKS  # noqa: E402
from baselines.caa import producer  # noqa: E402


class _Encoded(dict):
    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self


class _RecordingTokenizer:
    def __init__(self):
        self.rendered = []
        self.tokenized = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        text = messages[0]["content"]
        self.rendered.append(text)
        return text

    def __call__(self, texts, **kwargs):
        assert kwargs == {
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left",
        }
        self.tokenized.append(list(texts))
        input_ids = torch.tensor([[0, len(text)] for text in texts])
        return _Encoded(input_ids=input_ids)


class _Block(torch.nn.Module):
    """Stands in for a decoder block so a forward hook has something to fire."""

    def forward(self, hidden):
        return hidden


class _HiddenStateModel:
    """A model whose top returned hidden state is post-final-norm.

    That is what transformers actually returns, and it is a different space
    from the block output a direction is injected into, so the fake reproduces
    it: the last entry of ``hidden_states`` is scaled, and only the block's
    own output carries the value capture is supposed to read.
    """

    device = torch.device("cpu")

    def __init__(self):
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=2,
            _name_or_path="example/fake-model",
        )
        self.model = SimpleNamespace(layers=[_Block(), _Block()])

    def __call__(self, input_ids, output_hidden_states,
                 logits_to_keep=0):
        assert output_hidden_states is True
        base = input_ids.float().unsqueeze(-1)
        base = torch.cat((base, base + 0.5), dim=-1)
        final_block_output = base + 20
        self.model.layers[-1](final_block_output)      # fires the hook
        return SimpleNamespace(
            hidden_states=(base, base + 10, final_block_output * 100)
        )


def _pairs(n=4):
    return [
        {"positive": f"p{index}", "negative": f"n{index}"}
        for index in range(n)
    ]


def _direction_states(_model, _tok, texts):
    directions = {
        "p0": (1.0, 0.0),
        "p1": (3.0, 0.0),
        "p2": (0.0, 2.0),
        "p3": (0.0, 4.0),
        "n0": (0.0, 0.0),
        "n1": (0.0, 0.0),
        "n2": (0.0, 0.0),
        "n3": (0.0, 0.0),
    }
    rows = []
    for text in texts:
        base = torch.tensor(directions[text])
        rows.append(torch.stack((base, 2 * base, 3 * base)))
    return torch.stack(rows)


def test_last_token_states_renders_and_returns_all_layers_on_cpu():
    model = _HiddenStateModel()
    tokenizer = _RecordingTokenizer()

    states = capture.last_token_states(model, tokenizer, ["a", "abcd"])

    assert states.shape == (2, 3, 2)
    assert states.dtype == torch.float32
    assert states.device.type == "cpu"
    assert tokenizer.rendered == ["a", "abcd"]
    assert tokenizer.tokenized == [["a", "abcd"]]
    # the top index is the last block's own output (21.0), not the
    # post-final-norm state the model returned for it (2100.0)
    assert torch.equal(states[0, :, 0], torch.tensor([1.0, 11.0, 21.0]))
    assert torch.equal(states[1, :, 1], torch.tensor([4.5, 14.5, 24.5]))


def test_extract_streams_batches_and_saves_exact_mean(tmp_path, monkeypatch):
    model = _HiddenStateModel()
    calls = []

    def recording_states(model, tokenizer, texts):
        calls.append(list(texts))
        return _direction_states(model, tokenizer, texts)

    monkeypatch.setattr(
        caa, "artifact_path",
        lambda model, task: tmp_path / model / "caa" / f"{task}.pt",
    )
    monkeypatch.setattr(capture, "extraction_pairs", lambda task: _pairs())
    monkeypatch.setattr(capture, "last_token_states", recording_states)

    destination = producer.extract_task(
        model, object(), "fake-model", "no_comma", batch=2
    )
    record = torch.load(destination, weights_only=True)
    assert destination == (
        tmp_path / "fake-model" / "caa" / "no_comma.pt"
    )

    expected_base = torch.tensor([1.0, 1.5])
    expected_vector = torch.stack(
        (expected_base, 2 * expected_base, 3 * expected_base)
    )
    assert torch.allclose(record["vector"], expected_vector)
    assert record["split_half_cosine"] == pytest.approx([0.0, 0.0, 0.0])
    assert calls == [
        ["p0", "p1"],
        ["n0", "n1"],
        ["p2", "p3"],
        ["n2", "n3"],
    ]
    assert set(record) == {
        "vector",
        "method",
        "task",
        "model",
        "n_pairs",
        "extraction_position",
        "split_half_cosine",
    }
    assert record["method"] == "caa_diff_in_means"
    assert record["task"] == "no_comma"
    assert record["model"] == "example/fake-model"
    assert record["n_pairs"] == 4
    assert record["extraction_position"] == "last_prompt_token"

    calls.clear()
    with pytest.raises(FileExistsError, match="already exists"):
        producer.extract_task(
            model, object(), "fake-model", "no_comma", batch=2
        )
    assert calls == []

    producer.extract_task(
        model,
        object(),
        "fake-model",
        "no_comma",
        batch=2,
        force=True,
    )
    assert calls


def test_extract_rejects_invalid_inputs_before_saving(tmp_path, monkeypatch):
    model = _HiddenStateModel()
    monkeypatch.setattr(
        caa, "artifact_path",
        lambda model, task: tmp_path / model / "caa" / f"{task}.pt",
    )
    monkeypatch.setattr(capture, "extraction_pairs", lambda task: _pairs())
    monkeypatch.setattr(capture, "last_token_states", _direction_states)

    with pytest.raises(ValueError, match="Unknown task"):
        producer.extract_task(
            model, object(), "fake-model", "not_a_task", batch=2
        )
    with pytest.raises(ValueError, match="batch must be positive"):
        producer.extract_task(
            model, object(), "fake-model", "no_comma", batch=0
        )

    monkeypatch.setattr(
        capture, "extraction_pairs", lambda task: _pairs(n=1)
    )
    with pytest.raises(ValueError, match="at least two pairs"):
        producer.extract_task(
            model, object(), "fake-model", "no_comma", batch=1
        )
    assert not list(tmp_path.rglob("*.pt"))


def test_extract_rejects_bad_or_nonfinite_state_tensors(tmp_path, monkeypatch):
    model = _HiddenStateModel()
    monkeypatch.setattr(
        caa, "artifact_path",
        lambda model, task: tmp_path / model / "caa" / f"{task}.pt",
    )
    monkeypatch.setattr(capture, "extraction_pairs", lambda task: _pairs(n=2))

    def wrong_layers(_model, _tok, texts):
        return torch.zeros(len(texts), 2, 2)

    monkeypatch.setattr(capture, "last_token_states", wrong_layers)
    with pytest.raises(ValueError, match=r"expected \[2, 3, hidden\]"):
        producer.extract_task(
            model, object(), "fake-model", "no_comma", batch=2
        )

    def nonfinite(_model, _tok, texts):
        states = torch.ones(len(texts), 3, 2)
        states[0, 0, 0] = torch.nan
        return states

    monkeypatch.setattr(capture, "last_token_states", nonfinite)
    with pytest.raises(ValueError, match="NaN or Inf"):
        producer.extract_task(
            model, object(), "fake-model", "no_comma", batch=2
        )
    assert not list(tmp_path.rglob("*.pt"))


def test_atomic_save_removes_partial_temporary_file(tmp_path, monkeypatch):
    destination = tmp_path / "artifact.pt"

    def broken_save(_record, temporary):
        Path(temporary).write_bytes(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(producer.torch, "save", broken_save)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        producer._atomic_torch_save({"vector": torch.ones(1)}, destination)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_cli_refuses_existing_artifact_before_model_load(
    tmp_path, monkeypatch
):
    model_name = "example/Fake-Model"
    model_slug = slug(model_name)
    monkeypatch.setattr(
        caa, "artifact_path",
        lambda model, task: tmp_path / model / "caa" / f"{task}.pt",
    )
    existing = caa.artifact_path(model_slug, "no_comma")
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing")

    def unexpected_model_load(*_args, **_kwargs):
        pytest.fail("model loading must not start when an artifact exists")

    monkeypatch.setattr(
        produce.AutoTokenizer, "from_pretrained", unexpected_model_load
    )
    with pytest.raises(SystemExit):
        produce.main(["no_comma", "--model", model_name])


@pytest.mark.parametrize("value", ["0", "-3"])
def test_positive_int_rejects_nonpositive_values(value):
    with pytest.raises(Exception, match="positive integer"):
        produce._positive_int(value)


def test_cli_defaults_to_extracting_all_five_tasks():
    parser = produce.build_parser()
    args = parser.parse_args([])
    assert produce.resolve_tasks(parser, args.tasks) == list(TASKS)


def test_cli_rejects_an_unknown_task_name_before_any_work():
    parser = produce.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        produce.resolve_tasks(parser, ["no_coma"])
    assert excinfo.value.code == 2


class FakeTokenizer:
    def __init__(self):
        self.render_count = 0

    def apply_chat_template(self, messages, **_kwargs):
        self.render_count += 1
        return messages[0]["content"]

    def __call__(self, texts, **_kwargs):
        values = []
        for text in texts:
            index = int(text.rsplit("-", 1)[1])
            values.append(index + (10 if text.startswith("positive") else 0))
        input_ids = torch.tensor([[0, value] for value in values])
        return _Encoded(input_ids=input_ids)


class _Block_caa_producer_smoke(torch.nn.Module):
    """A decoder block for a forward hook to fire on."""

    def forward(self, hidden):
        return hidden


class FakeModel:
    device = torch.device("cpu")

    def __init__(self):
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=3,
            _name_or_path="example/fake-model",
        )
        self.model = SimpleNamespace(layers=[_Block_caa_producer_smoke(), _Block_caa_producer_smoke()])

    def __call__(self, input_ids, output_hidden_states,
                 logits_to_keep=0):
        assert output_hidden_states is True
        base = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        final_block_output = 3 * base
        self.model.layers[-1](final_block_output)
        # as transformers does, the top entry is the post-final-norm state
        return SimpleNamespace(
            hidden_states=(base, 2 * base, 100 * base)
        )


def test_fake_model_runs_pairs_to_reloadable_artifact(tmp_path, monkeypatch):
    pairs = [
        {"positive": f"positive-{index}", "negative": f"negative-{index}"}
        for index in range(4)
    ]
    monkeypatch.setattr(
        caa, "artifact_path",
        lambda model, task: tmp_path / model / "caa" / f"{task}.pt",
    )
    monkeypatch.setattr(capture, "extraction_pairs", lambda task: pairs)

    tokenizer = FakeTokenizer()
    destination = producer.extract_task(
        FakeModel(), tokenizer, "fake-model", "no_comma", batch=2
    )
    record = torch.load(destination, weights_only=True)

    assert tokenizer.render_count == 8
    assert destination == (
        tmp_path / "fake-model" / "caa" / "no_comma.pt"
    )
    assert record["vector"].shape == (3, 3)
    assert torch.equal(
        record["vector"],
        torch.tensor([
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0],
        ]),
    )
    assert record["split_half_cosine"] == pytest.approx([1.0, 1.0, 1.0])
    assert not list(destination.parent.glob("*.tmp"))


def _caa_intervention(model, vector, layer, alpha, positions="all"):
    """The context CAA installs, reached through its public interface.

    These tests check the arithmetic of the addition, so they ask for every
    position; which positions the benchmark actually steers is decided in
    baselines.shared.positions and tested there.
    """
    from baselines.shared.residual_add import add_direction

    caa.make_intervention(
        model, {"vector": vector}, {"layer": layer, "alpha": alpha}
    )
    return add_direction(
        model, alpha * vector[layer], layer, positions=positions
    )


class _TensorLayer(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, hidden):
        return hidden + self.offset


class _TupleLayer(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, hidden):
        return hidden + self.offset, "preserved-cache"


class _FakeModel:
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self):
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList([
                _TensorLayer(1.0),
                _TupleLayer(2.0),
            ])
        )
        self.config = SimpleNamespace(hidden_size=3)


def _vector():
    vector = torch.zeros(3, 3)
    vector[2] = torch.tensor([1.0, 2.0, 3.0])
    return vector


def _artifact(**overrides):
    record = {
        "vector": torch.zeros(3, 3),
        "extraction_position": "last_prompt_token",
        "method": "caa_diff_in_means",
        "task": "no_comma",
        "model": "Qwen/Qwen3-8B",
    }
    record.update(overrides)
    return record


def test_inject_uses_hidden_state_index_and_preserves_tuple_fields():
    model = _FakeModel()
    hidden = torch.zeros(2, 4, 3)

    untouched_first_layer = model.model.layers[0](hidden)
    assert torch.equal(untouched_first_layer, torch.ones_like(hidden))

    with _caa_intervention(model, _vector(), layer=2, alpha=0.5):
        steered, cache = model.model.layers[1](hidden)

    expected = hidden + 2.0 + torch.tensor([0.5, 1.0, 1.5])
    assert torch.equal(steered, expected)
    assert cache == "preserved-cache"

    restored, cache = model.model.layers[1](hidden)
    assert torch.equal(restored, hidden + 2.0)
    assert cache == "preserved-cache"


def test_inject_removes_hook_when_context_body_raises():
    model = _FakeModel()
    layer = model.model.layers[0]

    with pytest.raises(RuntimeError, match="simulated failure"):
        with _caa_intervention(model, _vector(), layer=1, alpha=1.0):
            assert len(layer._forward_hooks) == 1
            raise RuntimeError("simulated failure")

    assert len(layer._forward_hooks) == 0
    assert torch.equal(layer(torch.zeros(1, 1, 3)), torch.ones(1, 1, 3))


@pytest.mark.parametrize("layer", [0, 3, 1.5, True])
def test_inject_rejects_invalid_layer(layer):
    with pytest.raises(ValueError, match="layer must"):
        with _caa_intervention(_FakeModel(), _vector(), layer=layer, alpha=1.0):
            pass


@pytest.mark.parametrize(
    "vector",
    [
        torch.zeros(3),
        torch.zeros(2, 3),
        torch.zeros(3, 4),
    ],
)
def test_inject_rejects_vector_shape_mismatch(vector):
    with pytest.raises(ValueError, match="vector must have shape"):
        with _caa_intervention(_FakeModel(), vector, layer=1, alpha=1.0):
            pass


def test_load_vector_accepts_matching_caa_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        caa, "artifact_path",
        lambda model, task: tmp_path / model / "caa" / f"{task}.pt",
    )
    path = tmp_path / "qwen3-8b/caa/no_comma.pt"
    path.parent.mkdir(parents=True)
    torch.save(_artifact(), path)

    loaded = caa.load_artifact("qwen3-8b", "no_comma")

    assert loaded["task"] == "no_comma"
    assert loaded["method"] == "caa_diff_in_means"
    assert loaded["vector"].shape == (3, 3)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"method": "other"}, "not a CAA mean-difference"),
        ({"task": "lowercase"}, "does not match"),
        ({"model": "other/model"}, "does not match"),
        ({"vector": torch.zeros(3)}, "must have shape"),
    ],
)
def test_load_vector_rejects_wrong_identity(
    tmp_path, monkeypatch, override, message
):
    monkeypatch.setattr(
        caa, "artifact_path",
        lambda model, task: tmp_path / model / "caa" / f"{task}.pt",
    )
    path = tmp_path / "qwen3-8b/caa/no_comma.pt"
    path.parent.mkdir(parents=True)
    torch.save(_artifact(**override), path)

    with pytest.raises(ValueError, match=message):
        caa.load_artifact("qwen3-8b", "no_comma")


def test_by_default_the_question_is_read_as_written_and_the_answer_is_steered():
    """The final prompt position is included because the first generated token
    is written from it; everything the question occupies is left alone."""
    model = _FakeModel()
    prompt = torch.zeros(1, 4, 3)
    generated = torch.zeros(1, 1, 3)
    added = torch.tensor([1.0, 2.0, 3.0])
    intervention = caa.make_intervention(
        model, {"vector": _vector()}, {"layer": 2, "alpha": 1.0}
    )

    with intervention():
        during_prompt, _ = model.model.layers[1](prompt)
        during_generation, _ = model.model.layers[1](generated)

    steered = torch.zeros(1, 4, 3) + 2.0
    steered[0, -1] += added
    assert torch.equal(during_prompt, steered)
    assert torch.equal(during_generation, generated + 2.0 + added)


def _caa_intervention_caa_intervention_smoke(model, vector, layer, alpha, positions="all"):
    """The context CAA installs, reached through its public interface."""
    from baselines.shared.residual_add import add_direction

    caa.make_intervention(
        model, {"vector": vector}, {"layer": layer, "alpha": alpha}
    )
    return add_direction(
        model, alpha * vector[layer], layer, positions=positions
    )


class _Add(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, hidden):
        return hidden + self.value


class FakeModel_caa_intervention_smoke:
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self):
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList([_Add(1.0), _Add(2.0)])
        )
        self.config = SimpleNamespace(hidden_size=3)

    def __call__(self, hidden):
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


def test_saved_vector_loads_injects_and_restores_model(tmp_path, monkeypatch):
    vector = torch.zeros(3, 3)
    vector[2] = torch.tensor([1.0, -1.0, 0.5])
    path = tmp_path / "fake-model/caa/no_comma.pt"
    path.parent.mkdir(parents=True)
    torch.save({
        "vector": vector,
        "extraction_position": "last_prompt_token",
        "method": "caa_diff_in_means",
        "task": "no_comma",
        "model": "example/Fake-Model",
    }, path)
    monkeypatch.setattr(
        caa, "artifact_path",
        lambda model, task: tmp_path / model / "caa" / f"{task}.pt",
    )

    record = caa.load_artifact("fake-model", "no_comma")
    model = FakeModel_caa_intervention_smoke()
    hidden = torch.zeros(2, 4, 3)
    plain = model(hidden)

    with _caa_intervention_caa_intervention_smoke(model, record["vector"], layer=2, alpha=2.0):
        steered = model(hidden)

    restored = model(hidden)
    expected_delta = torch.tensor([2.0, -2.0, 1.0])
    assert torch.equal(plain, torch.full_like(hidden, 3.0))
    assert torch.equal(steered, plain + expected_delta)
    assert torch.equal(restored, plain)
