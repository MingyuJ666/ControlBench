"""ITI's per-head arithmetic. This is the step most likely to run without error
while doing a different experiment than intended: a wrong reshape still
produces numbers, just for the wrong head. Each test names a single head and
checks that only that head moved.
"""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import iti  # noqa: E402
from baselines.iti import producer  # noqa: E402
N_LAYERS, N_HEADS, HEAD_DIM = 2, 3, 4
from baselines.iti import heads  # noqa: E402
N_LAYERS_iti_heads, N_HEADS_iti_heads, HEAD_DIM_iti_heads = 3, 4, 2


def _ranking() -> list:
    """Heads ordered best first; layer 1 head 2 wins, layer 0 head 0 loses."""
    return [
        (1, 2, 0.95), (0, 1, 0.90), (1, 0, 0.80),
        (0, 2, 0.70), (1, 1, 0.60), (0, 0, 0.55),
    ]


def _artifact(**overrides) -> dict:
    direction = torch.zeros(N_LAYERS, N_HEADS, HEAD_DIM)
    direction[..., 0] = 1.0                     # unit vectors along axis 0
    record = {
        "direction": direction,
        "sigma": torch.full((N_LAYERS, N_HEADS), 2.0),
        "ranking": _ranking(),
        "extraction_position": "last_prompt_token",
        "method": "iti_head_probes",
        "task": "no_comma",
        "model": "org/Fake-Model",
    }
    record.update(overrides)
    return record


def test_selection_takes_the_globally_best_heads_not_the_best_per_layer():
    selected = iti.selected_heads(_artifact(), heads_k=3)
    assert selected == [(1, 2), (0, 1), (1, 0)]


def test_selection_refuses_more_heads_than_were_ranked():
    with pytest.raises(ValueError, match="exceeds the 6 heads that separate"):
        iti.selected_heads(_artifact(), heads_k=7)


def test_addition_is_alpha_times_sigma_along_each_head_direction():
    edits = iti.additions(_artifact(), {"heads_k": 2, "alpha": 3.0})

    assert set(edits) == {(1, 2), (0, 1)}
    for vector in edits.values():
        assert torch.allclose(vector, torch.tensor([6.0, 0.0, 0.0, 0.0]))


def test_reported_head_layers_use_the_shared_one_based_convention():
    """Internally a head's layer indexes model.model.layers; reported, it has
    to mean what every other baseline's layer means, or a cross-method table
    puts this method's L9 beside another's and calls them one site."""
    dose = iti.dose_metadata(_artifact(), {"heads_k": 2, "alpha": 3.0})

    assert iti.selected_heads(_artifact(), heads_k=2) == [(1, 2), (0, 1)]
    assert dose["site"]["heads"] == [[2, 2], [1, 1]]
    assert dose["site"]["layer_convention"] == "one-based hidden-state index"


def test_config_name_records_both_axes():
    assert iti.config_name({"heads_k": 48, "alpha": 15.0}) == "K48_a15.0"


def test_search_plan_is_one_joint_stage_over_both_axes():
    plan = iti.search_plan({"n_heads_total": 1152})
    assert len(plan.stages) == 1
    assert plan.stages[0].carry == ()
    assert plan.n_configs == 9
    pairs = {(c["heads_k"], c["alpha"]) for c in plan.stages[0].configs}
    assert pairs == {
        (k, a) for k in (16, 48, 128) for a in (5.0, 15.0, 30.0)
    }
    assert plan.as_record()["metadata"]["shape"] == "joint"


def test_search_plan_rejects_a_grid_that_would_not_fit_the_budget():
    """Truncating would leave the largest head counts silently untested."""
    with pytest.raises(ValueError, match="over the budget"):
        iti.search_plan(
            {"n_heads_total": 1152},
            {"heads_k": [1, 2, 3, 4], "alpha": [1.0, 2.0, 3.0]},
        )


def test_search_plan_rejects_asking_for_more_heads_than_the_model_has():
    with pytest.raises(ValueError, match="exceed the model's 6 heads"):
        iti.search_plan({"n_heads_total": 6}, {"heads_k": [8]})


def test_pair_split_keeps_both_members_of_a_pair_on_one_side():
    train, validation = producer.pair_split(10, fraction=0.2)
    assert train == list(range(8))
    assert validation == [8, 9]
    assert not set(train) & set(validation)


def test_pair_split_refuses_input_it_cannot_divide():
    with pytest.raises(ValueError, match="at least two pairs"):
        producer.pair_split(1)


def test_fit_head_returns_a_unit_direction_pointing_at_the_positives():
    generator = torch.Generator().manual_seed(0)
    positive = torch.randn(30, HEAD_DIM, generator=generator)
    negative = torch.randn(30, HEAD_DIM, generator=generator)
    positive[:, 1] += 5.0

    direction, model = producer.fit_head(positive, negative)

    assert torch.linalg.vector_norm(direction) == pytest.approx(1.0, abs=1e-5)
    assert direction[1] > 0.9
    assert producer.score_head(model, positive, negative) == pytest.approx(1.0)


def _saved(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(
        iti, "artifact_path",
        lambda model_slug, task: tmp_path / model_slug / f"{task}.pt",
    )
    path = tmp_path / "fake-model" / "no_comma.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_artifact(**overrides), path)


def test_load_artifact_accepts_a_matching_record(tmp_path, monkeypatch):
    _saved(tmp_path, monkeypatch)
    loaded = iti.load_artifact("fake-model", "no_comma")
    assert loaded["ranking"][0] == (1, 2, 0.95)


@pytest.mark.parametrize("overrides,message", [
    ({"method": "probe_logistic_regression"}, "is not iti_head_probes"),
    ({"task": "lowercase"}, "does not match"),
    ({"sigma": torch.ones(N_LAYERS)}, "one value per head"),
    ({"ranking": [(0, 0, 1.0)]}, "order every head exactly once"),
])
def test_load_artifact_rejects_a_broken_contract(
        tmp_path, monkeypatch, overrides, message):
    _saved(tmp_path, monkeypatch, **overrides)
    with pytest.raises(ValueError, match=message):
        iti.load_artifact("fake-model", "no_comma")


def test_load_artifact_rejects_directions_that_are_not_unit_vectors(
        tmp_path, monkeypatch):
    direction = torch.zeros(N_LAYERS, N_HEADS, HEAD_DIM)
    direction[..., 0] = 3.0
    _saved(tmp_path, monkeypatch, direction=direction)
    with pytest.raises(ValueError, match="is not a unit vector"):
        iti.load_artifact("fake-model", "no_comma")


class _FakeOProj(torch.nn.Module):
    """Stands in for o_proj: records what it was handed, unchanged."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, packed):
        self.seen = packed.clone()
        return packed


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        blocks = [
            torch.nn.Module() for _ in range(N_LAYERS_iti_heads)
        ]
        for index, block in enumerate(blocks):
            block.self_attn = torch.nn.Module()
            block.self_attn.o_proj = _FakeOProj()
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList(blocks)
        self.model = inner
        self.config = type(
            "Config", (),
            {"num_attention_heads": N_HEADS_iti_heads, "head_dim": HEAD_DIM_iti_heads,
             "hidden_size": N_HEADS_iti_heads * HEAD_DIM_iti_heads},
        )()
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def forward(self, hidden=None, **_kwargs):
        for block in self.model.layers:
            hidden = block.self_attn.o_proj(hidden)
        return hidden


def _packed(batch: int = 1, seq: int = 3) -> torch.Tensor:
    """Distinct value per (row, position, head, dim), so slices are traceable."""
    total = batch * seq * N_HEADS_iti_heads * HEAD_DIM_iti_heads
    return torch.arange(total, dtype=torch.float32).reshape(
        batch, seq, N_HEADS_iti_heads * HEAD_DIM_iti_heads
    )


def test_split_heads_puts_each_head_in_its_own_slice():
    packed = _packed(batch=1, seq=1)
    split = heads.split_heads(packed, N_HEADS_iti_heads, HEAD_DIM_iti_heads)

    assert split.shape == (1, 1, N_HEADS_iti_heads, HEAD_DIM_iti_heads)
    for head in range(N_HEADS_iti_heads):
        expected = packed[0, 0, head * HEAD_DIM_iti_heads:(head + 1) * HEAD_DIM_iti_heads]
        assert torch.equal(split[0, 0, head], expected)


def test_split_heads_rejects_a_width_that_is_not_heads_times_dim():
    with pytest.raises(ValueError, match="final dimension of 8"):
        heads.split_heads(torch.zeros(1, 7), N_HEADS_iti_heads, HEAD_DIM_iti_heads)


def test_capture_reads_the_final_position_of_every_layer():
    model = _FakeModel()
    packed = _packed(batch=2, seq=3)

    captured = heads.capture_last_token(model, {"hidden": packed})

    assert captured.shape == (2, N_LAYERS_iti_heads, N_HEADS_iti_heads, HEAD_DIM_iti_heads)
    expected = heads.split_heads(packed[:, -1, :], N_HEADS_iti_heads, HEAD_DIM_iti_heads)
    for layer in range(N_LAYERS_iti_heads):
        assert torch.equal(captured[:, layer], expected)


def test_addition_lands_in_exactly_one_head_of_one_layer():
    model = _FakeModel()
    packed = _packed(batch=1, seq=2)
    addition = torch.tensor([10.0, 20.0])

    with heads.add_to_heads(model, {(1, 2): addition}, positions="all"):
        model(hidden=packed.clone())

    untouched = model.model.layers[0].self_attn.o_proj.seen
    changed = model.model.layers[1].self_attn.o_proj.seen
    assert torch.equal(untouched, packed)

    difference = heads.split_heads(changed - packed, N_HEADS_iti_heads, HEAD_DIM_iti_heads)
    for head in range(N_HEADS_iti_heads):
        if head == 2:
            assert torch.equal(difference[0, 0, head], addition)
            assert torch.equal(difference[0, 1, head], addition)
        else:
            assert torch.equal(difference[0, 0, head], torch.zeros(HEAD_DIM_iti_heads))


def test_additions_to_several_heads_of_one_layer_do_not_overwrite_each_other():
    model = _FakeModel()
    packed = _packed(batch=1, seq=1)

    with heads.add_to_heads(model, {
        (0, 0): torch.tensor([1.0, 0.0]),
        (0, 3): torch.tensor([0.0, 5.0]),
    }, positions="all"):
        model(hidden=packed.clone())

    difference = heads.split_heads(
        model.model.layers[0].self_attn.o_proj.seen - packed, N_HEADS_iti_heads, HEAD_DIM_iti_heads
    )
    assert torch.equal(difference[0, 0, 0], torch.tensor([1.0, 0.0]))
    assert torch.equal(difference[0, 0, 3], torch.tensor([0.0, 5.0]))
    assert torch.equal(difference[0, 0, 1], torch.zeros(HEAD_DIM_iti_heads))


def test_hooks_are_removed_even_when_the_body_raises():
    model = _FakeModel()
    packed = _packed(batch=1, seq=1)

    with pytest.raises(RuntimeError, match="deliberate"):
        with heads.add_to_heads(model, {(2, 1): torch.ones(HEAD_DIM_iti_heads)}):
            raise RuntimeError("deliberate")

    model(hidden=packed.clone())
    assert torch.equal(model.model.layers[2].self_attn.o_proj.seen, packed)


@pytest.mark.parametrize("layer,head,message", [
    (N_LAYERS_iti_heads, 0, "layer must be in 0..2"),
    (-1, 0, "layer must be in 0..2"),
    (0, N_HEADS_iti_heads, "head must be in 0..3"),
])
def test_addition_rejects_a_head_that_does_not_exist(layer, head, message):
    with pytest.raises(ValueError, match=message):
        with heads.add_to_heads(_FakeModel(), {(layer, head): torch.ones(HEAD_DIM_iti_heads)}):
            pass


def test_addition_rejects_a_vector_of_the_wrong_width():
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        with heads.add_to_heads(_FakeModel(), {(0, 0): torch.ones(3)}):
            pass


def test_by_default_only_the_final_prompt_position_and_the_answer_move():
    """The first generated token is written from the final prompt position, so
    that one is included and the rest of the question is not."""
    model = _FakeModel()
    prompt = _packed(batch=1, seq=3)
    generated = _packed(batch=1, seq=1)
    addition = torch.tensor([7.0, 7.0])

    with heads.add_to_heads(model, {(0, 1): addition}):
        model(hidden=prompt.clone())
        during_prompt = model.model.layers[0].self_attn.o_proj.seen
        model(hidden=generated.clone())
        during_generation = model.model.layers[0].self_attn.o_proj.seen

    prompt_difference = heads.split_heads(
        during_prompt - prompt, N_HEADS_iti_heads, HEAD_DIM_iti_heads
    )
    assert torch.equal(prompt_difference[0, 0, 1], torch.zeros(HEAD_DIM_iti_heads))
    assert torch.equal(prompt_difference[0, 1, 1], torch.zeros(HEAD_DIM_iti_heads))
    assert torch.equal(prompt_difference[0, 2, 1], addition)

    difference = heads.split_heads(
        during_generation - generated, N_HEADS_iti_heads, HEAD_DIM_iti_heads
    )
    assert torch.equal(difference[0, 0, 1], addition)
    assert torch.equal(difference[0, 0, 0], torch.zeros(HEAD_DIM_iti_heads))
