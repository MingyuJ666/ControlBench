"""The SAE baseline: the feature ranking, the dictionary it is read from, and
the reconstruction gate that decides which layers may be searched at all.
"""
import json
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import sae  # noqa: E402
from baselines.sae import gate, producer  # noqa: E402
from baselines.sae import checkpoint, gate  # noqa: E402


def _gate_report(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(
        sae, "gate_report_path",
        lambda model_slug, tag="chat": tmp_path / f"gate.{tag}.json",
    )
    run = {
        "measure": gate.MEASURE,
        "model": "Qwen/Qwen3-8B",
        "tag": "chat",
        "passing_layers": [21, 27, 33],
    }
    run.update(overrides)
    (tmp_path / "gate.chat.json").write_text(
        json.dumps(run), encoding="utf-8"
    )


def test_only_layers_the_gate_admitted_can_be_searched(tmp_path, monkeypatch):
    _gate_report(tmp_path, monkeypatch)

    plan = sae.search_plan({"n_layers": 36, "model_slug": "qwen3-8b"})

    assert sorted({c["layer"] for c in plan.stages[0].configs}) == [21, 27, 33]
    assert sorted({c["alpha"] for c in plan.stages[0].configs}) == [1.0, 2.0, 4.0]
    assert plan.n_configs == 9


def test_the_plan_fits_the_budget_however_many_layers_the_gate_admits(
        monkeypatch):
    """The gate is a measurement, so how many layers it admits is not ours to
    assume. A strength grid that only fits while some layers fail would make
    the baseline unrunnable the day a re-run admitted one more."""
    for n_admitted in range(1, 6):
        admitted = list(range(9, 9 + n_admitted * 6, 6))
        monkeypatch.setattr(
            sae, "passing_layers", lambda ms, tag="chat", a=admitted: a
        )
        plan = sae.search_plan({"n_layers": 36, "model_slug": "qwen3-8b"})
        assert plan.n_configs <= sae.DEFAULT_BUDGET


def test_a_layer_whose_feature_failed_confirmation_is_not_searched(
        tmp_path, monkeypatch):
    """The gate and the confirmation half are two separate votes and both
    bind: a layer left with a zero direction steers with nothing, and an
    unsteered generation is coherent enough to win the search."""
    _gate_report(tmp_path, monkeypatch)
    artifact = {"indices_without_direction": [27]}

    plan = sae.search_plan(
        {"n_layers": 36, "model_slug": "qwen3-8b", "artifact": artifact}
    )

    assert sorted({c["layer"] for c in plan.stages[0].configs}) == [21, 33]


def test_a_report_from_an_earlier_measure_is_refused(tmp_path, monkeypatch):
    """FVU once depended on batch grouping; the field name did not change."""
    _gate_report(tmp_path, monkeypatch, measure="per-batch variance")

    with pytest.raises(SystemExit, match="re-run the gate"):
        sae.search_plan({"n_layers": 36, "model_slug": "qwen3-8b"})


def test_no_admitted_layer_is_reported_not_worked_around(tmp_path, monkeypatch):
    _gate_report(tmp_path, monkeypatch, passing_layers=[])

    with pytest.raises(SystemExit, match="reports that rather than a score"):
        sae.search_plan({"n_layers": 36, "model_slug": "qwen3-8b"})


def test_a_missing_gate_stops_the_search(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sae, "gate_report_path",
        lambda model_slug, tag="chat": tmp_path / "absent.json",
    )
    with pytest.raises(SystemExit, match="run the gate before selecting"):
        sae.search_plan({"n_layers": 36, "model_slug": "qwen3-8b"})


def test_separation_is_the_mean_activation_difference():
    """The paper's ranking. A feature large in both conditions cancels; one
    that fires more for the positives does not."""
    positive = torch.tensor([[50.0, 1.0], [50.0, 3.0]])
    negative = torch.tensor([[50.0, 0.0], [50.0, 0.0]])

    scores = producer.separation(positive, negative)

    assert scores.tolist() == [0.0, 2.0]
    assert int(torch.argmax(scores)) == 1


def test_the_standardized_score_is_recorded_but_does_not_rank():
    """Dividing by the within-group spread is the more careful statistic and
    would choose differently; it is not what the paper does."""
    generator = torch.Generator().manual_seed(0)
    positive = torch.rand(40, 2, generator=generator)
    negative = torch.rand(40, 2, generator=generator)
    positive[:, 0] += 5.0                        # big difference, huge spread
    positive[:, 0] += 40.0 * torch.rand(40, generator=generator)
    positive[:, 1] += 1.0                        # small difference, tight

    assert int(torch.argmax(producer.separation(positive, negative))) == 0
    assert int(torch.argmax(
        producer.standardized_separation(positive, negative)
    )) == 1


def test_a_feature_that_only_held_on_its_own_half_is_rejected():
    """The best of 65,536 features separates something; the other half decides
    whether it separates the instruction, and can say no."""
    generator = torch.Generator().manual_seed(1)
    positive = torch.rand(20, 4, generator=generator)
    negative = torch.rand(20, 4, generator=generator)
    positive[:, 2] += 3.0                        # real, holds on both halves
    positive[:10, 3] += 8.0                      # only in the selection half
    positive[10:, 3] -= 8.0                      # and reverses on the other

    selection, confirmation = producer.pair_split(20)
    chosen = producer.select_feature(
        positive, negative, selection, confirmation
    )

    assert chosen["feature"] == 3
    assert chosen["selection"] > 0
    assert chosen["confirmation"] < 0
    assert chosen["confirmed"] is False


def test_a_feature_that_holds_on_both_halves_is_kept():
    positive = torch.zeros(20, 2)
    negative = torch.zeros(20, 2)
    positive[:, 1] += 3.0

    selection, confirmation = producer.pair_split(20)
    chosen = producer.select_feature(
        positive, negative, selection, confirmation
    )

    assert chosen["feature"] == 1
    assert chosen["confirmed"] is True


def test_pair_split_keeps_a_pair_on_one_side_and_leaves_both_halves():
    selection, confirmation = producer.pair_split(10)
    assert selection == list(range(5))
    assert confirmation == list(range(5, 10))
    with pytest.raises(ValueError, match="at least two pairs"):
        producer.pair_split(1)


def _artifact(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(
        sae, "artifact_path",
        lambda model_slug, task: tmp_path / model_slug / f"{task}.pt",
    )
    direction = torch.zeros(3, 2)
    direction[2] = torch.tensor([1.0, 0.0])
    record = {
        "direction": direction,
        "max_act": torch.tensor([0.0, 0.0, 4.0]),
        "feature": torch.tensor([-1, -1, 4242]),
        "extraction_position": "last_prompt_token",
        "method": "sae_decoder_feature",
        "task": "no_comma",
        "model": "org/Fake-Model",
        "indices_without_direction": [0, 1],
    }
    record.update(overrides)
    path = tmp_path / "fake-model" / "no_comma.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, path)


def test_layers_the_gate_rejected_hold_no_direction(tmp_path, monkeypatch):
    _artifact(tmp_path, monkeypatch)
    loaded = sae.load_artifact("fake-model", "no_comma")
    assert loaded["indices_without_direction"] == [0, 1]


def test_a_rejected_layer_carrying_a_direction_is_refused(tmp_path, monkeypatch):
    direction = torch.zeros(3, 2)
    direction[1] = torch.tensor([1.0, 0.0])      # a layer the gate rejected
    direction[2] = torch.tensor([1.0, 0.0])
    _artifact(tmp_path, monkeypatch, direction=direction)
    with pytest.raises(ValueError, match="must hold zero directions"):
        sae.load_artifact("fake-model", "no_comma")


def test_strength_multiplies_the_feature_s_measured_maximum():
    artifact = {
        "direction": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        "max_act": torch.tensor([0.0, 4.0]),
        "feature": torch.tensor([-1, 7]),
    }
    assert torch.allclose(
        sae.delta(artifact, {"layer": 1, "alpha": 0.5}),
        torch.tensor([2.0, 0.0]),
    )
    dose = sae.dose_metadata(artifact, {"layer": 1, "alpha": 0.5})
    assert dose["max_act"] == 4.0
    assert dose["feature"] == 7
    assert dose["effective_delta_norm"] == pytest.approx(2.0)


def test_an_external_grid_cannot_reach_past_the_gate(tmp_path, monkeypatch):
    """Narrowing the search is a participant's right; widening it past the
    gate would make a mandatory gate advisory."""
    _gate_report(tmp_path, monkeypatch)

    narrowed = sae.search_plan(
        {"n_layers": 36, "model_slug": "qwen3-8b"}, {"layer": [27]}
    )
    assert {c["layer"] for c in narrowed.stages[0].configs} == {27}

    with pytest.raises(SystemExit, match="did not pass the reconstruction gate"):
        sae.search_plan(
            {"n_layers": 36, "model_slug": "qwen3-8b"}, {"layer": [9, 27]}
        )


def test_layer_numbering_round_trips_with_the_documented_offset():
    """Our index 1 is the first decoder block, the checkpoint calls it 0."""
    assert checkpoint.checkpoint_layer(1) == 0
    assert checkpoint.checkpoint_layer(9) == 8
    assert checkpoint.hidden_state_index(8) == 9
    for index in (1, 9, 21, 36):
        assert checkpoint.hidden_state_index(
            checkpoint.checkpoint_layer(index)
        ) == index


def test_layer_numbering_rejects_the_embedding_index():
    with pytest.raises(ValueError, match="embedding output"):
        checkpoint.checkpoint_layer(0)


def _sae(d_model: int = 4, d_sae: int = 6, k: int = 2) -> checkpoint.LoadedSae:
    torch.manual_seed(0)
    return checkpoint.LoadedSae(
        encoder_weight=torch.randn(d_sae, d_model),
        encoder_bias=torch.zeros(d_sae),
        decoder_weight=torch.randn(d_model, d_sae),
        decoder_bias=torch.zeros(d_model),
        k=k, hidden_state_index=9, checkpoint_layer=8,
        revision="abc", file_sha256="def",
    )


def test_encode_is_top_k_of_the_rectified_pre_activations():
    """Top-k ReLU, as the paper names it: rectify, then keep the largest k.
    Inference has to match what was trained."""
    sae = _sae()
    activations = torch.randn(3, sae.d_model)

    features = sae.encode(activations)

    assert features.shape == (3, sae.d_sae)
    # the checkpoint encodes the residual directly, without subtracting b_dec
    expected = torch.relu(
        activations @ sae.encoder_weight.T + sae.encoder_bias
    )
    _, indices = torch.topk(expected, sae.k, dim=-1)
    for row in range(activations.shape[0]):
        kept = indices[row]
        assert torch.allclose(features[row, kept], expected[row, kept])
        dropped = torch.ones(sae.d_sae, dtype=torch.bool)
        dropped[kept] = False
        assert torch.equal(features[row, dropped], torch.zeros(int(dropped.sum())))


def test_a_negative_pre_activation_contributes_nothing():
    """Rectification is the difference between this and the card's snippet."""
    sae = _sae(d_model=2, d_sae=3, k=2)
    sae.encoder_weight.zero_()
    sae.encoder_bias.copy_(torch.tensor([5.0, -1.0, -9.0]))

    features = sae.encode(torch.zeros(1, 2))

    assert features[0, 0] == 5.0
    assert features[0, 1] == 0.0
    assert features[0, 2] == 0.0


def test_reconstruct_composes_encode_and_decode():
    sae = _sae()
    activations = torch.randn(2, sae.d_model)
    assert torch.allclose(
        sae.reconstruct(activations), sae.decode(sae.encode(activations))
    )


def test_feature_direction_is_the_unit_decoder_column():
    sae = _sae()
    direction = sae.feature_direction(3)
    column = sae.decoder_weight[:, 3]
    assert torch.linalg.vector_norm(direction) == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(
        direction, column / torch.linalg.vector_norm(column), atol=1e-6
    )


def test_gate_verdict_needs_both_thresholds():
    passing = gate.GateResult(9, loss_recovered=0.85, fvu=0.20, cosine=0.95,
                              n_prompts=200, n_tokens=1000)
    assert passing.passed
    assert passing.as_record()["thresholds"]["loss_recovered_min"] == 0.80

    weak_loss = gate.GateResult(9, loss_recovered=0.70, fvu=0.20, cosine=0.95,
                                n_prompts=200, n_tokens=1000)
    high_error = gate.GateResult(9, loss_recovered=0.85, fvu=0.40, cosine=0.95,
                                 n_prompts=200, n_tokens=1000)
    assert not weak_loss.passed
    assert not high_error.passed


def test_gate_result_records_what_it_was_judged_against():
    record = gate.GateResult(21, 0.9, 0.1, 0.97, 200, 5000).as_record()
    assert record["hidden_state_index"] == 21
    assert record["passed"] is True
    assert record["thresholds"] == {
        "loss_recovered_min": 0.80, "fvu_max": 0.25
    }
