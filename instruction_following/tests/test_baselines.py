"""The registry every baseline is reached through, and the two that estimate a
residual-stream direction: probe and lat_pca. Also the position rule that
decides which token a direction is read at.
"""
import subprocess
import sys
from pathlib import Path
import pytest
import torch
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import baselines  # noqa: E402
from baselines.shared.base import SearchPlan, SearchStage, joint_plan  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import probe  # noqa: E402
from baselines.probe import producer  # noqa: E402
from baselines import lat_pca  # noqa: E402
from baselines.lat_pca import producer as producer_lat_pca
from baselines.shared import positions  # noqa: E402
DELTA = torch.tensor([1.0, 1.0])
from pipeline import produce  # noqa: E402


def test_registry_reports_known_and_planned_methods():
    assert baselines.keys() == [
        "caa", "probe", "lat_pca", "iti", "sae", "iti_disc", "sae_disc",
    ]
    report = baselines.availability()
    assert report["caa"] == "available"
    assert "not implemented" in report["reft"]


def test_registry_rejects_an_unknown_method_by_name():
    with pytest.raises(KeyError, match="unknown method 'nope'"):
        baselines.get("nope")


def test_selecting_one_method_does_not_import_the_others():
    """A missing optional dependency of one method must not break another."""
    program = (
        f"import sys; sys.path.insert(0, {str(SRC)!r});"
        "import baselines; baselines.get('caa');"
        "print(sorted(k for k in sys.modules if k.startswith('baselines.')))"
    )
    output = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True,
        check=True,
    ).stdout
    assert "baselines.caa" in output
    for planned in ("probe", "lat_pca", "iti", "sae", "reft"):
        assert f"baselines.{planned}" not in output


def test_stage_rejects_configurations_with_mismatched_axes():
    with pytest.raises(ValueError, match="mixes configuration axes"):
        SearchStage(name="s", configs=[{"layer": 1}, {"heads_k": 8}])


def test_stage_rejects_carrying_an_axis_it_does_not_have():
    with pytest.raises(ValueError, match="carries unknown axes"):
        SearchStage(name="s", configs=[{"layer": 1}], carry=("heads_k",))


def test_plan_refuses_to_exceed_its_budget():
    stage = SearchStage(name="s", configs=[{"layer": layer} for layer in (1, 2, 3)])
    with pytest.raises(ValueError, match="over the budget"):
        SearchPlan(stages=[stage], budget=2)


def test_a_joint_plan_pairs_every_site_with_every_strength():
    """One stage, no carried axis: nothing is chosen before anything else, so
    no assumption is made that the best site at one strength is the best site
    at another."""
    plan = joint_plan("layer", [9, 21], "alpha", [1.0, 2.0], budget=9)

    assert len(plan.stages) == 1
    assert plan.stages[0].carry == ()
    assert plan.stages[0].configs == [
        {"layer": 9, "alpha": 1.0}, {"layer": 9, "alpha": 2.0},
        {"layer": 21, "alpha": 1.0}, {"layer": 21, "alpha": 2.0},
    ]
    assert plan.n_configs == 4


def test_plan_record_is_serializable_for_the_calibration_summary():
    record = joint_plan("layer", [9], "alpha", [1.0], budget=9).as_record()
    assert record["metadata"]["shape"] == "joint"
    assert record["stages"][0]["carry"] == []
    assert record["n_configs"] == 1


def test_caa_search_plan_matches_the_reference_grid():
    caa = baselines.get("caa")
    plan = caa.search_plan({"n_layers": 36})
    pairs = [(c["layer"], c["alpha"]) for c in plan.stages[0].configs]
    assert pairs == [
        (9, 1.0), (9, 2.0), (9, 4.0),
        (21, 1.0), (21, 2.0), (21, 4.0),
        (33, 1.0), (33, 2.0), (33, 4.0),
    ]
    assert plan.n_configs == 9


def test_caa_search_plan_accepts_participant_grids():
    caa = baselines.get("caa")
    plan = caa.search_plan(
        {"n_layers": 36}, {"layer": [4, 8], "alpha": [1.0]}, budget=9
    )
    assert [(c["layer"], c["alpha"]) for c in plan.stages[0].configs] == [
        (4, 1.0), (8, 1.0)
    ]


@pytest.mark.parametrize("overrides,message", [
    ({"layer": [9, 9]}, "duplicates"),
    ({"layer": [0]}, "between 1 and 36"),
    ({"layer": [37]}, "between 1 and 36"),
    ({"alpha": [1.0, 1.0]}, "duplicates"),
])
def test_caa_search_plan_rejects_invalid_grids(overrides, message):
    with pytest.raises(ValueError, match=message):
        baselines.get("caa").search_plan({"n_layers": 36}, overrides)


def test_caa_config_name_is_the_published_directory_name():
    assert baselines.get("caa").config_name({"layer": 21, "alpha": 1.0}) == "L21_a1.0"


def test_caa_dose_metadata_scales_with_strength():
    caa = baselines.get("caa")
    artifact = {"vector": torch.zeros(3, 4)}
    artifact["vector"][2] = torch.tensor([3.0, 4.0, 0.0, 0.0])
    dose = caa.dose_metadata(artifact, {"layer": 2, "alpha": 2.0})
    assert dose["direction_norm"] == pytest.approx(5.0)
    assert dose["effective_delta_norm"] == pytest.approx(10.0)
    assert dose["site"] == {"component": "resid_post", "layer": 2}


def _separable(n: int = 40, hidden: int = 6, gap: float = 4.0):
    """Two clouds separated along axis 0, with noise on the other axes."""
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(n, hidden, generator=generator)
    positive = noise.clone()
    positive[:, 0] += gap
    negative = torch.randn(n, hidden, generator=generator)
    negative[:, 0] -= gap
    return positive, negative


def test_fit_layer_recovers_the_separating_axis_as_a_unit_direction():
    positive, negative = _separable()
    direction, sigma, accuracy = producer.fit_layer(positive, negative)

    assert torch.linalg.vector_norm(direction) == pytest.approx(1.0, abs=1e-5)
    assert abs(direction[0]) > 0.9          # dominated by the separating axis
    assert direction[0] > 0                 # points from negative to positive
    assert accuracy == pytest.approx(1.0)
    assert sigma > 0


def test_fit_layer_rejects_activations_it_cannot_separate():
    identical = torch.ones(8, 4)
    with pytest.raises(ValueError):
        producer.fit_layer(identical, identical.clone())


def test_delta_is_alpha_sigma_along_the_unit_direction():
    artifact = {
        "direction": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        "sigma": torch.tensor([1.0, 3.0, 1.0]),
    }
    delta = probe.delta(artifact, {"layer": 1, "alpha": 2.0})
    assert torch.allclose(delta, torch.tensor([0.0, 6.0]))


def test_dose_metadata_reports_sigma_scaled_magnitude():
    artifact = {
        "direction": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "sigma": torch.tensor([1.0, 2.5]),
    }
    dose = probe.dose_metadata(artifact, {"layer": 1, "alpha": 2.0})
    assert dose["method"] == "probe"
    assert dose["direction_norm"] == 1.0
    assert dose["projection_sigma"] == pytest.approx(2.5)
    assert dose["effective_delta_norm"] == pytest.approx(5.0)


def _artifact(tmp_path, monkeypatch, **overrides) -> dict:
    monkeypatch.setattr(
        probe, "artifact_path",
        lambda model_slug, task: tmp_path / model_slug / f"{task}.pt",
    )
    record = {
        "direction": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "sigma": torch.tensor([1.0, 2.0]),
        "extraction_position": "last_prompt_token",
        "method": "probe_logistic_regression",
        "task": "no_comma",
        "model": "org/Fake-Model",
    }
    record.update(overrides)
    path = tmp_path / "fake-model" / "no_comma.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, path)
    return record


def test_load_artifact_accepts_a_matching_probe(tmp_path, monkeypatch):
    _artifact(tmp_path, monkeypatch)
    loaded = probe.load_artifact("fake-model", "no_comma")
    assert loaded["task"] == "no_comma"
    assert loaded["sigma"].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("overrides,message", [
    ({"method": "caa_diff_in_means"}, "is not probe_logistic_regression"),
    ({"task": "lowercase"}, "does not match"),
    ({"model": "org/Other-Model"}, "does not match"),
    ({"sigma": torch.tensor([1.0])}, "one value per hidden-state index"),
    ({"direction": torch.tensor([[3.0, 4.0], [0.0, 1.0]])}, "unit vectors"),
])
def test_load_artifact_rejects_a_broken_contract(
        tmp_path, monkeypatch, overrides, message):
    _artifact(tmp_path, monkeypatch, **overrides)
    with pytest.raises(ValueError, match=message):
        probe.load_artifact("fake-model", "no_comma")


class _FakeModel:
    def __init__(self, n_layers: int = 2, hidden: int = 2):
        self.model = type("Inner", (), {"layers": list(range(n_layers))})()
        self.config = type("Config", (), {"hidden_size": hidden})()
        self.dtype = torch.float32
        self.device = torch.device("cpu")


def test_make_intervention_rejects_a_direction_of_the_wrong_shape():
    artifact = {
        "direction": torch.zeros(2, 2), "sigma": torch.ones(2),
    }
    with pytest.raises(ValueError, match="must have shape"):
        probe.make_intervention(
            _FakeModel(), artifact, {"layer": 1, "alpha": 1.0}
        )


def test_probe_and_caa_do_not_share_an_artifact_directory():
    assert probe.artifact_path("qwen3-8b", "no_comma").parent.name == "probe"


def test_an_index_with_identical_activations_has_no_contrast():
    """Index 0 is the embedding of a template token both prompts end with."""
    identical = torch.ones(4, 3)
    assert producer.has_contrast(identical, identical.clone()) is False
    assert producer.has_contrast(identical, identical + 1) is True


def test_load_artifact_accepts_zero_rows_only_where_contrast_was_absent(
        tmp_path, monkeypatch):
    direction = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    _artifact(
        tmp_path, monkeypatch, direction=direction,
        sigma=torch.tensor([0.0, 2.0]), indices_without_contrast=[0],
    )
    loaded = probe.load_artifact("fake-model", "no_comma")
    assert loaded["indices_without_contrast"] == [0]

    _artifact(
        tmp_path, monkeypatch, direction=direction,
        sigma=torch.tensor([0.0, 2.0]), indices_without_contrast=[],
    )
    with pytest.raises(ValueError, match="unit vectors"):
        probe.load_artifact("fake-model", "no_comma")


def test_load_artifact_rejects_a_contrast_free_index_holding_a_direction(
        tmp_path, monkeypatch):
    """A recorded-unusable index must not carry something steerable."""
    _artifact(
        tmp_path, monkeypatch,
        direction=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        sigma=torch.tensor([1.0, 2.0]), indices_without_contrast=[0],
    )
    with pytest.raises(ValueError, match="must hold zero directions"):
        probe.load_artifact("fake-model", "no_comma")


def _differences(n: int = 60, hidden: int = 5, separation: float = 5.0):
    """Pairs separated along axis 0, with unrelated noise elsewhere."""
    generator = torch.Generator().manual_seed(0)
    differences = 0.3 * torch.randn(n, hidden, generator=generator)
    differences[:, 0] += separation
    return differences


def test_first_component_is_the_axis_the_conditions_separate_along():
    """The paper alternates signs so the mean is near zero and the leading
    component is the separation itself; subtracting a same-signed mean would
    delete exactly that and leave the residual variation."""
    direction, explained = producer_lat_pca.first_component(_differences())

    assert torch.linalg.vector_norm(direction) == pytest.approx(1.0, abs=1e-5)
    assert abs(direction[0]) > 0.95
    assert explained > 0.9


def test_alternating_signs_flips_every_other_difference():
    differences = torch.ones(4, 2)
    alternated = producer_lat_pca.alternate_signs(differences)
    assert alternated[0].tolist() == [1.0, 1.0]
    assert alternated[1].tolist() == [-1.0, -1.0]
    assert alternated[2].tolist() == [1.0, 1.0]
    assert alternated[3].tolist() == [-1.0, -1.0]


def test_the_direction_points_the_way_the_positives_lie():
    """A principal axis is sign-free; steering is not. The vote is over pairs,
    so one extreme pair cannot decide the direction."""
    differences = _differences()

    direction, _ = producer_lat_pca.first_component(differences)
    assert float(((differences @ direction) > 0).float().mean()) > 0.5

    flipped, _ = producer_lat_pca.first_component(-differences)
    assert float((((-differences) @ flipped) > 0).float().mean()) > 0.5


def test_first_component_rejects_input_it_cannot_decompose():
    with pytest.raises(ValueError, match="at least two difference rows"):
        producer_lat_pca.first_component(torch.ones(1, 4))
    # Identical differences are not degenerate once signs alternate — they
    # become +d, -d, +d and vary along d. Only all-zero rows have no axis.
    with pytest.raises(ValueError, match="no variance"):
        producer_lat_pca.first_component(torch.zeros(5, 4))


def test_delta_is_alpha_sigma_along_the_unit_direction__lat_pca():
    artifact = {
        "direction": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "sigma": torch.tensor([1.0, 4.0]),
    }
    delta = lat_pca.delta(artifact, {"layer": 1, "alpha": 0.5})
    assert torch.allclose(delta, torch.tensor([0.0, 2.0]))


def test_dose_metadata_names_this_method():
    artifact = {
        "direction": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "sigma": torch.tensor([1.0, 2.0]),
    }
    dose = lat_pca.dose_metadata(artifact, {"layer": 1, "alpha": 1.0})
    assert dose["method"] == "lat_pca"
    assert dose["effective_delta_norm"] == pytest.approx(2.0)


def _artifact_lat_pca(tmp_path, monkeypatch, **overrides) -> None:
    monkeypatch.setattr(
        lat_pca, "artifact_path",
        lambda model_slug, task: tmp_path / model_slug / f"{task}.pt",
    )
    record = {
        "direction": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "sigma": torch.tensor([1.0, 2.0]),
        "extraction_position": "last_prompt_token",
        "method": "lat_pca_first_component",
        "task": "no_comma",
        "model": "org/Fake-Model",
        "producer_config": {"construction": producer_lat_pca.CONSTRUCTION},
    }
    record.update(overrides)
    path = tmp_path / "fake-model" / "no_comma.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, path)


def test_load_artifact_accepts_a_matching_record(tmp_path, monkeypatch):
    _artifact_lat_pca(tmp_path, monkeypatch)
    assert lat_pca.load_artifact("fake-model", "no_comma")["task"] == "no_comma"


def test_load_artifact_rejects_another_methods_artifact(tmp_path, monkeypatch):
    _artifact_lat_pca(tmp_path, monkeypatch, method="probe_logistic_regression")
    with pytest.raises(ValueError, match="is not lat_pca_first_component"):
        lat_pca.load_artifact("fake-model", "no_comma")


def test_search_plan_matches_the_reference_grid():
    plan = lat_pca.search_plan({"n_layers": 36})
    assert [(c["layer"], c["alpha"]) for c in plan.stages[0].configs] == [
        (9, 1.0), (9, 2.0), (9, 4.0),
        (21, 1.0), (21, 2.0), (21, 4.0),
        (33, 1.0), (33, 2.0), (33, 4.0),
    ]


def test_each_method_writes_to_its_own_artifact_directory():
    assert lat_pca.artifact_path("qwen3-8b", "no_comma").parent.name == "lat_pca"


def test_load_artifact_refuses_one_built_by_an_earlier_recipe(
        tmp_path, monkeypatch):
    """The direction changed while the method name did not, and nothing in the
    tensor shows which recipe made it."""
    _artifact_lat_pca(
        tmp_path, monkeypatch,
        producer_config={"construction": "centred differences"},
    )
    with pytest.raises(ValueError, match="produce it again"):
        lat_pca.load_artifact("fake-model", "no_comma")


def _hidden(seq: int, batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, seq, 2)


def _added(result: torch.Tensor) -> list[float]:
    """How much was added at each position of a single-row batch."""
    return result[0, :, 0].tolist()


def test_the_default_reaches_the_first_generated_token():
    """That token is written from the final prompt position, so a rule that
    starts after the prompt never touches it — and one capital letter fails
    the lowercase task however well the rest is steered."""
    mask = positions.PositionMask(positions.FROM_LAST_PROMPT_TOKEN)

    prefill = mask.apply(_hidden(5), DELTA)
    assert _added(prefill) == [0.0, 0.0, 0.0, 0.0, 1.0]

    for _ in range(3):                              # cached decoding steps
        assert _added(mask.apply(_hidden(1), DELTA)) == [1.0]


def test_generated_only_leaves_the_first_token_unsteered():
    """Kept as the comparison arm: it is what the benchmark ran before, and
    the difference between the two is the first token's causal weight."""
    mask = positions.PositionMask(positions.GENERATED_ONLY)

    assert _added(mask.apply(_hidden(5), DELTA)) == [0.0] * 5
    assert _added(mask.apply(_hidden(1), DELTA)) == [1.0]


def test_the_default_holds_when_generation_runs_without_a_cache():
    """Each pass then resends the whole sequence; everything before the final
    prompt position must still be untouched, or the setting would silently
    depend on the cache."""
    mask = positions.PositionMask(positions.FROM_LAST_PROMPT_TOKEN)

    assert _added(mask.apply(_hidden(4), DELTA)) == [0, 0, 0, 1]
    assert _added(mask.apply(_hidden(6), DELTA)) == [0, 0, 0, 1, 1, 1]


def test_last_prompt_token_touches_one_position_and_never_again():
    mask = positions.PositionMask(positions.LAST_PROMPT_TOKEN)

    assert _added(mask.apply(_hidden(4), DELTA)) == [0.0, 0.0, 0.0, 1.0]
    assert _added(mask.apply(_hidden(1), DELTA)) == [0.0]
    assert _added(mask.apply(_hidden(1), DELTA)) == [0.0]


def test_all_positions_mode_reproduces_the_earlier_behaviour():
    mask = positions.PositionMask(positions.ALL)

    assert _added(mask.apply(_hidden(3), DELTA)) == [1.0, 1.0, 1.0]
    assert _added(mask.apply(_hidden(1), DELTA)) == [1.0]


def test_every_row_of_a_batch_is_treated_alike():
    mask = positions.PositionMask(positions.FROM_LAST_PROMPT_TOKEN)
    mask.apply(_hidden(3, batch=4), DELTA)

    result = mask.apply(_hidden(1, batch=4), DELTA)
    assert result[:, :, 0].tolist() == [[1.0]] * 4


def test_the_default_is_the_smallest_set_reaching_every_output_token():
    assert positions.DEFAULT == positions.FROM_LAST_PROMPT_TOKEN


def test_an_unknown_mode_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="unknown position mode"):
        positions.PositionMask("everywhere")


def test_addition_broadcasts_over_a_per_head_layout():
    """ITI hooks a tensor whose last dimension packs every head together."""
    mask = positions.PositionMask(positions.FROM_LAST_PROMPT_TOKEN)
    packed_delta = torch.zeros(8)
    packed_delta[2:4] = 3.0

    mask.apply(torch.zeros(1, 5, 8), packed_delta)
    result = mask.apply(torch.zeros(1, 1, 8), packed_delta)
    assert result[0, 0].tolist() == [0, 0, 3, 3, 0, 0, 0, 0]


@pytest.mark.parametrize("key", ["caa", "probe", "lat_pca", "iti", "sae"])
def test_every_registered_baseline_resolves_its_producer(key):
    """The CLI reaches each producer, which no unit test used to exercise:
    a rename left this path naming a variable that no longer existed, and
    nothing failed until a real run tried to produce an artifact."""
    resolved = produce.producer_for(baselines.get(key))
    assert callable(resolved)
    assert resolved.__name__ == "produce_task"


def test_producer_lookup_names_the_baseline_that_lacks_one(monkeypatch):
    """The registered module has to be importable for the missing-producer
    branch to be reachable at all; without that the lookup died earlier, on
    the import, and any exception satisfied the test."""
    import sys
    from types import ModuleType

    module = ModuleType("baselines.nothing")
    module.KEY = "nothing"
    monkeypatch.setitem(sys.modules, "baselines.nothing", module)
    monkeypatch.setitem(
        sys.modules, "baselines.nothing.producer", ModuleType("empty")
    )

    with pytest.raises(SystemExit, match="has no producer.produce_task"):
        produce.producer_for(module)
