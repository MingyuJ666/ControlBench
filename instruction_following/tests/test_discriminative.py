"""Selecting for one instruction rather than for instructions in general."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import baselines  # noqa: E402
from baselines.shared.discriminative import (discriminative_separation,  # noqa: E402
                                             other_task_mean, standardized_gap)
from baselines.tasks_order import other_tasks  # noqa: E402


def test_a_unit_that_answers_to_every_instruction_scores_zero():
    """The failure this exists for: the SAE's ranking chose one feature for
    four of five tasks, because that feature reports that an instruction is
    present rather than which one."""
    general = torch.full((8, 1), 5.0)                 # fires the same for all
    specific = torch.cat([torch.full((8, 1), 5.0), torch.zeros(8, 1)], dim=1)

    target = torch.cat([general, specific[:, 1:]], dim=1)
    target[:, 1] = 9.0                                # specific unit fires here
    others = [torch.tensor([[5.0, 0.0]]).repeat(8, 1) for _ in range(4)]

    scores = discriminative_separation(target, others)

    assert scores[0] == pytest.approx(0.0)            # the general unit
    assert scores[1] == pytest.approx(9.0)            # the specific one
    assert int(torch.argmax(scores)) == 1


def test_every_other_task_counts_equally():
    """Pooling rows would let one very responsive rival dominate; the score
    averages tasks so all four constrain the choice."""
    target = torch.full((4, 1), 10.0)
    loud = torch.full((100, 1), 8.0)
    quiet = [torch.full((2, 1), 0.0) for _ in range(3)]

    assert float(other_task_mean([loud, *quiet])) == pytest.approx(2.0)


def test_the_effect_size_keeps_ordering_after_accuracy_saturates():
    """Two heads that both separate perfectly are indistinguishable to a
    proportion correct, and to AUC; they are not to an effect size."""
    generator = torch.Generator().manual_seed(0)
    narrow = torch.randn(50, 1, generator=generator) * 0.01
    wide = torch.randn(50, 1, generator=generator) * 0.01

    barely = standardized_gap(narrow + 0.1, wide - 0.1)
    clearly = standardized_gap(narrow + 8.0, wide - 8.0)

    assert float(clearly) > float(barely) * 10


def test_the_contrast_set_is_every_other_task_in_declared_order():
    assert other_tasks("no_comma") == [
        "lowercase", "fixed_ending", "question_ending", "one_sentence"
    ]
    with pytest.raises(ValueError, match="unknown task"):
        other_tasks("not_a_task")


@pytest.mark.parametrize("variant,parent", [("sae_disc", "sae"),
                                            ("iti_disc", "iti")])
def test_a_variant_shares_everything_but_the_selection(variant, parent,
                                                       monkeypatch):
    """The comparison is only meaningful if the two differ in one thing: the
    same sites, the same grid, the same dose, a different chosen unit."""
    a, b = baselines.get(variant), baselines.get(parent)
    # The sae plan reads its gate report from the data root, which holds run
    # outputs and is empty on a fresh clone. The claim under test, that the
    # two variants request one identical plan, does not depend on which
    # layers the gate admitted.
    monkeypatch.setattr(
        baselines.get("sae"), "passing_layers",
        lambda model_slug, tag="chat": [21, 27, 33],
    )
    context = {"n_layers": 36, "n_heads_total": 1152,
               "model_slug": "qwen3-8b", "task": "no_comma"}

    assert a.search_plan(context).as_record() == b.search_plan(context).as_record()
    assert a.ALPHAS == b.ALPHAS
    assert a.DEFAULT_BUDGET == b.DEFAULT_BUDGET


@pytest.mark.parametrize("variant,parent", [("sae_disc", "sae"),
                                            ("iti_disc", "iti")])
def test_a_variant_never_writes_where_its_parent_wrote(variant, parent):
    """Both rankings are results; neither may overwrite the other."""
    a, b = baselines.get(variant), baselines.get(parent)
    first = a.artifact_path("qwen3-8b", "no_comma")
    second = b.artifact_path("qwen3-8b", "no_comma")

    assert first != second
    assert first.parent.name == variant
    assert second.parent.name == parent
    assert a.ARTIFACT_METHOD != b.ARTIFACT_METHOD
