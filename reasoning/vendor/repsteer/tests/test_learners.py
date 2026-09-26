from types import SimpleNamespace

import torch

from repsteer.artifacts import DirectionArtifact, ProbeArtifact, SubspaceArtifact
from repsteer.capture import ActivationBatch, MemoryStore
from repsteer.data import ContrastivePairs
from repsteer.learners import (
    CAA,
    LAT,
    PCA,
    ActAdd,
    DiffMean,
    LinearProbe,
    lat_components,
    principal_components,
)
from repsteer.learners.pca import _orient_by_group_difference
from repsteer.positions import LastNonPaddingToken
from repsteer.sites import resid_post


class _CaptureModel:
    model_id = "tiny/capture"
    revision = "rev-1"
    config = SimpleNamespace(_commit_hash="rev-1")

    def __init__(self):
        self.calls = 0
        self.values = {
            "p1": torch.tensor([2.0, 0.0, 0.2]),
            "p2": torch.tensor([4.0, 0.2, 0.0]),
            "p3": torch.tensor([3.0, -0.1, 0.1]),
            "n1": torch.tensor([0.0, 0.0, -0.2]),
            "n2": torch.tensor([0.0, 0.2, 0.0]),
            "n3": torch.tensor([-1.0, -0.1, -0.1]),
        }

    def capture(self, request):
        self.calls += 1
        return ActivationBatch(
            torch.stack([self.values[value] for value in request.inputs]),
            request=request,
        )


def _data():
    return ContrastivePairs.from_records(
        [
            {"positive": "p1", "negative": "n1"},
            {"positive": "p2", "negative": "n2"},
            {"positive": "p3", "negative": "n3"},
        ]
    )


def _learner(learner_type, **kwargs):
    return learner_type(
        site=resid_post(0),
        positions=LastNonPaddingToken(),
        **kwargs,
    )


def test_direction_and_subspace_learners_have_deterministic_math():
    model = _CaptureModel()
    data = _data()

    diff = _learner(DiffMean).fit(model, data)
    actadd = _learner(ActAdd).fit(model, data)
    caa = _learner(CAA).fit(model, data)
    pca = _learner(PCA, n_components=2).fit(model, data)
    lat = _learner(LAT).fit(model, data)

    assert isinstance(diff, DirectionArtifact)
    assert isinstance(actadd, DirectionArtifact)
    assert isinstance(caa, DirectionArtifact)
    assert isinstance(pca, SubspaceArtifact) and pca.basis.shape == (2, 3)
    assert isinstance(lat, DirectionArtifact)

    # DiffMean keeps repsteer's own unit-norm convention (not a paper claim).
    assert torch.allclose(diff.direction.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.dot(diff.direction, torch.tensor([1.0, 0.0, 0.0])) > 0.99

    # ActAdd/CAA default to the raw paired mean-difference vector (no
    # normalization), matching the official CAA/ActAdd implementations. For
    # this equal-count paired fixture that raw vector equals
    # mean(positive) - mean(negative) exactly, i.e. the same axis as
    # DiffMean before DiffMean's own unit normalization.
    raw_expected = torch.tensor([10.0 / 3.0, 0.0, 0.2])
    assert torch.allclose(actadd.direction, raw_expected, atol=1e-6)
    assert torch.allclose(caa.direction, raw_expected, atol=1e-6)
    assert caa.metadata.method == "caa"
    assert not torch.allclose(actadd.direction.norm(), torch.tensor(1.0), atol=1e-3)
    assert torch.allclose(
        actadd.direction / actadd.direction.norm(), diff.direction, atol=1e-6
    )

    # normalize="l2" opts ActAdd/CAA back into DiffMean's unit-norm convention.
    actadd_unit = _learner(ActAdd, normalize="l2").fit(model, data)
    assert torch.allclose(actadd_unit.direction.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(actadd_unit.direction, diff.direction, atol=1e-6)

    assert diff.metadata.dataset_fingerprint == data.fingerprint
    assert diff.metadata.method == "diff_mean"
    assert pca.metadata.config["sign_alignment"] == "positive_negative_mean_difference"
    assert lat.metadata.config["pair_signing"] == "seeded_pair_shuffle"
    assert lat.metadata.config["sign_alignment"] == "positive_pairwise_vote"


def test_lat_emulates_official_shuffled_pair_order_before_centered_pca():
    positive = torch.tensor([[10.0, -3.0], [10.0, -1.0], [10.0, 1.0], [10.0, 3.0]])
    negative = torch.zeros_like(positive)

    shuffled, mean, _ = lat_components(
        positive,
        negative,
        pair_signs=torch.tensor([1.0, -1.0, 1.0, -1.0]),
    )
    preserved, _, _ = lat_components(
        positive,
        negative,
        shuffle_pair_order=False,
    )

    # Randomly swapping members inside official RepE pairs turns the common x
    # shift into PCA variance.  Keeping the original order instead leaves only
    # the y variation after PCA centering.
    assert torch.allclose(shuffled[0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(mean, torch.tensor([10.0, 0.0]), atol=1e-6)
    assert torch.allclose(preserved[0].abs(), torch.tensor([0.0, 1.0]), atol=1e-6)


def test_lat_uses_pairwise_label_votes_to_orient_components():
    positive = torch.tensor([[-10.0], [-9.0], [-8.0], [100.0]])
    negative = torch.zeros_like(positive)

    basis, _, _ = lat_components(
        positive,
        negative,
        shuffle_pair_order=False,
    )

    # Three of four positive activations fall below their paired negative
    # activation.  Official RepE resolves this PCA sign by pairwise label vote,
    # rather than by the magnitude-weighted mean difference (which is positive).
    assert torch.allclose(basis[0], torch.tensor([-1.0]), atol=1e-6)


def test_lat_keeps_official_raw_difference_scaling():
    positive = torch.tensor(
        [[100.0, 0.0], [-100.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, -1.0]]
    )
    negative = torch.zeros_like(positive)

    basis, _, _ = lat_components(
        positive,
        negative,
        shuffle_pair_order=False,
    )

    # Upstream PCA consumes raw differences.  Row-wise L2 normalization would
    # make the repeated y differences dominate, whereas the official estimator
    # correctly retains the larger x variation.
    assert torch.allclose(basis[0].abs(), torch.tensor([1.0, 0.0]), atol=1e-6)


def test_pca_orients_components_toward_the_positive_group():
    # The dominant pooled-variance axis is nearly the y-axis; its tiny x-tilt
    # is deterministically magnitude-signed (positive) by
    # canonicalize_component_signs regardless of the labels. That tilt
    # happens to disagree with mean(positive) - mean(negative), which points
    # toward negative x. A magnitude-only sign would silently keep the wrong
    # orientation for Add-style steering; the group-difference vote flips it.
    positive = torch.tensor([[-1.0, 9.0], [-1.0, -11.0], [-1.0, 4.0], [-1.0, -6.0]])
    negative = torch.tensor([[4.0, 12.0], [4.0, -8.0], [4.0, 6.0], [4.0, -4.0]])
    activations = torch.cat([positive, negative], dim=0)
    unit_weights = torch.ones(4)

    unsigned, _, _ = principal_components(activations, n_components=1)
    assert torch.allclose(
        unsigned[0], torch.tensor([0.0537038334, 0.9985568523]), atol=1e-6
    )

    oriented = _orient_by_group_difference(
        unsigned,
        positive,
        negative,
        positive_weights=unit_weights,
        negative_weights=unit_weights,
    )

    assert torch.allclose(
        oriented[0], torch.tensor([-0.0537038334, -0.9985568523]), atol=1e-6
    )
    group_difference = positive.mean(dim=0) - negative.mean(dim=0)
    assert torch.dot(oriented[0], group_difference) > 0

    # A tied (zero) alignment leaves the deterministic magnitude sign as-is.
    tied = _orient_by_group_difference(
        unsigned,
        positive,
        positive,
        positive_weights=unit_weights,
        negative_weights=unit_weights,
    )
    assert torch.allclose(tied, unsigned)


def test_linear_probe_separates_examples_and_capture_store_is_reused():
    model = _CaptureModel()
    data = _data()
    store = MemoryStore()
    learner = _learner(LinearProbe, max_iter=50)

    probe = learner.fit(model, data, store=store)
    calls_after_first_fit = model.calls
    repeated = learner.fit(model, data, store=store)

    assert isinstance(probe, ProbeArtifact)
    positives = torch.stack([model.values[value] for value in data.positives])
    negatives = torch.stack([model.values[value] for value in data.negatives])
    assert probe.probabilities(positives).mean() > probe.probabilities(negatives).mean()
    assert torch.equal(probe.weight, repeated.weight)
    assert model.calls == calls_after_first_fit == 2
