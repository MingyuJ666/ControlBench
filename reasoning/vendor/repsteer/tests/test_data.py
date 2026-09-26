import pytest

from repsteer.data import ContrastivePairs


def test_contrastive_pairs_paired_weights_and_fingerprint_are_stable():
    left = ContrastivePairs.from_records(
        [
            {"positive": "p1", "negative": "n1", "weight": 2, "topic": "a"},
            {"positive": "p2", "negative": "n2", "weight": 1},
        ],
        metadata={"source": "unit"},
    )
    right = ContrastivePairs.from_records(
        [
            {"negative": "n1", "positive": "p1", "topic": "a", "weight": 2},
            {"negative": "n2", "positive": "p2", "weight": 1},
        ],
        metadata={"source": "unit"},
    )

    assert left.paired
    assert left.pair_weights == (2.0, 1.0)
    assert left.fingerprint == right.fingerprint
    assert str(left.fingerprint).startswith("sha256:")


def test_unpaired_groups_validate_ambiguous_and_missing_data():
    dataset = ContrastivePairs.from_groups(["p1", "p2"], ["n"])
    assert not dataset.paired
    assert len(dataset.positive_examples) == 2
    assert len(dataset.negative_examples) == 1

    with pytest.raises(ValueError, match="cannot be empty"):
        ContrastivePairs.from_records([])
    with pytest.raises(ValueError, match="equal length"):
        ContrastivePairs(positives=["p"], negatives=["n", "n2"], paired=True)
