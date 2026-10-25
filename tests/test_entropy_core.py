import numpy as np
import pytest

from albench.al.entropy_core import (
    binary_entropy,
    build_minority_quotas,
    divproto_select,
    make_entropy_candidate,
)


def test_binary_entropy_is_maximal_at_half():
    values = binary_entropy(np.array([0.1, 0.5, 0.9]))

    assert values[1] == pytest.approx(np.log(2.0))
    assert values[1] > values[0]
    assert values[1] > values[2]


def test_binary_entropy_is_exact_at_probability_boundaries():
    assert binary_entropy(np.array([0.0, 1.0])).tolist() == [0.0, 0.0]


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf])
def test_binary_entropy_rejects_non_probabilities(value):
    with pytest.raises(ValueError, match="probabilities"):
        binary_entropy(np.array([value]))


def test_make_candidate_applies_enms_to_image_score():
    candidate = make_entropy_candidate(
        path="image.jpg",
        labels=np.array([0, 0, 1]),
        confidences=np.array([0.50, 0.55, 0.40]),
        features=np.array(
            [
                [1.00, 0.00],
                [0.99, 0.01],
                [0.00, 1.00],
            ]
        ),
    )

    expected = binary_entropy(np.array([0.50, 0.40])).sum()
    assert candidate.score == pytest.approx(expected)
    assert set(candidate.prototypes) == {0, 1}
    assert candidate.presence == {0: 0.55, 1: 0.40}
    assert candidate.detection_count == 3
    assert candidate.representative_count == 2


def test_make_candidate_rejects_undefined_cosine_feature():
    with pytest.raises(ValueError, match="non-zero norm"):
        make_entropy_candidate(
            path="image.jpg",
            labels=np.array([0]),
            confidences=np.array([0.5]),
            features=np.zeros((1, 2)),
        )


def test_make_candidate_rejects_zero_entropy_prototype_mass():
    with pytest.raises(ValueError, match=r"Eq\. \(3\) is undefined"):
        make_entropy_candidate(
            path="image.jpg",
            labels=np.array([0]),
            confidences=np.array([1.0]),
            features=np.array([[1.0, 0.0]]),
        )


def test_paper_exact_minority_count_for_eight_classes():
    quotas = build_minority_quotas(
        labeled_counts=np.arange(8),
        budget=8,
        nc=8,
    )

    assert set(quotas) == {0, 1, 2, 3}


def test_paper_default_alpha_rejects_five_classes():
    with pytest.raises(ValueError, match=r"alpha \* nc must be an integer"):
        build_minority_quotas(
            labeled_counts=np.arange(5),
            budget=5,
            nc=5,
        )


def test_divproto_returns_exact_unique_budget():
    candidates = []
    for index in range(10):
        feature = np.zeros((1, 8), dtype=float)
        feature[0, index % 8] = 1.0
        candidates.append(
            make_entropy_candidate(
                path=f"image_{index}.jpg",
                labels=np.array([index % 8]),
                confidences=np.array([0.45 + 0.01 * (index % 3)]),
                features=feature,
            )
        )

    picks = divproto_select(
        candidates=candidates,
        labeled_counts=np.arange(8),
        budget=4,
        nc=8,
    )

    assert len(picks) == 4
    assert len(set(picks)) == 4
