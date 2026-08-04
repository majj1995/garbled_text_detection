import numpy as np
import pytest

from poor_word.evaluation.metrics import (
    MVP_GATE,
    TRIAL_GATE,
    base_rate_precision,
    evaluate_thresholds,
)


def test_trial_gate_precision_at_point_one_percent_base_rate() -> None:
    precision = base_rate_precision(prevalence=0.001, recall=0.50, fpr=0.000025)
    assert precision == pytest.approx(0.9524263, rel=1e-6)


def test_zero_fpr_has_perfect_precision_when_recall_is_positive() -> None:
    assert base_rate_precision(prevalence=0.001, recall=0.4, fpr=0.0) == 1.0


def test_threshold_report_uses_raw_counts_and_production_prevalence() -> None:
    # Two detected anomalies out of five and one false alarm out of 10,000 normals.
    labels = np.asarray([1] * 5 + [0] * 10_000, dtype=np.int64)
    scores = np.asarray(
        [0.99, 0.96, 0.001, 0.001, 0.001]
        + [0.97]
        + [0.01] * 9_999,
        dtype=np.float64,
    )

    evaluation = evaluate_thresholds(labels, scores, prevalence=0.001)

    assert evaluation.mvp_gate.spec == MVP_GATE
    assert evaluation.mvp_gate.passed is True
    assert evaluation.mvp_gate.point.tp == 2
    assert evaluation.mvp_gate.point.fn == 3
    assert evaluation.mvp_gate.point.fp == 1
    assert evaluation.mvp_gate.point.tn == 9_999
    assert evaluation.mvp_gate.point.recall == pytest.approx(0.4)
    assert evaluation.mvp_gate.point.fpr == pytest.approx(0.0001)
    assert evaluation.mvp_gate.point.production_precision == pytest.approx(
        base_rate_precision(0.001, 0.4, 0.0001)
    )
    assert evaluation.trial_gate.spec == TRIAL_GATE
    assert evaluation.trial_gate.passed is False


def test_no_prediction_point_is_available_as_closest_gate_fallback() -> None:
    evaluation = evaluate_thresholds([0, 1], [0.9, 0.8], prevalence=0.001)

    assert evaluation.mvp_gate.passed is False
    assert evaluation.mvp_gate.point.threshold > 0.9
    assert evaluation.mvp_gate.point.tp == 0
    assert evaluation.mvp_gate.point.fp == 0
    assert evaluation.mvp_gate.normalized_constraint_violation == pytest.approx(1.0)


def test_small_zero_false_positive_set_is_inconclusive_not_passed() -> None:
    evaluation = evaluate_thresholds([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], prevalence=0.001)

    assert evaluation.mvp_gate.threshold_requirements_met is True
    assert evaluation.mvp_gate.sample_support_sufficient is False
    assert evaluation.mvp_gate.status == "inconclusive"
    assert evaluation.mvp_gate.passed is False


@pytest.mark.parametrize(
    ("prevalence", "recall", "fpr"),
    [(-0.1, 0.5, 0.1), (1.1, 0.5, 0.1), (0.1, -0.1, 0.1), (0.1, 0.5, 1.1)],
)
def test_base_rate_precision_rejects_invalid_probabilities(
    prevalence: float, recall: float, fpr: float
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        base_rate_precision(prevalence, recall, fpr)


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0, 1], [0.1], "equal non-zero length"),
        ([0, 2], [0.1, 0.2], "only 0 and 1"),
        ([0, 0], [0.1, 0.2], "both positive and negative"),
        ([0, 1], [0.1, float("nan")], "finite"),
    ],
)
def test_threshold_evaluation_rejects_invalid_arrays(
    labels: list[int], scores: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_thresholds(labels, scores, prevalence=0.001)
