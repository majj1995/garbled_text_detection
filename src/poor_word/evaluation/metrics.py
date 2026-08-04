import math
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]


class GateSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    max_fpr: float = Field(ge=0.0, le=1.0)
    min_recall: float = Field(ge=0.0, le=1.0)


MVP_GATE = GateSpec(name="offline_mvp", max_fpr=0.0001, min_recall=0.40)
TRIAL_GATE = GateSpec(name="pilot_trial", max_fpr=0.000025, min_recall=0.50)


class ThresholdPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    threshold: float
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    tn: int = Field(ge=0)
    fn: int = Field(ge=0)
    recall: float = Field(ge=0.0, le=1.0)
    fpr: float = Field(ge=0.0, le=1.0)
    production_precision: float = Field(ge=0.0, le=1.0)


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec: GateSpec
    passed: bool
    status: Literal["pass", "fail", "inconclusive"]
    threshold_requirements_met: bool
    sample_support_sufficient: bool
    required_negative_count: int = Field(gt=0)
    point: ThresholdPoint
    normalized_constraint_violation: float = Field(ge=0.0)


class ThresholdEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    prevalence: float = Field(gt=0.0, lt=1.0)
    positive_count: int = Field(gt=0)
    negative_count: int = Field(gt=0)
    auroc: float | None = Field(default=None, ge=0.0, le=1.0)
    aucpr: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_table: tuple[ThresholdPoint, ...]
    mvp_gate: GateResult
    trial_gate: GateResult


def _probability(name: str, value: float) -> float:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def base_rate_precision(prevalence: float, recall: float, fpr: float) -> float:
    """Calculate PPV at an explicit deployment prevalence.

    This is deliberately separate from precision observed on a sampled or balanced
    evaluation set. The latter does not estimate production alert quality when the
    anomaly base rate is 0.1%.
    """
    prevalence = _probability("prevalence", prevalence)
    recall = _probability("recall", recall)
    fpr = _probability("fpr", fpr)
    true_positive_mass = prevalence * recall
    false_positive_mass = (1.0 - prevalence) * fpr
    denominator = true_positive_mass + false_positive_mass
    return true_positive_mass / denominator if denominator > 0.0 else 0.0


def _validated_arrays(
    labels: NDArray[np.int64] | list[int],
    scores: NDArray[np.float64] | list[float],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    label_values = np.asarray(labels, dtype=np.int64)
    score_values = np.asarray(scores, dtype=np.float64)
    if label_values.ndim != 1 or score_values.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if len(label_values) != len(score_values) or len(label_values) == 0:
        raise ValueError("labels and scores must have equal non-zero length")
    if not np.all(np.isin(label_values, (0, 1))):
        raise ValueError("labels must contain only 0 and 1")
    if not np.all(np.isfinite(score_values)):
        raise ValueError("scores must be finite")
    positive_count = int(label_values.sum())
    if positive_count == 0 or positive_count == len(label_values):
        raise ValueError("labels must contain both positive and negative samples")
    return label_values, score_values


def _threshold_points(
    labels: NDArray[np.int64], scores: NDArray[np.float64], prevalence: float
) -> tuple[ThresholdPoint, ...]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)
    cumulative_fp = np.cumsum(1 - sorted_labels, dtype=np.int64)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], np.asarray([True])]
    )
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    sentinel = float(np.nextafter(sorted_scores[0], np.inf))
    if not np.isfinite(sentinel):
        raise ValueError("scores are too large to construct a no-prediction threshold")
    points: list[ThresholdPoint] = [
        ThresholdPoint(
            threshold=sentinel,
            tp=0,
            fp=0,
            tn=negative_count,
            fn=positive_count,
            recall=0.0,
            fpr=0.0,
            production_precision=0.0,
        )
    ]
    for index in group_ends:
        tp = int(cumulative_tp[index])
        fp = int(cumulative_fp[index])
        recall = tp / positive_count
        fpr = fp / negative_count
        points.append(
            ThresholdPoint(
                threshold=float(sorted_scores[index]),
                tp=tp,
                fp=fp,
                tn=negative_count - fp,
                fn=positive_count - tp,
                recall=recall,
                fpr=fpr,
                production_precision=base_rate_precision(prevalence, recall, fpr),
            )
        )
    return tuple(points)


def _gate_result(
    points: tuple[ThresholdPoint, ...], spec: GateSpec, negative_count: int
) -> GateResult:
    def violation(point: ThresholdPoint) -> float:
        fpr_violation = max(0.0, point.fpr / spec.max_fpr - 1.0)
        recall_violation = max(0.0, spec.min_recall - point.recall) / spec.min_recall
        return fpr_violation + recall_violation

    passing = [
        point
        for point in points
        if point.fpr <= spec.max_fpr and point.recall >= spec.min_recall
    ]
    required_negative_count = math.ceil(1.0 / spec.max_fpr)
    sample_support_sufficient = negative_count >= required_negative_count
    if passing:
        selected = min(
            passing,
            key=lambda point: (-point.recall, point.fpr, -point.threshold),
        )
        return GateResult(
            spec=spec,
            passed=sample_support_sufficient,
            status="pass" if sample_support_sufficient else "inconclusive",
            threshold_requirements_met=True,
            sample_support_sufficient=sample_support_sufficient,
            required_negative_count=required_negative_count,
            point=selected,
            normalized_constraint_violation=0.0,
        )
    selected = min(
        points,
        key=lambda point: (violation(point), -point.recall, point.fpr, -point.threshold),
    )
    return GateResult(
        spec=spec,
        passed=False,
        status="fail",
        threshold_requirements_met=False,
        sample_support_sufficient=sample_support_sufficient,
        required_negative_count=required_negative_count,
        point=selected,
        normalized_constraint_violation=violation(selected),
    )


def evaluate_thresholds(
    labels: NDArray[np.int64] | list[int],
    scores: NDArray[np.float64] | list[float],
    *,
    prevalence: float,
) -> ThresholdEvaluation:
    prevalence = _probability("prevalence", prevalence)
    if prevalence in {0.0, 1.0}:
        raise ValueError("prevalence must be strictly between 0 and 1")
    label_values, score_values = _validated_arrays(labels, scores)
    points = _threshold_points(label_values, score_values, prevalence)
    return ThresholdEvaluation(
        prevalence=prevalence,
        positive_count=int(label_values.sum()),
        negative_count=int(len(label_values) - label_values.sum()),
        auroc=cast(float, roc_auc_score(label_values, score_values)),
        aucpr=cast(float, average_precision_score(label_values, score_values)),
        threshold_table=points,
        mvp_gate=_gate_result(points, MVP_GATE, int(len(label_values) - label_values.sum())),
        trial_gate=_gate_result(
            points, TRIAL_GATE, int(len(label_values) - label_values.sum())
        ),
    )
