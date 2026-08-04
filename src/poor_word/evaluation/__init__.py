"""Base-rate-aware evaluation and reproducible MVP reporting."""

from poor_word.evaluation.metrics import base_rate_precision, evaluate_thresholds
from poor_word.evaluation.report import build_mvp_report, write_mvp_report

__all__ = [
    "base_rate_precision",
    "build_mvp_report",
    "evaluate_thresholds",
    "write_mvp_report",
]
