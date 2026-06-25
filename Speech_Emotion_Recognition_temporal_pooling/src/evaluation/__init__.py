"""Métricas, reporting y diagnósticos de evaluación."""

from src.evaluation.diagnostics import (
    build_oof_diagnostics,
    compute_actor_class_recall,
    compute_actor_metrics,
    compute_fold_metrics_from_predictions,
)
from src.evaluation.metrics import (
    METRIC_NAMES,
    compute_8_to_4_metrics,
    compute_cv_summary,
    compute_metrics,
    full_classification_report,
    get_confusion_matrix,
    map_emotions_to_quadrants,
)
from src.evaluation.reporting import save_cv_results, summarize_cv_results

__all__ = [
    "METRIC_NAMES",
    "build_oof_diagnostics",
    "compute_8_to_4_metrics",
    "compute_actor_class_recall",
    "compute_actor_metrics",
    "compute_cv_summary",
    "compute_fold_metrics_from_predictions",
    "compute_metrics",
    "full_classification_report",
    "get_confusion_matrix",
    "map_emotions_to_quadrants",
    "save_cv_results",
    "summarize_cv_results",
]
