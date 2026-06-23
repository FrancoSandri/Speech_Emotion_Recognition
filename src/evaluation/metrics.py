"""Métricas globales y consistentes para los experimentos SER."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.config.contracts import EMOTION_TO_QUADRANT
from src.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_NAMES = (
    "macro_f1",
    "balanced_accuracy",
    "uar",
    "weighted_f1",
    "accuracy",
)


def compute_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence | None = None,
) -> dict[str, float]:
    """Calcula el conjunto estándar de métricas usado por todos los modelos."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true e y_pred deben ser vectores unidimensionales.")
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener igual longitud y no estar vacíos.")

    labels_arg = list(labels) if labels is not None else None

    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels_arg,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "uar": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels_arg,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels_arg,
                average="weighted",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def map_emotions_to_quadrants(emotions: Sequence) -> np.ndarray:
    """Mapea etiquetas emocionales RAVDESS a cuadrantes de Russell."""
    emotions = np.asarray(emotions)
    unknown = sorted(set(emotions) - set(EMOTION_TO_QUADRANT))
    if unknown:
        raise ValueError(f"Emociones sin mapping a cuadrante: {unknown}")

    return np.asarray([EMOTION_TO_QUADRANT[value] for value in emotions])


def compute_8_to_4_metrics(
    y_true_quadrant: Sequence,
    y_pred_emotion: Sequence,
    quadrant_labels: Sequence[str] = ("Q1", "Q2", "Q3", "Q4"),
) -> tuple[dict[str, float], np.ndarray]:
    """
    Evalúa el escenario 8→4: predicciones de emoción se colapsan a cuadrante.

    Devuelve las métricas y las predicciones ya mapeadas.
    """
    y_pred_quadrant = map_emotions_to_quadrants(y_pred_emotion)
    metrics = compute_metrics(
        y_true=np.asarray(y_true_quadrant),
        y_pred=y_pred_quadrant,
        labels=quadrant_labels,
    )
    return metrics, y_pred_quadrant


def compute_cv_summary(
    fold_metrics: list[dict[str, float]],
    metric_names: Sequence[str] = METRIC_NAMES,
) -> dict[str, Any]:
    """Agrega media, desvío estándar, mínimo y máximo por métrica."""
    if not fold_metrics:
        return {}

    summary: dict[str, float] = {}
    for metric in metric_names:
        values = np.asarray(
            [row[metric] for row in fold_metrics if metric in row],
            dtype=float,
        )
        if len(values) == 0:
            continue

        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
        summary[f"{metric}_min"] = float(values.min())
        summary[f"{metric}_max"] = float(values.max())

    return summary


def full_classification_report(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence[str] | None = None,
    as_dataframe: bool = False,
) -> str | pd.DataFrame:
    """Genera el reporte completo por clase."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels) if labels is not None else None,
        output_dict=as_dataframe,
        zero_division=0,
    )
    if as_dataframe:
        return pd.DataFrame(report).T.round(3)
    return report


def get_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Devuelve una matriz de confusión etiquetada."""
    resolved_labels = (
        list(labels)
        if labels is not None
        else sorted(set(y_true) | set(y_pred))
    )
    matrix = confusion_matrix(y_true, y_pred, labels=resolved_labels)
    return pd.DataFrame(matrix, index=resolved_labels, columns=resolved_labels)


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    """Registra las métricas principales en una línea."""
    label = f"[{prefix}] " if prefix else ""
    logger.info(
        "%smacro_F1=%.4f | BalAcc=%.4f | UAR=%.4f | Acc=%.4f",
        label,
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
        metrics["uar"],
        metrics["accuracy"],
    )
