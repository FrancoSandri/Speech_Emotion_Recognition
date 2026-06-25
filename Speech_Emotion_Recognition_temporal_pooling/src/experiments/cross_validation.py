"""Cross-validation común para todas las representaciones y modelos."""

from __future__ import annotations

from time import perf_counter
from collections.abc import Mapping
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from src.config.contracts import (
    FOLD_SENTINEL,
    PARTITION_DEVELOPMENT,
    TARGET_EMOTION_ORIGINAL,
    TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
    TARGET_EMOTION_QUADRANT,
)
from src.evaluation.metrics import (
    compute_8_to_4_metrics,
    compute_cv_summary,
    compute_metrics,
    log_metrics,
    map_emotions_to_quadrants,
)
from src.features.feature_store import prepare_model_table
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_cv(
    pipeline_factory: Callable[[], Any],
    representation: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    target_col: str,
    protocol: Literal["speaker_dependent", "speaker_independent"],
    representation_name: str,
    model_name: str,
    refinement: str = "none",
    n_folds: int | None = None,
    experiment_name: str = "",
    fitted_metadata_fn: Callable[[Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Ejecuta CV usando exclusivamente los folds persistidos de development.

    ``pipeline_factory`` debe devolver un estimador nuevo por fold. Scaler,
    PCA u otras transformaciones aprendidas deben vivir dentro de ese pipeline.

    Para ``emotion_original_eval_quadrant`` el modelo se entrena en ocho
    emociones, conserva métricas de ocho clases y usa como métricas principales
    las predicciones colapsadas a cuatro cuadrantes.
    """
    valid_targets = {
        TARGET_EMOTION_ORIGINAL,
        TARGET_EMOTION_QUADRANT,
        TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
    }
    if target_col not in valid_targets:
        raise ValueError(f"Target no soportado: {target_col!r}")

    train_target = (
        TARGET_EMOTION_ORIGINAL
        if target_col == TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT
        else target_col
    )

    table, feature_cols = prepare_model_table(
        representation=representation,
        metadata=metadata,
        splits=splits,
        partition=PARTITION_DEVELOPMENT,
    )

    fold_col = f"fold_{protocol}"
    if fold_col not in table.columns:
        raise KeyError(f"Columna de folds inexistente: {fold_col!r}")
    if (table[fold_col] == FOLD_SENTINEL).any():
        raise ValueError(f"Development contiene FOLD_SENTINEL en {fold_col}.")

    folds = sorted(int(value) for value in table[fold_col].unique())
    if n_folds is not None and len(folds) != n_folds:
        raise ValueError(
            f"Se esperaban {n_folds} folds y se encontraron {len(folds)}: {folds}"
        )

    emotion_labels = sorted(table[TARGET_EMOTION_ORIGINAL].unique())
    quadrant_labels = ["Q1", "Q2", "Q3", "Q4"]
    direct_labels = emotion_labels if train_target == TARGET_EMOTION_ORIGINAL else quadrant_labels

    label = experiment_name or (
        f"{representation_name}_{model_name}_{target_col}_{protocol}_{refinement}"
    )
    logger.info("=== CV: %s ===", label)

    fold_metrics: list[dict[str, float]] = []
    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_idx in folds:
        val_mask = table[fold_col] == fold_idx
        train_mask = table[fold_col] != fold_idx

        X_train = table.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
        X_val = table.loc[val_mask, feature_cols].to_numpy(dtype=np.float32)
        y_train = table.loc[train_mask, train_target].to_numpy()
        y_val_train_space = table.loc[val_mask, train_target].to_numpy()

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(f"Fold {fold_idx} vacío para {protocol}.")

        pipeline = pipeline_factory()
        started = perf_counter()
        pipeline.fit(X_train, y_train)
        train_seconds = perf_counter() - started
        y_pred_train_space = pipeline.predict(X_val)

        prediction_frame = pd.DataFrame(
            {
                "file_id": table.loc[val_mask, "file_id"].to_numpy(),
                "fold": fold_idx,
                "y_true": y_val_train_space,
                "y_pred": y_pred_train_space,
            }
        )

        original_metrics: dict[str, float] | None = None
        if target_col == TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT:
            original_metrics = compute_metrics(
                y_val_train_space,
                y_pred_train_space,
                labels=emotion_labels,
            )
            metrics, y_pred_eval = compute_8_to_4_metrics(
                y_true_quadrant=table.loc[val_mask, TARGET_EMOTION_QUADRANT].to_numpy(),
                y_pred_emotion=y_pred_train_space,
                quadrant_labels=quadrant_labels,
            )
            prediction_frame["y_true_eval"] = table.loc[
                val_mask, TARGET_EMOTION_QUADRANT
            ].to_numpy()
            prediction_frame["y_pred_eval"] = y_pred_eval
        else:
            metrics = compute_metrics(
                y_val_train_space,
                y_pred_train_space,
                labels=direct_labels,
            )

        fold_metrics.append(metrics)
        log_metrics(metrics, prefix=f"Fold {fold_idx}")

        fitted_metadata = (
            dict(fitted_metadata_fn(pipeline))
            if fitted_metadata_fn is not None
            else {}
        )

        row: dict[str, Any] = {
            "representation": representation_name,
            "protocol": protocol,
            "target": target_col,
            "model": model_name,
            "refinement": refinement,
            "fold": fold_idx,
            "n_input_features": len(feature_cols),
            "n_features": len(feature_cols),
            "n_train": len(X_train),
            "n_validation": len(X_val),
            "train_seconds": float(train_seconds),
            **metrics,
            **fitted_metadata,
        }
        if original_metrics is not None:
            row.update(
                {f"original_{name}": value for name, value in original_metrics.items()}
            )

        result_rows.append(row)
        prediction_frame.insert(0, "experiment", label)
        prediction_frames.append(prediction_frame)

    summary = compute_cv_summary(fold_metrics)
    fold_results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    logger.info(
        "=== %s — macro_F1 %.4f ± %.4f ===",
        label,
        summary["macro_f1_mean"],
        summary["macro_f1_std"],
    )

    return {
        "experiment": label,
        "fold_metrics": fold_metrics,
        "fold_results": fold_results,
        "summary": summary,
        "predictions": predictions,
    }
