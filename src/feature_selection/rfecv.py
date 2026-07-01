"""RFECV nested sobre eGeMAPSv02 y folds persistidos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedGroupKFold

from src.config.contracts import (
    FOLD_SENTINEL,
    PARTITION_DEVELOPMENT,
    PROTOCOL_INDEPENDENT,
    TARGET_EMOTION_ORIGINAL,
    TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
    TARGET_EMOTION_QUADRANT,
)
from src.evaluation.metrics import (
    compute_8_to_4_metrics,
    compute_cv_summary,
    compute_metrics,
    log_metrics,
)
from src.features.feature_store import prepare_model_table
from src.models import build_linear_probe
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_nested_rfecv(
    representation: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    target_col: str,
    logistic_regression_params: Mapping[str, Any],
    seed: int,
    inner_folds: int = 3,
    step: int = 5,
    min_features_to_select: int = 10,
    n_jobs: int = 1,
    n_folds: int | None = None,
    fold_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """
    Ejecuta RFECV dentro de cada outer train y evalúa outer validation.

    Los grupos internos son ``actor_id`` (protocolo speaker-independent).
    """
    if inner_folds < 2:
        raise ValueError("inner_folds debe ser mayor o igual que 2.")
    if step < 1:
        raise ValueError("step debe ser positivo.")
    if min_features_to_select < 1:
        raise ValueError("min_features_to_select debe ser positivo.")

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
    group_col = "actor_id"

    table, feature_cols = prepare_model_table(
        representation=representation,
        metadata=metadata,
        splits=splits,
        partition=PARTITION_DEVELOPMENT,
    )

    fold_col = "fold_speaker_independent"
    if (table[fold_col] == FOLD_SENTINEL).any():
        raise ValueError(f"Development contiene FOLD_SENTINEL en {fold_col}.")

    folds = sorted(int(value) for value in table[fold_col].unique())
    if n_folds is not None and len(folds) != n_folds:
        raise ValueError(
            f"Se esperaban {n_folds} folds y se encontraron {len(folds)}: {folds}"
        )
    if fold_indices is not None:
        requested = set(int(value) for value in fold_indices)
        unknown = requested - set(folds)
        if unknown:
            raise ValueError(f"Folds inexistentes solicitados: {sorted(unknown)}")
        folds = [fold for fold in folds if fold in requested]

    emotion_labels = sorted(table[TARGET_EMOTION_ORIGINAL].unique())
    quadrant_labels = ["Q1", "Q2", "Q3", "Q4"]
    direct_labels = emotion_labels if train_target == TARGET_EMOTION_ORIGINAL else quadrant_labels

    fold_metrics: list[dict[str, float]] = []
    fold_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_idx in folds:
        val_mask = table[fold_col] == fold_idx
        train_mask = table[fold_col] != fold_idx

        X_train = table.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
        X_val = table.loc[val_mask, feature_cols].to_numpy(dtype=np.float32)
        y_train = table.loc[train_mask, train_target].to_numpy()
        y_val_train_space = table.loc[val_mask, train_target].to_numpy()
        groups_train = table.loc[train_mask, group_col].to_numpy()

        _validate_inner_cv(y_train, groups_train, inner_folds)

        inner_cv = StratifiedGroupKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + fold_idx,
        )
        estimator = build_linear_probe(
            params=logistic_regression_params,
            seed=seed + fold_idx,
        )
        selector = RFECV(
            estimator=estimator,
            step=step,
            min_features_to_select=min(min_features_to_select, len(feature_cols)),
            cv=inner_cv,
            scoring="balanced_accuracy",
            importance_getter="named_steps.classifier.coef_",
            n_jobs=n_jobs,
        )

        logger.info(
            "RFECV: %s | outer fold=%s",
            target_col,
            fold_idx,
        )
        started = perf_counter()
        selector.fit(X_train, y_train, groups=groups_train)
        train_seconds = perf_counter() - started
        y_pred_train_space = selector.predict(X_val)

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
        log_metrics(metrics, prefix=f"RFECV fold {fold_idx}")

        row: dict[str, Any] = {
            "representation": "egemaps",
            "protocol": PROTOCOL_INDEPENDENT,
            "target": target_col,
            "model": "logistic_regression",
            "refinement": "rfecv",
            "fold": fold_idx,
            "n_input_features": len(feature_cols),
            "n_features": int(selector.n_features_),
            "n_train": len(X_train),
            "n_validation": len(X_val),
            "train_seconds": float(train_seconds),
            "inner_best_balanced_accuracy": float(np.max(selector.cv_results_["mean_test_score"])),
            **metrics,
        }
        if original_metrics is not None:
            row.update(
                {f"original_{name}": value for name, value in original_metrics.items()}
            )
        fold_rows.append(row)

        for feature, selected, rank in zip(
            feature_cols,
            selector.support_,
            selector.ranking_,
        ):
            feature_rows.append(
                {
                    "protocol": PROTOCOL_INDEPENDENT,
                    "target": target_col,
                    "model": "logistic_regression",
                    "method": "rfecv",
                    "refinement": "rfecv",
                    "fold": fold_idx,
                    "feature": feature,
                    "importance": np.nan,
                    "rank": int(rank),
                    "selected": bool(selected),
                }
            )

        prediction_frame.insert(
            0,
            "experiment",
            f"egemaps_logistic_regression_{target_col}_{PROTOCOL_INDEPENDENT}_rfecv",
        )
        prediction_frames.append(prediction_frame)

    fold_results = pd.DataFrame(fold_rows)
    feature_results = pd.DataFrame(feature_rows)
    frequencies = (
        feature_results.groupby(
            ["protocol", "target", "model", "method", "feature"],
            observed=True,
        )["selected"]
        .mean()
        .rename("selection_frequency")
        .reset_index()
    )
    feature_results = feature_results.merge(
        frequencies,
        on=["protocol", "target", "model", "method", "feature"],
        how="left",
        validate="many_to_one",
    )

    return {
        "fold_results": fold_results,
        "feature_results": feature_results,
        "summary": compute_cv_summary(fold_metrics),
        "predictions": pd.concat(prediction_frames, ignore_index=True),
    }


def _validate_inner_cv(y: np.ndarray, groups: np.ndarray, n_splits: int) -> None:
    """Falla temprano si los grupos internos no permiten el CV solicitado."""
    if len(np.unique(groups)) < n_splits:
        raise ValueError(
            f"Inner CV requiere {n_splits} grupos y solo hay {len(np.unique(groups))}."
        )

    group_target = pd.DataFrame({"target": y, "group": groups}).drop_duplicates()
    groups_per_class = group_target.groupby("target", observed=True)["group"].nunique()
    insufficient = groups_per_class[groups_per_class < n_splits]
    if not insufficient.empty:
        raise ValueError(
            "No hay suficientes grupos por clase para inner CV: "
            f"{insufficient.to_dict()}"
        )
