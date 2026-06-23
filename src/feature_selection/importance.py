"""Importancia de features eGeMAPSv02 calculada dentro de outer CV."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config.contracts import (
    FOLD_SENTINEL,
    PARTITION_DEVELOPMENT,
    TARGET_EMOTION_ORIGINAL,
    TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
)
from src.features.feature_store import prepare_model_table
from src.models import build_linear_probe, build_random_forest
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_feature_importance(
    representation: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    target_col: str,
    protocol: Literal["speaker_dependent", "speaker_independent"],
    random_forest_params: Mapping[str, Any],
    logistic_regression_params: Mapping[str, Any],
    seed: int,
    top_k: int = 15,
    permutation_repeats: int = 10,
    n_folds: int | None = None,
    fold_indices: Sequence[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Calcula tres estimadores de importancia sobre eGeMAPS por outer fold.

    Métodos
    -------
    - ``rf_impurity``: importancia Gini del Random Forest ajustado en outer train.
    - ``rf_permutation``: caída de macro F1 al permutar outer validation.
    - ``logreg_coefficient``: media del coeficiente absoluto entre clases.

    Para 8→4 se estudian las features del clasificador entrenado en ocho
    emociones, porque esa es la tarea que aprende el modelo.
    """
    if top_k < 1:
        raise ValueError("top_k debe ser positivo.")
    if permutation_repeats < 1:
        raise ValueError("permutation_repeats debe ser positivo.")

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
    if fold_indices is not None:
        requested = set(int(value) for value in fold_indices)
        unknown = requested - set(folds)
        if unknown:
            raise ValueError(f"Folds inexistentes solicitados: {sorted(unknown)}")
        folds = [fold for fold in folds if fold in requested]

    records: list[dict[str, Any]] = []

    for fold_idx in folds:
        val_mask = table[fold_col] == fold_idx
        train_mask = table[fold_col] != fold_idx

        X_train = table.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
        X_val = table.loc[val_mask, feature_cols].to_numpy(dtype=np.float32)
        y_train = table.loc[train_mask, train_target].to_numpy()
        y_val = table.loc[val_mask, train_target].to_numpy()

        logger.info(
            "Feature importance: %s | %s | fold=%s",
            protocol,
            target_col,
            fold_idx,
        )

        random_forest = build_random_forest(
            params=random_forest_params,
            seed=seed + fold_idx,
        )
        random_forest.fit(X_train, y_train)

        permutation = permutation_importance(
            estimator=random_forest,
            X=X_val,
            y=y_val,
            scoring="f1_macro",
            n_repeats=permutation_repeats,
            random_state=seed + fold_idx,
            n_jobs=1,
        )

        linear_probe = build_linear_probe(
            params=logistic_regression_params,
            seed=seed + fold_idx,
        )
        linear_probe.fit(X_train, y_train)
        coefficients = linear_probe.named_steps["classifier"].coef_
        coefficient_importance = np.mean(np.abs(coefficients), axis=0)

        method_values = {
            "rf_impurity": np.asarray(random_forest.feature_importances_),
            "rf_permutation": np.asarray(permutation.importances_mean),
            "logreg_coefficient": np.asarray(coefficient_importance),
        }

        for method, values in method_values.items():
            ranks = _descending_ranks(values)
            model_name = (
                "logistic_regression"
                if method == "logreg_coefficient"
                else "random_forest"
            )

            for feature, importance, rank in zip(feature_cols, values, ranks):
                records.append(
                    {
                        "protocol": protocol,
                        "target": target_col,
                        "model": model_name,
                        "method": method,
                        "refinement": "importance",
                        "fold": fold_idx,
                        "feature": feature,
                        "importance": float(importance),
                        "rank": int(rank),
                        "selected": bool(rank <= top_k),
                    }
                )

    fold_results = pd.DataFrame.from_records(records)
    summary = summarize_feature_importance(fold_results)

    frequency = summary[
        ["protocol", "target", "model", "method", "feature", "selection_frequency"]
    ]
    fold_results = fold_results.merge(
        frequency,
        on=["protocol", "target", "model", "method", "feature"],
        how="left",
        validate="many_to_one",
    )

    return {
        "fold_results": fold_results,
        "summary": summary,
    }


def summarize_feature_importance(results: pd.DataFrame) -> pd.DataFrame:
    """Agrega importancia, ranking y frecuencia de top-k entre folds."""
    if results.empty:
        return pd.DataFrame()

    group_cols = ["protocol", "target", "model", "method", "feature"]
    summary = (
        results.groupby(group_cols, observed=True)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            rank_mean=("rank", "mean"),
            rank_std=("rank", "std"),
            selection_frequency=("selected", "mean"),
        )
        .reset_index()
    )

    return summary.sort_values(
        ["protocol", "target", "method", "rank_mean", "feature"]
    ).reset_index(drop=True)


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    """Ranking 1..N descendente, estable ante empates."""
    return (
        pd.Series(values)
        .rank(method="min", ascending=False)
        .astype(int)
        .to_numpy()
    )
