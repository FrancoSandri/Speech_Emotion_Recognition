"""Consolidación simple de resultados de cross-validation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import METRIC_NAMES
from src.utils.config import resolve_path

EXPERIMENT_COLUMNS = [
    "representation",
    "protocol",
    "target",
    "model",
    "refinement",
]


def summarize_cv_results(
    cv_results: pd.DataFrame,
    metric_names: Sequence[str] = METRIC_NAMES,
) -> pd.DataFrame:
    """Agrega media, desvío, mínimo y máximo por configuración."""
    if cv_results.empty:
        return pd.DataFrame()

    missing = set(EXPERIMENT_COLUMNS + ["fold"]) - set(cv_results.columns)
    if missing:
        raise KeyError(f"Faltan columnas en cv_results: {sorted(missing)}")

    available_metrics = [
        metric for metric in metric_names if metric in cv_results.columns
    ]
    available_metrics.extend(
        f"original_{metric}"
        for metric in metric_names
        if f"original_{metric}" in cv_results.columns
    )
    if not available_metrics:
        raise ValueError("cv_results no contiene métricas conocidas.")

    rows: list[dict] = []
    grouped = cv_results.groupby(EXPERIMENT_COLUMNS, observed=True, sort=False)

    for experiment_values, group in grouped:
        row = dict(zip(EXPERIMENT_COLUMNS, experiment_values))
        for metric in available_metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        ["protocol", "target", "macro_f1_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def save_cv_results(
    cv_results: pd.DataFrame,
    output_path: str | Path = "reports/cv_results.csv",
) -> Path:
    """Guarda todos los resultados por fold en un único CSV."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(path, index=False)
    return path


def upsert_cv_results(
    existing: pd.DataFrame,
    new_results: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega resultados reemplazando configuraciones/folds ya existentes."""
    if existing.empty:
        return new_results.copy().reset_index(drop=True)
    if new_results.empty:
        return existing.copy().reset_index(drop=True)

    key_cols = EXPERIMENT_COLUMNS + ["fold"]
    missing = set(key_cols) - set(existing.columns) | set(key_cols) - set(new_results.columns)
    if missing:
        raise KeyError(f"Faltan columnas para consolidar CV: {sorted(missing)}")

    new_keys = pd.MultiIndex.from_frame(new_results[key_cols])
    existing_keys = pd.MultiIndex.from_frame(existing[key_cols])
    retained = existing.loc[~existing_keys.isin(new_keys)]

    return pd.concat([retained, new_results], ignore_index=True, sort=False)


def upsert_feature_selection(
    existing: pd.DataFrame,
    new_results: pd.DataFrame,
) -> pd.DataFrame:
    """Consolida importancias y RFECV sin duplicar feature/fold/método."""
    if existing.empty:
        return new_results.copy().reset_index(drop=True)
    if new_results.empty:
        return existing.copy().reset_index(drop=True)

    key_cols = ["protocol", "target", "model", "method", "fold", "feature"]
    missing = set(key_cols) - set(existing.columns) | set(key_cols) - set(new_results.columns)
    if missing:
        raise KeyError(
            f"Faltan columnas para consolidar feature selection: {sorted(missing)}"
        )

    new_keys = pd.MultiIndex.from_frame(new_results[key_cols])
    existing_keys = pd.MultiIndex.from_frame(existing[key_cols])
    retained = existing.loc[~existing_keys.isin(new_keys)]

    return pd.concat([retained, new_results], ignore_index=True, sort=False)


def save_feature_selection(
    results: pd.DataFrame,
    output_path: str | Path = "reports/feature_selection.csv",
) -> Path:
    """Guarda importancias y selecciones por fold en un único CSV."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return path
