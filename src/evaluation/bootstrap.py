from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


MetricFunction = Callable[[pd.Series, pd.Series], float]


def cluster_bootstrap_metric(
    predictions: pd.DataFrame,
    metric_fn: MetricFunction,
    *,
    cluster_col: str = "actor",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Estima una métrica y su intervalo de confianza mediante bootstrap
    agrupado por hablante.

    En cada repetición se remuestrean hablantes con reemplazo y se
    conservan todos sus audios.
    """

    required_columns = {
        cluster_col,
        y_true_col,
        y_pred_col,
    }
    missing = required_columns.difference(predictions.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError("El DataFrame de predicciones está vacío.")

    clusters = predictions[cluster_col].dropna().unique()

    if len(clusters) < 2:
        raise ValueError(
            "Se necesitan al menos dos clusters para el bootstrap."
        )

    rng = np.random.default_rng(seed)
    bootstrap_scores = np.empty(n_bootstrap, dtype=float)

    grouped = {
        cluster: group
        for cluster, group in predictions.groupby(
            cluster_col,
            sort=False,
        )
    }

    for bootstrap_index in range(n_bootstrap):
        sampled_clusters = rng.choice(
            clusters,
            size=len(clusters),
            replace=True,
        )

        sampled_parts = [
            grouped[cluster]
            for cluster in sampled_clusters
        ]
        sampled_predictions = pd.concat(
            sampled_parts,
            ignore_index=True,
        )

        bootstrap_scores[bootstrap_index] = metric_fn(
            sampled_predictions[y_true_col],
            sampled_predictions[y_pred_col],
        )

    point_estimate = metric_fn(
        predictions[y_true_col],
        predictions[y_pred_col],
    )

    alpha = 1.0 - confidence_level
    lower = np.quantile(bootstrap_scores, alpha / 2.0)
    upper = np.quantile(
        bootstrap_scores,
        1.0 - alpha / 2.0,
    )

    return {
        "estimate": float(point_estimate),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "std_error": float(
            bootstrap_scores.std(ddof=1)
        ),
        "n_bootstrap": int(n_bootstrap),
        "n_clusters": int(len(clusters)),
    }