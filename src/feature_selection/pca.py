"""PCA ajustado dentro del pipeline de cada outer fold."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.linear_probe import build_logistic_classifier


def build_pca_linear_probe(
    logistic_regression_params: Mapping[str, Any],
    seed: int,
    variance_threshold: float,
) -> Pipeline:
    """Construye StandardScaler → PCA → LogisticRegression."""
    if not 0.0 < variance_threshold < 1.0:
        raise ValueError("variance_threshold debe estar entre 0 y 1.")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=float(variance_threshold),
                    svd_solver="full",
                ),
            ),
            (
                "classifier",
                build_logistic_classifier(
                    params=logistic_regression_params,
                    seed=seed,
                ),
            ),
        ]
    )


def get_pca_fold_metadata(pipeline: Pipeline) -> dict[str, float | int]:
    """Extrae dimensionalidad y varianza explicada del PCA ya ajustado."""
    pca = pipeline.named_steps["pca"]
    return {
        "n_features": int(pca.n_components_),
        "pca_n_components": int(pca.n_components_),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
