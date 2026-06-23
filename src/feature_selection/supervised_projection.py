"""Proyección supervisada lineal mediante LDA regularizada."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.linear_probe import build_logistic_classifier


class ShrinkageLDATransformer(BaseEstimator, TransformerMixin):
    """Convierte cada muestra en ejes discriminantes LDA regularizados.

    ``LinearDiscriminantAnalysis(solver='lsqr')`` permite shrinkage y es
    eficiente en representaciones de alta dimensión, pero no expone
    ``transform``. Este wrapper usa sus scores discriminantes y toma la última
    clase como referencia, produciendo ``n_classes - 1`` dimensiones.
    """

    def __init__(self, shrinkage: str | float | None = "auto") -> None:
        self.shrinkage = shrinkage

    def fit(self, X, y):
        self.estimator_ = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage=self.shrinkage,
        )
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = int(self.estimator_.n_features_in_)
        self.n_components_ = max(1, len(self.classes_) - 1)
        return self

    def transform(self, X):
        scores = np.asarray(self.estimator_.decision_function(X), dtype=float)
        if scores.ndim == 1:
            return scores.reshape(-1, 1)
        return scores[:, :-1] - scores[:, [-1]]


def build_lda_linear_probe(
    logistic_regression_params: Mapping[str, Any],
    seed: int,
    shrinkage: str | float | None = "auto",
) -> Pipeline:
    """Construye StandardScaler → LDA shrinkage → Logistic Regression."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "projection",
                ShrinkageLDATransformer(shrinkage=shrinkage),
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


def get_lda_fold_metadata(pipeline: Pipeline) -> dict[str, float | int]:
    """Extrae la dimensionalidad efectiva de la proyección ajustada."""
    projection = pipeline.named_steps["projection"]
    return {
        "n_features": int(projection.n_components_),
        "lda_n_components": int(projection.n_components_),
    }
