"""Logistic Regression usada como linear probe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_classifier(
    params: Mapping[str, Any],
    seed: int,
) -> LogisticRegression:
    """Construye el clasificador lineal sin preprocesamiento."""
    return LogisticRegression(
        C=float(params.get("C", 1.0)),
        max_iter=int(params.get("max_iter", 2000)),
        class_weight=params.get("class_weight", "balanced"),
        solver=str(params.get("solver", "lbfgs")),
        random_state=seed,
    )


def build_linear_probe(
    params: Mapping[str, Any],
    seed: int,
) -> Pipeline:
    """Construye StandardScaler + LogisticRegression dentro de un Pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                build_logistic_classifier(params=params, seed=seed),
            ),
        ]
    )
