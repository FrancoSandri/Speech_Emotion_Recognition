"""Logistic Regression Elastic Net para la rama interpretable eGeMAPS."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_elastic_net_search(
    inner_cv,
    C_values: Sequence[float],
    l1_ratios: Sequence[float],
    seed: int,
    max_iter: int = 5000,
    n_jobs: int = 1,
) -> GridSearchCV:
    """Construye un GridSearchCV pequeño y compatible con grupos internos."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "classifier__C": [float(value) for value in C_values],
            "classifier__l1_ratio": [float(value) for value in l1_ratios],
        },
        scoring="f1_macro",
        cv=inner_cv,
        refit=True,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )
