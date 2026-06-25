"""Constructor de un MLP shallow para features globales."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


class LabelEncodedMLPClassifier(ClassifierMixin, BaseEstimator):
    """MLPClassifier con codificación interna de targets categóricos."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        activation: str = "relu",
        alpha: float = 1e-4,
        batch_size: int = 32,
        learning_rate_init: float = 1e-3,
        max_iter: int = 300,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
        n_iter_no_change: int = 20,
        random_state: int | None = None,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.estimator_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y_encoded)
        return self

    def predict(self, X) -> np.ndarray:
        encoded = self.estimator_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, X) -> np.ndarray:
        return self.estimator_.predict_proba(X)


def build_mlp(
    params: Mapping[str, Any],
    seed: int,
) -> Pipeline:
    """
    Construye StandardScaler + MLP shallow.

    El early stopping reserva internamente una fracción de outer train;
    outer validation nunca participa del ajuste.
    """
    hidden_dims = tuple(int(value) for value in params.get("hidden_dims", [128, 64]))

    classifier = LabelEncodedMLPClassifier(
        hidden_layer_sizes=hidden_dims,
        activation=str(params.get("activation", "relu")),
        alpha=float(params.get("alpha", 1e-4)),
        batch_size=int(params.get("batch_size", 32)),
        learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
        max_iter=int(params.get("max_iter", 300)),
        early_stopping=bool(params.get("early_stopping", True)),
        validation_fraction=float(params.get("validation_fraction", 0.15)),
        n_iter_no_change=int(params.get("n_iter_no_change", 20)),
        random_state=seed,
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
