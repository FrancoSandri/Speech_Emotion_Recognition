"""Mezclas uniformes y aprendidas de representaciones wav2vec por capa."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class LayerStandardizer:
    """Parámetros aprendidos exclusivamente sobre outer train."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)


def reshape_layer_features(
    X: np.ndarray,
    n_layers: int,
    hidden_size: int,
    pooling: str,
) -> np.ndarray:
    """Restaura ``[samples, layers, channels]`` desde features aplanadas."""
    X = np.asarray(X, dtype=np.float32)
    channels = hidden_size if pooling == "mean" else hidden_size * 2
    expected = n_layers * channels
    if X.ndim != 2 or X.shape[1] != expected:
        raise ValueError(
            f"Forma incompatible para layer mixture: {X.shape}; "
            f"se esperaban {expected} features."
        )
    return X.reshape(len(X), n_layers, channels)


def fit_layer_standardizer(values: np.ndarray) -> LayerStandardizer:
    """Calcula media y escala por capa/canal usando solo train."""
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return LayerStandardizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
    )


def build_layer_mixture_network(
    n_layers: int,
    hidden_size: int,
    pooling: str,
    n_classes: int,
    dropout: float = 0.10,
):
    """Mezcla aprendida de estadísticas por capa y cabeza lineal.

    Se conserva para reproducir ``learned_layers_mean`` y
    ``learned_layers_mean_std`` del notebook temporal original.
    """
    import torch
    from torch import nn

    channels = hidden_size if pooling == "mean" else hidden_size * 2

    class ScalarLayerMixtureClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_logits = nn.Parameter(torch.zeros(n_layers))
            self.normalization = nn.LayerNorm(channels)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(channels, n_classes)

        def forward(self, layer_features):
            weights = torch.softmax(self.layer_logits, dim=0)
            pooled = torch.sum(layer_features * weights[None, :, None], dim=1)
            pooled = self.normalization(pooled)
            return self.classifier(self.dropout(pooled))

        def layer_weights(self):
            return torch.softmax(self.layer_logits.detach(), dim=0)

    return ScalarLayerMixtureClassifier()


def build_sequence_layer_mixture(
    n_layers: int,
    strategy: Literal["uniform", "learned"],
):
    """Construye una mezcla de capas para secuencias ``[B, L, T, D]``.

    La estrategia aprendida se inicializa exactamente en pesos uniformes.
    """
    import torch
    from torch import nn

    if n_layers < 1:
        raise ValueError("n_layers debe ser positivo.")
    if strategy not in {"uniform", "learned"}:
        raise ValueError(f"Estrategia de capas no soportada: {strategy!r}")

    class SequenceLayerMixture(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.strategy = strategy
            if strategy == "learned":
                self.layer_logits = nn.Parameter(torch.zeros(n_layers))
            else:
                self.register_buffer(
                    "uniform_weights",
                    torch.full((n_layers,), 1.0 / n_layers),
                )

        def layer_weights(self):
            if self.strategy == "learned":
                return torch.softmax(self.layer_logits, dim=0)
            return self.uniform_weights

        def forward(self, hidden_states):
            if hidden_states.ndim != 4:
                raise ValueError(
                    "hidden_states debe tener forma [batch, layers, time, hidden]."
                )
            if int(hidden_states.shape[1]) != n_layers:
                raise ValueError(
                    f"Se esperaban {n_layers} capas y se recibieron "
                    f"{hidden_states.shape[1]}."
                )
            weights = self.layer_weights().to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            mixed = torch.sum(
                hidden_states * weights[None, :, None, None],
                dim=1,
            )
            return mixed, weights

    return SequenceLayerMixture()
