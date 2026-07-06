"""Adaptación incremental de una cabeza Logistic Regression congelando features.

El ``StandardScaler`` del dominio fuente permanece fijo. Los coeficientes e
interceptos de la Logistic Regression se copian a una capa lineal de PyTorch y
solo esa capa se actualiza con datos del dominio destino.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _pipeline_parts(pipeline):
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Se esperaba un sklearn Pipeline.")
    try:
        scaler = pipeline.named_steps["scaler"]
        classifier = pipeline.named_steps["classifier"]
    except KeyError as error:
        raise KeyError(
            "El pipeline debe contener steps 'scaler' y 'classifier'."
        ) from error
    for attribute in ("coef_", "intercept_", "classes_"):
        if not hasattr(classifier, attribute):
            raise ValueError(
                f"El clasificador fuente no está ajustado: falta {attribute}."
            )
    return scaler, classifier


@dataclass
class AdaptedLinearHead:
    """Cabeza lineal adaptada con scaler fuente congelado."""

    scaler: Any
    model: Any
    classes: np.ndarray
    device: Any
    train_loss: float
    epochs: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        scaled = self.scaler.transform(X).astype(np.float32)
        tensor = torch.tensor(
            scaled,
            dtype=torch.float32,
            device=self.device,
        )
        self.model.eval()
        with torch.inference_mode():
            return (
                torch.softmax(self.model(tensor), dim=1)
                .cpu()
                .numpy()
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes[probabilities.argmax(axis=1)]


def initialize_head_from_sklearn(
    pipeline,
    *,
    device=None,
) -> AdaptedLinearHead:
    """Copia una Logistic Regression multiclase a ``torch.nn.Linear``."""

    import torch
    from torch import nn

    scaler, classifier = _pipeline_parts(pipeline)
    coefficients = np.asarray(classifier.coef_, dtype=np.float32)
    intercept = np.asarray(classifier.intercept_, dtype=np.float32)

    if coefficients.ndim != 2 or coefficients.shape[0] < 3:
        raise ValueError(
            "La adaptación está definida para una cabeza multiclase."
        )

    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    model = nn.Linear(
        coefficients.shape[1],
        coefficients.shape[0],
        bias=True,
    ).to(resolved_device)

    with torch.no_grad():
        model.weight.copy_(
            torch.tensor(coefficients, device=resolved_device)
        )
        model.bias.copy_(
            torch.tensor(intercept, device=resolved_device)
        )

    return AdaptedLinearHead(
        scaler=scaler,
        model=model,
        classes=np.asarray(classifier.classes_).astype(str),
        device=resolved_device,
        train_loss=float("nan"),
        epochs=0,
    )


def validate_head_equivalence(
    pipeline,
    head: AdaptedLinearHead,
    X: np.ndarray,
    *,
    atol: float = 1e-5,
) -> float:
    """Comprueba que la copia PyTorch reproduce las probabilidades sklearn."""

    expected = np.asarray(pipeline.predict_proba(X), dtype=float)
    observed = np.asarray(head.predict_proba(X), dtype=float)
    difference = float(np.max(np.abs(expected - observed)))
    if difference > atol:
        raise AssertionError(
            "La cabeza PyTorch no reproduce el pipeline fuente. "
            f"max_abs_difference={difference:.3e} > {atol:.3e}."
        )
    return difference


def _balanced_class_weights(
    y_encoded: np.ndarray,
    n_classes: int,
):
    import torch

    counts = np.bincount(y_encoded, minlength=n_classes).astype(np.float32)
    present = counts > 0
    weights = np.ones(n_classes, dtype=np.float32)
    weights[present] = (
        len(y_encoded)
        / (present.sum() * counts[present])
    )
    return torch.tensor(weights, dtype=torch.float32)


def adapt_linear_head(
    source_pipeline,
    X_target: np.ndarray,
    y_target: Sequence[str],
    *,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    seed: int,
    source_anchor: float = 0.0,
    gradient_clip_norm: float | None = 5.0,
) -> AdaptedLinearHead:
    """Actualiza solo ``W,b`` de la cabeza inicializada desde RAVDESS."""

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if len(X_target) == 0:
        raise ValueError("No hay ejemplos para adaptar la cabeza.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head = initialize_head_from_sklearn(source_pipeline)
    classes = list(head.classes)
    class_to_index = {
        label: index for index, label in enumerate(classes)
    }

    y_target = np.asarray(y_target).astype(str)
    unknown = sorted(set(y_target) - set(classes))
    if unknown:
        raise ValueError(
            f"Etiquetas destino fuera de la cabeza fuente: {unknown}."
        )

    y_encoded = np.asarray(
        [class_to_index[value] for value in y_target],
        dtype=np.int64,
    )
    X_scaled = head.scaler.transform(X_target).astype(np.float32)

    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y_encoded, dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    initial_weight = head.model.weight.detach().clone()
    initial_bias = head.model.bias.detach().clone()

    loss_fn = nn.CrossEntropyLoss(
        weight=_balanced_class_weights(
            y_encoded,
            len(classes),
        ).to(head.device)
    )
    optimizer = torch.optim.AdamW(
        head.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    final_loss = float("nan")
    for _ in range(int(epochs)):
        head.model.train()
        epoch_loss = 0.0
        n_seen = 0

        for features, labels in loader:
            features = features.to(head.device)
            labels = labels.to(head.device)

            optimizer.zero_grad(set_to_none=True)
            logits = head.model(features)
            loss = loss_fn(logits, labels)

            if source_anchor > 0:
                loss = loss + source_anchor * (
                    (head.model.weight - initial_weight).pow(2).mean()
                    + (head.model.bias - initial_bias).pow(2).mean()
                )

            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    head.model.parameters(),
                    max_norm=gradient_clip_norm,
                )
            optimizer.step()

            epoch_loss += float(loss.detach().cpu()) * len(labels)
            n_seen += len(labels)

        final_loss = epoch_loss / max(n_seen, 1)

    head.train_loss = float(final_loss)
    head.epochs = int(epochs)
    return head
