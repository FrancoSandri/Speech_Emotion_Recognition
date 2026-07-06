"""Visualizaciones compactas para la evaluación final."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _annotate_confusion(
    ax,
    normalized: np.ndarray,
    counts: np.ndarray,
) -> None:
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            count = int(counts[row, column])
            ax.text(
                column,
                row,
                f"{value:.2f}\n({count})",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.5 else "black",
            )


def plot_cv_test_comparison(
    comparison: pd.DataFrame,
    *,
    title: str = "Estimación outer-CV frente a evaluación final en test",
):
    """Grafica media±std de CV y balanced accuracy de test."""

    required = {
        "label",
        "cv_mean",
        "cv_std",
        "test_balanced_accuracy",
    }
    missing = required - set(comparison.columns)
    if missing:
        raise KeyError(f"Faltan columnas para CV-test: {sorted(missing)}")

    plot_data = comparison.reset_index(drop=True)
    positions = np.arange(len(plot_data))

    fig, ax = plt.subplots(
        figsize=(10, max(4.5, 1.1 * len(plot_data)))
    )
    ax.errorbar(
        plot_data["cv_mean"],
        positions - 0.12,
        xerr=plot_data["cv_std"],
        fmt="o",
        capsize=4,
        label="Outer-CV sobre development",
    )
    ax.scatter(
        plot_data["test_balanced_accuracy"],
        positions + 0.12,
        marker="x",
        s=80,
        label="Test speaker-independent",
        zorder=3,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(plot_data["label"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_test_confusion_matrices(
    predictions: pd.DataFrame,
    model_specs: Mapping[str, Mapping[str, object]],
    *,
    labels: Sequence[str],
):
    """Muestra recall normalizado y conteo absoluto en cada celda."""

    n_models = len(model_specs)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(8 * n_models, 5.7),
        squeeze=False,
    )
    axes = axes.ravel()

    image = None
    for ax, (model_name, spec) in zip(axes, model_specs.items()):
        selected = predictions.loc[
            predictions["model"].eq(model_name)
        ]
        counts = confusion_matrix(
            selected["y_true"],
            selected["y_pred"],
            labels=labels,
        )
        normalized = confusion_matrix(
            selected["y_true"],
            selected["y_pred"],
            labels=labels,
            normalize="true",
        )
        image = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase verdadera")
        ax.set_title(str(spec["label"]), fontsize=9)
        _annotate_confusion(ax, normalized, counts)

    if image is not None:
        fig.colorbar(
            image,
            ax=axes.tolist(),
            shrink=0.72,
            label="Recall normalizado por clase verdadera",
        )
    return fig, axes


def plot_quadrant_confusion(
    quadrant_predictions: pd.DataFrame,
    *,
    labels: Sequence[str] = ("Q1", "Q2", "Q3", "Q4"),
    title: str = "Test · emoción original proyectada a cuadrantes",
):
    """Grafica matriz de cuadrantes con proporción y soporte."""

    counts = confusion_matrix(
        quadrant_predictions["y_true_quadrant"],
        quadrant_predictions["y_pred_quadrant"],
        labels=labels,
    )
    normalized = confusion_matrix(
        quadrant_predictions["y_true_quadrant"],
        quadrant_predictions["y_pred_quadrant"],
        labels=labels,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Cuadrante verdadero")
    ax.set_title(title)
    _annotate_confusion(ax, normalized, counts)
    fig.colorbar(
        image,
        ax=ax,
        shrink=0.8,
        label="Recall normalizado",
    )
    fig.tight_layout()
    return fig, ax


def plot_actor_domain_comparison(
    development_actor_metrics: pd.DataFrame,
    test_actor_metrics: pd.DataFrame,
):
    """Compara actores OOF de development con actores del test final."""

    fig, ax = plt.subplots(figsize=(8, 5))

    if not development_actor_metrics.empty:
        dev_positions = np.linspace(
            -0.09,
            0.09,
            len(development_actor_metrics),
        )
        ax.scatter(
            dev_positions,
            development_actor_metrics["balanced_accuracy"],
            label="Development OOF",
        )
        ax.hlines(
            development_actor_metrics["balanced_accuracy"].mean(),
            -0.18,
            0.18,
            linestyles="--",
        )

    test_positions = 1 + np.linspace(
        -0.09,
        0.09,
        len(test_actor_metrics),
    )
    ax.scatter(
        test_positions,
        test_actor_metrics["balanced_accuracy"],
        marker="x",
        s=70,
        label="Test final",
    )
    for x_value, row in zip(
        test_positions,
        test_actor_metrics.itertuples(index=False),
    ):
        ax.annotate(
            str(row.actor_id),
            (x_value, row.balanced_accuracy),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Development\n(predicciones OOF)", "Test\n(modelo final)"]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy por actor")
    ax.set_title(
        "Variabilidad por hablante para la misma representación final"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax
