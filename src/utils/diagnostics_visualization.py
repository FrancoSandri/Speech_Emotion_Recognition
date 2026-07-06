"""Visualizaciones compactas para diagnóstico e interpretabilidad SER."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_actor_diagnostics(
    actor_metrics: pd.DataFrame,
    actor_class_recall: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    emotion_order: Sequence[str],
    primary_representation: str = "wav2vec",
    metric_col: str = "balanced_accuracy",
    title: str = "Diagnóstico de varianza",
):
    """Resume variabilidad por actor, clase y outer fold en tres paneles."""
    required_actor = {"representation", "actor_id", metric_col}
    required_recall = {"representation", "actor_id", "target_class", "recall"}
    required_fold = {"representation", "fold", metric_col}

    for frame, required, name in (
        (actor_metrics, required_actor, "actor_metrics"),
        (actor_class_recall, required_recall, "actor_class_recall"),
        (fold_metrics, required_fold, "fold_metrics"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} no contiene: {sorted(missing)}")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

    sns.stripplot(
        data=actor_metrics,
        x="representation",
        y=metric_col,
        jitter=0.16,
        alpha=0.8,
        ax=axes[0],
    )
    representation_order = [
        value for value in actor_metrics["representation"].drop_duplicates()
    ]
    means = (
        actor_metrics.groupby("representation", observed=True)[metric_col]
        .mean()
        .reindex(representation_order)
    )
    axes[0].scatter(
        np.arange(len(means)),
        means.to_numpy(),
        marker="D",
        s=70,
        label="Media",
        zorder=4,
    )
    axes[0].set_title("Balanced accuracy por actor")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].legend()

    primary_matrix = (
        actor_class_recall.loc[
            actor_class_recall["representation"].eq(primary_representation)
        ]
        .pivot(index="actor_id", columns="target_class", values="recall")
        .reindex(columns=list(emotion_order))
    )
    sns.heatmap(
        primary_matrix,
        vmin=0,
        vmax=1,
        cmap="viridis",
        ax=axes[1],
        cbar_kws={"label": "Recall OOF"},
    )
    axes[1].set_title(
        f"{primary_representation}: recall actor × emoción"
    )
    axes[1].set_xlabel("Emoción")
    axes[1].set_ylabel("Actor")

    representations = [
        value for value in fold_metrics["representation"].drop_duplicates()
    ]
    offsets = np.linspace(-0.08, 0.08, max(1, len(representations)))
    for offset, representation in zip(offsets, representations):
        group = (
            fold_metrics.loc[
                fold_metrics["representation"].eq(representation)
            ]
            .sort_values("fold")
        )
        x = group["fold"].to_numpy(dtype=float) + offset
        axes[2].scatter(
            x,
            group[metric_col],
            s=55,
            label=representation,
        )
        for x_value, (_, row) in zip(x, group.iterrows()):
            actors = row.get("validation_actors", "")
            axes[2].annotate(
                actors,
                (x_value, row[metric_col]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
            )
    axes[2].set_title("Balanced accuracy por outer fold")
    axes[2].set_xlabel("Fold")
    axes[2].set_ylabel("Balanced accuracy")
    axes[2].legend()

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_family_importance_stability(
    family_importance: pd.DataFrame,
    *,
    target: str,
    family_order: Sequence[str],
    method_order: Sequence[str],
    title: str,
):
    """Muestra importancia normalizada media y dispersión entre outer folds."""
    required = {
        "target",
        "method",
        "family_label",
        "importance_normalized",
    }
    missing = required - set(family_importance.columns)
    if missing:
        raise KeyError(
            f"family_importance no contiene: {sorted(missing)}"
        )

    focus = family_importance.loc[
        family_importance["target"].eq(target)
    ].copy()
    stability = (
        focus.groupby(["method", "family_label"], observed=True)[
            "importance_normalized"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    offsets = np.linspace(-0.18, 0.18, len(method_order))
    y = np.arange(len(family_order))

    for offset, method in zip(offsets, method_order):
        group = (
            stability.loc[stability["method"].eq(method)]
            .set_index("family_label")
            .reindex(family_order)
        )
        ax.errorbar(
            group["mean"],
            y + offset,
            xerr=group["std"].fillna(0),
            fmt="o",
            capsize=3,
            label=method,
        )

    ax.set_yticks(y, family_order)
    ax.set_xlabel("Importancia normalizada media ± desvío")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_family_ablations(
    ablation_summary: pd.DataFrame,
    *,
    target: str,
    family_order: Sequence[str],
    target_label: str,
):
    """Separa información aislada y aporte complementario de cada familia."""
    required = {
        "target",
        "experiment",
        "family_label",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "delta_mean",
        "delta_std",
    }
    missing = required - set(ablation_summary.columns)
    if missing:
        raise KeyError(
            f"ablation_summary no contiene: {sorted(missing)}"
        )

    focus = ablation_summary.loc[
        ablation_summary["target"].eq(target)
    ].copy()

    baseline_rows = focus.loc[
        focus["experiment"].eq("all_features")
    ]
    baseline = (
        float(baseline_rows["balanced_accuracy_mean"].iloc[0])
        if not baseline_rows.empty
        else np.nan
    )

    family_only = (
        focus.loc[focus["experiment"].eq("family_only")]
        .set_index("family_label")
        .reindex(family_order)
    )
    leave_out = (
        focus.loc[focus["experiment"].eq("leave_one_family_out")]
        .set_index("family_label")
        .reindex(family_order)
    )

    y = np.arange(len(family_order))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    axes[0].errorbar(
        family_only["balanced_accuracy_mean"],
        y,
        xerr=family_only["balanced_accuracy_std"].fillna(0),
        fmt="o",
        capsize=3,
    )
    if np.isfinite(baseline):
        axes[0].axvline(
            baseline,
            linestyle="--",
            linewidth=1,
            label="eGeMAPS completo",
        )
        axes[0].legend()
    axes[0].set_yticks(y, family_order)
    axes[0].set_xlabel("Balanced accuracy media ± desvío")
    axes[0].set_title("Cada familia por separado")

    axes[1].errorbar(
        leave_out["delta_mean"],
        y,
        xerr=leave_out["delta_std"].fillna(0),
        fmt="o",
        capsize=3,
    )
    axes[1].axvline(0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Δ balanced accuracy al remover la familia")
    axes[1].set_title("Aporte complementario")

    fig.suptitle(
        f"Ablaciones de familias eGeMAPS — {target_label}",
        y=1.02,
    )
    fig.tight_layout()
    return fig, axes


def plot_projection_tradeoff(
    summary: pd.DataFrame,
    *,
    title: str,
):
    """Sintetiza rendimiento, estabilidad y dimensionalidad efectiva."""
    required = {
        "representation",
        "refinement",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "n_features_mean",
    }
    missing = required - set(summary.columns)
    if missing:
        raise KeyError(f"summary no contiene: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for representation, group in summary.groupby(
        "representation",
        observed=True,
    ):
        sizes = 80 + 45 * np.log10(
            group["n_features_mean"].clip(lower=1)
        )
        ax.scatter(
            group["balanced_accuracy_mean"],
            group["balanced_accuracy_std"],
            s=sizes,
            alpha=0.85,
            label=representation,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["refinement"],
                (
                    row["balanced_accuracy_mean"],
                    row["balanced_accuracy_std"],
                ),
                xytext=(6, 5),
                textcoords="offset points",
            )

    ax.set_xlabel("Balanced accuracy media")
    ax.set_ylabel("Desvío estándar entre folds")
    ax.set_title(title)
    ax.legend(title="Representación")
    fig.tight_layout()
    return fig, ax


def normalized_confusion_matrix(
    predictions: pd.DataFrame,
    *,
    labels: Sequence[str],
    true_col: str = "y_true",
    pred_col: str = "y_pred",
) -> np.ndarray:
    """Calcula una matriz de confusión normalizada por clase verdadera."""
    required = {true_col, pred_col}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions no contiene: {sorted(missing)}")

    return confusion_matrix(
        predictions[true_col],
        predictions[pred_col],
        labels=list(labels),
        normalize="true",
    )


def plot_confusion_comparison(
    baseline_predictions: pd.DataFrame,
    refined_predictions: pd.DataFrame,
    *,
    labels: Sequence[str],
    representation_name: str,
    refinement_label: str = "LDA shrinkage",
):
    """Compara matrices OOF baseline, refinamiento y su diferencia."""
    cm_baseline = normalized_confusion_matrix(
        baseline_predictions,
        labels=labels,
    )
    cm_refined = normalized_confusion_matrix(
        refined_predictions,
        labels=labels,
    )
    delta = cm_refined - cm_baseline

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    sns.heatmap(
        cm_baseline,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    sns.heatmap(
        cm_refined,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )

    limit = max(float(abs(delta.min())), float(abs(delta.max())))
    if limit == 0:
        limit = 1e-12
    sns.heatmap(
        delta,
        annot=True,
        fmt="+.2f",
        center=0,
        vmin=-limit,
        vmax=limit,
        cmap="coolwarm",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[2],
    )

    axes[0].set_title("Baseline")
    axes[1].set_title(refinement_label)
    axes[2].set_title(f"Cambio: {refinement_label} − baseline")

    for ax in axes:
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase verdadera")
        ax.tick_params(axis="x", rotation=40)

    fig.suptitle(
        f"Matrices OOF normalizadas — {representation_name}",
        y=1.02,
    )
    fig.tight_layout()
    return fig, axes
