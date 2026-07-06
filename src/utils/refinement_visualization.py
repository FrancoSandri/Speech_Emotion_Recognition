from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.evaluation.oof_predictions import configuration_mask
from src.utils.visualization import (
    PROTOCOL_LABELS,
    TARGET_LABELS,
    configuration_label,
    plot_confusion_delta,
    plot_normalized_confusion,
)


def _metric_label(metric: str) -> str:
    return metric.replace("_", " ").capitalize()


def plot_ranked_configurations(
    summary: pd.DataFrame,
    *,
    protocol: str,
    metric: str,
    top_n: int = 10,
) -> Figure | None:
    """Plot the highest-ranked configurations for one protocol."""

    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"
    required = {
        "protocol",
        "representation",
        "target",
        "model",
        "refinement",
        metric_mean,
        metric_std,
    }
    if summary.empty or not required.issubset(summary.columns):
        return None

    data = (
        summary.loc[summary["protocol"].eq(protocol)]
        .sort_values([metric_mean, metric_std], ascending=[False, True])
        .head(top_n)
        .copy()
    )
    if data.empty:
        return None

    data["configuration"] = data.apply(configuration_label, axis=1)
    data["target_label"] = data["target"].map(TARGET_LABELS).fillna(data["target"])
    data["plot_label"] = data["target_label"] + " · " + data["configuration"]
    data = data.sort_values(metric_mean)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.55 * len(data))))
    ax.errorbar(
        data[metric_mean],
        data["plot_label"],
        xerr=data[metric_std],
        fmt="o",
        capsize=3,
    )
    ax.set_xlabel(f"{_metric_label(metric)} medio ± desvío entre folds")
    ax.set_title(
        f"Mejores configuraciones · {PROTOCOL_LABELS.get(protocol, protocol)}"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_feature_importance_summary(
    importance_summary: pd.DataFrame,
    *,
    protocol: str,
    target: str,
    top_k: int = 15,
) -> Figure | None:
    """Plot the leading eGeMAPS features for each importance method."""

    if importance_summary.empty:
        return None

    data = importance_summary.loc[
        importance_summary["protocol"].eq(protocol)
        & importance_summary["target"].eq(target)
    ].copy()
    if data.empty:
        return None

    methods = list(data["method"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(7 * len(methods), 7),
        squeeze=False,
    )

    for ax, method in zip(axes.ravel(), methods):
        method_data = (
            data.loc[data["method"].eq(method)]
            .sort_values("importance_mean", ascending=False)
            .head(top_k)
            .sort_values("importance_mean")
        )
        ax.errorbar(
            method_data["importance_mean"],
            method_data["feature"],
            xerr=method_data["importance_std"],
            fmt="o",
            capsize=3,
        )
        ax.set_title(str(method))
        ax.set_xlabel("Importancia media ± desvío")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"Top eGeMAPS — {TARGET_LABELS.get(target, target)} · "
        f"{PROTOCOL_LABELS.get(protocol, protocol)}",
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def plot_rfecv_diagnostics(
    rfecv_rows: pd.DataFrame,
    feature_selection: pd.DataFrame,
    *,
    metric: str,
    stability_protocol: str,
    stability_target: str,
    stability_top_k: int = 20,
) -> list[Figure]:
    """Plot RFECV dimensionality, performance, and selection stability."""

    if rfecv_rows.empty:
        return []

    figures: list[Figure] = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for (protocol, target), group in rfecv_rows.groupby(
        ["protocol", "target"],
        observed=True,
    ):
        label = (
            f"{PROTOCOL_LABELS.get(protocol, protocol)} · "
            f"{TARGET_LABELS.get(target, target)}"
        )
        axes[0].scatter(
            group["n_features"],
            group[metric],
            label=label,
            alpha=0.85,
        )

    axes[0].set_xlabel("Features seleccionadas")
    axes[0].set_ylabel(f"{_metric_label(metric)} del outer fold")
    axes[0].set_title("RFECV: dimensionalidad vs rendimiento")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    rfecv_counts = (
        rfecv_rows.groupby(["protocol", "target"], observed=True)["n_features"]
        .apply(list)
    )
    labels = [
        f"{PROTOCOL_LABELS.get(protocol, protocol)}\n"
        f"{TARGET_LABELS.get(target, target)}"
        for protocol, target in rfecv_counts.index
    ]
    axes[1].boxplot(rfecv_counts.tolist(), tick_labels=labels)
    axes[1].set_ylabel("Cantidad de features")
    axes[1].set_title("Variación del subset entre outer folds")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figures.append(fig)

    if feature_selection.empty:
        return figures

    stability = feature_selection.loc[
        feature_selection["method"].eq("rfecv")
        & feature_selection["protocol"].eq(stability_protocol)
        & feature_selection["target"].eq(stability_target)
    ]
    if stability.empty:
        return figures

    stability = (
        stability.groupby("feature", observed=True)
        .agg(
            selection_frequency=("selected", "mean"),
            rank_mean=("rank", "mean"),
        )
        .reset_index()
        .sort_values(
            ["selection_frequency", "rank_mean"],
            ascending=[False, True],
        )
        .head(stability_top_k)
        .sort_values("selection_frequency")
    )

    stability_fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(stability["feature"], stability["selection_frequency"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Frecuencia de selección entre outer folds")
    ax.set_title(
        "Estabilidad RFECV — "
        f"{TARGET_LABELS.get(stability_target, stability_target)} · "
        f"{PROTOCOL_LABELS.get(stability_protocol, stability_protocol)}"
    )
    ax.grid(axis="x", alpha=0.25)
    stability_fig.tight_layout()
    figures.append(stability_fig)
    return figures


def plot_pca_diagnostics(
    pca_rows: pd.DataFrame,
    *,
    representations: Sequence[str],
    protocol: str,
    metric: str,
) -> Figure | None:
    """Plot the number of learned PCA components against outer-fold score."""

    if pca_rows.empty:
        return None

    data = pca_rows.loc[pca_rows["protocol"].eq(protocol)].copy()
    if data.empty:
        return None

    fig, axes = plt.subplots(
        1,
        len(representations),
        figsize=(7 * len(representations), 5),
        squeeze=False,
    )

    for ax, representation in zip(axes.ravel(), representations):
        rep_data = data.loc[data["representation"].eq(representation)]
        for (target, refinement), group in rep_data.groupby(
            ["target", "refinement"],
            observed=True,
        ):
            ax.scatter(
                group["pca_n_components"],
                group[metric],
                label=f"{TARGET_LABELS.get(target, target)} · {refinement}",
                alpha=0.85,
            )

        ax.set_xlabel("Componentes PCA aprendidos")
        ax.set_ylabel(f"{_metric_label(metric)} del outer fold")
        ax.set_title(str(representation))
        ax.grid(alpha=0.25)
        if not rep_data.empty:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"PCA: compresión vs rendimiento — "
        f"{PROTOCOL_LABELS.get(protocol, protocol)}",
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def plot_best_baseline_confusions(
    best_baselines: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    *,
    target_order: Sequence[str],
) -> Figure | None:
    """Plot normalized OOF confusion matrices for the best baseline per target."""

    if best_baselines.empty or oof_predictions.empty:
        return None

    available_targets = set(best_baselines["target"])
    targets = [target for target in target_order if target in available_targets]
    if not targets:
        return None

    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(6 * len(targets), 5.5),
        squeeze=False,
    )

    image = None
    for ax, target in zip(axes.ravel(), targets):
        config_row = best_baselines.loc[best_baselines["target"].eq(target)].iloc[0]
        predictions = oof_predictions.loc[
            configuration_mask(oof_predictions, config_row)
        ]
        image, _, _ = plot_normalized_confusion(
            predictions,
            title=(
                f"{TARGET_LABELS.get(target, target)}\n"
                f"{configuration_label(config_row)}"
            ),
            ax=ax,
        )

    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.02,
            pad=0.02,
            label="Proporción dentro de la clase verdadera",
        )
    fig.suptitle(
        "Mejores baselines — matrices OOF normalizadas",
        y=1.03,
        fontsize=14,
    )
    return fig


def plot_controlled_refinement_confusions(
    track_configs: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    *,
    target: str,
    protocol: str,
) -> list[Figure]:
    """Plot baseline, best refinement, and their normalized confusion delta."""

    if track_configs.empty or oof_predictions.empty:
        return []

    figures: list[Figure] = []
    for representation in track_configs["representation"].drop_duplicates():
        family = track_configs.loc[
            track_configs["representation"].eq(representation)
        ]
        baseline = family.loc[family["refinement"].astype(str).eq("none")]
        refined = family.loc[~family["refinement"].astype(str).eq("none")]
        if baseline.empty or refined.empty:
            continue

        baseline_row = baseline.iloc[0]
        refined_row = refined.iloc[0]
        baseline_predictions = oof_predictions.loc[
            configuration_mask(oof_predictions, baseline_row)
        ]
        refined_predictions = oof_predictions.loc[
            configuration_mask(oof_predictions, refined_row)
        ]

        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        image, _, labels = plot_normalized_confusion(
            baseline_predictions,
            title=f"Baseline\n{configuration_label(baseline_row)}",
            ax=axes[0],
        )
        plot_normalized_confusion(
            refined_predictions,
            title=f"Refinamiento\n{configuration_label(refined_row)}",
            labels=labels,
            ax=axes[1],
        )
        delta_image, _ = plot_confusion_delta(
            baseline_predictions,
            refined_predictions,
            title="Cambio normalizado\nrefinamiento − baseline",
            labels=labels,
            ax=axes[2],
        )

        fig.colorbar(
            image,
            ax=axes[:2].tolist(),
            fraction=0.025,
            pad=0.02,
            label="Proporción por clase verdadera",
        )
        fig.colorbar(
            delta_image,
            ax=axes[2],
            fraction=0.05,
            pad=0.04,
            label="Cambio",
        )
        fig.suptitle(
            f"Track controlado — {representation} · "
            f"{TARGET_LABELS.get(target, target)} · "
            f"{PROTOCOL_LABELS.get(protocol, protocol)}",
            y=1.03,
            fontsize=14,
        )
        figures.append(fig)

    return figures
