"""Visualizaciones para zero-shot y adaptación de cabeza."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_zero_shot(
    zero_shot: pd.DataFrame,
):
    plot_data = zero_shot.sort_values(
        ["domain", "balanced_accuracy"]
    ).reset_index(drop=True)
    labels = (
        plot_data["domain"]
        + " · "
        + plot_data["representation"]
    )
    positions = np.arange(len(plot_data))

    fig, ax = plt.subplots(
        figsize=(10, max(4.5, 0.7 * len(plot_data)))
    )
    ax.scatter(
        plot_data["balanced_accuracy"],
        positions,
        s=70,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy zero-shot")
    ax.set_title(
        "Transferencia sin adaptación de la cabeza RAVDESS"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def _curve_summary(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        results.groupby(
            [
                "domain",
                "representation",
                "n_adaptation_speakers",
            ],
            observed=True,
            as_index=False,
        )[metric]
        .agg(["mean", "std"])
        .reset_index()
    )


def plot_target_learning_curves(
    results: pd.DataFrame,
):
    summary = _curve_summary(
        results,
        "target_balanced_accuracy",
    )
    domains = summary["domain"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1,
        len(domains),
        figsize=(7 * len(domains), 5),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, domain in zip(axes, domains):
        domain_data = summary.loc[
            summary["domain"].eq(domain)
        ]
        for representation, group in domain_data.groupby(
            "representation",
            observed=True,
        ):
            group = group.sort_values(
                "n_adaptation_speakers"
            )
            x = group["n_adaptation_speakers"].to_numpy()
            mean = group["mean"].to_numpy()
            std = group["std"].fillna(0).to_numpy()
            ax.plot(x, mean, marker="o", label=representation)
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                alpha=0.18,
            )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Hablantes usados para adaptar")
        ax.set_ylabel("Balanced accuracy en target test")
        ax.set_title(domain)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(
        "Curvas de adaptación de la cabeza logística",
        y=1.02,
    )
    fig.tight_layout()
    return fig, axes


def plot_source_retention(
    results: pd.DataFrame,
):
    summary = _curve_summary(
        results,
        "source_balanced_accuracy",
    )
    domains = summary["domain"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1,
        len(domains),
        figsize=(7 * len(domains), 5),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, domain in zip(axes, domains):
        domain_data = summary.loc[
            summary["domain"].eq(domain)
        ]
        for representation, group in domain_data.groupby(
            "representation",
            observed=True,
        ):
            group = group.sort_values(
                "n_adaptation_speakers"
            )
            x = group["n_adaptation_speakers"].to_numpy()
            mean = group["mean"].to_numpy()
            std = group["std"].fillna(0).to_numpy()
            ax.plot(x, mean, marker="o", label=representation)
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                alpha=0.18,
            )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Hablantes usados para adaptar")
        ax.set_ylabel("Balanced accuracy en RAVDESS Speech test")
        ax.set_title(domain)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(
        "Retención del dominio fuente después de adaptar",
        y=1.02,
    )
    fig.tight_layout()
    return fig, axes


def plot_transfer_tradeoff(
    results: pd.DataFrame,
):
    adapted = results.loc[
        results["stage"].eq("adapted")
    ].copy()
    summary = (
        adapted.groupby(
            [
                "domain",
                "representation",
                "n_adaptation_speakers",
            ],
            observed=True,
            as_index=False,
        )[["target_gain", "source_change"]]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for (domain, representation), group in summary.groupby(
        ["domain", "representation"],
        observed=True,
    ):
        ax.plot(
            group["target_gain"],
            group["source_change"],
            marker="o",
            label=f"{domain} · {representation}",
        )
        for row in group.itertuples(index=False):
            ax.annotate(
                str(row.n_adaptation_speakers),
                (row.target_gain, row.source_change),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Ganancia de BA en dominio target")
    ax.set_ylabel("Cambio de BA en RAVDESS Speech")
    ax.set_title(
        "Trade-off entre adaptación y retención\n"
        "(etiquetas = número de hablantes)"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def _annotate_rectangular(
    ax,
    normalized: np.ndarray,
    counts: np.ndarray,
) -> None:
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}\n({int(counts[row, column])})",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.5 else "black",
            )


def plot_rectangular_confusion(
    counts: np.ndarray,
    normalized: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    *,
    title: str,
):
    fig, ax = plt.subplots(
        figsize=(
            max(8, 0.9 * len(column_labels)),
            max(5, 0.8 * len(row_labels)),
        )
    )
    image = ax.imshow(
        normalized,
        cmap="viridis",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(
        column_labels,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Salida de la cabeza RAVDESS")
    ax.set_ylabel("Clase verdadera armonizada")
    ax.set_title(title)
    _annotate_rectangular(ax, normalized, counts)
    fig.colorbar(
        image,
        ax=ax,
        label="Recall normalizado por fila",
    )
    fig.tight_layout()
    return fig, ax
