"""Visualizaciones compactas para experimentos temporales wav2vec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_metric_summary(
    summary: pd.DataFrame,
    *,
    metric_label: str = "Balanced accuracy",
    title: str = "Representaciones wav2vec multicapa",
):
    """Grafica media y desvío entre outer folds."""

    if summary.empty:
        raise ValueError("summary está vacío.")

    plot_data = summary.sort_values("metric_mean").reset_index(drop=True)
    y = np.arange(len(plot_data))

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.65 * len(plot_data))))
    ax.errorbar(
        plot_data["metric_mean"],
        y,
        xerr=plot_data["metric_std"].fillna(0),
        fmt="o",
        capsize=5,
    )
    ax.set_yticks(y, plot_data["refinement"])
    ax.set_xlabel(f"{metric_label} media ± desvío entre outer folds")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_fold_and_seed_variability(
    all_results: pd.DataFrame,
    *,
    configurations: Sequence[str],
    metric: str = "balanced_accuracy",
    metric_label: str = "Balanced accuracy",
):
    """Muestra resultados por fold y variabilidad de inicialización."""

    if all_results.empty:
        raise ValueError("all_results está vacío.")

    result_type = all_results.get(
        "result_type",
        pd.Series(index=all_results.index, dtype="string"),
    )
    ensemble = all_results.loc[~result_type.eq("seed")].copy()
    seed_rows = all_results.loc[result_type.eq("seed")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    present = [
        name
        for name in configurations
        if name in set(ensemble["refinement"])
    ]
    offsets = np.linspace(-0.18, 0.18, max(len(present), 1))

    for offset, refinement in zip(offsets, present):
        group = ensemble.loc[
            ensemble["refinement"].eq(refinement)
        ].sort_values("fold")
        axes[0].scatter(
            group["fold"].to_numpy(dtype=float) + offset,
            group[metric],
            label=refinement,
        )
    axes[0].set_title(f"{metric_label} por outer fold")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel(metric_label)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    seed_summary = (
        seed_rows.groupby(["refinement", "fold"], observed=True)[metric]
        .std(ddof=0)
        .reset_index(name="seed_std")
    )
    present_seed = [
        name
        for name in configurations
        if name in set(seed_summary["refinement"])
    ]
    for index, refinement in enumerate(present_seed):
        values = seed_summary.loc[
            seed_summary["refinement"].eq(refinement), "seed_std"
        ]
        x = np.full(len(values), index, dtype=float)
        axes[1].scatter(x, values, alpha=0.75)
        if len(values):
            axes[1].hlines(
                values.mean(),
                index - 0.22,
                index + 0.22,
                linewidth=2,
            )
    axes[1].set_xticks(range(len(present_seed)), present_seed, rotation=25)
    axes[1].set_ylabel(f"Desvío de {metric_label.lower()} entre seeds")
    axes[1].set_title("Variabilidad de inicialización por fold")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    return fig, axes


def plot_layer_weights(
    layer_weights: pd.DataFrame,
    *,
    configurations: Sequence[str],
    protocol: str,
    target: str,
    n_layers: int,
):
    """Grafica pesos aprendidos y la referencia uniforme ``1 / n_layers``."""

    subset = layer_weights.loc[
        layer_weights["protocol"].eq(protocol)
        & layer_weights["target"].eq(target)
        & layer_weights["configuration"].isin(configurations)
    ].copy()
    if subset.empty:
        raise ValueError("No hay pesos de capa para las configuraciones pedidas.")

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for configuration, group in subset.groupby(
        "configuration", observed=True
    ):
        curve = group.groupby("layer", observed=True)["weight"].agg(
            ["mean", "std"]
        )
        x = curve.index.to_numpy()
        mean = curve["mean"].to_numpy()
        std = curve["std"].fillna(0).to_numpy()
        ax.plot(x, mean, marker="o", label=configuration)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)

    ax.axhline(
        1.0 / n_layers,
        linestyle="--",
        linewidth=1.3,
        label=f"Promedio uniforme (1/{n_layers})",
    )
    ax.set_xlabel(
        "Índice de nivel representacional "
        "(0 = feature projection cuando está incluida)"
    )
    ax.set_ylabel("Peso softmax")
    ax.set_title("Selección aprendida de niveles wav2vec")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_attention_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    configurations: Sequence[str],
    protocol: str,
    target: str,
):
    """Resume concentración temporal para predicciones ensemble."""

    subset = diagnostics.loc[
        diagnostics["protocol"].eq(protocol)
        & diagnostics["target"].eq(target)
        & diagnostics["seed"].eq(-1)
        & diagnostics["configuration"].isin(configurations)
    ].copy()
    if subset.empty:
        raise ValueError("No hay diagnósticos ensemble de atención.")

    metrics = [
        ("attention_entropy_normalized", "Entropía normalizada"),
        ("max_attention", "Máximo peso temporal"),
        (
            "frames_fraction_50pct_mass",
            "Fracción de frames para 50 % de masa",
        ),
    ]
    present = [
        name
        for name in configurations
        if name in set(subset["configuration"])
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (column, title) in zip(axes, metrics):
        values = [
            subset.loc[subset["configuration"].eq(name), column].to_numpy()
            for name in present
        ]
        ax.boxplot(values, tick_labels=present)
        ax.tick_params(axis="x", rotation=22)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, axes


def plot_confusion_comparison(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    *,
    labels: Sequence[str],
    baseline_name: str,
    candidate_name: str,
):
    """Compara matrices normalizadas y su diferencia por clase verdadera."""

    if baseline_predictions.empty or candidate_predictions.empty:
        raise ValueError("Faltan predicciones para comparar matrices.")

    cm_baseline = confusion_matrix(
        baseline_predictions["y_true"],
        baseline_predictions["y_pred"],
        labels=labels,
        normalize="true",
    )
    cm_candidate = confusion_matrix(
        candidate_predictions["y_true"],
        candidate_predictions["y_pred"],
        labels=labels,
        normalize="true",
    )
    matrices = [
        cm_baseline,
        cm_candidate,
        cm_candidate - cm_baseline,
    ]
    titles = [
        f"Referencia: {baseline_name}",
        f"Candidato: {candidate_name}",
        "Cambio: candidato − referencia",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
    for index, (ax, matrix, title) in enumerate(
        zip(axes, matrices, titles)
    ):
        differential = index == 2
        limit = max(float(np.abs(matrix).max()), 1e-6)
        image = ax.imshow(
            matrix,
            cmap="coolwarm" if differential else "viridis",
            vmin=-limit if differential else 0,
            vmax=limit if differential else 1,
        )
        for row in range(len(labels)):
            for col in range(len(labels)):
                text = (
                    f"{matrix[row, col]:+.2f}"
                    if differential
                    else f"{matrix[row, col]:.2f}"
                )
                ax.text(
                    col,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title)
        ax.set_xlabel("Predicción")
        if index == 0:
            ax.set_ylabel("Clase verdadera")
        fig.colorbar(image, ax=ax, fraction=0.046)

    fig.tight_layout()
    return fig, axes


def _parse_probability_vector(probabilities) -> np.ndarray | None:
    """Normaliza probabilidades persistidas o devuelve ``None`` si faltan.

    Los CSV combinan configuraciones que sí guardan probabilidades con otras
    que no lo hacen. En estas últimas, pandas representa el valor ausente como
    ``NaN`` (float), aunque la columna exista en el DataFrame completo.
    """

    if probabilities is None:
        return None

    if isinstance(probabilities, str):
        value = probabilities.strip()
        if not value or value.lower() in {"nan", "none", "null"}:
            return None
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    elif isinstance(probabilities, (list, tuple, np.ndarray)):
        parsed = probabilities
    else:
        try:
            if bool(pd.isna(probabilities)):
                return None
        except (TypeError, ValueError):
            return None
        return None

    try:
        values = np.asarray(parsed, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if values.size == 0 or not np.isfinite(values).all():
        return None
    return values


def _probability_of_true_label(
    probabilities,
    true_label: str,
    labels: Sequence[str],
) -> float:
    """Obtiene P(clase verdadera) cuando el vector está disponible."""

    values = _parse_probability_vector(probabilities)
    if values is None:
        return np.nan

    try:
        index = list(map(str, labels)).index(str(true_label))
    except ValueError:
        return np.nan

    return float(values[index]) if index < len(values) else np.nan


def select_temporal_examples(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    *,
    attention_weights: Mapping[str, np.ndarray],
    candidate_name: str,
    labels: Sequence[str],
    max_examples: int = 3,
) -> pd.DataFrame:
    """Selecciona ejemplos deterministas y prioriza cambios informativos."""

    base_columns = ["file_id", "y_true", "y_pred"]
    candidate_columns = ["file_id", "y_pred"]
    if "probabilities" in baseline_predictions.columns:
        base_columns.append("probabilities")
    if "probabilities" in candidate_predictions.columns:
        candidate_columns.append("probabilities")

    comparison = baseline_predictions[base_columns].merge(
        candidate_predictions[candidate_columns],
        on="file_id",
        suffixes=("_base", "_candidate"),
        validate="one_to_one",
    )
    comparison["case"] = np.select(
        [
            comparison["y_pred_base"].ne(comparison["y_true"])
            & comparison["y_pred_candidate"].eq(comparison["y_true"]),
            comparison["y_pred_base"].eq(comparison["y_true"])
            & comparison["y_pred_candidate"].eq(comparison["y_true"]),
        ],
        ["Corregido por atención", "Acierto compartido"],
        default="Error persistente",
    )

    key_prefix = f"{candidate_name}::"
    comparison = comparison.loc[
        comparison["file_id"].astype(str).map(
            lambda value: key_prefix + value in attention_weights
        )
    ].copy()

    if {
        "probabilities_base",
        "probabilities_candidate",
    }.issubset(comparison.columns):
        comparison["true_probability_base"] = comparison.apply(
            lambda row: _probability_of_true_label(
                row["probabilities_base"], row["y_true"], labels
            ),
            axis=1,
        )
        comparison["true_probability_candidate"] = comparison.apply(
            lambda row: _probability_of_true_label(
                row["probabilities_candidate"], row["y_true"], labels
            ),
            axis=1,
        )
        comparison["probability_gain"] = (
            comparison["true_probability_candidate"]
            - comparison["true_probability_base"]
        )
        # Si el baseline no persistió probabilidades, no se inventa un delta.
        # Para ordenar ejemplos se usa como respaldo la confianza del candidato.
        comparison["selection_score"] = (
            comparison["probability_gain"]
            .where(comparison["probability_gain"].notna())
            .fillna(comparison["true_probability_candidate"])
            .fillna(0.0)
        )
    else:
        comparison["probability_gain"] = np.nan
        comparison["selection_score"] = 0.0

    case_order = [
        "Corregido por atención",
        "Acierto compartido",
        "Error persistente",
    ]
    selected: list[pd.DataFrame] = []
    for case in case_order:
        group = comparison.loc[comparison["case"].eq(case)].sort_values(
            ["selection_score", "file_id"],
            ascending=[False, True],
        )
        if not group.empty:
            selected.append(group.head(1))
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True).head(max_examples)


def plot_temporal_examples(
    examples: pd.DataFrame,
    *,
    metadata: pd.DataFrame,
    attention_weights: Mapping[str, np.ndarray],
    candidate_name: str,
    resolve_path_fn,
):
    """Contrasta atención y RMS en ejemplos OOF seleccionados."""

    if examples.empty:
        raise ValueError("No hay ejemplos temporales para graficar.")

    meta_index = metadata.set_index("file_id")
    fig, axes = plt.subplots(
        len(examples),
        1,
        figsize=(12, 3.2 * len(examples)),
        squeeze=False,
    )

    for ax, (_, row) in zip(axes[:, 0], examples.iterrows()):
        file_id = str(row["file_id"])
        weights = attention_weights[f"{candidate_name}::{file_id}"]
        duration = float(meta_index.loc[file_id, "duration_trimmed_s"])
        attention_time = np.linspace(
            0.0,
            duration,
            len(weights),
            endpoint=False,
        )
        ax.plot(
            attention_time,
            weights / max(float(weights.max()), 1e-8),
            label="Atención normalizada",
        )

        audio_path = Path(
            resolve_path_fn(meta_index.loc[file_id, "file_path_trimmed"])
        )
        if audio_path.exists():
            import librosa

            waveform, sample_rate = librosa.load(
                audio_path,
                sr=None,
                mono=True,
            )
            hop_length = 512
            rms = librosa.feature.rms(
                y=waveform,
                hop_length=hop_length,
            )[0]
            rms_time = librosa.times_like(
                rms,
                sr=sample_rate,
                hop_length=hop_length,
            )
            valid = rms_time <= duration
            ax.plot(
                rms_time[valid],
                rms[valid] / max(float(rms.max()), 1e-8),
                alpha=0.65,
                label="RMS normalizado",
            )

        ax.set_title(
            f"{row['case']} · {file_id} · "
            f"true={row['y_true']} · "
            f"base={row['y_pred_base']} · "
            f"att={row['y_pred_candidate']}"
        )
        ax.set_xlabel("Tiempo aproximado [s]")
        ax.set_ylabel("Magnitud normalizada")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig, axes
