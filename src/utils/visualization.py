import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

EXPERIMENT_COLUMNS = [
    "representation",
    "protocol",
    "target",
    "model",
    "refinement",
]

TARGET_LABELS = {
    "emotion_original": "8 emociones",
    "emotion_quadrant": "4 cuadrantes directos",
    "emotion_original_eval_quadrant": "8→4 cuadrantes",
}

PROTOCOL_LABELS = {
    "speaker_dependent": "Speaker-dependent",
    "speaker_independent": "Speaker-independent",
}


def configuration_label(row):
    refinement = str(row["refinement"])
    suffix = "" if refinement == "none" else f" · {refinement}"
    return f"{row['representation']} · {row['model']}{suffix}"


def plot_top_configurations(summary, protocol="speaker_independent", top_n=8):
    data = summary.query("protocol == @protocol").copy()
    if data.empty:
        print("No hay resultados para graficar.")
        return

    targets = [target for target in TARGET_LABELS if target in set(data["target"])]
    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(7 * len(targets), max(5, top_n * 0.48)),
        squeeze=False,
    )

    for ax, target in zip(axes.ravel(), targets):
        target_data = (
            data.query("target == @target")
            .sort_values(
                ["macro_f1_mean", "macro_f1_std"],
                ascending=[False, True],
            )
            .head(top_n)
            .copy()
        )
        target_data["configuration"] = target_data.apply(
            configuration_label,
            axis=1,
        )
        target_data = target_data.sort_values("macro_f1_mean")

        ax.errorbar(
            target_data["macro_f1_mean"],
            target_data["configuration"],
            xerr=target_data["macro_f1_std"],
            fmt="o",
            capsize=3,
        )
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("Macro F1 medio ± desvío")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"Mejores configuraciones — {PROTOCOL_LABELS.get(protocol, protocol)}",
        y=1.02,
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_protocol_scatter(summary):
    data = summary.pivot_table(
        index=["representation", "target", "model", "refinement"],
        columns="protocol",
        values="macro_f1_mean",
    ).reset_index()

    required = {"speaker_dependent", "speaker_independent"}
    if not required.issubset(data.columns):
        print("Faltan resultados de alguno de los protocolos.")
        return

    targets = [target for target in TARGET_LABELS if target in set(data["target"])]
    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(6 * len(targets), 5),
        squeeze=False,
    )

    for ax, target in zip(axes.ravel(), targets):
        target_data = data.query("target == @target").copy()
        ax.scatter(
            target_data["speaker_independent"],
            target_data["speaker_dependent"],
            alpha=0.8,
        )

        lower = float(
            min(
                target_data["speaker_independent"].min(),
                target_data["speaker_dependent"].min(),
            )
        )
        upper = float(
            max(
                target_data["speaker_independent"].max(),
                target_data["speaker_dependent"].max(),
            )
        )
        margin = max(0.02, (upper - lower) * 0.08)
        ax.plot(
            [lower - margin, upper + margin],
            [lower - margin, upper + margin],
            linestyle="--",
            linewidth=1,
        )
        ax.set_xlim(lower - margin, upper + margin)
        ax.set_ylim(lower - margin, upper + margin)
        ax.set_xlabel("Macro F1 speaker-independent")
        ax.set_ylabel("Macro F1 speaker-dependent")
        ax.set_title(TARGET_LABELS[target])
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Brecha de generalización entre protocolos\n"
        "Los puntos sobre la diagonal rinden mejor con hablantes conocidos",
        y=1.05,
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_8_to_4_tradeoff(cv_results):
    required = {"original_macro_f1", "macro_f1"}
    if cv_results.empty or not required.issubset(cv_results.columns):
        print("No hay métricas 8→4 disponibles.")
        return

    data = (
        cv_results
        .query("target == 'emotion_original_eval_quadrant'")
        .groupby(
            ["representation", "protocol", "model", "refinement"],
            observed=True,
        )
        .agg(
            original_macro_f1=("original_macro_f1", "mean"),
            quadrant_macro_f1=("macro_f1", "mean"),
        )
        .reset_index()
    )

    if data.empty:
        print("No hay resultados 8→4 disponibles.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for protocol, group in data.groupby("protocol", observed=True):
        ax.scatter(
            group["original_macro_f1"],
            group["quadrant_macro_f1"],
            label=PROTOCOL_LABELS.get(protocol, protocol),
            alpha=0.85,
        )

    ax.set_xlabel("Macro F1 original — 8 emociones")
    ax.set_ylabel("Macro F1 después del colapso — 4 cuadrantes")
    ax.set_title("Efecto del colapso 8→4")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_normalized_confusion(
    predictions,
    title,
    labels=None,
    ax=None,
):
    if predictions.empty:
        raise ValueError("No hay predicciones para la matriz de confusión.")

    use_eval_space = {
        "y_true_eval",
        "y_pred_eval",
    }.issubset(predictions.columns) and predictions["y_true_eval"].notna().any()

    true_col = "y_true_eval" if use_eval_space else "y_true"
    pred_col = "y_pred_eval" if use_eval_space else "y_pred"

    y_true = predictions[true_col].astype(str)
    y_pred = predictions[pred_col].astype(str)

    resolved_labels = (
        list(labels)
        if labels is not None
        else sorted(set(y_true) | set(y_pred))
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=resolved_labels,
        normalize="true",
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(5, len(resolved_labels) * 0.8), max(4, len(resolved_labels) * 0.7))
        )

    image = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(resolved_labels)))
    ax.set_yticks(range(len(resolved_labels)))
    ax.set_xticklabels(resolved_labels, rotation=45, ha="right")
    ax.set_yticklabels(resolved_labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase verdadera")
    ax.set_title(title)

    threshold = 0.5
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    return image, matrix, resolved_labels


def plot_confusion_delta(
    baseline_predictions,
    refined_predictions,
    title,
    labels=None,
    ax=None,
):
    use_eval_space = (
        {"y_true_eval", "y_pred_eval"}.issubset(baseline_predictions.columns)
        and baseline_predictions["y_true_eval"].notna().any()
    )
    true_col = "y_true_eval" if use_eval_space else "y_true"
    pred_col = "y_pred_eval" if use_eval_space else "y_pred"

    if labels is None:
        labels = sorted(
            set(baseline_predictions[true_col].astype(str))
            | set(baseline_predictions[pred_col].astype(str))
            | set(refined_predictions[true_col].astype(str))
            | set(refined_predictions[pred_col].astype(str))
        )

    base_matrix = confusion_matrix(
        baseline_predictions[true_col].astype(str),
        baseline_predictions[pred_col].astype(str),
        labels=labels,
        normalize="true",
    )
    refined_matrix = confusion_matrix(
        refined_predictions[true_col].astype(str),
        refined_predictions[pred_col].astype(str),
        labels=labels,
        normalize="true",
    )
    delta = refined_matrix - base_matrix

    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, len(labels) * 0.8), max(4, len(labels) * 0.7)))

    limit = max(0.05, float(np.abs(delta).max()))
    image = ax.imshow(delta, vmin=-limit, vmax=limit, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase verdadera")
    ax.set_title(title)

    for row_idx in range(delta.shape[0]):
        for col_idx in range(delta.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{delta[row_idx, col_idx]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    return image, delta