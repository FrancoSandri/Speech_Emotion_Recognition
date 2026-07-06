"""Evaluación zero-shot y curvas de adaptación de cabeza entre dominios."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.models.linear_head_adaptation import adapt_linear_head


@dataclass(frozen=True)
class DomainMatrix:
    X: np.ndarray
    y: np.ndarray
    file_ids: np.ndarray
    speaker_ids: np.ndarray
    emotion_native: np.ndarray


def build_domain_matrix(
    table: pd.DataFrame,
    feature_columns: Sequence[str],
) -> DomainMatrix:
    """Convierte una tabla alineada a matrices y metadatos."""

    required = {
        "file_id",
        "speaker_id",
        "emotion_original",
        "emotion_native",
        *feature_columns,
    }
    missing = required - set(table.columns)
    if missing:
        raise KeyError(
            f"Faltan columnas para DomainMatrix: {sorted(missing)[:10]}."
        )

    return DomainMatrix(
        X=table[list(feature_columns)].to_numpy(dtype=np.float32),
        y=table["emotion_original"].astype(str).to_numpy(),
        file_ids=table["file_id"].astype(str).to_numpy(),
        speaker_ids=table["speaker_id"].astype(str).to_numpy(),
        emotion_native=table["emotion_native"].astype(str).to_numpy(),
    )


def _balanced_accuracy_over_observed_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    observed_labels: Sequence[str],
) -> float:
    """Calcula BA como recall macro sobre las clases presentes en ``y_true``.

    Es matemáticamente equivalente a ``balanced_accuracy_score`` en este
    escenario, pero evita el warning de sklearn cuando la cabeza fuente
    predice clases que no tienen soporte en el dominio destino.

    Una predicción hacia una clase ausente sigue contando como falso negativo
    de la clase verdadera y, por lo tanto, reduce su recall.
    """

    return float(
        recall_score(
            y_true,
            y_pred,
            labels=list(observed_labels),
            average="macro",
            zero_division=0,
        )
    )


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    source_labels: Sequence[str],
) -> dict[str, float]:
    """Métricas target con BA como medida principal."""

    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true e y_pred deben ser vectores unidimensionales.")
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError(
            "y_true e y_pred deben tener igual longitud y no estar vacíos."
        )

    observed_labels = sorted(set(y_true))
    predicted_only_labels = sorted(set(y_pred) - set(observed_labels))

    return {
        "balanced_accuracy": _balanced_accuracy_over_observed_classes(
            y_true,
            y_pred,
            observed_labels,
        ),
        "macro_f1_observed": float(
            f1_score(
                y_true,
                y_pred,
                labels=observed_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1_source_space": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(source_labels),
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "unsupported_prediction_rate": float(
            np.mean(~np.isin(y_pred, observed_labels))
        ),
        "n_unsupported_prediction_classes": int(
            len(predicted_only_labels)
        ),
        "n_samples": int(len(y_true)),
        "n_observed_classes": int(len(observed_labels)),
    }


def prediction_frame(
    matrix: DomainMatrix,
    y_pred: Sequence[str],
    *,
    domain: str,
    representation: str,
    stage: str,
    run_id: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": matrix.file_ids,
            "speaker_id": matrix.speaker_ids,
            "emotion_native": matrix.emotion_native,
            "y_true": matrix.y,
            "y_pred": np.asarray(y_pred).astype(str),
            "correct": matrix.y == np.asarray(y_pred).astype(str),
            "domain": domain,
            "representation": representation,
            "stage": stage,
            "run_id": run_id,
        }
    )


def evaluate_zero_shot(
    source_pipelines: Mapping[str, Any],
    target_tests: Mapping[str, Mapping[str, DomainMatrix]],
    *,
    source_labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa las cabezas fuente sin adaptación en cada dominio."""

    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    for domain, representations in target_tests.items():
        for representation, matrix in representations.items():
            pipeline = source_pipelines[representation]
            y_pred = pipeline.predict(matrix.X).astype(str)
            run_id = f"zero_shot::{domain}::{representation}"
            rows.append(
                {
                    "domain": domain,
                    "representation": representation,
                    "stage": "zero_shot",
                    "n_adaptation_speakers": 0,
                    "n_adaptation_samples": 0,
                    "repeat": -1,
                    "seed": -1,
                    "run_id": run_id,
                    **evaluate_predictions(
                        matrix.y,
                        y_pred,
                        source_labels=source_labels,
                    ),
                }
            )
            predictions.append(
                prediction_frame(
                    matrix,
                    y_pred,
                    domain=domain,
                    representation=representation,
                    stage="zero_shot",
                    run_id=run_id,
                )
            )

    return (
        pd.DataFrame(rows),
        pd.concat(predictions, ignore_index=True, sort=False),
    )


def _resolve_speaker_sizes(
    requested: Sequence[int | str],
    n_available: int,
) -> list[int]:
    values: list[int] = []
    for value in requested:
        resolved = n_available if str(value).lower() == "all" else int(value)
        if resolved < 0:
            raise ValueError("Los tamaños no pueden ser negativos.")
        if resolved <= n_available:
            values.append(resolved)
    values.extend([0, n_available])
    return sorted(set(values))


def _sample_speakers(
    available: Sequence[str],
    n_speakers: int,
    *,
    seed: int,
) -> list[str]:
    available = sorted(map(str, available))
    if n_speakers == len(available):
        return available
    rng = np.random.default_rng(seed)
    return sorted(
        rng.choice(
            available,
            size=n_speakers,
            replace=False,
        ).tolist()
    )


def run_head_adaptation_curve(
    *,
    domain: str,
    representation: str,
    source_pipeline,
    adaptation_pool: DomainMatrix,
    target_test: DomainMatrix,
    source_test: DomainMatrix,
    source_labels: Sequence[str],
    speaker_sizes: Sequence[int | str],
    repeats: int,
    optimization_seeds: Sequence[int],
    subset_seed: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    source_anchor: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapta solo la cabeza y evalúa target y retención fuente."""

    available_speakers = sorted(set(adaptation_pool.speaker_ids))
    sizes = _resolve_speaker_sizes(
        speaker_sizes,
        len(available_speakers),
    )

    zero_target_pred = source_pipeline.predict(target_test.X).astype(str)
    zero_source_pred = source_pipeline.predict(source_test.X).astype(str)
    zero_target_metrics = evaluate_predictions(
        target_test.y,
        zero_target_pred,
        source_labels=source_labels,
    )
    zero_source_metrics = evaluate_predictions(
        source_test.y,
        zero_source_pred,
        source_labels=source_labels,
    )

    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    zero_run_id = f"zero_shot::{domain}::{representation}"
    rows.append(
        {
            "domain": domain,
            "representation": representation,
            "stage": "zero_shot",
            "n_adaptation_speakers": 0,
            "n_adaptation_samples": 0,
            "repeat": -1,
            "seed": -1,
            "adaptation_speakers": "[]",
            "train_loss": np.nan,
            "run_id": zero_run_id,
            "target_balanced_accuracy": zero_target_metrics[
                "balanced_accuracy"
            ],
            "target_macro_f1_observed": zero_target_metrics[
                "macro_f1_observed"
            ],
            "target_accuracy": zero_target_metrics["accuracy"],
            "unsupported_prediction_rate": zero_target_metrics[
                "unsupported_prediction_rate"
            ],
            "source_balanced_accuracy": zero_source_metrics[
                "balanced_accuracy"
            ],
            "target_gain": 0.0,
            "source_change": 0.0,
        }
    )
    prediction_frames.append(
        prediction_frame(
            target_test,
            zero_target_pred,
            domain=domain,
            representation=representation,
            stage="zero_shot",
            run_id=zero_run_id,
        )
    )

    for n_speakers in [value for value in sizes if value > 0]:
        effective_repeats = 1 if n_speakers == len(available_speakers) else repeats

        for repeat in range(effective_repeats):
            selected_speakers = _sample_speakers(
                available_speakers,
                n_speakers,
                seed=subset_seed + 10_000 * n_speakers + repeat,
            )
            adaptation_mask = np.isin(
                adaptation_pool.speaker_ids,
                selected_speakers,
            )
            X_adapt = adaptation_pool.X[adaptation_mask]
            y_adapt = adaptation_pool.y[adaptation_mask]

            for seed in optimization_seeds:
                run_id = (
                    f"adapt::{domain}::{representation}::"
                    f"n{n_speakers}::r{repeat}::s{seed}"
                )
                adapted = adapt_linear_head(
                    source_pipeline,
                    X_adapt,
                    y_adapt,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    seed=int(seed),
                    source_anchor=source_anchor,
                )
                target_pred = adapted.predict(target_test.X).astype(str)
                source_pred = adapted.predict(source_test.X).astype(str)

                target_metrics = evaluate_predictions(
                    target_test.y,
                    target_pred,
                    source_labels=source_labels,
                )
                source_metrics = evaluate_predictions(
                    source_test.y,
                    source_pred,
                    source_labels=source_labels,
                )

                rows.append(
                    {
                        "domain": domain,
                        "representation": representation,
                        "stage": "adapted",
                        "n_adaptation_speakers": int(n_speakers),
                        "n_adaptation_samples": int(len(y_adapt)),
                        "repeat": int(repeat),
                        "seed": int(seed),
                        "adaptation_speakers": json.dumps(
                            selected_speakers
                        ),
                        "train_loss": float(adapted.train_loss),
                        "run_id": run_id,
                        "target_balanced_accuracy": target_metrics[
                            "balanced_accuracy"
                        ],
                        "target_macro_f1_observed": target_metrics[
                            "macro_f1_observed"
                        ],
                        "target_accuracy": target_metrics["accuracy"],
                        "unsupported_prediction_rate": target_metrics[
                            "unsupported_prediction_rate"
                        ],
                        "source_balanced_accuracy": source_metrics[
                            "balanced_accuracy"
                        ],
                        "target_gain": (
                            target_metrics["balanced_accuracy"]
                            - zero_target_metrics["balanced_accuracy"]
                        ),
                        "source_change": (
                            source_metrics["balanced_accuracy"]
                            - zero_source_metrics["balanced_accuracy"]
                        ),
                    }
                )
                prediction_frames.append(
                    prediction_frame(
                        target_test,
                        target_pred,
                        domain=domain,
                        representation=representation,
                        stage="adapted",
                        run_id=run_id,
                    )
                )

    return (
        pd.DataFrame(rows),
        pd.concat(prediction_frames, ignore_index=True, sort=False),
    )


def summarize_adaptation_curve(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Resume media y dispersión por dominio, representación y tamaño."""

    metrics = [
        "target_balanced_accuracy",
        "source_balanced_accuracy",
        "target_gain",
        "source_change",
        "unsupported_prediction_rate",
    ]
    grouped = (
        results.groupby(
            [
                "domain",
                "representation",
                "n_adaptation_speakers",
            ],
            observed=True,
            as_index=False,
        )[metrics]
        .agg(["mean", "std", "count"])
    )
    grouped.columns = [
        "_".join(
            [str(part) for part in column if str(part)]
        )
        for column in grouped.columns.to_flat_index()
    ]
    return grouped


def select_representative_max_run(
    results: pd.DataFrame,
    *,
    domain: str,
    representation: str,
) -> str:
    """Selecciona el run máximo más cercano a la BA media, sin elegir el mejor."""

    subset = results.loc[
        results["domain"].eq(domain)
        & results["representation"].eq(representation)
        & results["stage"].eq("adapted")
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No hay runs adaptados para {domain}/{representation}."
        )

    max_size = int(subset["n_adaptation_speakers"].max())
    subset = subset.loc[
        subset["n_adaptation_speakers"].eq(max_size)
    ].copy()
    mean_score = subset["target_balanced_accuracy"].mean()
    subset["distance_to_mean"] = (
        subset["target_balanced_accuracy"] - mean_score
    ).abs()
    return str(
        subset.sort_values(
            ["distance_to_mean", "run_id"]
        ).iloc[0]["run_id"]
    )


def rectangular_confusion(
    predictions: pd.DataFrame,
    *,
    source_labels: Sequence[str],
    row_labels: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Matriz rectangular: clases verdaderas target × salidas RAVDESS."""

    resolved_rows = (
        list(row_labels)
        if row_labels is not None
        else [
            label
            for label in source_labels
            if label in set(predictions["y_true"])
        ]
    )
    resolved_columns = list(source_labels)

    counts = confusion_matrix(
        predictions["y_true"],
        predictions["y_pred"],
        labels=[*resolved_rows, *[
            label for label in resolved_columns
            if label not in resolved_rows
        ]],
    )
    # sklearn produce una matriz cuadrada; extraemos filas target y columnas fuente.
    all_labels = [
        *resolved_rows,
        *[
            label for label in resolved_columns
            if label not in resolved_rows
        ],
    ]
    row_indices = [all_labels.index(label) for label in resolved_rows]
    column_indices = [
        all_labels.index(label) for label in resolved_columns
    ]
    rectangular_counts = counts[np.ix_(row_indices, column_indices)]
    denominators = rectangular_counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        rectangular_counts,
        denominators,
        out=np.zeros_like(rectangular_counts, dtype=float),
        where=denominators > 0,
    )
    return (
        rectangular_counts,
        normalized,
        resolved_rows,
        resolved_columns,
    )
