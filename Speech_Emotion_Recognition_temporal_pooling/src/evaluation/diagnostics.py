"""Diagnósticos compactos sobre predicciones out-of-fold.

Las funciones trabajan exclusivamente con predicciones OOF del development
pool. No entrenan modelos ni acceden a los test finales.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics


PREDICTION_ALIASES = {
    "y_true": ("y_true", "target_true"),
    "y_pred": ("y_pred", "target_pred"),
}


def _resolve_prediction_column(frame: pd.DataFrame, canonical: str) -> str:
    for candidate in PREDICTION_ALIASES[canonical]:
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        f"No se encontró una columna compatible con {canonical!r}. "
        f"Opciones esperadas: {PREDICTION_ALIASES[canonical]}"
    )


def build_oof_diagnostics(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Une predicciones OOF y metadata mediante ``file_id``.

    La salida normaliza los nombres a ``target_true`` y ``target_pred`` y
    agrega una columna booleana ``correct``.
    """
    if predictions.empty:
        return pd.DataFrame()
    if "file_id" not in predictions.columns or "file_id" not in metadata.columns:
        raise KeyError("predictions y metadata deben contener file_id.")
    if predictions["file_id"].duplicated().any():
        duplicates = int(predictions["file_id"].duplicated().sum())
        raise ValueError(
            "Las predicciones OOF deben contener una fila por file_id y "
            f"configuración. Se detectaron {duplicates} duplicados."
        )
    if metadata["file_id"].duplicated().any():
        raise ValueError("metadata contiene file_id duplicados.")

    true_col = _resolve_prediction_column(predictions, "y_true")
    pred_col = _resolve_prediction_column(predictions, "y_pred")

    metadata_columns = [
        column
        for column in (
            "file_id",
            "actor_id",
            "sex",
            "emotion_original",
            "emotion_quadrant",
            "intensity",
            "statement",
            "repetition_id",
            "duration_trimmed_s",
        )
        if column in metadata.columns
    ]

    merged = predictions.merge(
        metadata[metadata_columns],
        on="file_id",
        how="left",
        validate="many_to_one",
    )
    if merged["actor_id"].isna().any():
        raise ValueError("Existen predicciones OOF sin metadata asociada.")

    merged = merged.rename(
        columns={true_col: "target_true", pred_col: "target_pred"}
    )
    merged["correct"] = merged["target_true"].eq(merged["target_pred"])
    return merged


def compute_actor_metrics(
    diagnostics: pd.DataFrame,
    group_columns: Sequence[str] = ("representation", "actor_id"),
) -> pd.DataFrame:
    """Calcula métricas por actor para cada configuración seleccionada."""
    if diagnostics.empty:
        return pd.DataFrame()

    required = set(group_columns) | {"target_true", "target_pred"}
    missing = required - set(diagnostics.columns)
    if missing:
        raise KeyError(f"Faltan columnas para métricas por actor: {sorted(missing)}")

    rows: list[dict] = []
    grouped = diagnostics.groupby(list(group_columns), observed=True, sort=False)
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        labels = sorted(group["target_true"].dropna().unique())
        row.update(compute_metrics(group["target_true"], group["target_pred"], labels))
        row["n_samples"] = int(len(group))
        row["n_errors"] = int((~group["correct"]).sum())
        rows.append(row)

    return pd.DataFrame(rows)


def compute_actor_class_recall(
    diagnostics: pd.DataFrame,
    group_columns: Sequence[str] = ("representation", "actor_id"),
) -> pd.DataFrame:
    """Calcula recall por actor y clase verdadera."""
    if diagnostics.empty:
        return pd.DataFrame()

    required = set(group_columns) | {"target_true", "target_pred"}
    missing = required - set(diagnostics.columns)
    if missing:
        raise KeyError(f"Faltan columnas para recall actor×clase: {sorted(missing)}")

    rows: list[dict] = []
    grouped = diagnostics.groupby(
        [*group_columns, "target_true"],
        observed=True,
        sort=False,
    )
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        base_keys = keys[:-1]
        target_class = keys[-1]
        row = dict(zip(group_columns, base_keys))
        row["target_class"] = target_class
        row["recall"] = float(group["target_pred"].eq(target_class).mean())
        row["support"] = int(len(group))
        rows.append(row)

    return pd.DataFrame(rows)


def compute_fold_metrics_from_predictions(
    diagnostics: pd.DataFrame,
    group_columns: Sequence[str] = ("representation", "fold"),
) -> pd.DataFrame:
    """Reconstruye métricas por fold a partir de predicciones OOF."""
    if diagnostics.empty:
        return pd.DataFrame()

    required = set(group_columns) | {"target_true", "target_pred"}
    missing = required - set(diagnostics.columns)
    if missing:
        raise KeyError(f"Faltan columnas para métricas por fold: {sorted(missing)}")

    rows: list[dict] = []
    grouped = diagnostics.groupby(list(group_columns), observed=True, sort=False)
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        labels = sorted(group["target_true"].dropna().unique())
        row.update(compute_metrics(group["target_true"], group["target_pred"], labels))
        row["n_validation"] = int(len(group))
        if "actor_id" in group.columns:
            row["validation_actors"] = ", ".join(
                str(value) for value in sorted(group["actor_id"].unique())
            )
        rows.append(row)

    return pd.DataFrame(rows)
