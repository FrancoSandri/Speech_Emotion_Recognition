"""Persistencia y síntesis de experimentos temporales wav2vec.

El módulo centraliza la carga, consolidación y selección de artefactos. No
entrena modelos y no accede al conjunto de test final.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd


RESULT_KEYS = [
    "representation",
    "protocol",
    "target",
    "model",
    "refinement",
    "fold",
    "seed",
    "result_type",
]
PREDICTION_KEYS = [
    "file_id",
    "representation",
    "protocol",
    "target",
    "model",
    "refinement",
]
PREDICTION_LOAD_KEYS = [*PREDICTION_KEYS, "fold"]
LAYER_WEIGHT_KEYS = [
    "configuration",
    "protocol",
    "target",
    "fold",
    "seed",
    "layer",
]
DIAGNOSTIC_KEYS = [
    "file_id",
    "configuration",
    "protocol",
    "target",
    "seed",
]


@dataclass(frozen=True)
class TemporalArtifactPaths:
    """Rutas de los artefactos persistidos por el notebook temporal."""

    results: Path
    predictions: Path
    layer_weights: Path
    attention_diagnostics: Path
    attention_weights: Path


@dataclass
class TemporalArtifacts:
    """Estado consolidado de resultados temporales."""

    results: pd.DataFrame
    predictions: pd.DataFrame
    layer_weights: pd.DataFrame
    attention_diagnostics: pd.DataFrame
    attention_weights: dict[str, np.ndarray]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Carga un CSV o devuelve un DataFrame vacío."""

    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Guarda un DataFrame creando el directorio de destino."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _deduplicate_loaded_rows(
    frame: pd.DataFrame,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Elimina repeticiones exactas de clave en artefactos persistidos.

    Stores generados por versiones previas del notebook pueden conservar más
    de una ejecución de la misma configuración. Se mantiene la fila más
    reciente según el orden del CSV. Las configuraciones diferentes no se
    mezclan porque la deduplicación usa la clave experimental completa.
    """

    if frame.empty or not set(keys).issubset(frame.columns):
        return frame.copy().reset_index(drop=True)

    return (
        frame.drop_duplicates(subset=list(keys), keep="last")
        .reset_index(drop=True)
    )


def upsert_rows(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Reemplaza filas con la misma clave y conserva el resto del artefacto."""

    if existing.empty:
        return new.copy().reset_index(drop=True)
    if new.empty:
        return existing.copy().reset_index(drop=True)

    missing = (set(keys) - set(existing.columns)) | (
        set(keys) - set(new.columns)
    )
    if missing:
        raise KeyError(
            "Faltan claves para consolidar resultados: "
            f"{sorted(missing)}"
        )

    old_index = pd.MultiIndex.from_frame(
        existing.loc[:, list(keys)].fillna("<NA>")
    )
    new_index = pd.MultiIndex.from_frame(
        new.loc[:, list(keys)].fillna("<NA>")
    )
    retained = existing.loc[~old_index.isin(new_index)]
    return pd.concat([retained, new], ignore_index=True, sort=False)


def _normalize_layer_weight_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``configuration`` a stores antiguos de layer weights."""

    if frame.empty or "configuration" in frame.columns:
        return frame
    if "pooling" not in frame.columns:
        raise KeyError(
            "El store de layer weights no contiene configuration ni pooling."
        )
    normalized = frame.copy()
    normalized["configuration"] = (
        "learned_layers_" + normalized["pooling"].astype(str)
    )
    return normalized


def load_temporal_artifacts(
    paths: TemporalArtifactPaths,
    load_attention_weights_fn,
) -> TemporalArtifacts:
    """Carga todos los artefactos temporales con compatibilidad de esquema."""

    layer_weights = _normalize_layer_weight_schema(
        load_csv(paths.layer_weights)
    )
    return TemporalArtifacts(
        results=_deduplicate_loaded_rows(
            load_csv(paths.results),
            RESULT_KEYS,
        ),
        predictions=_deduplicate_loaded_rows(
            load_csv(paths.predictions),
            PREDICTION_LOAD_KEYS,
        ),
        layer_weights=_deduplicate_loaded_rows(
            layer_weights,
            LAYER_WEIGHT_KEYS,
        ),
        attention_diagnostics=_deduplicate_loaded_rows(
            load_csv(paths.attention_diagnostics),
            DIAGNOSTIC_KEYS,
        ),
        attention_weights=load_attention_weights_fn(paths.attention_weights),
    )


def persist_temporal_artifacts(
    artifacts: TemporalArtifacts,
    paths: TemporalArtifactPaths,
    save_attention_weights_fn,
) -> None:
    """Persiste el estado consolidado de experimentos temporales."""

    save_csv(artifacts.results, paths.results)
    save_csv(artifacts.predictions, paths.predictions)
    save_csv(artifacts.layer_weights, paths.layer_weights)
    save_csv(artifacts.attention_diagnostics, paths.attention_diagnostics)
    save_attention_weights_fn(
        artifacts.attention_weights,
        paths.attention_weights,
    )


def merge_experiment_output(
    artifacts: TemporalArtifacts,
    output: Mapping[str, Any],
) -> TemporalArtifacts:
    """Consolida la salida de un experimento sin usar variables globales."""

    merged = TemporalArtifacts(
        results=upsert_rows(
            artifacts.results,
            output.get("fold_results", pd.DataFrame()),
            RESULT_KEYS,
        ),
        predictions=upsert_rows(
            artifacts.predictions,
            output.get("predictions", pd.DataFrame()),
            PREDICTION_KEYS,
        ),
        layer_weights=upsert_rows(
            artifacts.layer_weights,
            output.get("layer_weights", pd.DataFrame()),
            LAYER_WEIGHT_KEYS,
        ),
        attention_diagnostics=upsert_rows(
            artifacts.attention_diagnostics,
            output.get("attention_diagnostics", pd.DataFrame()),
            DIAGNOSTIC_KEYS,
        ),
        attention_weights=dict(artifacts.attention_weights),
    )
    merged.attention_weights.update(output.get("attention_weights", {}))
    return merged


def select_primary_results(
    results: pd.DataFrame,
    *,
    protocol: str,
    target: str,
    configurations: Sequence[str],
    include_seed_rows: bool = False,
) -> pd.DataFrame:
    """Filtra las configuraciones primarias de un experimento temporal."""

    if results.empty:
        return pd.DataFrame()

    required = {"protocol", "target", "refinement"}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(
            f"Faltan columnas para filtrar resultados: {sorted(missing)}"
        )

    selected = results.loc[
        results["protocol"].eq(protocol)
        & results["target"].eq(target)
        & results["refinement"].isin(configurations)
    ].copy()

    if (
        not include_seed_rows
        and "result_type" in selected.columns
    ):
        selected = selected.loc[~selected["result_type"].eq("seed")]

    return selected.reset_index(drop=True)


def summarize_primary_results(
    primary_results: pd.DataFrame,
    *,
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Resume rendimiento y estabilidad entre outer folds."""

    if primary_results.empty:
        return pd.DataFrame()
    if metric not in primary_results.columns:
        raise KeyError(f"La métrica {metric!r} no está disponible.")

    return (
        primary_results.groupby("refinement", observed=True)
        .agg(
            metric_mean=(metric, "mean"),
            metric_std=(metric, "std"),
            worst_fold=(metric, "min"),
            best_fold=(metric, "max"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["metric_mean", "metric_std"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def select_best_configuration(
    summary: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    """Selecciona por métrica media y, ante empate, menor desvío."""

    if summary.empty:
        return None
    subset = summary.loc[summary["refinement"].isin(candidates)].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(
        ["metric_mean", "metric_std"],
        ascending=[False, True],
    )
    return str(subset.iloc[0]["refinement"])


def _align_predictions_with_current_splits(
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    fold_column: str,
) -> pd.DataFrame:
    """Conserva únicamente la predicción correspondiente al fold vigente.

    Los artefactos generados con versiones anteriores podían acumular una fila
    del mismo audio por cada asignación histórica de fold porque ``fold`` se
    utilizaba como parte de la clave de persistencia. La identidad OOF real es
    ``file_id + configuración``; el fold es un atributo de esa predicción.
    """

    required_predictions = {"file_id", "fold"}
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_predictions:
        raise KeyError(
            "Faltan columnas para alinear predicciones OOF: "
            f"{sorted(missing_predictions)}"
        )

    required_splits = {"file_id", fold_column}
    missing_splits = required_splits - set(splits.columns)
    if missing_splits:
        raise KeyError(
            "Faltan columnas en splits para alinear predicciones OOF: "
            f"{sorted(missing_splits)}"
        )
    if splits["file_id"].duplicated().any():
        raise ValueError("splits debe contener una única fila por file_id.")

    expected = splits[["file_id", fold_column]].rename(
        columns={fold_column: "_expected_fold"}
    )
    aligned = predictions.merge(
        expected,
        on="file_id",
        how="left",
        validate="many_to_one",
    )
    if aligned["_expected_fold"].isna().any():
        missing_ids = (
            aligned.loc[aligned["_expected_fold"].isna(), "file_id"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Hay predicciones OOF sin asignación en splits. Ejemplos: "
            f"{missing_ids}"
        )

    observed_fold = pd.to_numeric(aligned["fold"], errors="coerce")
    expected_fold = pd.to_numeric(aligned["_expected_fold"], errors="coerce")
    valid = observed_fold.eq(expected_fold)
    aligned = aligned.loc[valid].drop(columns="_expected_fold")

    if aligned.empty:
        raise ValueError(
            "Ninguna predicción OOF coincide con la asignación de folds "
            "vigente. Revise que el CSV y splits.parquet pertenezcan a la "
            "misma ejecución experimental."
        )
    return aligned


def select_oof_predictions(
    predictions: pd.DataFrame,
    *,
    protocol: str,
    target: str,
    refinement: str,
    representation: str | None = None,
    model: str | None = None,
    splits: pd.DataFrame | None = None,
    fold_column: str | None = None,
) -> pd.DataFrame:
    """Recupera una única configuración de predicciones OOF ensemble.

    ``refinement`` no siempre identifica por sí solo una configuración. Los
    filtros de representación y modelo evitan mezclar ejecuciones distintas.
    Cuando se reciben ``splits`` y ``fold_column``, las filas legacy se alinean
    con la asignación de folds vigente antes de eliminar duplicados.
    """

    if predictions.empty:
        return pd.DataFrame()

    required = {"file_id", "protocol", "target", "refinement"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(
            f"Faltan columnas para filtrar predicciones: {sorted(missing)}"
        )

    mask = (
        predictions["protocol"].eq(protocol)
        & predictions["target"].eq(target)
        & predictions["refinement"].eq(refinement)
    )
    if representation is not None:
        if "representation" not in predictions.columns:
            raise KeyError(
                "Se solicitó filtrar por representation, pero la columna "
                "no existe en el artefacto OOF."
            )
        mask &= predictions["representation"].eq(representation)
    if model is not None:
        if "model" not in predictions.columns:
            raise KeyError(
                "Se solicitó filtrar por model, pero la columna no existe "
                "en el artefacto OOF."
            )
        mask &= predictions["model"].eq(model)

    selected = predictions.loc[mask].copy()
    if selected.empty:
        return selected

    configuration_columns = [
        column
        for column in ("representation", "model")
        if column in selected.columns
    ]
    configurations = selected[configuration_columns].drop_duplicates()
    if len(configurations) > 1:
        raise ValueError(
            "El filtro OOF seleccionó más de una configuración. "
            "Especifique representation y model. Configuraciones: "
            f"{configurations.to_dict(orient='records')}"
        )

    if (splits is None) != (fold_column is None):
        raise ValueError(
            "splits y fold_column deben proporcionarse juntos."
        )
    if splits is not None and fold_column is not None:
        selected = _align_predictions_with_current_splits(
            selected,
            splits,
            fold_column=fold_column,
        )

    duplicate_count = int(selected["file_id"].duplicated(keep=False).sum())
    if duplicate_count:
        warnings.warn(
            "Se encontraron filas OOF legacy repetidas para la misma "
            "configuración y file_id. Se conserva la última fila persistida "
            f"({duplicate_count} filas involucradas).",
            RuntimeWarning,
            stacklevel=2,
        )

    # El fold no forma parte de la identidad OOF: cada archivo debe tener una
    # única predicción por configuración, aunque haya cambiado de fold entre
    # versiones históricas de splits.
    selected = selected.drop_duplicates(
        subset=["file_id"],
        keep="last",
    )

    sort_columns = [
        column for column in ("fold", "file_id") if column in selected.columns
    ]
    if sort_columns:
        selected = selected.sort_values(sort_columns)
    return selected.reset_index(drop=True)