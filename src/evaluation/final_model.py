"""Evaluación final reproducible sobre un test speaker-independent.

El módulo separa explícitamente:

1. auditoría de particiones;
2. preparación exclusiva de development;
3. refit final;
4. apertura única del test;
5. evaluación, análisis y persistencia.

No realiza selección de hiperparámetros ni ajusta decisiones usando el test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_metrics,
    full_classification_report,
    map_emotions_to_quadrants,
)
from src.features.feature_store import prepare_model_table
from src.utils.config import resolve_path


@dataclass(frozen=True)
class PartitionDataset:
    """Matriz y metadata alineadas para una representación y partición."""

    X: np.ndarray
    y: np.ndarray
    file_ids: np.ndarray
    actor_ids: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(len(self.y))

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def audit_speaker_independent_split(
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    development_partition: str,
    test_partition: str,
) -> dict[str, Any]:
    """Valida exclusión por archivo y actor sin materializar features de test."""

    for name, frame in {"metadata": metadata, "splits": splits}.items():
        if "file_id" not in frame.columns:
            raise KeyError(f"{name} no contiene file_id.")
        if frame["file_id"].isna().any() or frame["file_id"].duplicated().any():
            raise ValueError(f"{name} debe contener file_id únicos y no nulos.")

    table = metadata[["file_id", "actor_id"]].merge(
        splits[["file_id", "partition"]],
        on="file_id",
        how="inner",
        validate="one_to_one",
    )

    development = table.loc[table["partition"].eq(development_partition)]
    test = table.loc[table["partition"].eq(test_partition)]

    if development.empty or test.empty:
        raise ValueError(
            "Development y test deben contener observaciones. "
            f"development={len(development)}, test={len(test)}."
        )

    development_ids = set(development["file_id"].astype(str))
    test_ids = set(test["file_id"].astype(str))
    if not development_ids.isdisjoint(test_ids):
        raise ValueError("Development y test comparten file_id.")

    development_actors = set(development["actor_id"])
    test_actors = set(test["actor_id"])
    overlap = development_actors.intersection(test_actors)
    if overlap:
        raise ValueError(
            "El test no es speaker-independent. Actores compartidos: "
            f"{sorted(overlap)}"
        )

    return {
        "n_development": int(len(development)),
        "n_test": int(len(test)),
        "n_development_actors": int(len(development_actors)),
        "n_test_actors": int(len(test_actors)),
        "development_actors": sorted(development_actors),
        "test_actors": sorted(test_actors),
    }


def prepare_partition_datasets(
    model_specs: Mapping[str, Mapping[str, Any]],
    representations: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    partition: str,
    target_col: str,
) -> dict[str, PartitionDataset]:
    """Prepara una sola matriz por representación y la comparte entre modelos."""

    cache: dict[str, PartitionDataset] = {}
    datasets: dict[str, PartitionDataset] = {}

    for model_name, spec in model_specs.items():
        representation_name = str(spec["representation"])
        if representation_name not in representations:
            raise KeyError(
                f"No existe la representación {representation_name!r} "
                f"requerida por {model_name!r}."
            )

        if representation_name not in cache:
            table, feature_names = prepare_model_table(
                representation=representations[representation_name],
                metadata=metadata,
                splits=splits,
                partition=partition,
            )
            if target_col not in table.columns:
                raise KeyError(f"Target inexistente: {target_col!r}.")
            if "actor_id" not in table.columns:
                raise KeyError("La tabla alineada no contiene actor_id.")

            cache[representation_name] = PartitionDataset(
                X=table[feature_names].to_numpy(dtype=np.float32),
                y=table[target_col].astype(str).to_numpy(),
                file_ids=table["file_id"].astype(str).to_numpy(),
                actor_ids=table["actor_id"].to_numpy(),
                feature_names=tuple(feature_names),
            )

        datasets[model_name] = cache[representation_name]

    return datasets


def validate_expected_labels(
    dataset: PartitionDataset,
    expected_labels: Sequence[str],
) -> None:
    """Comprueba la cobertura de clases al abrir formalmente el test."""

    observed = set(map(str, np.unique(dataset.y)))
    expected = set(map(str, expected_labels))
    if observed != expected:
        raise ValueError(
            "Las etiquetas de la partición no coinciden con las esperadas. "
            f"faltantes={sorted(expected-observed)}, "
            f"extras={sorted(observed-expected)}."
        )


def fit_final_pipelines(
    model_specs: Mapping[str, Mapping[str, Any]],
    development_data: Mapping[str, PartitionDataset],
) -> dict[str, Any]:
    """Ajusta cada pipeline una vez sobre todo development."""

    fitted: dict[str, Any] = {}
    for model_name, spec in model_specs.items():
        if model_name not in development_data:
            raise KeyError(f"Faltan datos development para {model_name!r}.")
        factory = spec.get("factory")
        if not callable(factory):
            raise TypeError(f"factory no es callable para {model_name!r}.")

        pipeline = factory()
        dataset = development_data[model_name]
        pipeline.fit(dataset.X, dataset.y)
        fitted[model_name] = pipeline

    return fitted


def _probability_frame(
    pipeline: Any,
    X: np.ndarray,
    labels: Sequence[str],
) -> pd.DataFrame:
    """Devuelve probabilidades alineadas al orden global de etiquetas."""

    columns = [f"probability_{label}" for label in labels]
    result = pd.DataFrame(np.nan, index=np.arange(len(X)), columns=columns)

    if not hasattr(pipeline, "predict_proba"):
        return result

    probabilities = np.asarray(pipeline.predict_proba(X), dtype=float)
    classes = [str(value) for value in pipeline.classes_]
    class_to_index = {label: index for index, label in enumerate(classes)}

    for label in labels:
        if str(label) in class_to_index:
            result[f"probability_{label}"] = probabilities[
                :, class_to_index[str(label)]
            ]
    return result


def evaluate_final_pipelines(
    fitted_pipelines: Mapping[str, Any],
    model_specs: Mapping[str, Mapping[str, Any]],
    test_data: Mapping[str, PartitionDataset],
    *,
    labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa una vez cada pipeline y devuelve resumen y predicciones auditables."""

    summary_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    reference_ids: np.ndarray | None = None
    reference_labels: np.ndarray | None = None

    for model_name, pipeline in fitted_pipelines.items():
        dataset = test_data[model_name]

        if reference_ids is None:
            reference_ids = dataset.file_ids
            reference_labels = dataset.y
        else:
            if not np.array_equal(reference_ids, dataset.file_ids):
                raise ValueError(
                    "Los modelos no están evaluándose sobre los mismos file_id."
                )
            if not np.array_equal(reference_labels, dataset.y):
                raise ValueError(
                    "Las etiquetas verdaderas difieren entre representaciones."
                )

        y_pred = np.asarray(pipeline.predict(dataset.X)).astype(str)
        metrics = compute_metrics(dataset.y, y_pred, labels=labels)

        summary_rows.append(
            {
                "model": model_name,
                "label": str(model_specs[model_name]["label"]),
                "representation": str(
                    model_specs[model_name]["representation"]
                ),
                "n_test": dataset.n_samples,
                "n_features": dataset.n_features,
                **metrics,
            }
        )

        frame = pd.DataFrame(
            {
                "file_id": dataset.file_ids,
                "actor_id": dataset.actor_ids,
                "model": model_name,
                "label": str(model_specs[model_name]["label"]),
                "representation": str(
                    model_specs[model_name]["representation"]
                ),
                "y_true": dataset.y,
                "y_pred": y_pred,
                "correct": dataset.y == y_pred,
            }
        )
        frame = pd.concat(
            [
                frame.reset_index(drop=True),
                _probability_frame(pipeline, dataset.X, labels),
            ],
            axis=1,
        )
        prediction_frames.append(frame)

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("balanced_accuracy", ascending=False)
        .reset_index(drop=True)
    )
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    return summary, predictions


def build_classification_report_table(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    labels: Sequence[str],
) -> pd.DataFrame:
    """Construye el reporte por clase para una configuración final."""

    selected = predictions.loc[predictions["model"].eq(model_name)]
    if selected.empty:
        raise ValueError(f"No hay predicciones para {model_name!r}.")

    report = full_classification_report(
        selected["y_true"],
        selected["y_pred"],
        labels=labels,
        as_dataframe=True,
    )
    return report


def compute_actor_metrics(
    predictions: pd.DataFrame,
    *,
    model_name: str | None = None,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calcula métricas por actor sin usar ``groupby.apply``."""

    selected = predictions
    if model_name is not None and "model" in selected.columns:
        selected = selected.loc[selected["model"].eq(model_name)]

    required = {"actor_id", "y_true", "y_pred"}
    missing = required - set(selected.columns)
    if missing:
        raise KeyError(f"Faltan columnas actor-level: {sorted(missing)}.")

    rows: list[dict[str, Any]] = []
    for actor_id, group in selected.groupby(
        "actor_id",
        observed=True,
        sort=True,
    ):
        metrics = compute_metrics(
            group["y_true"],
            group["y_pred"],
            labels=labels,
        )
        rows.append(
            {
                "actor_id": actor_id,
                "n_samples": int(len(group)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def compute_quadrant_analysis(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    quadrant_labels: Sequence[str] = ("Q1", "Q2", "Q3", "Q4"),
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Mapea la salida de emoción a cuadrantes y calcula métricas."""

    selected = predictions.loc[
        predictions["model"].eq(model_name)
    ].copy()
    if selected.empty:
        raise ValueError(f"No hay predicciones para {model_name!r}.")

    selected["y_true_quadrant"] = map_emotions_to_quadrants(
        selected["y_true"]
    )
    selected["y_pred_quadrant"] = map_emotions_to_quadrants(
        selected["y_pred"]
    )
    metrics = compute_metrics(
        selected["y_true_quadrant"],
        selected["y_pred_quadrant"],
        labels=quadrant_labels,
    )
    return selected, metrics


def load_cv_reference_table(
    source_paths: Mapping[str, str | Path],
    configurations: Mapping[str, Mapping[str, Any]],
    *,
    metric: str = "balanced_accuracy",
    fold_column: str = "fold",
) -> pd.DataFrame:
    """Recupera referencias CV exactas y usa desvío muestral entre folds."""

    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []

    for model_name, config in configurations.items():
        source_name = str(config["source"])
        if source_name not in source_paths:
            raise KeyError(f"Fuente CV desconocida: {source_name!r}.")

        if source_name not in frames:
            path = resolve_path(source_paths[source_name])
            if not path.exists():
                raise FileNotFoundError(f"Resultados CV no encontrados: {path}")
            frames[source_name] = pd.read_csv(path)

        selected = frames[source_name].copy()
        for column, value in dict(config["filters"]).items():
            if column not in selected.columns:
                raise KeyError(
                    f"{source_name} no contiene la columna {column!r} "
                    f"requerida por {model_name!r}."
                )
            selected = selected.loc[
                selected[column].astype(str).eq(str(value))
            ]

        if selected.empty:
            raise ValueError(
                f"No se encontraron filas CV para {model_name!r}: "
                f"{dict(config['filters'])}"
            )
        if metric not in selected.columns or fold_column not in selected.columns:
            raise KeyError(
                f"Faltan {metric!r} o {fold_column!r} en {source_name}."
            )

        fold_values: list[float] = []
        for fold, group in selected.groupby(fold_column, sort=True):
            unique_values = group[metric].dropna().astype(float).unique()
            if len(unique_values) != 1:
                raise ValueError(
                    f"El fold {fold} de {model_name!r} tiene múltiples "
                    f"valores de {metric}: {unique_values.tolist()}"
                )
            fold_values.append(float(unique_values[0]))

        values = np.asarray(fold_values, dtype=float)
        rows.append(
            {
                "model": model_name,
                "cv_mean": float(values.mean()),
                "cv_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "cv_min": float(values.min()),
                "cv_max": float(values.max()),
                "n_folds": int(len(values)),
            }
        )

    return pd.DataFrame(rows)


def build_cv_test_gap_table(
    cv_reference: pd.DataFrame,
    test_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compara estimación outer-CV y test sin convertir el gap en significancia."""

    required_cv = {"model", "cv_mean", "cv_std"}
    required_test = {"model", "label", "balanced_accuracy"}
    if missing := required_cv - set(cv_reference.columns):
        raise KeyError(f"Faltan columnas CV: {sorted(missing)}")
    if missing := required_test - set(test_summary.columns):
        raise KeyError(f"Faltan columnas test: {sorted(missing)}")

    comparison = cv_reference.merge(
        test_summary[
            ["model", "label", "balanced_accuracy"]
        ].rename(
            columns={"balanced_accuracy": "test_balanced_accuracy"}
        ),
        on="model",
        how="inner",
        validate="one_to_one",
    )
    comparison["gap_absolute"] = (
        comparison["test_balanced_accuracy"]
        - comparison["cv_mean"]
    )
    return comparison.sort_values(
        "test_balanced_accuracy",
        ascending=False,
    ).reset_index(drop=True)


def select_current_oof_predictions(
    predictions: pd.DataFrame | str | Path,
    *,
    filters: Mapping[str, Any],
    splits: pd.DataFrame,
    fold_column: str,
    development_partition: str,
) -> pd.DataFrame:
    """Selecciona OOF de la configuración exacta y los folds vigentes."""

    frame = (
        pd.read_csv(resolve_path(predictions))
        if isinstance(predictions, (str, Path))
        else predictions.copy()
    )
    selected = frame.copy()
    for column, value in filters.items():
        if column not in selected.columns:
            raise KeyError(f"Predicciones OOF sin columna {column!r}.")
        selected = selected.loc[
            selected[column].astype(str).eq(str(value))
        ]

    if selected.empty:
        raise ValueError(
            f"No hay predicciones OOF para los filtros: {dict(filters)}"
        )
    if "file_id" not in selected.columns:
        raise KeyError("Las predicciones OOF no contienen file_id.")
    if fold_column not in splits.columns or "partition" not in splits.columns:
        raise KeyError("splits no contiene fold o partition requeridos.")

    current = splits.loc[
        splits["partition"].eq(development_partition),
        ["file_id", fold_column],
    ].copy()
    current["file_id"] = current["file_id"].astype(str)
    current = current.rename(columns={fold_column: "current_fold"})

    selected["file_id"] = selected["file_id"].astype(str)
    if "fold" in selected.columns:
        selected = selected.rename(columns={"fold": "stored_fold"})

    selected = selected.merge(
        current,
        on="file_id",
        how="inner",
        validate="many_to_one",
    )

    if "stored_fold" in selected.columns:
        stored = pd.to_numeric(selected["stored_fold"], errors="coerce")
        current_fold = pd.to_numeric(
            selected["current_fold"],
            errors="coerce",
        )
        selected = selected.loc[stored.eq(current_fold)].copy()

    selected = selected.drop_duplicates()
    conflict_columns = [
        column
        for column in ("y_true", "y_pred")
        if column in selected.columns
    ]
    if selected.duplicated("file_id", keep=False).any():
        conflicts = (
            selected.groupby("file_id")[conflict_columns]
            .nunique(dropna=False)
            .max(axis=1)
        )
        conflicting_ids = conflicts[conflicts > 1].index.tolist()
        if conflicting_ids:
            raise ValueError(
                "Predicciones OOF conflictivas para los folds vigentes. "
                f"Ejemplos: {conflicting_ids[:10]}"
            )
        selected = selected.drop_duplicates(
            subset=["file_id"],
            keep="last",
        )

    expected_ids = set(current["file_id"])
    observed_ids = set(selected["file_id"])
    if observed_ids != expected_ids:
        raise ValueError(
            "La configuración OOF no cubre exactamente development. "
            f"faltantes={len(expected_ids-observed_ids)}, "
            f"extras={len(observed_ids-expected_ids)}."
        )

    return (
        selected.assign(fold=selected["current_fold"])
        .drop(columns=["current_fold"], errors="ignore")
        .sort_values(["fold", "file_id"])
        .reset_index(drop=True)
    )


def persist_final_artifacts(
    *,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    classification_report: pd.DataFrame,
    final_pipeline: Any,
    manifest: Mapping[str, Any],
    reports_dir: str | Path,
    models_dir: str | Path,
    final_model_name: str,
) -> dict[str, Path]:
    """Persiste pipeline, predicciones, reporte, features y manifest."""

    reports_path = resolve_path(reports_dir)
    models_path = resolve_path(models_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": reports_path / "test_results_summary.csv",
        "predictions": reports_path / "test_predictions.csv",
        "classification_report": (
            reports_path / "test_classification_report_final_model.csv"
        ),
        "pipeline": models_path / f"{final_model_name}.joblib",
        "manifest": reports_path / "final_model_manifest.json",
    }

    summary.to_csv(paths["summary"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    classification_report.to_csv(paths["classification_report"])
    joblib.dump(final_pipeline, paths["pipeline"])

    serializable_manifest = dict(manifest)
    serializable_manifest["artifact_paths"] = {
        name: path.as_posix()
        for name, path in paths.items()
    }
    with paths["manifest"].open("w", encoding="utf-8") as stream:
        json.dump(
            serializable_manifest,
            stream,
            ensure_ascii=False,
            indent=2,
            default=lambda value: (
                value.item()
                if isinstance(value, np.generic)
                else str(value)
            ),
        )

    return paths
