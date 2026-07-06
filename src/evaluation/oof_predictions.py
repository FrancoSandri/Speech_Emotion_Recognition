from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from src.experiments.baselines import make_model_factory
from src.experiments.cross_validation import run_cv
from src.feature_selection import (
    build_pca_linear_probe,
    get_pca_fold_metadata,
    run_nested_rfecv,
)
from src.feature_selection.supervised_projection import (
    build_lda_linear_probe,
    get_lda_fold_metadata,
)

EXPERIMENT_COLUMNS: tuple[str, ...] = (
    "representation",
    "protocol",
    "target",
    "model",
    "refinement",
)


def _as_dict(config_row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    """Return a plain dictionary from a mapping or pandas row."""

    if hasattr(config_row, "to_dict"):
        return dict(config_row.to_dict())
    return dict(config_row)


def _config_value(config: Any, key: str) -> Any:
    """Read a value from mappings, OmegaConf nodes, or simple objects."""

    if isinstance(config, Mapping):
        return config[key]
    return getattr(config, key)


def _validate_columns(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    frame_name: str,
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} no contiene las columnas requeridas: {missing}")


def parse_pca_variance_threshold(refinement: str) -> float:
    """Convert labels such as ``pca_95`` to the variance threshold ``0.95``."""

    if not refinement.startswith("pca_"):
        raise ValueError(f"Refinamiento PCA inválido: {refinement!r}")

    try:
        percentage = float(refinement.removeprefix("pca_"))
    except ValueError as error:
        raise ValueError(
            f"No se pudo interpretar {refinement!r}; se esperaba, por ejemplo, 'pca_95'."
        ) from error

    if not 0 < percentage <= 100:
        raise ValueError(
            "El porcentaje de varianza PCA debe estar en el intervalo (0, 100]."
        )

    return percentage / 100.0


def collect_oof_predictions(
    config_row: Mapping[str, Any] | pd.Series,
    *,
    representations: Mapping[str, Any],
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    model_configs: Any,
    seed: int,
    n_folds: int,
    rfecv_config: Any | None = None,
    lda_config: Any | None = None,
    experiment_columns: Sequence[str] = EXPERIMENT_COLUMNS,
) -> pd.DataFrame:
    """Run one experiment configuration and return its OOF predictions.

    Every dependency previously taken from notebook globals is supplied as an
    argument. Preprocessing and feature refinement remain inside the outer CV
    routines used by the project.
    """

    row = _as_dict(config_row)
    missing = [column for column in experiment_columns if column not in row]
    if missing:
        raise ValueError(f"La configuración no contiene las columnas: {missing}")

    representation_name = str(row["representation"])
    target = str(row["target"])
    model = str(row["model"])
    refinement = str(row["refinement"])

    if representation_name not in representations:
        raise KeyError(f"Representación desconocida: {representation_name!r}")

    representation = representations[representation_name]

    if refinement == "none":
        factory = make_model_factory(
            model_name=model,
            model_config=_config_value(model_configs, model),
            seed=seed,
        )
        output = run_cv(
            pipeline_factory=factory,
            representation=representation,
            metadata=metadata,
            splits=splits,
            target_col=target,
            representation_name=representation_name,
            model_name=model,
            refinement="none",
            n_folds=n_folds,
        )

    elif refinement.startswith("pca_"):
        if model != "logistic_regression":
            raise ValueError("PCA solo está definido para logistic_regression.")

        factory = partial(
            build_pca_linear_probe,
            logistic_regression_params=_config_value(
                model_configs,
                "logistic_regression",
            ),
            seed=seed,
            variance_threshold=parse_pca_variance_threshold(refinement),
        )
        output = run_cv(
            pipeline_factory=factory,
            representation=representation,
            metadata=metadata,
            splits=splits,
            target_col=target,
            representation_name=representation_name,
            model_name=model,
            refinement=refinement,
            n_folds=n_folds,
            fitted_metadata_fn=get_pca_fold_metadata,
        )

    elif refinement == "lda_shrinkage":
        if model != "logistic_regression":
            raise ValueError(
                "LDA shrinkage solo está definido para logistic_regression."
            )
        if lda_config is None:
            raise ValueError(
                "Se requiere lda_config para recolectar OOF con LDA shrinkage."
            )

        factory = partial(
            build_lda_linear_probe,
            logistic_regression_params=_config_value(
                model_configs,
                "logistic_regression",
            ),
            seed=seed,
            shrinkage=_config_value(lda_config, "shrinkage"),
        )
        output = run_cv(
            pipeline_factory=factory,
            representation=representation,
            metadata=metadata,
            splits=splits,
            target_col=target,
            representation_name=representation_name,
            model_name=model,
            refinement=refinement,
            n_folds=n_folds,
            fitted_metadata_fn=get_lda_fold_metadata,
        )

    elif refinement == "rfecv":
        if rfecv_config is None:
            raise ValueError(
                "Se requiere rfecv_config para recolectar OOF con RFECV."
            )
        if representation_name != "egemaps" or model != "logistic_regression":
            raise ValueError(
                "RFECV solo está definido para eGeMAPS + Logistic Regression."
            )

        output = run_nested_rfecv(
            representation=representation,
            metadata=metadata,
            splits=splits,
            target_col=target,
            logistic_regression_params=_config_value(
                model_configs,
                "logistic_regression",
            ),
            seed=seed,
            inner_folds=_config_value(rfecv_config, "inner_folds"),
            step=_config_value(rfecv_config, "step"),
            min_features_to_select=_config_value(
                rfecv_config,
                "min_features_to_select",
            ),
            n_jobs=_config_value(rfecv_config, "n_jobs"),
            n_folds=n_folds,
        )

    else:
        raise ValueError(f"Refinamiento no soportado: {refinement!r}")

    predictions = output["predictions"].copy()
    for column in experiment_columns:
        predictions[column] = row[column]

    return predictions


def configuration_mask(
    frame: pd.DataFrame,
    config_row: Mapping[str, Any] | pd.Series,
    experiment_columns: Sequence[str] = EXPERIMENT_COLUMNS,
) -> pd.Series:
    """Return the rows belonging to one experimental configuration."""

    row = _as_dict(config_row)
    _validate_columns(frame, experiment_columns, frame_name="frame")

    mask = pd.Series(True, index=frame.index, dtype=bool)
    for column in experiment_columns:
        expected = "<NA>" if pd.isna(row[column]) else str(row[column])
        observed = frame[column].astype("string").fillna("<NA>")
        mask &= observed.eq(expected)
    return mask


def load_or_collect_oof(
    selected_configs: pd.DataFrame,
    *,
    cache_path: str | Path,
    collect_fn: Callable[[Mapping[str, Any]], pd.DataFrame],
    experiment_columns: Sequence[str] = EXPERIMENT_COLUMNS,
    id_column: str = "file_id",
    verbose: bool = True,
) -> pd.DataFrame:
    """Load cached OOF predictions and compute only missing configurations."""

    _validate_columns(
        selected_configs,
        experiment_columns,
        frame_name="selected_configs",
    )

    path = Path(cache_path)
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not cached.empty:
        _validate_columns(
            cached,
            [*experiment_columns, id_column],
            frame_name="caché OOF",
        )

    collected: list[pd.DataFrame] = [cached] if not cached.empty else []
    unique_configs = (
        selected_configs.loc[:, list(experiment_columns)]
        .drop_duplicates()
        .to_dict(orient="records")
    )

    for config_row in unique_configs:
        already_cached = (
            not cached.empty
            and configuration_mask(cached, config_row, experiment_columns).any()
        )
        if already_cached:
            continue

        if verbose:
            description = " | ".join(
                f"{column}={config_row[column]}" for column in experiment_columns
            )
            print(f"Calculando OOF: {description}")

        collected.append(collect_fn(config_row))

    if not collected:
        return pd.DataFrame()

    result = pd.concat(collected, ignore_index=True, sort=False)
    _validate_columns(
        result,
        [*experiment_columns, id_column],
        frame_name="predicciones OOF",
    )
    result = result.drop_duplicates(
        subset=[*experiment_columns, id_column],
        keep="last",
    ).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result


def select_best_baselines(
    summary: pd.DataFrame,
    *,
    protocol: str,
    metric: str,
) -> pd.DataFrame:
    """Select the best unrefined configuration for every target."""

    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"
    _validate_columns(
        summary,
        [*EXPERIMENT_COLUMNS, metric_mean, metric_std],
        frame_name="summary",
    )

    candidates = summary.loc[
        summary["protocol"].eq(protocol)
        & summary["refinement"].astype(str).eq("none")
    ]

    return (
        candidates.sort_values(
            ["target", metric_mean, metric_std],
            ascending=[True, False, True],
        )
        .groupby("target", observed=True, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def select_controlled_refinement_tracks(
    summary: pd.DataFrame,
    *,
    protocol: str,
    target: str,
    representations: Sequence[str],
    metric: str,
    model: str = "logistic_regression",
) -> pd.DataFrame:
    """Select one baseline/refinement pair per representation."""

    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"
    _validate_columns(
        summary,
        [*EXPERIMENT_COLUMNS, metric_mean, metric_std],
        frame_name="summary",
    )

    controlled = summary.loc[
        summary["protocol"].eq(protocol)
        & summary["target"].eq(target)
        & summary["model"].eq(model)
    ]

    rows: list[pd.Series] = []
    for representation in representations:
        family = controlled.loc[controlled["representation"].eq(representation)]
        baseline = family.loc[family["refinement"].astype(str).eq("none")]
        refined = family.loc[~family["refinement"].astype(str).eq("none")]

        if baseline.empty or refined.empty:
            continue

        sort_columns = [metric_mean, metric_std]
        sort_order = [False, True]
        rows.append(baseline.sort_values(sort_columns, ascending=sort_order).iloc[0])
        rows.append(refined.sort_values(sort_columns, ascending=sort_order).iloc[0])

    return pd.DataFrame(rows).reset_index(drop=True)


def combine_configurations(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine configuration tables into one unique collection request."""

    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=list(EXPERIMENT_COLUMNS))

    for index, frame in enumerate(valid_frames):
        _validate_columns(
            frame,
            EXPERIMENT_COLUMNS,
            frame_name=f"configuraciones[{index}]",
        )

    return (
        pd.concat(
            [frame.loc[:, list(EXPERIMENT_COLUMNS)] for frame in valid_frames],
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
