"""Orquestación de feature importance, RFECV y PCA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import pandas as pd

from src.config.contracts import (
    PROTOCOL_INDEPENDENT,
    TARGET_EMOTION_ORIGINAL,
    TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
    TARGET_EMOTION_QUADRANT,
)
from src.evaluation.reporting import summarize_cv_results
from src.experiments.cross_validation import run_cv
from src.feature_selection import (
    build_pca_linear_probe,
    get_pca_fold_metadata,
    run_feature_importance,
    run_nested_rfecv,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PROTOCOLS = (PROTOCOL_INDEPENDENT,)
DEFAULT_TARGETS = (
    TARGET_EMOTION_ORIGINAL,
    TARGET_EMOTION_QUADRANT,
    TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
)


def run_feature_importance_grid(
    egemaps: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    model_configs: Mapping[str, Mapping[str, Any]],
    seed: int,
    protocols: Sequence[str] = DEFAULT_PROTOCOLS,
    targets: Sequence[str] = DEFAULT_TARGETS,
    top_k: int = 15,
    permutation_repeats: int = 10,
    n_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Ejecuta importancias eGeMAPS para los protocolos y targets indicados."""
    frames: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []

    for protocol in protocols:
        for target in targets:
            output = run_feature_importance(
                representation=egemaps,
                metadata=metadata,
                splits=splits,
                target_col=target,
                random_forest_params=model_configs["random_forest"],
                logistic_regression_params=model_configs["logistic_regression"],
                seed=seed,
                top_k=top_k,
                permutation_repeats=permutation_repeats,
                n_folds=n_folds,
            )
            frames.append(output["fold_results"])
            summaries.append(output["summary"])

    return {
        "feature_results": pd.concat(frames, ignore_index=True),
        "summary": pd.concat(summaries, ignore_index=True),
    }


def run_rfecv_grid(
    egemaps: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    logistic_regression_config: Mapping[str, Any],
    seed: int,
    protocols: Sequence[str] = DEFAULT_PROTOCOLS,
    targets: Sequence[str] = DEFAULT_TARGETS,
    inner_folds: int = 3,
    step: int = 5,
    min_features_to_select: int = 10,
    n_jobs: int = 1,
    n_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Ejecuta RFECV nested únicamente sobre eGeMAPSv02."""
    cv_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []

    for protocol in protocols:
        for target in targets:
            output = run_nested_rfecv(
                representation=egemaps,
                metadata=metadata,
                splits=splits,
                target_col=target,
                logistic_regression_params=logistic_regression_config,
                seed=seed,
                inner_folds=inner_folds,
                step=step,
                min_features_to_select=min_features_to_select,
                n_jobs=n_jobs,
                n_folds=n_folds,
            )
            cv_frames.append(output["fold_results"])
            feature_frames.append(output["feature_results"])

    fold_results = pd.concat(cv_frames, ignore_index=True)
    return {
        "fold_results": fold_results,
        "summary": summarize_cv_results(fold_results),
        "feature_results": pd.concat(feature_frames, ignore_index=True),
    }


def run_pca_grid(
    representations: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    logistic_regression_config: Mapping[str, Any],
    seed: int,
    representation_names: Sequence[str] = ("egemaps", "wav2vec"),
    protocols: Sequence[str] = DEFAULT_PROTOCOLS,
    targets: Sequence[str] = DEFAULT_TARGETS,
    variance_thresholds: Sequence[float] = (0.90, 0.95),
    n_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Evalúa StandardScaler → PCA → LogReg con el motor común de CV."""
    frames: list[pd.DataFrame] = []

    for representation_name in representation_names:
        if representation_name not in representations:
            raise KeyError(f"Representación faltante: {representation_name!r}")

        for protocol in protocols:
            for target in targets:
                for threshold in variance_thresholds:
                    threshold = float(threshold)
                    refinement = f"pca_{int(round(threshold * 100))}"
                    factory = partial(
                        build_pca_linear_probe,
                        logistic_regression_params=logistic_regression_config,
                        seed=seed,
                        variance_threshold=threshold,
                    )

                    result = run_cv(
                        pipeline_factory=factory,
                        representation=representations[representation_name],
                        metadata=metadata,
                        splits=splits,
                        target_col=target,
                        representation_name=representation_name,
                        model_name="logistic_regression",
                        refinement=refinement,
                        n_folds=n_folds,
                        fitted_metadata_fn=get_pca_fold_metadata,
                    )
                    frames.append(result["fold_results"])

    fold_results = pd.concat(frames, ignore_index=True)
    return {
        "fold_results": fold_results,
        "summary": summarize_cv_results(fold_results),
    }
