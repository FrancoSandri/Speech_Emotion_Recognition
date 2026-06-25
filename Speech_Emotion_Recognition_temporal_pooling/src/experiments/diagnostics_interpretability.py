"""Orquestación compacta de ablaciones eGeMAPS y proyección LDA."""

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
from src.feature_selection.supervised_projection import (
    build_lda_linear_probe,
    get_lda_fold_metadata,
)
from src.features.egemaps_families import (
    FAMILY_LABELS,
    build_family_mapping,
    family_feature_names,
)
from src.models.linear_probe import build_linear_probe


def _subset_representation(
    representation: pd.DataFrame,
    selected_features: Sequence[str],
) -> pd.DataFrame:
    missing = set(selected_features) - set(representation.columns)
    if missing:
        raise KeyError(f"Features inexistentes en la representación: {sorted(missing)}")
    return representation[["file_id", *selected_features]].copy()


def run_family_ablations(
    egemaps: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    logistic_regression_config: Mapping[str, Any],
    seed: int,
    protocols: Sequence[str] = (PROTOCOL_INDEPENDENT,),
    targets: Sequence[str] = (
        TARGET_EMOTION_ORIGINAL,
        TARGET_EMOTION_QUADRANT,
    ),
    n_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Ejecuta all-features, family-only y leave-one-family-out."""
    feature_names = [column for column in egemaps.columns if column != "file_id"]
    build_family_mapping(feature_names)
    frames: list[pd.DataFrame] = []

    factory = partial(
        build_linear_probe,
        params=dict(logistic_regression_config),
        seed=seed,
    )

    for protocol in protocols:
        for target in targets:
            baseline = run_cv(
                pipeline_factory=factory,
                representation=egemaps,
                metadata=metadata,
                splits=splits,
                target_col=target,
                protocol=protocol,
                representation_name="egemaps",
                model_name="logistic_regression",
                refinement="all_features",
                n_folds=n_folds,
            )["fold_results"]
            baseline["experiment"] = "all_features"
            baseline["family"] = "all"
            baseline["family_label"] = "Todas las familias"
            frames.append(baseline)

            for family, family_label in FAMILY_LABELS.items():
                own_features = family_feature_names(feature_names, family)
                other_features = [
                    feature for feature in feature_names if feature not in own_features
                ]

                for experiment, selected in (
                    ("family_only", own_features),
                    ("leave_one_family_out", other_features),
                ):
                    subset = _subset_representation(egemaps, selected)
                    result = run_cv(
                        pipeline_factory=factory,
                        representation=subset,
                        metadata=metadata,
                        splits=splits,
                        target_col=target,
                        protocol=protocol,
                        representation_name="egemaps",
                        model_name="logistic_regression",
                        refinement=f"{experiment}:{family}",
                        n_folds=n_folds,
                    )["fold_results"]
                    result["experiment"] = experiment
                    result["family"] = family
                    result["family_label"] = family_label
                    frames.append(result)

    fold_results = pd.concat(frames, ignore_index=True, sort=False)

    baseline = fold_results.loc[
        fold_results["experiment"].eq("all_features"),
        ["protocol", "target", "fold", "macro_f1"],
    ].rename(columns={"macro_f1": "baseline_macro_f1"})
    fold_results = fold_results.merge(
        baseline,
        on=["protocol", "target", "fold"],
        how="left",
        validate="many_to_one",
    )
    fold_results["delta_macro_f1"] = (
        fold_results["macro_f1"] - fold_results["baseline_macro_f1"]
    )

    return {
        "fold_results": fold_results,
        "summary": summarize_family_ablations(fold_results),
    }


def summarize_family_ablations(results: pd.DataFrame) -> pd.DataFrame:
    """Resume media y desvío de rendimiento y delta por familia."""
    if results.empty:
        return pd.DataFrame()
    grouping = [
        "protocol",
        "target",
        "experiment",
        "family",
        "family_label",
    ]
    return (
        results.groupby(grouping, observed=True, sort=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            delta_macro_f1_mean=("delta_macro_f1", "mean"),
            delta_macro_f1_std=("delta_macro_f1", "std"),
            n_features_mean=("n_features", "mean"),
        )
        .reset_index()
    )


def run_lda_grid(
    representations: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    logistic_regression_config: Mapping[str, Any],
    seed: int,
    representation_names: Sequence[str] = ("egemaps", "wav2vec"),
    protocols: Sequence[str] = (PROTOCOL_INDEPENDENT,),
    targets: Sequence[str] = (
        TARGET_EMOTION_ORIGINAL,
        TARGET_EMOTION_QUADRANT,
        TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
    ),
    shrinkage: str | float | None = "auto",
    n_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Evalúa LDA con shrinkage usando el motor común de outer CV."""
    frames: list[pd.DataFrame] = []

    for representation_name in representation_names:
        if representation_name not in representations:
            raise KeyError(f"Representación faltante: {representation_name!r}")

        factory = partial(
            build_lda_linear_probe,
            logistic_regression_params=dict(logistic_regression_config),
            seed=seed,
            shrinkage=shrinkage,
        )

        for protocol in protocols:
            for target in targets:
                output = run_cv(
                    pipeline_factory=factory,
                    representation=representations[representation_name],
                    metadata=metadata,
                    splits=splits,
                    target_col=target,
                    protocol=protocol,
                    representation_name=representation_name,
                    model_name="logistic_regression",
                    refinement="lda_shrinkage",
                    n_folds=n_folds,
                    fitted_metadata_fn=get_lda_fold_metadata,
                )
                frames.append(output["fold_results"])

    fold_results = pd.concat(frames, ignore_index=True, sort=False)
    return {
        "fold_results": fold_results,
        "summary": summarize_cv_results(fold_results),
    }
