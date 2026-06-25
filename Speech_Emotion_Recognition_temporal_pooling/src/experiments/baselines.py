"""Orquestación de la matriz de baselines de Sprint 3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Callable

import pandas as pd

from src.config.contracts import (
    PROTOCOL_DEPENDENT,
    PROTOCOL_INDEPENDENT,
    TARGET_EMOTION_ORIGINAL,
    TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
    TARGET_EMOTION_QUADRANT,
)
from src.evaluation.reporting import summarize_cv_results
from src.experiments.cross_validation import run_cv
from src.models import build_linear_probe, build_mlp, build_random_forest
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_REPRESENTATIONS = ("egemaps", "wav2vec")
DEFAULT_PROTOCOLS = (PROTOCOL_DEPENDENT, PROTOCOL_INDEPENDENT)
DEFAULT_TARGETS = (
    TARGET_EMOTION_ORIGINAL,
    TARGET_EMOTION_QUADRANT,
    TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
)
DEFAULT_MODELS = ("random_forest", "logistic_regression", "mlp")

MODEL_BUILDERS: dict[str, Callable[..., Any]] = {
    "random_forest": build_random_forest,
    "logistic_regression": build_linear_probe,
    "mlp": build_mlp,
}


def make_model_factory(
    model_name: str,
    model_config: Mapping[str, Any],
    seed: int,
) -> Callable[[], Any]:
    """Devuelve una fábrica que crea un estimador nuevo por fold."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Modelo desconocido: {model_name!r}. "
            f"Opciones: {sorted(MODEL_BUILDERS)}"
        )

    return partial(
        MODEL_BUILDERS[model_name],
        params=dict(model_config),
        seed=seed,
    )


def run_baseline_grid(
    representations: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    model_configs: Mapping[str, Mapping[str, Any]],
    seed: int,
    representation_names: Sequence[str] = DEFAULT_REPRESENTATIONS,
    protocols: Sequence[str] = DEFAULT_PROTOCOLS,
    targets: Sequence[str] = DEFAULT_TARGETS,
    models: Sequence[str] = DEFAULT_MODELS,
    n_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Ejecuta las configuraciones solicitadas usando el mismo motor de CV.

    No guarda artefactos automáticamente. El notebook decide cuándo persistir
    el CSV consolidado.
    """
    missing_representations = set(representation_names) - set(representations)
    if missing_representations:
        raise KeyError(
            "Faltan representaciones: "
            f"{sorted(missing_representations)}"
        )

    missing_models = set(models) - set(model_configs)
    if missing_models:
        raise KeyError(
            f"Falta configuración para modelos: {sorted(missing_models)}"
        )

    fold_frames: list[pd.DataFrame] = []

    total = (
        len(representation_names)
        * len(protocols)
        * len(targets)
        * len(models)
    )
    current = 0

    for representation_name in representation_names:
        representation = representations[representation_name]

        for protocol in protocols:
            for target in targets:
                for model_name in models:
                    current += 1
                    logger.info(
                        "Baseline %s/%s: %s | %s | %s | %s",
                        current,
                        total,
                        representation_name,
                        protocol,
                        target,
                        model_name,
                    )

                    factory = make_model_factory(
                        model_name=model_name,
                        model_config=model_configs[model_name],
                        seed=seed,
                    )

                    result = run_cv(
                        pipeline_factory=factory,
                        representation=representation,
                        metadata=metadata,
                        splits=splits,
                        target_col=target,
                        protocol=protocol,
                        representation_name=representation_name,
                        model_name=model_name,
                        refinement="none",
                        n_folds=n_folds,
                    )
                    fold_frames.append(result["fold_results"])

    fold_results = pd.concat(fold_frames, ignore_index=True)
    summary = summarize_cv_results(fold_results)

    return {
        "fold_results": fold_results,
        "summary": summary,
    }
