"""Extracción y alineación de representaciones para nuevos dominios."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.features.extract_egemaps import load_or_extract_egemaps
from src.features.wav2vec_temporal import (
    build_static_representation,
    extract_or_load_temporal_features,
)
from src.utils.config import resolve_path


def extract_or_load_domain_representations(
    metadata: pd.DataFrame,
    *,
    features_dir: str | Path,
    model_name: str,
    revision: str,
    target_sample_rate: int,
    include_feature_projection: bool,
    sequence_dtype: str = "float16",
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Genera eGeMAPS y wav2vec ``average_mean_std`` para un dominio."""

    features_root = resolve_path(features_dir)
    features_root.mkdir(parents=True, exist_ok=True)

    egemaps = load_or_extract_egemaps(
        metadata=metadata,
        output_path=features_root / "egemaps.parquet",
        path_col="file_path_trimmed",
        overwrite=overwrite,
    )

    statistics = extract_or_load_temporal_features(
        metadata=metadata,
        statistics_path=features_root / "wav2vec_layer_statistics.npz",
        sequences_dir=features_root / "wav2vec_sequences",
        multilayer_sequences_dir=None,
        path_col="file_path_trimmed",
        model_name=model_name,
        revision=revision,
        target_sample_rate=target_sample_rate,
        include_feature_projection=include_feature_projection,
        sequence_dtype=sequence_dtype,
        overwrite=overwrite,
    )
    wav2vec_temporal = build_static_representation(
        statistics,
        layer_strategy="average",
        pooling="mean_std",
    )

    return {
        "egemaps": egemaps,
        "wav2vec_temporal": wav2vec_temporal,
    }


def align_target_representation(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    representation_name: str,
) -> pd.DataFrame:
    """Reordena target con el contrato exacto de features fuente."""

    if "file_id" not in source or "file_id" not in target:
        raise KeyError("source y target deben contener file_id.")

    source_columns = [
        column for column in source.columns if column != "file_id"
    ]
    target_columns = [
        column for column in target.columns if column != "file_id"
    ]

    missing = set(source_columns) - set(target_columns)
    extra = set(target_columns) - set(source_columns)
    if missing or extra:
        raise ValueError(
            f"Espacio {representation_name!r} incompatible. "
            f"faltantes={sorted(missing)[:10]}, "
            f"extras={sorted(extra)[:10]}."
        )

    return target[["file_id", *source_columns]].copy()


def align_domain_representations(
    source_representations: Mapping[str, pd.DataFrame],
    target_representations: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Valida y alinea todas las representaciones target."""

    missing = set(source_representations) - set(target_representations)
    if missing:
        raise KeyError(
            f"Representaciones target ausentes: {sorted(missing)}."
        )

    return {
        name: align_target_representation(
            source_representations[name],
            target_representations[name],
            representation_name=name,
        )
        for name in source_representations
    }
