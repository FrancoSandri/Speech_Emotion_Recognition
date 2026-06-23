"""
Acceso unificado a las representaciones extraídas.

Centraliza la alineación por ``file_id`` y entrega, por defecto, únicamente
el development pool. Los tests solo se cargan cuando se solicita de forma
explícita, lo que reduce el riesgo de usarlos accidentalmente en Sprint 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.config.contracts import (
    FOLD_SENTINEL,
    PARTITION_DEVELOPMENT,
)
from src.utils.config import resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

REPR_EGEMAPS = "egemaps"
REPR_WAV2VEC = "wav2vec"
VALID_REPRS = {REPR_EGEMAPS, REPR_WAV2VEC}


def load_representation(name: str, features_dir: str | Path) -> pd.DataFrame:
    """Carga ``egemaps.parquet`` o ``wav2vec.parquet``."""
    if name not in VALID_REPRS:
        raise ValueError(
            f"Representación desconocida: {name!r}. Opciones: {sorted(VALID_REPRS)}"
        )

    path = resolve_path(Path(features_dir) / f"{name}.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Features {name!r} no encontradas en {path}.")

    representation = pd.read_parquet(path)
    _validate_unique_file_ids(representation, name)

    logger.info(
        "Cargado '%s': %s muestras, %s features.",
        name,
        len(representation),
        len(representation.columns) - 1,
    )
    return representation


def prepare_model_table(
    representation: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    partition: str = PARTITION_DEVELOPMENT,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Alinea metadata, splits y features mediante una relación uno a uno.

    Parameters
    ----------
    partition:
        Por defecto carga solo ``development``. Sprint 3 no debe pasar otra
        partición; Sprint 4 podrá solicitar los tests explícitamente.
    """
    for name, dataframe in {
        "metadata": metadata,
        "splits": splits,
        "representation": representation,
    }.items():
        _validate_unique_file_ids(dataframe, name)

    metadata_ids = set(metadata["file_id"])
    splits_ids = set(splits["file_id"])
    representation_ids = set(representation["file_id"])

    if metadata_ids != splits_ids:
        raise ValueError(
            "metadata y splits no contienen exactamente los mismos file_id. "
            f"Faltan en splits={len(metadata_ids - splits_ids)}, "
            f"sobran en splits={len(splits_ids - metadata_ids)}."
        )

    if metadata_ids != representation_ids:
        raise ValueError(
            "metadata y representación no contienen exactamente los mismos file_id. "
            f"Faltan features={len(metadata_ids - representation_ids)}, "
            f"sobran features={len(representation_ids - metadata_ids)}."
        )

    merged = metadata.merge(
        splits,
        on="file_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        representation,
        on="file_id",
        how="inner",
        validate="one_to_one",
    )

    subset = merged.loc[merged["partition"] == partition].copy()
    if subset.empty:
        raise ValueError(f"Partición {partition!r} vacía tras alinear los datos.")

    feature_cols = [column for column in representation.columns if column != "file_id"]
    if not feature_cols:
        raise ValueError("La representación no contiene columnas de features.")

    non_numeric = [
        column
        for column in feature_cols
        if not pd.api.types.is_numeric_dtype(subset[column])
    ]
    if non_numeric:
        raise TypeError(f"Features no numéricas: {non_numeric[:5]}")

    values = subset[feature_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("La representación contiene NaN o valores infinitos.")

    return subset.reset_index(drop=True), feature_cols


def get_feature_matrix(
    representation: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    target_col: str,
    partition: str = PARTITION_DEVELOPMENT,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Devuelve ``X``, ``y`` y ``file_id`` para una partición."""
    table, feature_cols = prepare_model_table(
        representation=representation,
        metadata=metadata,
        splits=splits,
        partition=partition,
    )

    if target_col not in table.columns:
        raise KeyError(f"Target inexistente: {target_col!r}")

    X = table[feature_cols].to_numpy(dtype=np.float32)
    y = table[target_col].to_numpy()
    file_ids = table["file_id"].copy()

    logger.info(
        "get_feature_matrix [%s, %s]: X=%s",
        partition,
        target_col,
        X.shape,
    )
    return X, y, file_ids


def get_fold_data(
    representation: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    target_col: str,
    fold_idx: int,
    protocol: Literal["speaker_dependent", "speaker_independent"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve train y validation para un fold persistido."""
    table, feature_cols = prepare_model_table(
        representation=representation,
        metadata=metadata,
        splits=splits,
        partition=PARTITION_DEVELOPMENT,
    )

    if target_col not in table.columns:
        raise KeyError(f"Target inexistente: {target_col!r}")

    fold_col = f"fold_{protocol}"
    if fold_col not in table.columns:
        raise KeyError(f"Columna de folds inexistente: {fold_col!r}")

    val_mask = table[fold_col] == fold_idx
    train_mask = (table[fold_col] != fold_idx) & (table[fold_col] != FOLD_SENTINEL)

    if not val_mask.any() or not train_mask.any():
        raise ValueError(f"Fold {fold_idx} inválido o vacío para {protocol}.")

    X_train = table.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
    y_train = table.loc[train_mask, target_col].to_numpy()
    X_val = table.loc[val_mask, feature_cols].to_numpy(dtype=np.float32)
    y_val = table.loc[val_mask, target_col].to_numpy()

    return X_train, y_train, X_val, y_val


def _validate_unique_file_ids(dataframe: pd.DataFrame, name: str) -> None:
    if "file_id" not in dataframe.columns:
        raise KeyError(f"{name} no contiene la columna 'file_id'.")
    if dataframe["file_id"].isna().any():
        raise ValueError(f"{name} contiene file_id nulos.")
    if dataframe["file_id"].duplicated().any():
        raise ValueError(f"{name} contiene file_id duplicados.")
