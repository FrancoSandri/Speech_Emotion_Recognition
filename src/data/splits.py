"""
src/data/splits.py
------------------
Construcción de las particiones globales y los folds de CV.

Particiones:
  - test_speaker_independent : actores completamente aislados
  - development              : todo lo restante

Folds (persisten en splits.parquet):
  - fold_speaker_independent : StratifiedGroupKFold(groups=actor_id)

El valor -1 (FOLD_SENTINEL) indica que el registro pertenece a test
y no debe aparecer en folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

from src.config.contracts import (
    PARTITION_DEVELOPMENT,
    PARTITION_TEST_INDEPENDENT,
    FOLD_SENTINEL,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def assign_partitions(
    metadata: pd.DataFrame,
    test_independent_actors: list[int],
    seed: int = 42,
    target_col: str = "emotion_original",
) -> pd.DataFrame:
    """
    Asigna a cada registro una partición global sin separar repeticiones
    pertenecientes al mismo utterance_group_id.

    El test speaker-independent se define mediante actores reservados.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata completa. Debe contener actor_id, file_id,
        utterance_group_id y la columna indicada en target_col.
    test_independent_actors : list[int]
        IDs de actores reservados para test_speaker_independent.
    seed : int, default=42
        Semilla utilizada por StratifiedGroupKFold.
    target_col : str, default="emotion_original"
        Target usado para preservar aproximadamente la distribución de clases.

    Returns
    -------
    pd.DataFrame
        Copia de metadata con la columna partition añadida.

    Raises
    ------
    ValueError
        Si la configuración es inválida.
    KeyError
        Si faltan columnas requeridas.
    AssertionError
        Si las particiones resultantes presentan solapamientos inválidos.
    """
    required_columns = {
        "file_id",
        "actor_id",
        "utterance_group_id",
        target_col,
    }
    missing_columns = required_columns.difference(metadata.columns)

    if missing_columns:
        raise KeyError(
            "Faltan columnas requeridas para construir las particiones: "
            f"{sorted(missing_columns)}"
        )

    if metadata["file_id"].duplicated().any():
        duplicated = metadata.loc[
            metadata["file_id"].duplicated(keep=False),
            "file_id",
        ].unique()

        raise ValueError(
            "file_id debe ser único. "
            f"Se encontraron {len(duplicated)} IDs duplicados."
        )

    if not test_independent_actors:
        raise ValueError(
            "test_independent_actors debe contener al menos un actor."
        )

    df = metadata.copy()

    # Comprobar que cada utterance_group_id pertenece a un único actor y target.
    group_consistency = (
        df.groupby("utterance_group_id", observed=True)
        .agg(
            n_actors=("actor_id", "nunique"),
            n_targets=(target_col, "nunique"),
        )
    )

    inconsistent_groups = group_consistency[
        (group_consistency["n_actors"] != 1)
        | (group_consistency["n_targets"] != 1)
    ]

    if not inconsistent_groups.empty:
        raise ValueError(
            f"{len(inconsistent_groups)} utterance_group_id no son "
            "consistentes respecto de actor o target."
        )

    df["partition"] = PARTITION_DEVELOPMENT

    # ------------------------------------------------------------------
    # Test speaker-independent: actores completamente reservados.
    # ------------------------------------------------------------------
    independent_mask = df["actor_id"].isin(test_independent_actors)

    if not independent_mask.any():
        raise ValueError(
            "Ningún actor de test_independent_actors aparece en metadata."
        )

    missing_independent_actors = sorted(
        set(test_independent_actors).difference(df["actor_id"].unique())
    )

    if missing_independent_actors:
        raise ValueError(
            "Los siguientes actores configurados para test independent "
            f"no aparecen en metadata: {missing_independent_actors}"
        )

    df.loc[
        independent_mask,
        "partition",
    ] = PARTITION_TEST_INDEPENDENT

    # ------------------------------------------------------------------
    # Validaciones estructurales.
    # ------------------------------------------------------------------
    group_partition_counts = (
        df.groupby("utterance_group_id", observed=True)["partition"]
        .nunique()
    )

    shared_groups = group_partition_counts[group_partition_counts > 1]

    if not shared_groups.empty:
        raise AssertionError(
            f"{len(shared_groups)} utterance_group_id aparecen en más "
            "de una partición."
        )

    development_actors = set(
        df.loc[
            df["partition"] == PARTITION_DEVELOPMENT,
            "actor_id",
        ]
    )
    independent_actors = set(
        df.loc[
            df["partition"] == PARTITION_TEST_INDEPENDENT,
            "actor_id",
        ]
    )

    shared_independent_actors = independent_actors.intersection(
        development_actors
    )

    if shared_independent_actors:
        raise AssertionError(
            "Los siguientes actores del test speaker-independent aparecen "
            f"en otras particiones: {sorted(shared_independent_actors)}"
        )

    # Mantiene cualquier validación general ya implementada en el proyecto.
    _assert_disjoint(df)

    counts = df["partition"].value_counts()
    group_counts = (
        df.groupby("partition", observed=True)["utterance_group_id"]
        .nunique()
    )

    logger.info(
        "Particiones asignadas:\n%s\n\nGrupos por partición:\n%s",
        counts.to_string(),
        group_counts.to_string(),
    )

    return df


def _assert_disjoint(df: pd.DataFrame) -> None:
    """Verifica que las particiones sean completamente disjuntas."""
    file_ids_by_partition = {
        p: set(df.loc[df["partition"] == p, "file_id"])
        for p in [PARTITION_DEVELOPMENT, PARTITION_TEST_INDEPENDENT]
    }

    overlap_dev_indep = (
        file_ids_by_partition[PARTITION_DEVELOPMENT]
        & file_ids_by_partition[PARTITION_TEST_INDEPENDENT]
    )

    assert not overlap_dev_indep, f"Overlap development ∩ test_indep: {overlap_dev_indep}"
    logger.info("✓ Particiones disjuntas verificadas.")


def build_cv_folds(
    metadata_with_partition: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    target_col: str = "emotion_original",
) -> pd.DataFrame:
    """
    Construye los folds de CV speaker-independent.

    Solo opera sobre el development pool. El test recibe FOLD_SENTINEL (-1).

    Parameters
    ----------
    metadata_with_partition : pd.DataFrame
        Metadata con columna 'partition' ya asignada.
    n_splits : int
        Número de folds (K).
    seed : int
        Semilla para reproducibilidad.
    target_col : str
        Columna de target para estratificar (usualmente emotion_original).

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: file_id, partition, fold_speaker_independent.
    """
    df = metadata_with_partition.copy()
    df["fold_speaker_independent"] = FOLD_SENTINEL

    dev_mask = df["partition"] == PARTITION_DEVELOPMENT
    # Orden canónico para reproducibilidad independiente del orden de metadata.
    dev_df = df.loc[dev_mask].sort_values("file_id").copy()

    y_dev = dev_df[target_col].values

    # -------------------------------------------------------------------------
    # Folds speaker-independent: grupos = actor_id
    # -------------------------------------------------------------------------
    groups_indep = dev_df["actor_id"].values

    fold_indep = _fit_stratified_group_kfold(
        y=y_dev,
        groups=groups_indep,
        n_splits=n_splits,
        seed=seed,
        label="speaker_independent",
    )
    df.loc[dev_df.index, "fold_speaker_independent"] = fold_indep

    return df[["file_id", "partition", "fold_speaker_independent"]]


def _fit_stratified_group_kfold(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    label: str,
) -> np.ndarray:
    """
    Intenta StratifiedGroupKFold; cae a GroupKFold si falla (con warning).
    Devuelve array de asignaciones de fold (0-indexed).
    """
    fold_assignments = np.full(len(y), FOLD_SENTINEL, dtype=int)

    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (_, val_idx) in enumerate(cv.split(y, y, groups=groups)):
            fold_assignments[val_idx] = fold_idx

        # Verificar que todas las clases estén en cada fold
        _audit_fold_balance(y, fold_assignments, n_splits, label)
        logger.info(f"✓ StratifiedGroupKFold [{label}] completado con {n_splits} folds.")

    except Exception as e:
        logger.warning(
            f"StratifiedGroupKFold [{label}] falló: {e}. "
            f"Cayendo a GroupKFold — DOCUMENTAR EN NOTEBOOK."
        )
        cv = GroupKFold(n_splits=n_splits)
        for fold_idx, (_, val_idx) in enumerate(cv.split(y, y, groups=groups)):
            fold_assignments[val_idx] = fold_idx

    return fold_assignments


def _audit_fold_balance(
    y: np.ndarray,
    fold_assignments: np.ndarray,
    n_splits: int,
    label: str,
) -> None:
    """Verifica que todas las clases estén presentes en cada fold de validación."""
    all_classes = set(np.unique(y))
    for fold_idx in range(n_splits):
        val_mask = fold_assignments == fold_idx
        val_classes = set(np.unique(y[val_mask]))
        missing = all_classes - val_classes
        if missing:
            logger.warning(
                f"[{label}] Fold {fold_idx} no contiene clases: {missing}. "
                "Considerar reducir n_splits o revisar distribución."
            )


def build_splits_dataframe(
    metadata: pd.DataFrame,
    test_independent_actors: list[int],
    n_splits: int = 5,
    seed: int = 42,
    target_col: str = "emotion_original",
) -> pd.DataFrame:
    """Construye particiones globales y folds persistidos de CV."""
    df_part = assign_partitions(
        metadata=metadata,
        test_independent_actors=test_independent_actors,
        seed=seed,
        target_col=target_col,
    )

    splits_df = build_cv_folds(
        metadata_with_partition=df_part,
        n_splits=n_splits,
        seed=seed,
        target_col=target_col,
    )

    logger.info(
        "splits.parquet construido: %s registros.\n"
        "  development: %s\n"
        "  test_independent: %s",
        len(splits_df),
        (splits_df["partition"] == PARTITION_DEVELOPMENT).sum(),
        (splits_df["partition"] == PARTITION_TEST_INDEPENDENT).sum(),
    )
    return splits_df
