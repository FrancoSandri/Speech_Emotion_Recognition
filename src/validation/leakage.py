"""
src/validation/leakage.py
-------------------------
Auditoría de leakage para Sprint 1.

Verifica:
  1. Separación de actores entre development y test_independent.
  2. Ningún utterance_group_id se comparte entre development y test_dependent.
  3. Ausencia de file_id overlap entre las tres particiones.
  4. Que ningún archivo de test aparezca en los folds de CV.
  5. Que todos los archivos de development tengan fold asignado.
  6. Que test_independent contenga ambos sexos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config.contracts import (
    PARTITION_DEVELOPMENT,
    PARTITION_TEST_DEPENDENT,
    PARTITION_TEST_INDEPENDENT,
    FOLD_SENTINEL,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LeakageReport:
    """Resultado de la auditoría de leakage."""
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.errors:
            return "FAIL"
        if self.warnings:
            return "PASS WITH WARNINGS"
        return "PASS"

    def __str__(self) -> str:
        lines = [f"LeakageReport — {self.status}"]
        for e in self.errors:
            lines.append(f"  [ERROR] {e}")
        for w in self.warnings:
            lines.append(f"  [WARN]  {w}")
        if not self.errors and not self.warnings:
            lines.append("  Todas las verificaciones pasaron.")
        return "\n".join(lines)


def audit_leakage(
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
) -> LeakageReport:
    """
    Ejecuta la auditoría completa de leakage.


    Parameters
    ----------
    metadata : pd.DataFrame
        metadata.parquet con actor_id, utterance_group_id, file_id, sex.
    splits : pd.DataFrame
        splits.parquet con file_id, partition, fold_speaker_dependent,
        fold_speaker_independent.

    Returns
    -------
    LeakageReport
    """
    df = metadata.merge(
        splits, on="file_id", how="inner", validate="one_to_one"
    )
    errors = []
    warnings = []

    # ------------------------------------------------------------------
    # 1. Actores del test_independent no deben aparecer en development
    # ------------------------------------------------------------------
    actors_indep = set(
        df.loc[df["partition"] == PARTITION_TEST_INDEPENDENT, "actor_id"]
    )
    actors_dev = set(
        df.loc[df["partition"] == PARTITION_DEVELOPMENT, "actor_id"]
    )
    actor_overlap = actors_indep & actors_dev
    if actor_overlap:
        errors.append(
            f"Actores en test_independent Y development: {sorted(actor_overlap)}"
        )
    else:
        logger.info("✓ [1] Actores test_independent completamente aislados de development.")

    # ------------------------------------------------------------------
    # 2. Development y test dependent deben ser disjuntos por archivo y grupo.
    # ------------------------------------------------------------------
    file_ids_dep = set(
        df.loc[df["partition"] == PARTITION_TEST_DEPENDENT, "file_id"]
    )
    file_ids_dev = set(
        df.loc[df["partition"] == PARTITION_DEVELOPMENT, "file_id"]
    )
    file_overlap_dep_dev = file_ids_dep & file_ids_dev
    if file_overlap_dep_dev:
        errors.append(
            f"file_id en test_dependent Y development: "
            f"{len(file_overlap_dep_dev)} archivos. "
            f"Primeros 5: {sorted(file_overlap_dep_dev)[:5]}"
        )
    else:
        logger.info("✓ [2] Ningún file_id de test_dependent aparece en development.")

    ugroups_dep = set(
        df.loc[df["partition"] == PARTITION_TEST_DEPENDENT, "utterance_group_id"]
    )
    ugroups_dev = set(
        df.loc[df["partition"] == PARTITION_DEVELOPMENT, "utterance_group_id"]
    )
    shared_groups = ugroups_dev & ugroups_dep

    if shared_groups:
        errors.append(
            f"{len(shared_groups)} utterance_group_id compartidos entre "
            "development y test_speaker_dependent."
        )


    dependent_actors = set(
        df.loc[df["partition"] == PARTITION_TEST_DEPENDENT, "actor_id"]
    )
    unknown_dependent_actors = dependent_actors - actors_dev
    if unknown_dependent_actors:
        errors.append(
            "Actores de test_dependent ausentes en development: "
            f"{sorted(unknown_dependent_actors)}"
        )

    # ------------------------------------------------------------------
    # 3. File overlap entre las tres particiones
    # ------------------------------------------------------------------
    for p1, p2 in [
        (PARTITION_DEVELOPMENT,    PARTITION_TEST_DEPENDENT),
        (PARTITION_DEVELOPMENT,    PARTITION_TEST_INDEPENDENT),
        (PARTITION_TEST_DEPENDENT, PARTITION_TEST_INDEPENDENT),
    ]:
        ids_p1 = set(df.loc[df["partition"] == p1, "file_id"])
        ids_p2 = set(df.loc[df["partition"] == p2, "file_id"])
        overlap = ids_p1 & ids_p2
        if overlap:
            errors.append(f"file_id overlap {p1} ∩ {p2}: {len(overlap)} archivos.")
        else:
            logger.info(f"✓ [3] Sin file overlap: {p1} ∩ {p2} = ∅")

    # ------------------------------------------------------------------
    # 4. Archivos de test no deben tener fold asignado (!= FOLD_SENTINEL)
    # ------------------------------------------------------------------
    test_mask = df["partition"].isin(
        [PARTITION_TEST_DEPENDENT, PARTITION_TEST_INDEPENDENT]
    )
    for fold_col in ["fold_speaker_dependent", "fold_speaker_independent"]:
        leaked = df[test_mask & (df[fold_col] != FOLD_SENTINEL)]
        if not leaked.empty:
            errors.append(
                f"Archivos de test con fold asignado en '{fold_col}': "
                f"{len(leaked)} registros."
            )
        else:
            logger.info(f"✓ [4] Archivos de test tienen FOLD_SENTINEL en '{fold_col}'.")

    # ------------------------------------------------------------------
    # 5. Todos los archivos de development deben tener fold asignado
    # ------------------------------------------------------------------
    dev_mask = df["partition"] == PARTITION_DEVELOPMENT
    for fold_col in ["fold_speaker_dependent", "fold_speaker_independent"]:
        missing_fold = df[dev_mask & (df[fold_col] == FOLD_SENTINEL)]
        if not missing_fold.empty:
            errors.append(
                f"Archivos de development SIN fold en '{fold_col}': "
                f"{len(missing_fold)} registros."
            )
        else:
            logger.info(f"✓ [5] Todos los archivos de development tienen fold en '{fold_col}'.")

    # ------------------------------------------------------------------
    # 6. Balance de actores por sexo en test speaker-independent
    # ------------------------------------------------------------------
    independent_actor_sex = (
        df.loc[
            df["partition"] == PARTITION_TEST_INDEPENDENT,
            ["actor_id", "sex"],
        ]
        .drop_duplicates()
        .sort_values("actor_id")
    )

    # Cada actor debe tener una única etiqueta de sexo.
    sex_per_actor = independent_actor_sex.groupby("actor_id")["sex"].nunique()

    inconsistent_actors = sex_per_actor[sex_per_actor != 1]

    if not inconsistent_actors.empty:
        errors.append(
            "Existen actores con más de una etiqueta de sexo en "
            "test_speaker_independent: "
            f"{inconsistent_actors.index.tolist()}."
        )
    else:
        sex_counts = (
            independent_actor_sex["sex"]
            .value_counts()
            .to_dict()
        )

        expected_sex_counts = {
            "male": 3,
            "female": 3,
        }

        if sex_counts != expected_sex_counts:
            errors.append(
                "Balance incorrecto de actores en test_speaker_independent. "
                f"Esperado: {expected_sex_counts}. "
                f"Obtenido: {sex_counts}."
            )
        else:
            logger.info(
                "✓ [6] test_speaker_independent contiene "
                "3 actores male y 3 actores female."
            )

    passed = len(errors) == 0
    report = LeakageReport(passed=passed, errors=errors, warnings=warnings)
    logger.info(f"\n{report}")
    return report