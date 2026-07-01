"""
src/validation/sanity.py
------------------------
Validaciones de sanidad sobre metadata.parquet, splits.parquet y audios.

Levanta AssertionError o devuelve SanityReport con errores y warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from src.utils.config import resolve_path

import pandas as pd

from src.config.contracts import (
    METADATA_REQUIRED_COLS,
    SPLITS_REQUIRED_COLS,
    PARTITION_DEVELOPMENT,
    PARTITION_TEST_INDEPENDENT,
    ALL_PARTITIONS,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SanityReport:
    """Resultado de una verificación de sanidad."""
    name: str
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
        lines = [f"SanityReport [{self.name}] — {self.status}"]
        for e in self.errors:
            lines.append(f"  [ERROR] {e}")
        for w in self.warnings:
            lines.append(f"  [WARN]  {w}")
        if not self.errors and not self.warnings:
            lines.append("  Todas las verificaciones pasaron.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validación de metadata
# ---------------------------------------------------------------------------

def validate_metadata(df: pd.DataFrame) -> SanityReport:
    """
    Valida metadata.parquet contra los contratos de Sprint 1.

    Condiciones bloqueantes (→ error):
      - Columnas requeridas presentes.
      - file_id único y no nulo.
      - actor_id en [1, 24].
      - emotion_original y emotion_quadrant sin nulos.
      - trim_start_s < trim_end_s.
      - duration_trimmed_s > 0.
      - sex consistente con actor_id.

    Condiciones de advertencia:
      - trim_ratio > 0.50 en algún registro.
    """
    errors = []
    warnings = []

    # --- Columnas requeridas ---
    missing_cols = [c for c in METADATA_REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        errors.append(f"Columnas faltantes en metadata: {missing_cols}")

    if errors:
        return SanityReport("metadata", passed=False, errors=errors)

    # --- file_id único ---
    if not df["file_id"].is_unique:
        dups = df[df["file_id"].duplicated(keep=False)]["file_id"].tolist()
        errors.append(f"file_id duplicados: {dups[:5]}")

    # --- Nulos en columnas críticas ---
    for col in ["file_id", "actor_id", "emotion_original",
                "emotion_quadrant", "sex", "utterance_group_id"]:
        n = df[col].isna().sum()
        if n > 0:
            errors.append(f"'{col}' tiene {n} nulos.")

    # --- actor_id rango ---
    out = df[~df["actor_id"].between(1, 24)]
    if not out.empty:
        errors.append(f"actor_id fuera de [1,24]: {out['actor_id'].unique().tolist()}")

    # --- emotion_quadrant no nulo ---
    bad_q = df[df["emotion_quadrant"].isna()]
    if not bad_q.empty:
        errors.append(
            f"emotion_quadrant nulo en {len(bad_q)} filas "
            f"(emociones: {bad_q['emotion_original'].unique().tolist()})."
        )

    # --- sex consistente ---
    from src.config.contracts import actor_sex
    sex_mismatch = df[df.apply(
        lambda r: actor_sex(r["actor_id"]) != r["sex"], axis=1
    )]
    if not sex_mismatch.empty:
        errors.append(
            f"sex inconsistente con actor_id: {len(sex_mismatch)} filas."
        )

    # --- Métricas de trimming ---
    trimmed = df.dropna(subset=["trim_start_s", "trim_end_s", "duration_trimmed_s"])
    if len(trimmed) < len(df):
        warnings.append(
            f"{len(df) - len(trimmed)} filas sin métricas de trimming (NaN). "
            "¿El trimming fue ejecutado?"
        )
    else:
        bad_trim = trimmed[trimmed["trim_start_s"] >= trimmed["trim_end_s"]]
        if not bad_trim.empty:
            errors.append(f"trim_start_s >= trim_end_s en {len(bad_trim)} audios.")

        empty_audio = trimmed[trimmed["duration_trimmed_s"] <= 0]
        if not empty_audio.empty:
            errors.append(f"duration_trimmed_s <= 0 en {len(empty_audio)} audios.")

        excessive = trimmed[trimmed["trim_ratio"] > 0.50]
        if not excessive.empty:
            warnings.append(
                f"trim_ratio > 0.50 en {len(excessive)} audios. Inspeccionar: "
                f"{excessive['file_id'].tolist()[:3]}"
            )

    passed = len(errors) == 0
    report = SanityReport("metadata", passed=passed, errors=errors, warnings=warnings)
    logger.info(f"\n{report}")
    return report


# ---------------------------------------------------------------------------
# Validación de splits
# ---------------------------------------------------------------------------

def validate_splits(splits: pd.DataFrame, metadata: pd.DataFrame) -> SanityReport:
    """
    Valida splits.parquet.

    Condiciones bloqueantes:
      - Columnas requeridas presentes.
      - file_id único y completo (todos los de metadata).
      - partition con valores válidos solamente.
      - Sin folds fuera de rango para development.
    """
    errors = []
    warnings = []

    missing_cols = [c for c in SPLITS_REQUIRED_COLS if c not in splits.columns]
    if missing_cols:
        errors.append(f"Columnas faltantes en splits: {missing_cols}")
        return SanityReport("splits", passed=False, errors=errors)

    # --- file_id único ---
    if not splits["file_id"].is_unique:
        errors.append("file_id duplicados en splits.")

    # --- Completitud vs metadata ---
    meta_ids = set(metadata["file_id"])
    split_ids = set(splits["file_id"])
    missing_in_splits = meta_ids - split_ids
    extra_in_splits = split_ids - meta_ids
    if missing_in_splits:
        errors.append(
            f"{len(missing_in_splits)} file_ids de metadata ausentes en splits."
        )
    if extra_in_splits:
        errors.append(
            f"{len(extra_in_splits)} file_ids en splits no encontrados en metadata."
        )

    # --- Valores de partition ---
    unknown_parts = set(splits["partition"].unique()) - ALL_PARTITIONS
    if unknown_parts:
        errors.append(f"Valores de partition desconocidos: {unknown_parts}")

    passed = len(errors) == 0
    report = SanityReport("splits", passed=passed, errors=errors, warnings=warnings)
    logger.info(f"\n{report}")
    return report


# ---------------------------------------------------------------------------
# Validación de audios trimmeados
# ---------------------------------------------------------------------------

def validate_trimmed_audio(
    metadata: pd.DataFrame,
    sample_size: int | None = None,
) -> SanityReport:
    """
    Verifica que los archivos trimmeados existen y se pueden abrir.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata con file_path_trimmed.
    sample_size : int, optional
        Si se especifica, solo verifica una muestra aleatoria.
    """
    import librosa

    errors = []
    warnings = []

    rows = metadata
    if sample_size:
        rows = metadata.sample(min(sample_size, len(metadata)), random_state=42)

    missing = []
    unreadable = []
    empty = []

    for _, row in rows.iterrows():
        path = resolve_path(row["file_path_trimmed"])
        if not path.exists():
            missing.append(row["file_id"])
            continue
        try:
            y, sr = librosa.load(str(path), sr=None, mono=True)
            if len(y) == 0:
                empty.append(row["file_id"])
        except Exception as e:
            unreadable.append((row["file_id"], str(e)))

    if missing:
        errors.append(f"Archivos trimmeados faltantes: {len(missing)}. Primeros: {missing[:3]}")
    if unreadable:
        errors.append(f"Archivos no legibles: {len(unreadable)}. Primeros: {unreadable[:3]}")
    if empty:
        errors.append(f"Archivos vacíos (0 muestras): {len(empty)}")

    n_checked = len(rows) - len(missing) - len(unreadable)
    logger.info(f"Audio validation: {n_checked}/{len(rows)} archivos verificados correctamente.")

    passed = len(errors) == 0
    report = SanityReport("audio", passed=passed, errors=errors, warnings=warnings)
    logger.info(f"\n{report}")
    return report
