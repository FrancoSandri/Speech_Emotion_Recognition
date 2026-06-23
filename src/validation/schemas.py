"""
src/validation/schemas.py
-------------------------
Definición y validación de schemas para los DataFrames del proyecto.

Cada schema especifica:
  - columnas requeridas y sus tipos esperados
  - rangos válidos para columnas numéricas
  - conjuntos válidos para columnas categóricas

Uso:
    from src.validation.schemas import validate_schema
    validate_schema(df, "metadata")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

from src.config.contracts import (
    ALL_PARTITIONS,
    RAVDESS_EMOTION_CODES,
    FOLD_SENTINEL,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Definición de schemas
# ---------------------------------------------------------------------------

@dataclass
class ColumnSpec:
    """Especificación de una columna."""
    dtype_family: str                    # 'int', 'float', 'str', 'bool'
    nullable: bool = False
    valid_values: set | None = None      # Para categóricas
    min_val: float | None = None
    max_val: float | None = None


METADATA_SCHEMA: dict[str, ColumnSpec] = {
    "file_id":             ColumnSpec("str",   nullable=False),
    "file_path_raw":       ColumnSpec("str",   nullable=False),
    "file_path_trimmed":   ColumnSpec("str",   nullable=False),
    "actor_id":            ColumnSpec("int",   nullable=False, min_val=1, max_val=24),
    "sex":                 ColumnSpec("str",   nullable=False, valid_values={"male", "female"}),
    "emotion_code":        ColumnSpec("int",   nullable=False, min_val=1, max_val=8),
    "emotion_original":    ColumnSpec("str",   nullable=False,
                                      valid_values=set(RAVDESS_EMOTION_CODES.values())),
    "intensity_code":      ColumnSpec("int",   nullable=False, min_val=1, max_val=2),
    "intensity":           ColumnSpec("str",   nullable=False,
                                      valid_values={"normal", "strong"}),
    "statement_code":      ColumnSpec("int",   nullable=False, min_val=1, max_val=2),
    "statement":           ColumnSpec("str",   nullable=False,
                                      valid_values={"kids", "dogs"}),
    "repetition_id":       ColumnSpec("int",   nullable=False, min_val=1, max_val=2),
    "utterance_group_id":  ColumnSpec("str",   nullable=False),
    "emotion_quadrant":    ColumnSpec("str",   nullable=False,
                                      valid_values={"Q1", "Q2", "Q3", "Q4"}),
    "duration_raw_s":      ColumnSpec("float", nullable=True,  min_val=0.0),
    "duration_trimmed_s":  ColumnSpec("float", nullable=True,  min_val=0.0),
    "trim_ratio":          ColumnSpec("float", nullable=True,  min_val=0.0, max_val=1.0),
    "trim_start_s":        ColumnSpec("float", nullable=True,  min_val=0.0),
    "trim_end_s":          ColumnSpec("float", nullable=True,  min_val=0.0),
}

SPLITS_SCHEMA: dict[str, ColumnSpec] = {
    "file_id":                    ColumnSpec("str", nullable=False),
    "partition":                  ColumnSpec("str", nullable=False,
                                             valid_values=ALL_PARTITIONS),
    "fold_speaker_dependent":     ColumnSpec("int", nullable=False, min_val=-1),
    "fold_speaker_independent":   ColumnSpec("int", nullable=False, min_val=-1),
}


# ---------------------------------------------------------------------------
# Validador genérico
# ---------------------------------------------------------------------------

def validate_schema(
    df: pd.DataFrame,
    schema_name: str,
    raise_on_error: bool = True,
) -> list[str]:
    """
    Valida un DataFrame contra un schema predefinido.

    Parameters
    ----------
    df : pd.DataFrame
    schema_name : str
        'metadata' o 'splits'
    raise_on_error : bool
        Si True, lanza ValueError al primer error. Si False, devuelve lista.

    Returns
    -------
    list[str]
        Lista de errores encontrados (vacía si todo OK).
    """
    schemas = {
        "metadata": METADATA_SCHEMA,
        "splits":   SPLITS_SCHEMA,
    }

    if schema_name not in schemas:
        raise ValueError(f"Schema desconocido: '{schema_name}'. "
                         f"Opciones: {list(schemas.keys())}")

    schema = schemas[schema_name]
    errors: list[str] = []

    for col, spec in schema.items():
        # --- Columna existe ---
        if col not in df.columns:
            errors.append(f"[{col}] Columna faltante.")
            continue

        series = df[col]

        # --- Nulos ---
        n_null = series.isna().sum()
        if not spec.nullable and n_null > 0:
            errors.append(f"[{col}] {n_null} nulos no permitidos.")

        # Trabajar solo con valores no nulos para el resto
        non_null = series.dropna()
        if len(non_null) == 0:
            continue

        # --- Tipo ---
        if spec.dtype_family == "int":
            if not pd.api.types.is_integer_dtype(series):
                # Tolerar float con .0 (conversión automática)
                if pd.api.types.is_float_dtype(series):
                    if not (non_null == non_null.astype(int)).all():
                        errors.append(f"[{col}] Esperado int, tiene floats no enteros.")
                else:
                    errors.append(f"[{col}] Esperado int, dtype={series.dtype}.")

        elif spec.dtype_family == "float":
            if not pd.api.types.is_numeric_dtype(series):
                errors.append(f"[{col}] Esperado float/numeric, dtype={series.dtype}.")

        elif spec.dtype_family == "str":
            if not (pd.api.types.is_string_dtype(series) or
                    pd.api.types.is_object_dtype(series)):
                errors.append(f"[{col}] Esperado str/object, dtype={series.dtype}.")

        # --- Rango numérico ---
        if spec.min_val is not None:
            try:
                below = (non_null.astype(float) < spec.min_val).sum()
                if below > 0:
                    errors.append(f"[{col}] {below} valores < {spec.min_val}.")
            except Exception:
                pass

        if spec.max_val is not None:
            try:
                above = (non_null.astype(float) > spec.max_val).sum()
                if above > 0:
                    errors.append(f"[{col}] {above} valores > {spec.max_val}.")
            except Exception:
                pass

        # --- Valores válidos (categórico) ---
        if spec.valid_values is not None:
            unknown = set(non_null.unique()) - spec.valid_values
            if unknown:
                errors.append(
                    f"[{col}] Valores desconocidos: {unknown}. "
                    f"Permitidos: {spec.valid_values}."
                )

    if errors:
        msg = f"Schema '{schema_name}' — {len(errors)} error(es):\n" + \
              "\n".join(f"  • {e}" for e in errors)
        if raise_on_error:
            raise ValueError(msg)
        else:
            logger.error(msg)
    else:
        logger.info(f"✓ Schema '{schema_name}': OK ({len(df)} filas, "
                    f"{len(df.columns)} columnas).")

    return errors
