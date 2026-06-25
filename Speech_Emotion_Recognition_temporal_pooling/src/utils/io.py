"""
src/utils/io.py
---------------
Funciones de lectura/escritura de artefactos del proyecto.
Centraliza el manejo de paths y formatos para que los notebooks
no tengan paths hardcodeados.
"""

from pathlib import Path

import pandas as pd

from src.utils.config import resolve_path


def load_metadata(
    metadata_path: str | Path,
) -> pd.DataFrame:
    """Carga metadata.parquet."""
    return pd.read_parquet(
        resolve_path(metadata_path)
    )


def load_splits(
    splits_path: str | Path,
) -> pd.DataFrame:
    """Carga splits.parquet."""
    return pd.read_parquet(
        resolve_path(splits_path)
    )


def load_merged(
    metadata_path: str | Path,
    splits_path: str | Path,
) -> pd.DataFrame:
    """
    Carga y une metadata y splits mediante una relación uno a uno.
    """
    metadata = load_metadata(metadata_path)
    splits = load_splits(splits_path)

    return metadata.merge(
        splits,
        on="file_id",
        how="inner",
        validate="one_to_one",
    )


def save_parquet(
    df: pd.DataFrame,
    path: str | Path,
    **kwargs,
) -> None:
    """Guarda un DataFrame Parquet."""
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, **kwargs)



def save_markdown(content: str, path: str | Path) -> None:
    """Guarda contenido Markdown, creando directorios si no existen."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    """Crea el directorio (y padres) si no existe. Devuelve el Path."""
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
