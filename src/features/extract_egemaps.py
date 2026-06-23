"""
src/features/extract_egemaps.py
--------------------------------
Extracción de features eGeMAPSv02 Functionals usando openSMILE.

Salida: un DataFrame con una fila por file_id y 88 columnas acústicas.
Los nombres originales de openSMILE se preservan (sin renombrar).

Requisito: pip install opensmile
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import opensmile as smile

from src.utils.logging import get_logger
from src.utils.config import resolve_path

logger = get_logger(__name__)


def extract_egemaps_single(
    file_path: str | Path,
    config_name: str = "eGeMAPSv02",
    feature_level: str = "Functionals",
) -> np.ndarray:
    """
    Extrae eGeMAPSv02 Functionals de un único archivo de audio.

    Parameters
    ----------
    file_path : str | Path
    config_name : str
        Nombre del config openSMILE (default: 'eGeMAPSv02').
    feature_level : str
        Nivel de feature (default: 'Functionals').

    Returns
    -------
    np.ndarray
        Vector de features de forma (88,).
    """
    audio_path = resolve_path(file_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio no encontrado: {audio_path}"
        )

    features = smile.process_file(str(audio_path))
    return features.values[0]  # shape: (88,)


def extract_egemaps_batch(
    metadata: pd.DataFrame,
    path_col: str = "file_path_trimmed",
    config_name: str = "eGeMAPSv02",
    feature_level: str = "Functionals",
    verbose_every: int = 100,
) -> pd.DataFrame:
    """
    Extrae eGeMAPSv02 para todos los archivos en metadata.

    Parameters
    ----------
    metadata : pd.DataFrame
        Debe contener columnas 'file_id' y path_col.
    path_col : str
        Columna con el path al audio (default: 'file_path_trimmed').
    config_name : str
    feature_level : str
    verbose_every : int
        Log cada N archivos.

    Returns
    -------
    pd.DataFrame
        file_id como índice + 88 columnas de features.
        Guardado en data/processed/features/egemaps.parquet.
    """
    import opensmile

    smile = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, config_name),
        feature_level=getattr(opensmile.FeatureLevel, feature_level),
    )

    feature_names = smile.feature_names
    records = []
    errors = []

    for i, row in metadata.iterrows():
        if i % verbose_every == 0:
            logger.info(f"eGeMAPS: {i}/{len(metadata)} — {row['file_id']}")

        try:
            audio_path = resolve_path(row[path_col])

            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio no encontrado: {audio_path}"
                )

            feats = smile.process_file(str(audio_path))            
            vec = feats.values[0]
            records.append({"file_id": row["file_id"], **dict(zip(feature_names, vec))})
        except Exception as e:
            logger.error(f"Error en {row['file_id']}: {e}")
            errors.append(row["file_id"])

    if errors:
        logger.warning(f"eGeMAPS falló en {len(errors)} archivos: {errors[:5]}")

    df = pd.DataFrame(records)
    logger.info(
        f"eGeMAPS extraído: {len(df)} archivos, {len(feature_names)} features. "
        f"Errores: {len(errors)}."
    )
    return df


def load_or_extract_egemaps(
    metadata: pd.DataFrame,
    output_path: str | Path,
    path_col: str = "file_path_trimmed",
    overwrite: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Carga egemaps.parquet si existe, sino extrae y guarda.

    Parameters
    ----------
    metadata : pd.DataFrame
    output_path : str | Path
        Ruta de destino del parquet.
    path_col : str
    overwrite : bool
    **kwargs : dict
        Pasados a extract_egemaps_batch.

    Returns
    -------
    pd.DataFrame
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        logger.info(f"Cargando eGeMAPS desde caché: {output_path}")
        return pd.read_parquet(output_path)

    logger.info("Extrayendo eGeMAPS (puede tardar varios minutos)...")
    df = extract_egemaps_batch(metadata, path_col=path_col, **kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"eGeMAPS guardado: {output_path}")

    return df
