"""
src/data/audio_cleaning.py
--------------------------
Trimming de silencios extremos sobre los audios RAVDESS.

Procedimiento:
  1. Cargar audio con librosa (sin normalización).
  2. Calcular RMS por frames.
  3. Detectar primera y última región activa por umbral en dB.
  4. Agregar padding en ambos extremos.
  5. Guardar sin modificar amplitud.

NO se aplica: normalización RMS/LUFS/peak, compresión, noise reduction,
eliminación de silencios internos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from src.utils.logging import get_logger
from src.utils.config import resolve_path

logger = get_logger(__name__)


@dataclass
class TrimResult:
    """Resultado del trimming de un archivo de audio."""
    file_id: str
    file_path_raw: str
    file_path_trimmed: str
    duration_raw_s: float
    duration_trimmed_s: float
    trim_start_s: float
    trim_end_s: float
    trim_ratio: float          # fracción eliminada respecto al original
    warning_excessive: bool    # True si trim_ratio > threshold


def _rms_db(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """
    Calcula RMS en dB por frames.

    Parameters
    ----------
    y : np.ndarray
        Señal de audio mono.
    frame_length : int
        Longitud de frame en muestras.
    hop_length : int
        Salto entre frames en muestras.

    Returns
    -------
    np.ndarray
        RMS en dB por frame.
    """
    # Usar librosa para el cálculo de RMS
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # Convertir a dB; añadir pequeño epsilon para evitar log(0)
    rms_db = 20 * np.log10(rms + 1e-9)
    return rms_db


def trim_audio(
    file_path_raw: str | Path,
    file_path_trimmed: str | Path,
    sample_rate: int = 48000,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    threshold_db: float = -35.0,
    padding_ms: float = 150.0,
    trim_ratio_warning: float = 0.50,
    overwrite: bool = False,
) -> TrimResult:
    """
    Aplica trimming de silencios extremos a un archivo de audio.

    Parameters
    ----------
    file_path_raw : str | Path
        Path al audio original RAVDESS.
    file_path_trimmed : str | Path
        Path de destino del audio recortado.
    sample_rate : int
        Sample rate original (48 000 Hz para RAVDESS).
    frame_length_ms : float
        Longitud de frame en ms.
    hop_length_ms : float
        Salto entre frames en ms.
    threshold_db : float
        Umbral en dB relativo al máximo RMS.
    padding_ms : float
        Padding en ms a agregar en cada extremo.
    trim_ratio_warning : float
        Umbral de advertencia: trim_ratio > este valor genera warning.
    overwrite : bool
        Si False y el archivo trimmeado ya existe, lo omite.

    Returns
    -------
    TrimResult
    """
    stored_raw_path = Path(file_path_raw).as_posix()
    stored_trimmed_path = Path(file_path_trimmed).as_posix()

    file_path_raw = resolve_path(file_path_raw)
    file_path_trimmed = resolve_path(file_path_trimmed)

    if not file_path_raw.exists():
        raise FileNotFoundError(f"Audio no encontrado: {file_path_raw}")

    # --- Cargar sin normalización (always_2d=False devuelve mono directamente) ---
    y, sr = librosa.load(str(file_path_raw), sr=sample_rate, mono=True)
    duration_raw_s = len(y) / sr

    # --- Calcular RMS por frames ---
    frame_length = int(frame_length_ms * sr / 1000)
    hop_length   = int(hop_length_ms   * sr / 1000)

    rms_db = _rms_db(y, frame_length=frame_length, hop_length=hop_length)

    # --- Umbral relativo al máximo RMS ---
    max_rms_db = rms_db.max()
    abs_threshold_db = max_rms_db + threshold_db  # threshold_db es negativo

    # --- Detectar región activa ---
    active_frames = np.where(rms_db >= abs_threshold_db)[0]

    if len(active_frames) == 0:
        logger.warning(
            f"No se detectó región activa en {file_path_raw.name}. "
            "Se conserva el audio completo."
        )
        first_frame, last_frame = 0, len(rms_db) - 1
    else:
        first_frame = active_frames[0]
        last_frame  = active_frames[-1]

    # --- Convertir frames a muestras ---
    trim_start_sample = first_frame * hop_length
    trim_end_sample   = min(last_frame * hop_length + frame_length, len(y))

    # --- Agregar padding ---
    padding_samples = int(padding_ms * sr / 1000)
    start_sample = max(0, trim_start_sample - padding_samples)
    end_sample   = min(len(y), trim_end_sample + padding_samples)

    trim_start_s = start_sample / sr
    trim_end_s   = end_sample / sr

    # --- Extraer segmento ---
    y_trimmed = y[start_sample:end_sample]

    if len(y_trimmed) == 0:
        raise ValueError(
            f"Audio trimmeado vacío: {file_path_raw.name}. "
            "Revisar threshold_db o archivo."
        )

    duration_trimmed_s = len(y_trimmed) / sr
    trim_ratio = 1.0 - (duration_trimmed_s / duration_raw_s)

    # --- Guardar sin modificar amplitud ---
    if overwrite or not file_path_trimmed.exists():
        file_path_trimmed.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(file_path_trimmed), y_trimmed, sr)

    warning_excessive = trim_ratio > trim_ratio_warning
    if warning_excessive:
        logger.warning(
            f"trim_ratio={trim_ratio:.2f} > {trim_ratio_warning} "
            f"en {file_path_raw.name}. Inspeccionar."
        )

    return TrimResult(
        file_id=file_path_raw.stem,
        file_path_raw=stored_raw_path,
        file_path_trimmed=stored_trimmed_path,
        duration_raw_s=round(duration_raw_s, 4),
        duration_trimmed_s=round(duration_trimmed_s, 4),
        trim_start_s=round(trim_start_s, 4),
        trim_end_s=round(trim_end_s, 4),
        trim_ratio=round(trim_ratio, 4),
        warning_excessive=warning_excessive,
    )


def process_all_audio(
    metadata: pd.DataFrame,
    sample_rate: int = 48000,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    threshold_db: float = -35.0,
    padding_ms: float = 150.0,
    trim_ratio_warning: float = 0.50,
    overwrite: bool = False,
    verbose_every: int = 100,
) -> pd.DataFrame:
    """
    Aplica trimming a todos los audios en metadata y devuelve
    un DataFrame con las métricas de trimming.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata con columnas file_path_raw y file_path_trimmed.
    ... (ver trim_audio para el resto de parámetros)

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: file_id, duration_raw_s, duration_trimmed_s,
        trim_ratio, trim_start_s, trim_end_s, warning_excessive.
    """
    results = []
    errors = []

    for i, row in metadata.iterrows():
        if i % verbose_every == 0:
            logger.info(f"Trimming {i}/{len(metadata)} — {row['file_id']}")

        try:
            result = trim_audio(
                file_path_raw=row["file_path_raw"],
                file_path_trimmed=row["file_path_trimmed"],
                sample_rate=sample_rate,
                frame_length_ms=frame_length_ms,
                hop_length_ms=hop_length_ms,
                threshold_db=threshold_db,
                padding_ms=padding_ms,
                trim_ratio_warning=trim_ratio_warning,
                overwrite=overwrite,
            )
            results.append(result.__dict__)
        except Exception as e:
            logger.error(f"Error procesando {row['file_id']}: {e}")
            errors.append({"file_id": row["file_id"], "error": str(e)})

    if errors:
        logger.error(f"Errores en {len(errors)} archivos: {[e['file_id'] for e in errors]}")

    trim_df = pd.DataFrame(results)
    warnings_count = trim_df["warning_excessive"].sum()
    logger.info(
        f"Trimming completado: {len(trim_df)} archivos procesados, "
        f"{warnings_count} advertencias de trim_ratio > {trim_ratio_warning}."
    )
    return trim_df


def merge_trim_metrics(metadata: pd.DataFrame, trim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une las métricas de trimming con la metadata original.

    Parameters
    ----------
    metadata : pd.DataFrame
    trim_df : pd.DataFrame
        Salida de process_all_audio().

    Returns
    -------
    pd.DataFrame
        Metadata actualizada con columnas de trimming.
    """
    trim_cols = ["file_id", "duration_raw_s", "duration_trimmed_s",
                 "trim_ratio", "trim_start_s", "trim_end_s"]

    # Reemplazar las columnas NaN inicializadas en build_metadata
    metadata = metadata.drop(
        columns=[c for c in trim_cols[1:] if c in metadata.columns],
        errors="ignore"
    )
    return metadata.merge(trim_df[trim_cols], on="file_id", how="left")
