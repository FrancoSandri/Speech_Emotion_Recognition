"""
src/data/metadata.py
--------------------
Descubrimiento de archivos RAVDESS, parseo del naming convention
y construcción de targets (emotion_original, emotion_quadrant).

Naming convention RAVDESS (audio-only, speech):
  03-01-{emo}-{int}-{stmt}-{rep}-{actor}.wav
  [0] modality      03 = audio_only
  [1] vocal_channel 01 = speech
  [2] emotion       01-08
  [3] intensity     01-02
  [4] statement     01-02
  [5] repetition    01-02
  [6] actor         01-24
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.contracts import (
    RAVDESS_EMOTION_CODES,
    RAVDESS_INTENSITY_CODES,
    RAVDESS_STATEMENT_CODES,
    RAVDESS_VOCAL_CHANNEL_CODES,
    RAVDESS_MODALITY_CODES,
    EMOTION_TO_QUADRANT,
    actor_sex,
)
from src.utils.logging import get_logger
from src.utils.config import to_project_relative

logger = get_logger(__name__)


def discover_audio_files(raw_dir: str | Path, modality: str = "audio_only") -> list[Path]:
    """
    Descubre todos los .wav del subset seleccionado dentro de raw_dir.

    Parameters
    ----------
    raw_dir : str | Path
        Directorio raíz de RAVDESS (contiene Actor_01/, Actor_02/, ...).
    modality : str
        'audio_only' filtra por primer segmento '03' en el nombre.

    Returns
    -------
    list[Path]
        Lista ordenada de paths de audio.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir no encontrado: {raw_dir}")

    modality_prefix = {"audio_only": "03", "video_only": "02", "full_av": "01"}
    prefix = modality_prefix.get(modality)

    wav_files = sorted(raw_dir.rglob("*.wav"))

    if prefix:
        wav_files = [f for f in wav_files if f.stem.split("-")[0] == prefix]

    logger.info(f"Archivos .wav descubiertos ({modality}): {len(wav_files)}")
    return wav_files


def parse_filename(filepath: Path) -> Optional[dict]:
    """
    Parsea un nombre de archivo RAVDESS y devuelve un dict con todos
    los campos codificados. Devuelve None si el nombre no es válido.

    Parameters
    ----------
    filepath : Path
        Path al archivo .wav.

    Returns
    -------
    dict | None
    """
    stem = filepath.stem
    parts = stem.split("-")

    if len(parts) != 7:
        logger.warning(f"Nombre inesperado (esperados 7 segmentos): {filepath.name}")
        return None

    try:
        modality_code      = int(parts[0])
        vocal_channel_code = int(parts[1])
        emotion_code       = int(parts[2])
        intensity_code     = int(parts[3])
        statement_code     = int(parts[4])
        repetition_id      = int(parts[5])
        actor_id           = int(parts[6])
    except ValueError:
        logger.warning(f"No se pudieron convertir a int los segmentos: {filepath.name}")
        return None

    # Validaciones de rango
    if emotion_code not in RAVDESS_EMOTION_CODES:
        logger.warning(f"emotion_code fuera de rango [{emotion_code}]: {filepath.name}")
        return None
    if actor_id < 1 or actor_id > 24:
        logger.warning(f"actor_id fuera de rango [{actor_id}]: {filepath.name}")
        return None

    emotion_name = RAVDESS_EMOTION_CODES[emotion_code]
    # neutral solo existe en intensidad 01 → normalizar
    if emotion_code == 1 and intensity_code == 2:
        logger.debug(f"neutral con intensity=2 (imposible en RAVDESS): {filepath.name}")

    return {
        "file_id":            stem,
        "file_path_raw":      str(filepath),
        "modality_code":      modality_code,
        "modality":           RAVDESS_MODALITY_CODES.get(modality_code, "unknown"),
        "vocal_channel_code": vocal_channel_code,
        "vocal_channel":      RAVDESS_VOCAL_CHANNEL_CODES.get(vocal_channel_code, "unknown"),
        "emotion_code":       emotion_code,
        "emotion_original":   emotion_name,
        "intensity_code":     intensity_code,
        "intensity":          RAVDESS_INTENSITY_CODES.get(intensity_code, "unknown"),
        "statement_code":     statement_code,
        "statement":          RAVDESS_STATEMENT_CODES.get(statement_code, "unknown"),
        "repetition_id":      repetition_id,
        "actor_id":           actor_id,
        "sex":                actor_sex(actor_id),
    }


def build_utterance_group_id(row: pd.Series) -> str:
    """
    Construye el utterance_group_id como identificador de todas las
    repeticiones equivalentes de un mismo enunciado.

    Definición: actor_id + emotion_original + intensity + statement
    Las dos repeticiones del mismo grupo deben mantenerse juntas.

    Parameters
    ----------
    row : pd.Series
        Fila de metadata con los campos relevantes.

    Returns
    -------
    str
        Identificador de grupo, p.ej. 'a01_happy_strong_dogs'.
    """
    return (
        f"a{row['actor_id']:02d}"
        f"_{row['emotion_original']}"
        f"_{row['intensity']}"
        f"_{row['statement']}"
    )


def build_metadata(raw_dir: str | Path, audio_trimmed_dir: str | Path) -> pd.DataFrame:
    """
    Construye el DataFrame de metadata completo.

    Incluye:
    - Parseo de todos los archivos RAVDESS speech (modality=audio_only)
    - Targets: emotion_original y emotion_quadrant
    - utterance_group_id
    - file_path_trimmed (path esperado; el audio se genera en audio_cleaning)

    Las métricas de trimming (duration_raw_s, duration_trimmed_s, etc.)
    se añaden posteriormente por audio_cleaning.py mediante merge.

    Parameters
    ----------
    raw_dir : str | Path
        Directorio raíz de RAVDESS.
    audio_trimmed_dir : str | Path
        Directorio donde se guardarán los audios trimmeados.

    Returns
    -------
    pd.DataFrame
        Metadata con todos los campos obligatorios definidos en contracts.py,
        excepto las métricas de trimming (se añaden luego).
    """
    raw_dir = Path(raw_dir)
    audio_trimmed_dir = Path(audio_trimmed_dir)

    audio_files = discover_audio_files(raw_dir, modality="audio_only")

    records = []
    failed = []
    for fp in audio_files:
        parsed = parse_filename(fp)
        if parsed is None:
            failed.append(str(fp))
            continue
        records.append(parsed)

    if failed:
        logger.error(f"Archivos sin parsear ({len(failed)}): {failed}")
        raise ValueError(
            f"{len(failed)} archivos no pudieron parsearse. "
            "Sprint 1 no puede completarse con archivos sin parsear."
        )

    df = pd.DataFrame(records)

    # Persistir paths relativos, no paths absolutos de la máquina.
    df["file_path_raw"] = df["file_path_raw"].map(
        to_project_relative
    )

    # --- Targets ---
    df["emotion_quadrant"] = (
        df["emotion_original"]
        .map(EMOTION_TO_QUADRANT)
    )

    # --- utterance_group_id ---
    df["utterance_group_id"] = df.apply(
        build_utterance_group_id,
        axis=1,
    )

    # --- file_path_trimmed ---
    def _trimmed_path(row: pd.Series) -> str:
        actor_folder = f"Actor_{row['actor_id']:02d}"

        trimmed_path = (
            audio_trimmed_dir
            / actor_folder
            / f"{row['file_id']}.wav"
        )

        return to_project_relative(trimmed_path)


    df["file_path_trimmed"] = df.apply(
        _trimmed_path,
        axis=1,
    )

    # --- Inicializar columnas de trimming como NaN (se rellenan después) ---
    for col in ["duration_raw_s", "duration_trimmed_s", "trim_ratio",
                "trim_start_s", "trim_end_s"]:
        df[col] = float("nan")

    # --- Ordenar y resetear índice ---
    df = df.sort_values("file_id").reset_index(drop=True)

    logger.info(f"Metadata construida: {len(df)} registros, "
                f"{df['actor_id'].nunique()} actores, "
                f"{df['emotion_original'].nunique()} emociones.")

    return df


def validate_metadata_basic(df: pd.DataFrame) -> None:
    """
    Validaciones básicas de integridad sobre la metadata.
    Lanza AssertionError si alguna condición falla.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata construida por build_metadata().
    """
    # file_id único
    assert df["file_id"].is_unique, "file_id duplicados detectados."

    # Sin nulos en campos críticos
    critical = ["file_id", "actor_id", "emotion_original",
                "emotion_quadrant", "sex", "utterance_group_id"]
    for col in critical:
        nulls = df[col].isna().sum()
        assert nulls == 0, f"Columna '{col}' tiene {nulls} nulos."

    # Rango de actores
    assert df["actor_id"].between(1, 24).all(), "actor_id fuera del rango [1, 24]."

    # Emociones válidas
    valid_emotions = set(RAVDESS_EMOTION_CODES.values())
    unknown = set(df["emotion_original"].unique()) - valid_emotions
    assert not unknown, f"Emociones desconocidas: {unknown}"

    # Cuadrantes válidos
    valid_quadrants = {"Q1", "Q2", "Q3", "Q4"}
    unknown_q = set(df["emotion_quadrant"].unique()) - valid_quadrants
    assert not unknown_q, f"Cuadrantes desconocidos: {unknown_q}"

    # Sexo consistente con actor_id
    sex_check = df.apply(lambda r: actor_sex(r["actor_id"]) == r["sex"], axis=1)
    assert sex_check.all(), "Inconsistencia entre actor_id y sex."

    logger.info("✓ Validaciones básicas de metadata: OK")
