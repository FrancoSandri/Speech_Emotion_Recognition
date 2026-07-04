"""
src/config/contracts.py
-----------------------
Contratos compartidos del proyecto: columnas requeridas, nombres oficiales
de targets, protocolos y paths. No contiene lógica de procesamiento.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Columnas requeridas en metadata.parquet
# ---------------------------------------------------------------------------
METADATA_REQUIRED_COLS = [
    "file_id",
    "file_path_raw",
    "file_path_trimmed",
    "actor_id",
    "sex",
    "modality",
    "vocal_channel",
    "emotion_code",
    "emotion_original",
    "intensity_code",
    "intensity",
    "statement_code",
    "statement",
    "repetition_id",
    "utterance_group_id",
    "emotion_quadrant",
    # Métricas de trimming
    "duration_raw_s",
    "duration_trimmed_s",
    "trim_ratio",
    "trim_start_s",
    "trim_end_s",
]

# ---------------------------------------------------------------------------
# Columnas requeridas en splits.parquet
# ---------------------------------------------------------------------------
SPLITS_REQUIRED_COLS = [
    "file_id",
    "partition",
    "fold_speaker_independent",
]

# ---------------------------------------------------------------------------
# Nombres oficiales de targets
# ---------------------------------------------------------------------------
TARGET_EMOTION_ORIGINAL = "emotion_original"
TARGET_EMOTION_QUADRANT = "emotion_quadrant"
TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT = "emotion_original_eval_quadrant"

ALL_TARGETS = {
    TARGET_EMOTION_ORIGINAL,
    TARGET_EMOTION_QUADRANT,
    TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
}

# ---------------------------------------------------------------------------
# Nombres oficiales de protocolos de evaluación
# ---------------------------------------------------------------------------
PROTOCOL_INDEPENDENT = "speaker_independent"

# ---------------------------------------------------------------------------
# Valores oficiales de la columna `partition`
# ---------------------------------------------------------------------------
PARTITION_DEVELOPMENT      = "development"
PARTITION_TEST_INDEPENDENT = "test_speaker_independent"

ALL_PARTITIONS = {
    PARTITION_DEVELOPMENT,
    PARTITION_TEST_INDEPENDENT,
}

# ---------------------------------------------------------------------------
# Fold sentinel: -1 indica que el registro no pertenece al pool de CV
# ---------------------------------------------------------------------------
FOLD_SENTINEL = -1

# ---------------------------------------------------------------------------
# Mapping emoción RAVDESS
# ---------------------------------------------------------------------------
RAVDESS_EMOTION_CODES: dict[int, str] = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

RAVDESS_INTENSITY_CODES: dict[int, str] = {
    1: "normal",
    2: "strong",
}

RAVDESS_STATEMENT_CODES: dict[int, str] = {
    1: "kids",   # "Kids are talking by the door"
    2: "dogs",   # "Dogs are sitting by the door"
}

RAVDESS_VOCAL_CHANNEL_CODES: dict[int, str] = {
    1: "speech",
    2: "song",
}

RAVDESS_MODALITY_CODES: dict[int, str] = {
    1: "full_av",
    2: "video_only",
    3: "audio_only",
}

# Convención oficial RAVDESS:
# actores impares = masculino
# actores pares = femenino
def actor_sex(actor_id: int) -> str:
    """
    Devuelve el sexo del actor según la codificación oficial de RAVDESS.

    Parameters
    ----------
    actor_id : int
        Identificador del actor, dentro del intervalo [1, 24].

    Returns
    -------
    str
        "male" para actores impares y "female" para actores pares.

    Raises
    ------
    ValueError
        Si actor_id está fuera del rango válido.
    """
    if not 1 <= actor_id <= 24:
        raise ValueError(
            f"actor_id fuera del rango válido [1, 24]: {actor_id}"
        )

    return "male" if actor_id % 2 != 0 else "female"

# Mapping nombre de emoción → cuadrante Russell
EMOTION_TO_QUADRANT: dict[str, str] = {
    "neutral":   "Q4",
    "calm":      "Q4",
    "happy":     "Q1",
    "surprised": "Q1",
    "angry":     "Q2",
    "fearful":   "Q2",
    "disgust":   "Q2",
    "sad":       "Q3",
}

QUADRANT_LABELS: dict[str, str] = {
    "Q1": "High Valence / High Arousal",
    "Q2": "Low Valence / High Arousal",
    "Q3": "Low Valence / Low Arousal",
    "Q4": "High Valence / Low Arousal",
}
