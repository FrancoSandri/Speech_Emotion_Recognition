"""
Extracción de embeddings wav2vec2-base congelados.

Pipeline:
    audio trimmed
    → mono
    → resampling a 16 kHz
    → feature extractor oficial
    → last_hidden_state
    → mean pooling temporal

Salida:
    una fila por file_id
    768 dimensiones por archivo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from src.utils.config import resolve_path
from src.utils.logging import get_logger


logger = get_logger(__name__)

MODEL_NAME = "facebook/wav2vec2-base"
MODEL_REVISION = "966365c3dccea13c4bda090c093d08527a2200c4"
TARGET_SR = 16_000
EXPECTED_DIM = 768
LAYER_NAME = "last_hidden_state"
POOLING_NAME = "mean"


def _load_model_and_feature_extractor(
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
):
    """
    Carga el feature extractor y el encoder wav2vec congelado.
    """
    logger.info(
        "Cargando wav2vec: %s, revision=%s",
        model_name,
        revision,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name,
        revision=revision,
    )

    model = Wav2Vec2Model.from_pretrained(
        model_name,
        revision=revision,
    )

    # Encoder completamente congelado.
    model.requires_grad_(False)
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    logger.info(
        "Modelo cargado en %s. hidden_size=%s",
        device,
        model.config.hidden_size,
    )

    return feature_extractor, model, device


def extract_wav2vec_single(
    file_path: str | Path,
    feature_extractor,
    model,
    device,
    target_sample_rate: int = TARGET_SR,
) -> np.ndarray:
    """
    Extrae un embedding global de un único audio.

    El audio se procesa individualmente, por lo que no se introduce
    padding temporal. El embedding se obtiene promediando todos los
    vectores temporales de last_hidden_state.

    Returns
    -------
    np.ndarray
        Vector float32 de dimensión model.config.hidden_size.
    """
    audio_path = resolve_path(file_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio no encontrado: {audio_path}"
        )

    waveform, _ = librosa.load(
        str(audio_path),
        sr=target_sample_rate,
        mono=True,
    )

    if waveform.size == 0:
        raise ValueError(
            f"Audio vacío: {audio_path}"
        )

    if not np.isfinite(waveform).all():
        raise ValueError(
            f"Audio con NaN o infinitos: {audio_path}"
        )

    inputs = feature_extractor(
        waveform,
        sampling_rate=target_sample_rate,
        return_tensors="pt",
        padding=False,
    )

    input_values = inputs["input_values"].to(device)

    with torch.inference_mode():
        outputs = model(
            input_values=input_values,
            return_dict=True,
        )

    # (1, T, hidden_size) → (T, hidden_size)
    hidden_states = outputs.last_hidden_state.squeeze(0)

    if hidden_states.ndim != 2:
        raise RuntimeError(
            "Dimensión inesperada de last_hidden_state: "
            f"{tuple(hidden_states.shape)}"
        )

    embedding = (
        hidden_states
        .mean(dim=0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if not np.isfinite(embedding).all():
        raise ValueError(
            f"Embedding con NaN o infinitos: {audio_path}"
        )

    return embedding


def validate_wav2vec_features(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    expected_dim: int = EXPECTED_DIM,
) -> None:
    """
    Valida alineación, dimensión e integridad de los embeddings.
    """
    if "file_id" not in features.columns:
        raise ValueError(
            "wav2vec features no contiene la columna 'file_id'."
        )

    if features["file_id"].duplicated().any():
        duplicated = features.loc[
            features["file_id"].duplicated(keep=False),
            "file_id",
        ].unique()

        raise ValueError(
            f"wav2vec contiene {len(duplicated)} file_id duplicados."
        )

    expected_ids = set(metadata["file_id"])
    observed_ids = set(features["file_id"])

    missing_ids = expected_ids - observed_ids
    extra_ids = observed_ids - expected_ids

    if missing_ids or extra_ids:
        raise ValueError(
            "Desalineación entre metadata y wav2vec. "
            f"Faltantes={len(missing_ids)}, "
            f"extras={len(extra_ids)}."
        )

    feature_columns = [
        column
        for column in features.columns
        if column != "file_id"
    ]

    if len(feature_columns) != expected_dim:
        raise ValueError(
            "Dimensión wav2vec incorrecta. "
            f"Esperada={expected_dim}, "
            f"obtenida={len(feature_columns)}."
        )

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]

    if non_numeric:
        raise TypeError(
            "Columnas wav2vec no numéricas: "
            f"{non_numeric[:5]}"
        )

    values = features[feature_columns].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "wav2vec contiene NaN o valores infinitos."
        )


def extract_wav2vec_batch(
    metadata: pd.DataFrame,
    path_col: str = "file_path_trimmed",
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
    target_sample_rate: int = TARGET_SR,
    layer: str = LAYER_NAME,
    pooling: str = POOLING_NAME,
    expected_dim: int = EXPECTED_DIM,
    verbose_every: int = 50,
) -> pd.DataFrame:
    """
    Extrae embeddings wav2vec para todos los registros de metadata.

    El procesamiento es individual para evitar padding temporal y mantener
    el mean pooling comparable entre audios.
    """
    required_columns = {
        "file_id",
        path_col,
    }
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise KeyError(
            "Faltan columnas requeridas para wav2vec: "
            f"{sorted(missing_columns)}"
        )

    if metadata["file_id"].duplicated().any():
        raise ValueError(
            "metadata contiene file_id duplicados."
        )

    if layer != LAYER_NAME:
        raise ValueError(
            f"Layer no soportada: {layer!r}. "
            f"Se esperaba {LAYER_NAME!r}."
        )

    if pooling != POOLING_NAME:
        raise ValueError(
            f"Pooling no soportado: {pooling!r}. "
            f"Se esperaba {POOLING_NAME!r}."
        )

    feature_extractor, model, device = (
        _load_model_and_feature_extractor(
            model_name=model_name,
            revision=revision,
        )
    )

    model_dim = int(model.config.hidden_size)

    if model_dim != expected_dim:
        raise ValueError(
            "hidden_size del modelo distinto al configurado. "
            f"Modelo={model_dim}, configuración={expected_dim}."
        )

    feature_columns = [
        f"wav2vec_{index}"
        for index in range(model_dim)
    ]

    records: list[dict] = []
    errors: list[tuple[str, str]] = []

    metadata_ordered = metadata.reset_index(drop=True)

    for position, row in metadata_ordered.iterrows():
        if position % verbose_every == 0:
            logger.info(
                "wav2vec: %s/%s — %s",
                position,
                len(metadata_ordered),
                row["file_id"],
            )

        try:
            embedding = extract_wav2vec_single(
                file_path=row[path_col],
                feature_extractor=feature_extractor,
                model=model,
                device=device,
                target_sample_rate=target_sample_rate,
            )

            records.append(
                {
                    "file_id": row["file_id"],
                    **dict(zip(feature_columns, embedding)),
                }
            )

        except Exception as exc:
            logger.error(
                "Error wav2vec en %s: %s",
                row["file_id"],
                exc,
            )
            errors.append(
                (row["file_id"], str(exc))
            )

    # Nunca devolver ni guardar una extracción parcial.
    if errors:
        examples = errors[:5]

        raise RuntimeError(
            f"wav2vec falló en {len(errors)} archivos. "
            f"Primeros errores: {examples}"
        )

    features = pd.DataFrame.from_records(records)

    validate_wav2vec_features(
        features=features,
        metadata=metadata,
        expected_dim=expected_dim,
    )

    logger.info(
        "wav2vec extraído: %s archivos, %s dimensiones.",
        len(features),
        expected_dim,
    )

    return features


def load_or_extract_wav2vec(
    metadata: pd.DataFrame,
    output_path: str | Path,
    path_col: str = "file_path_trimmed",
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
    target_sample_rate: int = TARGET_SR,
    layer: str = LAYER_NAME,
    pooling: str = POOLING_NAME,
    expected_dim: int = EXPECTED_DIM,
    overwrite: bool = False,
    verbose_every: int = 50,
) -> pd.DataFrame:
    """
    Carga un Parquet válido o ejecuta una extracción completa.
    """
    resolved_output_path = resolve_path(output_path)

    if resolved_output_path.exists() and not overwrite:
        logger.info(
            "Cargando wav2vec desde caché: %s",
            resolved_output_path,
        )

        features = pd.read_parquet(
            resolved_output_path
        )

        validate_wav2vec_features(
            features=features,
            metadata=metadata,
            expected_dim=expected_dim,
        )

        return features

    features = extract_wav2vec_batch(
        metadata=metadata,
        path_col=path_col,
        model_name=model_name,
        revision=revision,
        target_sample_rate=target_sample_rate,
        layer=layer,
        pooling=pooling,
        expected_dim=expected_dim,
        verbose_every=verbose_every,
    )

    resolved_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        resolved_output_path,
        index=False,
    )

    logger.info(
        "wav2vec guardado: %s",
        resolved_output_path,
    )

    return features