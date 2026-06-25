"""Artefactos temporales congelados de wav2vec para pooling y capas.

El módulo extrae una única vez hidden states sin targets ni folds. A partir de
ellos guarda:

- media y desvío temporal por capa en un NPZ alineado por ``file_id``;
- la secuencia de la última capa en un ``.pt`` por audio.

Las transformaciones aprendidas se entrenan posteriormente dentro de outer CV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

MODEL_NAME = "facebook/wav2vec2-base"
MODEL_REVISION = "966365c3dccea13c4bda090c093d08527a2200c4"
TARGET_SR = 16_000


def _load_model_and_feature_extractor(model_name: str, revision: str):
    """Carga perezosamente processor y encoder para no exigir transformers al importar."""
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name, revision=revision
    )
    model = Wav2Vec2Model.from_pretrained(model_name, revision=revision)
    model.requires_grad_(False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return feature_extractor, model, device
from src.utils.config import resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

LayerStrategy = Literal["last", "average"]
PoolingStrategy = Literal["mean", "mean_std"]


@dataclass(frozen=True)
class LayerStatistics:
    """Estadísticas temporales wav2vec alineadas por archivo."""

    file_ids: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_files(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_layers(self) -> int:
        return int(self.means.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.means.shape[2])


def _validate_metadata(metadata: pd.DataFrame, path_col: str) -> None:
    required = {"file_id", path_col}
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"Faltan columnas para extracción temporal: {sorted(missing)}")
    if metadata["file_id"].isna().any() or metadata["file_id"].duplicated().any():
        raise ValueError("metadata debe contener file_id únicos y no nulos.")


def _safe_torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibilidad con versiones antiguas de torch
        return torch.load(path, map_location="cpu")


def extract_temporal_single(
    file_path: str | Path,
    feature_extractor,
    model,
    device,
    target_sample_rate: int = TARGET_SR,
    include_feature_projection: bool = True,
) -> tuple[np.ndarray, np.ndarray, "Any"]:
    """Extrae media/desvío por capa y secuencia de la última capa."""
    import librosa
    import torch

    audio_path = resolve_path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

    waveform, _ = librosa.load(
        str(audio_path),
        sr=target_sample_rate,
        mono=True,
    )
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Audio vacío o inválido: {audio_path}")

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
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) < 2:
        raise RuntimeError("wav2vec no devolvió hidden states por capa.")

    selected = hidden_states if include_feature_projection else hidden_states[1:]
    # Cada tensor: [1, frames, hidden_size]
    stacked = torch.stack([state.squeeze(0) for state in selected], dim=0)
    means = stacked.mean(dim=1)
    stds = stacked.std(dim=1, correction=0)
    last_sequence = hidden_states[-1].squeeze(0).detach().cpu()

    means_np = means.detach().cpu().numpy().astype(np.float32)
    stds_np = stds.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(means_np).all() or not np.isfinite(stds_np).all():
        raise ValueError(f"Hidden states inválidos: {audio_path}")

    return means_np, stds_np, last_sequence


def validate_layer_statistics(
    stats: LayerStatistics,
    expected_file_ids: Iterable[str] | None = None,
) -> None:
    """Valida alineación, forma y valores de un artefacto de capas."""
    if stats.means.ndim != 3 or stats.stds.ndim != 3:
        raise ValueError("means y stds deben tener shape [files, layers, hidden].")
    if stats.means.shape != stats.stds.shape:
        raise ValueError("means y stds deben tener la misma forma.")
    if len(stats.file_ids) != stats.means.shape[0]:
        raise ValueError("file_ids no coincide con la primera dimensión.")
    if len(np.unique(stats.file_ids)) != len(stats.file_ids):
        raise ValueError("El artefacto temporal contiene file_id duplicados.")
    if not np.isfinite(stats.means).all() or not np.isfinite(stats.stds).all():
        raise ValueError("El artefacto temporal contiene NaN o infinitos.")
    if (stats.stds < 0).any():
        raise ValueError("Se encontraron desvíos temporales negativos.")

    if expected_file_ids is not None:
        expected = set(map(str, expected_file_ids))
        observed = set(map(str, stats.file_ids.tolist()))
        if expected != observed:
            raise ValueError(
                "Desalineación de layer statistics: "
                f"faltantes={len(expected-observed)}, extras={len(observed-expected)}."
            )


def save_layer_statistics(stats: LayerStatistics, output_path: str | Path) -> Path:
    """Guarda estadísticas por capa con metadata de procedencia."""
    validate_layer_statistics(stats)
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        file_ids=stats.file_ids.astype(str),
        layer_means=stats.means.astype(np.float32),
        layer_stds=stats.stds.astype(np.float32),
        metadata_json=np.asarray(json.dumps(stats.metadata, ensure_ascii=False)),
    )
    return path


def load_layer_statistics(
    input_path: str | Path,
    expected_file_ids: Iterable[str] | None = None,
) -> LayerStatistics:
    """Carga y valida estadísticas por capa."""
    path = resolve_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Layer statistics no encontradas: {path}")

    with np.load(path, allow_pickle=False) as data:
        metadata_raw = data["metadata_json"].item()
        stats = LayerStatistics(
            file_ids=data["file_ids"].astype(str),
            means=data["layer_means"].astype(np.float32),
            stds=data["layer_stds"].astype(np.float32),
            metadata=json.loads(str(metadata_raw)),
        )
    validate_layer_statistics(stats, expected_file_ids=expected_file_ids)
    return stats


def extract_or_load_temporal_features(
    metadata: pd.DataFrame,
    statistics_path: str | Path,
    sequences_dir: str | Path,
    path_col: str = "file_path_trimmed",
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
    target_sample_rate: int = TARGET_SR,
    include_feature_projection: bool = True,
    sequence_dtype: Literal["float16", "float32"] = "float16",
    overwrite: bool = False,
    verbose_every: int = 25,
) -> LayerStatistics:
    """Extrae o carga los artefactos temporales completos.

    Si el NPZ ya existe y ``overwrite=False``, se valida y se devuelve. La
    presencia de secuencias faltantes se verifica de forma explícita.
    """
    import torch

    _validate_metadata(metadata, path_col)
    stats_path = resolve_path(statistics_path)
    seq_dir = resolve_path(sequences_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)

    expected_ids = metadata["file_id"].astype(str).tolist()
    if stats_path.exists() and not overwrite:
        stats = load_layer_statistics(stats_path, expected_file_ids=expected_ids)
        missing_sequences = [
            file_id for file_id in expected_ids if not (seq_dir / f"{file_id}.pt").exists()
        ]
        if missing_sequences:
            raise FileNotFoundError(
                f"Faltan {len(missing_sequences)} secuencias wav2vec. "
                "Ejecute con overwrite=True para regenerar los artefactos."
            )
        return stats

    feature_extractor, model, device = _load_model_and_feature_extractor(
        model_name=model_name,
        revision=revision,
    )

    ordered = metadata.sort_values("file_id").reset_index(drop=True)
    all_means: list[np.ndarray] = []
    all_stds: list[np.ndarray] = []
    errors: list[tuple[str, str]] = []

    tensor_dtype = torch.float16 if sequence_dtype == "float16" else torch.float32

    for position, row in ordered.iterrows():
        file_id = str(row["file_id"])
        if position % verbose_every == 0:
            logger.info("wav2vec temporal: %s/%s — %s", position, len(ordered), file_id)
        try:
            means, stds, last_sequence = extract_temporal_single(
                file_path=row[path_col],
                feature_extractor=feature_extractor,
                model=model,
                device=device,
                target_sample_rate=target_sample_rate,
                include_feature_projection=include_feature_projection,
            )
            all_means.append(means)
            all_stds.append(stds)
            torch.save(last_sequence.to(dtype=tensor_dtype), seq_dir / f"{file_id}.pt")
        except Exception as exc:  # se reportan todos los archivos fallidos juntos
            logger.error("Error temporal wav2vec en %s: %s", file_id, exc)
            errors.append((file_id, str(exc)))

    if errors:
        raise RuntimeError(
            f"La extracción temporal falló en {len(errors)} archivos. "
            f"Primeros errores: {errors[:5]}"
        )

    means_array = np.stack(all_means).astype(np.float32)
    stds_array = np.stack(all_stds).astype(np.float32)
    stats = LayerStatistics(
        file_ids=ordered["file_id"].astype(str).to_numpy(),
        means=means_array,
        stds=stds_array,
        metadata={
            "model_name": model_name,
            "revision": revision,
            "sample_rate": int(target_sample_rate),
            "include_feature_projection": bool(include_feature_projection),
            "n_layers": int(means_array.shape[1]),
            "hidden_size": int(means_array.shape[2]),
            "sequence_dtype": sequence_dtype,
        },
    )
    save_layer_statistics(stats, stats_path)
    return stats


def build_static_representation(
    stats: LayerStatistics,
    layer_strategy: LayerStrategy,
    pooling: PoolingStrategy,
) -> pd.DataFrame:
    """Convierte layer statistics en una representación tabular estática."""
    validate_layer_statistics(stats)
    if layer_strategy == "last":
        means = stats.means[:, -1, :]
        stds = stats.stds[:, -1, :]
    elif layer_strategy == "average":
        means = stats.means.mean(axis=1)
        stds = stats.stds.mean(axis=1)
    else:
        raise ValueError(f"Estrategia de capas no soportada: {layer_strategy!r}")

    if pooling == "mean":
        matrix = means
        prefix = f"wav2vec_{layer_strategy}_mean"
    elif pooling == "mean_std":
        matrix = np.concatenate([means, stds], axis=1)
        prefix = f"wav2vec_{layer_strategy}_mean_std"
    else:
        raise ValueError(f"Pooling no soportado: {pooling!r}")

    columns = [f"{prefix}_{idx}" for idx in range(matrix.shape[1])]
    frame = pd.DataFrame(matrix.astype(np.float32), columns=columns)
    frame.insert(0, "file_id", stats.file_ids.astype(str))
    return frame


def build_flat_layer_representation(
    stats: LayerStatistics,
    pooling: PoolingStrategy,
) -> pd.DataFrame:
    """Aplana las estadísticas por capa para el estimador de mezcla aprendida."""
    validate_layer_statistics(stats)
    pieces = [stats.means]
    if pooling == "mean_std":
        pieces.append(stats.stds)
    elif pooling != "mean":
        raise ValueError(f"Pooling no soportado: {pooling!r}")

    matrix = np.concatenate(pieces, axis=2).reshape(stats.n_files, -1)
    columns = [f"layer_stat_{idx}" for idx in range(matrix.shape[1])]
    frame = pd.DataFrame(matrix.astype(np.float32), columns=columns)
    frame.insert(0, "file_id", stats.file_ids.astype(str))
    return frame


def validate_sequence_store(
    file_ids: Iterable[str],
    sequences_dir: str | Path,
    expected_hidden_size: int | None = None,
) -> pd.DataFrame:
    """Valida el store ragged y devuelve un índice compacto."""
    seq_dir = resolve_path(sequences_dir)
    rows: list[dict[str, Any]] = []
    for file_id in map(str, file_ids):
        path = seq_dir / f"{file_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Secuencia faltante: {path}")
        tensor = _safe_torch_load(path)
        if getattr(tensor, "ndim", None) != 2:
            raise ValueError(f"Secuencia inválida para {file_id}: shape={getattr(tensor, 'shape', None)}")
        if expected_hidden_size is not None and int(tensor.shape[1]) != expected_hidden_size:
            raise ValueError(
                f"Hidden size inválido para {file_id}: {tensor.shape[1]} != {expected_hidden_size}"
            )
        if not bool(tensor.isfinite().all()):
            raise ValueError(f"Secuencia con NaN o infinitos: {file_id}")
        rows.append(
            {
                "file_id": file_id,
                "sequence_path": path.as_posix(),
                "n_frames": int(tensor.shape[0]),
                "hidden_size": int(tensor.shape[1]),
            }
        )
    return pd.DataFrame(rows)
