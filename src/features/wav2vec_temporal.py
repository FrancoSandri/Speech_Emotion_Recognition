"""Artefactos temporales congelados de wav2vec.

El módulo extrae hidden states sin targets ni folds y persiste tres vistas
alineadas por ``file_id``:

- media y desvío temporal por capa en un único NPZ;
- secuencia de la última capa para compatibilidad con experimentos previos;
- secuencia completa ``[capas, frames, hidden]`` para atención multicapa.

Todas las transformaciones aprendidas se ajustan posteriormente dentro de
outer cross-validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from src.utils.config import resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = "facebook/wav2vec2-base"
MODEL_REVISION = "966365c3dccea13c4bda090c093d08527a2200c4"
TARGET_SR = 16_000

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


def _load_model_and_feature_extractor(model_name: str, revision: str):
    """Carga perezosamente processor y encoder congelado."""
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name,
        revision=revision,
    )
    model = Wav2Vec2Model.from_pretrained(model_name, revision=revision)
    model.requires_grad_(False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return feature_extractor, model, device


def _validate_metadata(metadata: pd.DataFrame, path_col: str) -> None:
    required = {"file_id", path_col}
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"Faltan columnas para extracción temporal: {sorted(missing)}")
    if metadata["file_id"].isna().any() or metadata["file_id"].duplicated().any():
        raise ValueError("metadata debe contener file_id únicos y no nulos.")


def _safe_torch_load(path: str | Path):
    import torch

    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu")


def extract_multilayer_sequence(
    file_path: str | Path,
    feature_extractor,
    model,
    device,
    target_sample_rate: int = TARGET_SR,
    include_feature_projection: bool = True,
):
    """Extrae hidden states frame-level de todas las capas.

    Returns
    -------
    torch.Tensor
        Tensor CPU float32 con forma ``[n_layers, n_frames, hidden_size]``.
    """
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
    stacked = torch.stack([state.squeeze(0) for state in selected], dim=0)
    stacked = stacked.detach().cpu().to(dtype=torch.float32)

    if stacked.ndim != 3 or not bool(torch.isfinite(stacked).all()):
        raise ValueError(
            f"Hidden states inválidos para {audio_path}: shape={tuple(stacked.shape)}"
        )
    return stacked


def save_multilayer_sequence(
    hidden_states,
    file_id: str,
    output_path: str | Path,
    model_name: str,
    revision: str,
    sample_rate: int,
    dtype: Literal["float16", "float32"] = "float16",
) -> Path:
    """Guarda una secuencia multicapa autocontenida."""
    import torch

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states debe tener forma [layers, frames, hidden].")
    tensor_dtype = torch.float16 if dtype == "float16" else torch.float32
    tensor = hidden_states.detach().cpu().to(dtype=tensor_dtype)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Hidden states no finitos para {file_id}.")

    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_states": tensor,
            "file_id": str(file_id),
            "n_layers": int(tensor.shape[0]),
            "n_frames": int(tensor.shape[1]),
            "hidden_size": int(tensor.shape[2]),
            "model_name": str(model_name),
            "model_revision": str(revision),
            "sample_rate": int(sample_rate),
        },
        path,
    )
    return path


def load_multilayer_sequence(
    input_path: str | Path,
    expected_file_id: str | None = None,
    expected_n_layers: int | None = None,
    expected_hidden_size: int | None = None,
):
    """Carga y valida un tensor multicapa."""
    import torch

    path = resolve_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Secuencia multicapa no encontrada: {path}")

    payload = _safe_torch_load(path)
    if isinstance(payload, torch.Tensor):
        hidden_states = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict) and "hidden_states" in payload:
        hidden_states = payload["hidden_states"]
        metadata = {key: value for key, value in payload.items() if key != "hidden_states"}
    else:
        raise ValueError(f"Formato multicapa inválido: {path}")

    hidden_states = hidden_states.to(dtype=torch.float32)
    if hidden_states.ndim != 3 or not bool(torch.isfinite(hidden_states).all()):
        raise ValueError(f"Tensor multicapa inválido: {path}")
    if expected_file_id is not None:
        observed = str(metadata.get("file_id", expected_file_id))
        if observed != str(expected_file_id):
            raise ValueError(f"file_id inconsistente en {path}: {observed}")
    if expected_n_layers is not None and int(hidden_states.shape[0]) != expected_n_layers:
        raise ValueError(
            f"Número de capas inválido en {path}: "
            f"{hidden_states.shape[0]} != {expected_n_layers}"
        )
    if expected_hidden_size is not None and int(hidden_states.shape[2]) != expected_hidden_size:
        raise ValueError(
            f"Hidden size inválido en {path}: "
            f"{hidden_states.shape[2]} != {expected_hidden_size}"
        )
    return hidden_states, metadata


def extract_temporal_single(
    file_path: str | Path,
    feature_extractor,
    model,
    device,
    target_sample_rate: int = TARGET_SR,
    include_feature_projection: bool = True,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Extrae media/desvío por capa y secuencia de la última capa."""
    stacked = extract_multilayer_sequence(
        file_path=file_path,
        feature_extractor=feature_extractor,
        model=model,
        device=device,
        target_sample_rate=target_sample_rate,
        include_feature_projection=include_feature_projection,
    )
    means = stacked.mean(dim=1).numpy().astype(np.float32)
    stds = stacked.std(dim=1, correction=0).numpy().astype(np.float32)
    return means, stds, stacked[-1].clone()


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


def validate_sequence_store(
    file_ids: Iterable[str],
    sequences_dir: str | Path,
    expected_hidden_size: int | None = None,
) -> pd.DataFrame:
    """Valida el store de secuencias de última capa."""
    seq_dir = resolve_path(sequences_dir)
    rows: list[dict[str, Any]] = []
    for file_id in map(str, file_ids):
        path = seq_dir / f"{file_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Secuencia faltante: {path}")
        tensor = _safe_torch_load(path)
        if getattr(tensor, "ndim", None) != 2:
            raise ValueError(
                f"Secuencia inválida para {file_id}: "
                f"shape={getattr(tensor, 'shape', None)}"
            )
        if expected_hidden_size is not None and int(tensor.shape[1]) != expected_hidden_size:
            raise ValueError(
                f"Hidden size inválido para {file_id}: "
                f"{tensor.shape[1]} != {expected_hidden_size}"
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


def validate_multilayer_store(
    file_ids: Iterable[str],
    sequences_dir: str | Path,
    expected_n_layers: int | None = None,
    expected_hidden_size: int | None = None,
) -> pd.DataFrame:
    """Valida el store multicapa y devuelve un índice alineable."""
    seq_dir = resolve_path(sequences_dir)
    rows: list[dict[str, Any]] = []
    for file_id in map(str, file_ids):
        path = seq_dir / f"{file_id}.pt"
        hidden_states, metadata = load_multilayer_sequence(
            path,
            expected_file_id=file_id,
            expected_n_layers=expected_n_layers,
            expected_hidden_size=expected_hidden_size,
        )
        rows.append(
            {
                "file_id": file_id,
                "multilayer_sequence_path": path.as_posix(),
                "n_layers": int(hidden_states.shape[0]),
                "n_frames": int(hidden_states.shape[1]),
                "hidden_size": int(hidden_states.shape[2]),
                "model_revision": metadata.get("model_revision"),
            }
        )
    return pd.DataFrame(rows)


def extract_or_load_temporal_features(
    metadata: pd.DataFrame,
    statistics_path: str | Path,
    sequences_dir: str | Path,
    multilayer_sequences_dir: str | Path | None = None,
    path_col: str = "file_path_trimmed",
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
    target_sample_rate: int = TARGET_SR,
    include_feature_projection: bool = True,
    sequence_dtype: Literal["float16", "float32"] = "float16",
    overwrite: bool = False,
    verbose_every: int = 25,
) -> LayerStatistics:
    """Extrae o completa artefactos temporales sin repetir trabajo innecesario."""
    import torch

    _validate_metadata(metadata, path_col)
    stats_path = resolve_path(statistics_path)
    seq_dir = resolve_path(sequences_dir)
    multi_dir = (
        resolve_path(multilayer_sequences_dir)
        if multilayer_sequences_dir is not None
        else None
    )
    seq_dir.mkdir(parents=True, exist_ok=True)
    if multi_dir is not None:
        multi_dir.mkdir(parents=True, exist_ok=True)

    ordered = metadata.sort_values("file_id").reset_index(drop=True)
    expected_ids = ordered["file_id"].astype(str).tolist()
    stats_existing = stats_path.exists() and not overwrite
    stats = (
        load_layer_statistics(stats_path, expected_file_ids=expected_ids)
        if stats_existing
        else None
    )

    missing_last = [
        file_id
        for file_id in expected_ids
        if overwrite or not (seq_dir / f"{file_id}.pt").exists()
    ]
    missing_multi = []
    if multi_dir is not None:
        missing_multi = [
            file_id
            for file_id in expected_ids
            if overwrite or not (multi_dir / f"{file_id}.pt").exists()
        ]

    need_stats = stats is None
    ids_to_process = set(expected_ids if need_stats else []) | set(missing_last) | set(missing_multi)
    if not ids_to_process:
        return stats  # type: ignore[return-value]

    feature_extractor, model, device = _load_model_and_feature_extractor(
        model_name=model_name,
        revision=revision,
    )

    means_by_id: dict[str, np.ndarray] = {}
    stds_by_id: dict[str, np.ndarray] = {}
    errors: list[tuple[str, str]] = []
    tensor_dtype = torch.float16 if sequence_dtype == "float16" else torch.float32

    process_rows = ordered.loc[ordered["file_id"].astype(str).isin(ids_to_process)]
    for position, row in enumerate(process_rows.itertuples(index=False), start=1):
        file_id = str(row.file_id)
        if position == 1 or position % verbose_every == 0:
            logger.info(
                "wav2vec temporal: %s/%s — %s",
                position,
                len(process_rows),
                file_id,
            )
        try:
            stacked = extract_multilayer_sequence(
                file_path=getattr(row, path_col),
                feature_extractor=feature_extractor,
                model=model,
                device=device,
                target_sample_rate=target_sample_rate,
                include_feature_projection=include_feature_projection,
            )
            if need_stats:
                means_by_id[file_id] = stacked.mean(dim=1).numpy().astype(np.float32)
                stds_by_id[file_id] = (
                    stacked.std(dim=1, correction=0).numpy().astype(np.float32)
                )
            if file_id in missing_last:
                torch.save(
                    stacked[-1].to(dtype=tensor_dtype),
                    seq_dir / f"{file_id}.pt",
                )
            if multi_dir is not None and file_id in missing_multi:
                save_multilayer_sequence(
                    stacked,
                    file_id=file_id,
                    output_path=multi_dir / f"{file_id}.pt",
                    model_name=model_name,
                    revision=revision,
                    sample_rate=target_sample_rate,
                    dtype=sequence_dtype,
                )
        except Exception as exc:
            logger.error("Error temporal wav2vec en %s: %s", file_id, exc)
            errors.append((file_id, str(exc)))

    if errors:
        raise RuntimeError(
            f"La extracción temporal falló en {len(errors)} archivos. "
            f"Primeros errores: {errors[:5]}"
        )

    if need_stats:
        means_array = np.stack([means_by_id[file_id] for file_id in expected_ids])
        stds_array = np.stack([stds_by_id[file_id] for file_id in expected_ids])
        stats = LayerStatistics(
            file_ids=np.asarray(expected_ids),
            means=means_array.astype(np.float32),
            stds=stds_array.astype(np.float32),
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

    validate_sequence_store(
        expected_ids,
        seq_dir,
        expected_hidden_size=stats.hidden_size,
    )
    if multi_dir is not None:
        validate_multilayer_store(
            expected_ids,
            multi_dir,
            expected_n_layers=stats.n_layers,
            expected_hidden_size=stats.hidden_size,
        )
    return stats


def build_static_representation(
    stats: LayerStatistics,
    layer_strategy: LayerStrategy,
    pooling: PoolingStrategy,
) -> pd.DataFrame:
    """Convierte estadísticas por capa en una representación tabular."""
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
    """Aplana estadísticas por capa para la mezcla aprendida tabular."""
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


class MultiLayerSequenceDataset:
    """Dataset lazy para tensores ``[layers, frames, hidden]``."""

    def __init__(
        self,
        file_ids: Sequence[str],
        sequence_paths: Sequence[str | Path],
        labels: Sequence[int] | None = None,
    ) -> None:
        if len(file_ids) != len(sequence_paths):
            raise ValueError("file_ids y sequence_paths deben tener igual longitud.")
        if labels is not None and len(labels) != len(file_ids):
            raise ValueError("labels debe tener igual longitud que file_ids.")
        self.file_ids = list(map(str, file_ids))
        self.sequence_paths = [str(path) for path in sequence_paths]
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, index: int):
        hidden_states, _ = load_multilayer_sequence(
            self.sequence_paths[index],
            expected_file_id=self.file_ids[index],
        )
        item = {
            "file_id": self.file_ids[index],
            "hidden_states": hidden_states,
        }
        if self.labels is not None:
            item["label"] = int(self.labels[index])
        return item


def collate_multilayer_sequences(batch):
    """Padding temporal para una tanda de secuencias multicapa."""
    import torch

    if not batch:
        raise ValueError("No se puede colatear un batch vacío.")
    n_layers = int(batch[0]["hidden_states"].shape[0])
    hidden_size = int(batch[0]["hidden_states"].shape[2])
    lengths = torch.tensor(
        [int(item["hidden_states"].shape[1]) for item in batch],
        dtype=torch.long,
    )
    max_frames = int(lengths.max())
    padded = torch.zeros(
        len(batch),
        n_layers,
        max_frames,
        hidden_size,
        dtype=torch.float32,
    )
    for index, item in enumerate(batch):
        tensor = item["hidden_states"]
        if tensor.shape[0] != n_layers or tensor.shape[2] != hidden_size:
            raise ValueError("Todas las secuencias deben compartir capas y hidden size.")
        padded[index, :, : tensor.shape[1], :] = tensor

    time_index = torch.arange(max_frames)[None, :]
    mask = time_index < lengths[:, None]
    output = {
        "file_ids": [item["file_id"] for item in batch],
        "hidden_states": padded,
        "mask": mask,
        "lengths": lengths,
    }
    if "label" in batch[0]:
        output["labels"] = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long,
        )
    return output
