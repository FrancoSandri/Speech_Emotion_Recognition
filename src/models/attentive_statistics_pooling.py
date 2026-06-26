"""Attentive statistics pooling para secuencias wav2vec congeladas."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from src.models.layer_mixture import build_sequence_layer_mixture


def _safe_torch_load(path: str | Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu")


class Wav2VecSequenceDataset:
    """Dataset legacy de secuencias de última capa."""

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
        import torch

        sequence = _safe_torch_load(self.sequence_paths[index])
        if isinstance(sequence, dict):
            sequence = sequence.get("hidden_states", sequence)
        sequence = sequence.to(dtype=torch.float32)
        item = {
            "file_id": self.file_ids[index],
            "sequence": sequence,
        }
        if self.labels is not None:
            item["label"] = int(self.labels[index])
        return item


def collate_wav2vec_sequences(batch):
    """Padding para secuencias legacy de última capa."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    sequences = [item["sequence"] for item in batch]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    time_index = torch.arange(padded.shape[1])[None, :]
    mask = time_index < lengths[:, None]

    output = {
        "file_ids": [item["file_id"] for item in batch],
        "sequences": padded,
        "mask": mask,
        "lengths": lengths,
    }
    if "label" in batch[0]:
        output["labels"] = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long,
        )
    return output


def build_attentive_statistics_pooling(
    input_dim: int,
    attention_hidden_dim: int,
    eps: float = 1e-5,
):
    """Construye un pooling temporal independiente del clasificador."""
    import torch
    from torch import nn

    class AttentiveStatisticsPooling(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(input_dim, attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(attention_hidden_dim, 1),
            )

        def forward(self, sequences, mask):
            if sequences.ndim != 3:
                raise ValueError("sequences debe tener forma [batch, time, hidden].")
            if mask.ndim != 2 or mask.shape[:2] != sequences.shape[:2]:
                raise ValueError("mask debe tener forma [batch, time].")

            scores = self.attention(sequences).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
            weights = weights * mask.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)

            mean = torch.sum(weights.unsqueeze(-1) * sequences, dim=1)
            centered = sequences - mean.unsqueeze(1)
            variance = torch.sum(
                weights.unsqueeze(-1) * centered.square(),
                dim=1,
            )
            std = torch.sqrt(torch.clamp(variance, min=eps))
            return torch.cat([mean, std], dim=-1), weights

    return AttentiveStatisticsPooling()


def build_attentive_statistics_classifier(
    input_dim: int,
    attention_hidden_dim: int,
    n_classes: int,
    dropout: float = 0.10,
    eps: float = 1e-5,
):
    """Clasificador legacy sobre una secuencia ya construida."""
    from torch import nn

    class AttentiveStatisticsClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pooling = build_attentive_statistics_pooling(
                input_dim=input_dim,
                attention_hidden_dim=attention_hidden_dim,
                eps=eps,
            )
            self.normalization = nn.LayerNorm(input_dim * 2)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(input_dim * 2, n_classes)

        def pool(self, sequences, mask):
            return self.pooling(sequences, mask)

        def forward(self, sequences, mask, return_attention: bool = False):
            pooled, weights = self.pooling(sequences, mask)
            logits = self.classifier(self.dropout(self.normalization(pooled)))
            if return_attention:
                return logits, weights
            return logits

    return AttentiveStatisticsClassifier()


def build_multilayer_attentive_statistics_classifier(
    n_layers: int,
    input_dim: int,
    attention_hidden_dim: int,
    n_classes: int,
    layer_strategy: Literal["uniform", "learned"],
    dropout: float = 0.50,
    eps: float = 1e-5,
):
    """Integra mezcla multicapa y attentive statistics pooling.

    ``uniform`` implementa ``average_attention_statistics`` y ``learned``
    implementa ``learned_layers_attention_statistics``. La mezcla aprendida se
    inicializa en el promedio uniforme para que ambos modelos compartan el mismo
    punto de partida representacional.
    """
    from torch import nn

    class MultiLayerAttentiveStatisticsClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_strategy = layer_strategy
            self.layer_mixture = build_sequence_layer_mixture(
                n_layers=n_layers,
                strategy=layer_strategy,
            )
            self.pooling = build_attentive_statistics_pooling(
                input_dim=input_dim,
                attention_hidden_dim=attention_hidden_dim,
                eps=eps,
            )
            self.normalization = nn.LayerNorm(input_dim * 2)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(input_dim * 2, n_classes)

        def layer_weights(self):
            return self.layer_mixture.layer_weights()

        def forward(
            self,
            hidden_states,
            mask,
            return_attention: bool = False,
            return_layer_weights: bool = False,
        ):
            sequence, layer_weights = self.layer_mixture(hidden_states)
            pooled, attention_weights = self.pooling(sequence, mask)
            logits = self.classifier(self.dropout(self.normalization(pooled)))

            if return_attention or return_layer_weights:
                output = {"logits": logits}
                if return_attention:
                    output["attention_weights"] = attention_weights
                if return_layer_weights:
                    output["layer_weights"] = layer_weights
                return output
            return logits

    return MultiLayerAttentiveStatisticsClassifier()
