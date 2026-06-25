"""Pooling estadístico atento sobre secuencias wav2vec congeladas."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _safe_torch_load(path: str | Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class Wav2VecSequenceDataset:
    """Dataset liviano que carga una secuencia por ``file_id`` bajo demanda."""

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

        sequence = _safe_torch_load(self.sequence_paths[index]).to(dtype=torch.float32)
        item = {
            "file_id": self.file_ids[index],
            "sequence": sequence,
        }
        if self.labels is not None:
            item["label"] = int(self.labels[index])
        return item


def collate_wav2vec_sequences(batch):
    """Aplica padding y devuelve una máscara booleana de frames válidos."""
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
            [item["label"] for item in batch], dtype=torch.long
        )
    return output


def build_attentive_statistics_classifier(
    input_dim: int,
    attention_hidden_dim: int,
    n_classes: int,
    dropout: float = 0.10,
    eps: float = 1e-5,
):
    """Construye atención temporal + media/desvío ponderados + clasificador."""
    import torch
    from torch import nn

    class AttentiveStatisticsClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(input_dim, attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(attention_hidden_dim, 1),
            )
            self.normalization = nn.LayerNorm(input_dim * 2)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(input_dim * 2, n_classes)

        def pool(self, sequences, mask):
            scores = self.attention(sequences).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)

            mean = torch.sum(weights.unsqueeze(-1) * sequences, dim=1)
            centered = sequences - mean.unsqueeze(1)
            variance = torch.sum(
                weights.unsqueeze(-1) * centered.square(), dim=1
            )
            std = torch.sqrt(torch.clamp(variance, min=eps))
            return torch.cat([mean, std], dim=-1), weights

        def forward(self, sequences, mask, return_attention: bool = False):
            pooled, weights = self.pool(sequences, mask)
            logits = self.classifier(
                self.dropout(self.normalization(pooled))
            )
            if return_attention:
                return logits, weights
            return logits

    return AttentiveStatisticsClassifier()
