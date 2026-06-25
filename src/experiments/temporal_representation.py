"""Experimentos controlados de pooling temporal, capas y Elastic Net.

Reutiliza los folds persistidos, las métricas globales y el feature store del
proyecto. Los artefactos de test no son accesibles desde estas funciones.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.config.contracts import (
    PARTITION_DEVELOPMENT,
    PROTOCOL_DEPENDENT,
    PROTOCOL_INDEPENDENT,
    TARGET_EMOTION_ORIGINAL,
    TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
    TARGET_EMOTION_QUADRANT,
)
from src.evaluation.metrics import compute_8_to_4_metrics, compute_metrics
from src.experiments.cross_validation import run_cv
from src.features.egemaps_families import FAMILY_LABELS, build_family_mapping
from src.features.feature_store import prepare_model_table
from src.features.wav2vec_temporal import (
    LayerStatistics,
    build_flat_layer_representation,
    build_static_representation,
    validate_sequence_store,
)
from src.models.attentive_statistics_pooling import (
    Wav2VecSequenceDataset,
    build_attentive_statistics_classifier,
    collate_wav2vec_sequences,
)
from src.models.elastic_net import build_elastic_net_search
from src.models.layer_mixture import (
    build_layer_mixture_network,
    fit_layer_standardizer,
    reshape_layer_features,
)
from src.models.linear_probe import build_linear_probe
from src.utils.config import resolve_path
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


@dataclass(frozen=True)
class TargetSpec:
    train_target: str
    train_labels: list[str]
    eval_labels: list[str]


def _target_spec(table: pd.DataFrame, target: str) -> TargetSpec:
    emotions = sorted(table[TARGET_EMOTION_ORIGINAL].astype(str).unique())
    quadrants = ["Q1", "Q2", "Q3", "Q4"]
    if target == TARGET_EMOTION_ORIGINAL:
        return TargetSpec(TARGET_EMOTION_ORIGINAL, emotions, emotions)
    if target == TARGET_EMOTION_QUADRANT:
        return TargetSpec(TARGET_EMOTION_QUADRANT, quadrants, quadrants)
    if target == TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT:
        return TargetSpec(TARGET_EMOTION_ORIGINAL, emotions, quadrants)
    raise ValueError(f"Target no soportado: {target!r}")


def _evaluate_target(
    target: str,
    y_true_train_space: np.ndarray,
    y_pred_train_space: np.ndarray,
    y_true_quadrant: np.ndarray,
    spec: TargetSpec,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, float] | None]:
    if target == TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT:
        original = compute_metrics(
            y_true_train_space,
            y_pred_train_space,
            labels=spec.train_labels,
        )
        metrics, y_pred_eval = compute_8_to_4_metrics(
            y_true_quadrant=y_true_quadrant,
            y_pred_emotion=y_pred_train_space,
            quadrant_labels=spec.eval_labels,
        )
        return metrics, y_true_quadrant, y_pred_eval, original

    metrics = compute_metrics(
        y_true_train_space,
        y_pred_train_space,
        labels=spec.eval_labels,
    )
    return metrics, y_true_train_space, y_pred_train_space, None


def _group_column(protocol: str) -> str:
    if protocol == PROTOCOL_INDEPENDENT:
        return "actor_id"
    if protocol == PROTOCOL_DEPENDENT:
        return "utterance_group_id"
    raise ValueError(f"Protocolo no soportado: {protocol!r}")


def _fold_column(protocol: str) -> str:
    return f"fold_{protocol}"


def _inner_split(
    table: pd.DataFrame,
    y: np.ndarray,
    protocol: str,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = table[_group_column(protocol)].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(np.zeros(len(table)), y, groups=groups))
    return np.asarray(train_idx), np.asarray(val_idx)


def _class_weights(y_encoded: np.ndarray, n_classes: int):
    import torch

    counts = np.bincount(y_encoded, minlength=n_classes).astype(np.float32)
    weights = len(y_encoded) / np.maximum(counts, 1.0)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    labels = list(range(n_classes))
    return float(compute_metrics(y_true, y_pred, labels=labels)["macro_f1"])


def _train_tabular_neural(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[Any, int, float]:
    """Entrena un modelo tabular torch con full-batch y early stopping."""
    import torch
    from torch import nn

    set_global_seed(seed)
    device = _device()
    model = model.to(device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = None if X_val is None else torch.tensor(X_val, dtype=torch.float32, device=device)

    loss_fn = nn.CrossEntropyLoss(
        weight=_class_weights(y_train, int(y_train.max()) + 1).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_state = deepcopy(model.state_dict())
    best_score = -np.inf
    best_epoch = 1
    stale = 0
    epochs = fixed_epochs or max_epochs

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if fixed_epochs is not None:
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            continue

        model.eval()
        with torch.inference_mode():
            val_pred = model(X_val_t).argmax(dim=1).cpu().numpy()
        score = _torch_macro_f1(y_val, val_pred, int(y_train.max()) + 1)
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, float(best_score if np.isfinite(best_score) else np.nan)


def _predict_tabular_neural(model, X: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.inference_mode():
        logits = model(torch.tensor(X, dtype=torch.float32, device=_device()))
        return torch.softmax(logits, dim=1).cpu().numpy()


def run_static_pooling_grid(
    stats: LayerStatistics,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    logistic_regression_config: Mapping[str, Any],
    seed: int,
    layer_strategies: Sequence[str] = ("last", "average"),
    poolings: Sequence[str] = ("mean", "mean_std"),
    protocols: Sequence[str] = (PROTOCOL_INDEPENDENT,),
    targets: Sequence[str] = (TARGET_EMOTION_ORIGINAL,),
    n_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Evalúa poolings no entrenables con el motor estándar de CV."""
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for layer_strategy in layer_strategies:
        for pooling in poolings:
            representation = build_static_representation(stats, layer_strategy, pooling)
            for protocol in protocols:
                for target in targets:
                    output = run_cv(
                        pipeline_factory=lambda: build_linear_probe(
                            params=logistic_regression_config,
                            seed=seed,
                        ),
                        representation=representation,
                        metadata=metadata,
                        splits=splits,
                        target_col=target,
                        protocol=protocol,
                        representation_name="wav2vec_temporal",
                        model_name="logistic_regression",
                        refinement=f"{layer_strategy}_{pooling}",
                        n_folds=n_folds,
                    )
                    fold_frames.append(output["fold_results"])
                    pred = output["predictions"].copy()
                    pred["representation"] = "wav2vec_temporal"
                    pred["protocol"] = protocol
                    pred["target"] = target
                    pred["model"] = "logistic_regression"
                    pred["refinement"] = f"{layer_strategy}_{pooling}"
                    prediction_frames.append(pred)

    return {
        "fold_results": pd.concat(fold_frames, ignore_index=True, sort=False),
        "predictions": pd.concat(prediction_frames, ignore_index=True, sort=False),
    }


def run_layer_mixture_grid(
    stats: LayerStatistics,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    seeds: Sequence[int],
    poolings: Sequence[str] = ("mean", "mean_std"),
    protocols: Sequence[str] = (PROTOCOL_INDEPENDENT,),
    targets: Sequence[str] = (TARGET_EMOTION_ORIGINAL,),
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    max_epochs: int = 250,
    patience: int = 20,
    inner_folds: int = 3,
    n_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Entrena mezclas escalares de capas y separa seed/fold variability."""
    all_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    weights_rows: list[dict[str, Any]] = []

    for pooling in poolings:
        representation = build_flat_layer_representation(stats, pooling)
        table, feature_cols = prepare_model_table(representation, metadata, splits)
        for protocol in protocols:
            fold_col = _fold_column(protocol)
            folds = sorted(table[fold_col].unique())
            if n_folds is not None and len(folds) != n_folds:
                raise ValueError(f"Folds inesperados para {protocol}: {folds}")

            for target in targets:
                spec = _target_spec(table, target)
                encoder = LabelEncoder().fit(spec.train_labels)
                for fold in folds:
                    train_mask = table[fold_col] != fold
                    val_mask = ~train_mask
                    outer_train = table.loc[train_mask].reset_index(drop=True)
                    outer_val = table.loc[val_mask].reset_index(drop=True)
                    X_outer_train = reshape_layer_features(
                        outer_train[feature_cols].to_numpy(np.float32),
                        stats.n_layers,
                        stats.hidden_size,
                        pooling,
                    )
                    X_outer_val = reshape_layer_features(
                        outer_val[feature_cols].to_numpy(np.float32),
                        stats.n_layers,
                        stats.hidden_size,
                        pooling,
                    )
                    y_outer_train = encoder.transform(outer_train[spec.train_target])
                    y_outer_val_labels = outer_val[spec.train_target].astype(str).to_numpy()

                    inner_train_idx, inner_val_idx = _inner_split(
                        outer_train,
                        y_outer_train,
                        protocol,
                        n_splits=inner_folds,
                        seed=42,
                    )
                    seed_probabilities: list[np.ndarray] = []
                    seed_metrics: list[float] = []

                    for seed in seeds:
                        inner_standardizer = fit_layer_standardizer(
                            X_outer_train[inner_train_idx]
                        )
                        X_inner_train = inner_standardizer.transform(
                            X_outer_train[inner_train_idx]
                        )
                        X_inner_val = inner_standardizer.transform(
                            X_outer_train[inner_val_idx]
                        )
                        model = build_layer_mixture_network(
                            n_layers=stats.n_layers,
                            hidden_size=stats.hidden_size,
                            pooling=pooling,
                            n_classes=len(encoder.classes_),
                        )
                        model, best_epoch, _ = _train_tabular_neural(
                            model,
                            X_inner_train,
                            y_outer_train[inner_train_idx],
                            X_inner_val,
                            y_outer_train[inner_val_idx],
                            learning_rate,
                            weight_decay,
                            max_epochs,
                            patience,
                            seed,
                        )

                        outer_standardizer = fit_layer_standardizer(X_outer_train)
                        X_train_scaled = outer_standardizer.transform(X_outer_train)
                        X_val_scaled = outer_standardizer.transform(X_outer_val)
                        final_model = build_layer_mixture_network(
                            n_layers=stats.n_layers,
                            hidden_size=stats.hidden_size,
                            pooling=pooling,
                            n_classes=len(encoder.classes_),
                        )
                        started = perf_counter()
                        final_model, _, _ = _train_tabular_neural(
                            final_model,
                            X_train_scaled,
                            y_outer_train,
                            None,
                            None,
                            learning_rate,
                            weight_decay,
                            max_epochs,
                            patience,
                            seed,
                            fixed_epochs=best_epoch,
                        )
                        elapsed = perf_counter() - started
                        probabilities = _predict_tabular_neural(final_model, X_val_scaled)
                        seed_probabilities.append(probabilities)
                        pred_labels = encoder.inverse_transform(probabilities.argmax(axis=1))
                        metrics, _, _, original = _evaluate_target(
                            target,
                            y_outer_val_labels,
                            pred_labels,
                            outer_val[TARGET_EMOTION_QUADRANT].astype(str).to_numpy(),
                            spec,
                        )
                        seed_metrics.append(metrics["macro_f1"])
                        row = {
                            "representation": "wav2vec_layer_statistics",
                            "protocol": protocol,
                            "target": target,
                            "model": "layer_mixture",
                            "refinement": f"learned_layers_{pooling}",
                            "fold": int(fold),
                            "seed": int(seed),
                            "result_type": "seed",
                            "n_input_features": int(stats.n_layers * stats.hidden_size * (2 if pooling == 'mean_std' else 1)),
                            "n_features": int(stats.hidden_size * (2 if pooling == 'mean_std' else 1)),
                            "best_epoch": int(best_epoch),
                            "train_seconds": float(elapsed),
                            **metrics,
                        }
                        if original is not None:
                            row.update({f"original_{k}": v for k, v in original.items()})
                        all_rows.append(row)

                        for layer_idx, weight in enumerate(
                            final_model.layer_weights().cpu().numpy()
                        ):
                            weights_rows.append(
                                {
                                    "protocol": protocol,
                                    "target": target,
                                    "pooling": pooling,
                                    "fold": int(fold),
                                    "seed": int(seed),
                                    "layer": int(layer_idx),
                                    "weight": float(weight),
                                }
                            )

                    ensemble_prob = np.mean(seed_probabilities, axis=0)
                    ensemble_pred = encoder.inverse_transform(ensemble_prob.argmax(axis=1))
                    metrics, y_true_eval, y_pred_eval, original = _evaluate_target(
                        target,
                        y_outer_val_labels,
                        ensemble_pred,
                        outer_val[TARGET_EMOTION_QUADRANT].astype(str).to_numpy(),
                        spec,
                    )
                    ensemble_row = {
                        "representation": "wav2vec_layer_statistics",
                        "protocol": protocol,
                        "target": target,
                        "model": "layer_mixture",
                        "refinement": f"learned_layers_{pooling}",
                        "fold": int(fold),
                        "seed": -1,
                        "result_type": "ensemble",
                        "n_input_features": int(stats.n_layers * stats.hidden_size * (2 if pooling == 'mean_std' else 1)),
                        "n_features": int(stats.hidden_size * (2 if pooling == 'mean_std' else 1)),
                        "seed_macro_f1_std": float(np.std(seed_metrics, ddof=0)),
                        **metrics,
                    }
                    if original is not None:
                        ensemble_row.update({f"original_{k}": v for k, v in original.items()})
                    all_rows.append(ensemble_row)
                    predictions.append(
                        pd.DataFrame(
                            {
                                "file_id": outer_val["file_id"].astype(str),
                                "protocol": protocol,
                                "target": target,
                                "model": "layer_mixture",
                                "refinement": f"learned_layers_{pooling}",
                                "fold": int(fold),
                                "y_true": y_true_eval,
                                "y_pred": y_pred_eval,
                            }
                        )
                    )

    return {
        "fold_results": pd.DataFrame(all_rows),
        "predictions": pd.concat(predictions, ignore_index=True, sort=False),
        "layer_weights": pd.DataFrame(weights_rows),
    }


def _make_loader(dataset, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_wav2vec_sequences,
        num_workers=0,
    )


def _train_attention_model(
    model,
    train_loader,
    val_loader,
    n_classes: int,
    class_weights,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    seed: int,
    fixed_epochs: int | None = None,
):
    import torch
    from torch import nn

    set_global_seed(seed)
    device = _device()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = deepcopy(model.state_dict())
    best_score = -np.inf
    best_epoch = 1
    stale = 0
    epochs = fixed_epochs or max_epochs

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["sequences"].to(device),
                batch["mask"].to(device),
            )
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()

        if fixed_epochs is not None:
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            continue

        probabilities, labels, _, _ = _predict_attention(model, val_loader)
        pred = probabilities.argmax(axis=1)
        score = _torch_macro_f1(labels, pred, n_classes)
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, float(best_score if np.isfinite(best_score) else np.nan)


def _predict_attention(model, loader):
    import torch

    device = _device()
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    file_ids: list[str] = []
    attentions: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for batch in loader:
            logits, weights = model(
                batch["sequences"].to(device),
                batch["mask"].to(device),
                return_attention=True,
            )
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            if "labels" in batch:
                labels.append(batch["labels"].numpy())
            file_ids.extend(batch["file_ids"])
            for index, file_id in enumerate(batch["file_ids"]):
                length = int(batch["lengths"][index])
                attentions[file_id] = weights[index, :length].cpu().numpy().astype(np.float32)
    y = np.concatenate(labels) if labels else np.empty(0, dtype=int)
    return np.concatenate(probabilities), y, file_ids, attentions


def run_attentive_pooling_cv(
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    sequences_dir: str | Path,
    seeds: Sequence[int],
    protocols: Sequence[str] = (PROTOCOL_INDEPENDENT,),
    targets: Sequence[str] = (TARGET_EMOTION_ORIGINAL,),
    input_dim: int = 768,
    attention_hidden_dim: int = 128,
    dropout: float = 0.10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 16,
    inner_folds: int = 3,
    n_folds: int | None = None,
) -> dict[str, Any]:
    """Evalúa attentive statistics pooling con ensemble de seeds por fold."""
    development = metadata.merge(splits, on="file_id", validate="one_to_one")
    development = development.loc[
        development["partition"].eq(PARTITION_DEVELOPMENT)
    ].copy()
    sequence_index = validate_sequence_store(
        development["file_id"], sequences_dir, expected_hidden_size=input_dim
    )
    table = development.merge(sequence_index, on="file_id", validate="one_to_one")

    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    attention_records: dict[str, np.ndarray] = {}

    for protocol in protocols:
        fold_col = _fold_column(protocol)
        folds = sorted(table[fold_col].unique())
        if n_folds is not None and len(folds) != n_folds:
            raise ValueError(f"Folds inesperados para {protocol}: {folds}")

        for target in targets:
            spec = _target_spec(table, target)
            encoder = LabelEncoder().fit(spec.train_labels)
            for fold in folds:
                outer_train = table.loc[table[fold_col] != fold].reset_index(drop=True)
                outer_val = table.loc[table[fold_col] == fold].reset_index(drop=True)
                y_outer_train = encoder.transform(outer_train[spec.train_target])
                inner_train_idx, inner_val_idx = _inner_split(
                    outer_train,
                    y_outer_train,
                    protocol,
                    inner_folds,
                    seed=42,
                )
                seed_probabilities: list[np.ndarray] = []
                seed_macro_f1: list[float] = []
                seed_attention: list[dict[str, np.ndarray]] = []

                for seed in seeds:
                    inner_train = outer_train.iloc[inner_train_idx]
                    inner_val = outer_train.iloc[inner_val_idx]
                    y_inner_train = y_outer_train[inner_train_idx]
                    y_inner_val = y_outer_train[inner_val_idx]
                    train_dataset = Wav2VecSequenceDataset(
                        inner_train["file_id"], inner_train["sequence_path"], y_inner_train
                    )
                    val_dataset = Wav2VecSequenceDataset(
                        inner_val["file_id"], inner_val["sequence_path"], y_inner_val
                    )
                    train_loader = _make_loader(train_dataset, batch_size, True)
                    val_loader = _make_loader(val_dataset, batch_size, False)
                    model = build_attentive_statistics_classifier(
                        input_dim,
                        attention_hidden_dim,
                        len(encoder.classes_),
                        dropout,
                    )
                    model, best_epoch, _ = _train_attention_model(
                        model,
                        train_loader,
                        val_loader,
                        len(encoder.classes_),
                        _class_weights(y_inner_train, len(encoder.classes_)),
                        learning_rate,
                        weight_decay,
                        max_epochs,
                        patience,
                        seed,
                    )

                    outer_train_dataset = Wav2VecSequenceDataset(
                        outer_train["file_id"],
                        outer_train["sequence_path"],
                        y_outer_train,
                    )
                    outer_val_labels_encoded = encoder.transform(
                        outer_val[spec.train_target]
                    )
                    outer_val_dataset = Wav2VecSequenceDataset(
                        outer_val["file_id"],
                        outer_val["sequence_path"],
                        outer_val_labels_encoded,
                    )
                    outer_train_loader = _make_loader(
                        outer_train_dataset, batch_size, True
                    )
                    outer_val_loader = _make_loader(
                        outer_val_dataset, batch_size, False
                    )
                    final_model = build_attentive_statistics_classifier(
                        input_dim,
                        attention_hidden_dim,
                        len(encoder.classes_),
                        dropout,
                    )
                    started = perf_counter()
                    final_model, _, _ = _train_attention_model(
                        final_model,
                        outer_train_loader,
                        None,
                        len(encoder.classes_),
                        _class_weights(y_outer_train, len(encoder.classes_)),
                        learning_rate,
                        weight_decay,
                        max_epochs,
                        patience,
                        seed,
                        fixed_epochs=best_epoch,
                    )
                    elapsed = perf_counter() - started
                    probs, _, ordered_ids, attentions = _predict_attention(
                        final_model, outer_val_loader
                    )
                    order = pd.Index(ordered_ids).get_indexer(outer_val["file_id"].astype(str))
                    if (order < 0).any():
                        raise RuntimeError("No se pudieron alinear predicciones de atención.")
                    probs = probs[order]
                    attentions = {
                        file_id: attentions[file_id]
                        for file_id in outer_val["file_id"].astype(str)
                    }
                    seed_probabilities.append(probs)
                    seed_attention.append(attentions)
                    pred_labels = encoder.inverse_transform(probs.argmax(axis=1))
                    y_true_labels = outer_val[spec.train_target].astype(str).to_numpy()
                    metrics, _, _, original = _evaluate_target(
                        target,
                        y_true_labels,
                        pred_labels,
                        outer_val[TARGET_EMOTION_QUADRANT].astype(str).to_numpy(),
                        spec,
                    )
                    seed_macro_f1.append(metrics["macro_f1"])
                    row = {
                        "representation": "wav2vec_sequence",
                        "protocol": protocol,
                        "target": target,
                        "model": "attentive_statistics",
                        "refinement": "last_layer_attention_mean_std",
                        "fold": int(fold),
                        "seed": int(seed),
                        "result_type": "seed",
                        "n_input_features": input_dim,
                        "n_features": input_dim * 2,
                        "best_epoch": int(best_epoch),
                        "train_seconds": float(elapsed),
                        **metrics,
                    }
                    if original is not None:
                        row.update({f"original_{k}": v for k, v in original.items()})
                    rows.append(row)

                ensemble_probs = np.mean(seed_probabilities, axis=0)
                ensemble_pred = encoder.inverse_transform(ensemble_probs.argmax(axis=1))
                y_true_labels = outer_val[spec.train_target].astype(str).to_numpy()
                metrics, y_true_eval, y_pred_eval, original = _evaluate_target(
                    target,
                    y_true_labels,
                    ensemble_pred,
                    outer_val[TARGET_EMOTION_QUADRANT].astype(str).to_numpy(),
                    spec,
                )
                ensemble_row = {
                    "representation": "wav2vec_sequence",
                    "protocol": protocol,
                    "target": target,
                    "model": "attentive_statistics",
                    "refinement": "last_layer_attention_mean_std",
                    "fold": int(fold),
                    "seed": -1,
                    "result_type": "ensemble",
                    "n_input_features": input_dim,
                    "n_features": input_dim * 2,
                    "seed_macro_f1_std": float(np.std(seed_macro_f1, ddof=0)),
                    **metrics,
                }
                if original is not None:
                    ensemble_row.update({f"original_{k}": v for k, v in original.items()})
                rows.append(ensemble_row)
                predictions.append(
                    pd.DataFrame(
                        {
                            "file_id": outer_val["file_id"].astype(str),
                            "protocol": protocol,
                            "target": target,
                            "model": "attentive_statistics",
                            "refinement": "last_layer_attention_mean_std",
                            "fold": int(fold),
                            "y_true": y_true_eval,
                            "y_pred": y_pred_eval,
                        }
                    )
                )
                for file_id in outer_val["file_id"].astype(str):
                    stacked = np.stack([item[file_id] for item in seed_attention])
                    attention_records[file_id] = stacked.mean(axis=0).astype(np.float32)

    return {
        "fold_results": pd.DataFrame(rows),
        "predictions": pd.concat(predictions, ignore_index=True, sort=False),
        "attention_weights": attention_records,
    }


def run_egemaps_elastic_net(
    egemaps: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    C_values: Sequence[float],
    l1_ratios: Sequence[float],
    seed: int,
    protocols: Sequence[str] = (PROTOCOL_INDEPENDENT,),
    targets: Sequence[str] = (
        TARGET_EMOTION_ORIGINAL,
        TARGET_EMOTION_QUADRANT,
    ),
    inner_folds: int = 3,
    max_iter: int = 5000,
    n_jobs: int = 1,
    n_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Ejecuta Elastic Net nested y agrega coeficientes por familia."""
    table, feature_cols = prepare_model_table(egemaps, metadata, splits)
    family_map = build_family_mapping(feature_cols)
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    family_rows: list[dict[str, Any]] = []

    for protocol in protocols:
        fold_col = _fold_column(protocol)
        folds = sorted(table[fold_col].unique())
        if n_folds is not None and len(folds) != n_folds:
            raise ValueError(f"Folds inesperados para {protocol}: {folds}")
        group_col = _group_column(protocol)

        for target in targets:
            spec = _target_spec(table, target)
            for fold in folds:
                train = table.loc[table[fold_col] != fold].reset_index(drop=True)
                val = table.loc[table[fold_col] == fold].reset_index(drop=True)
                X_train = train[feature_cols].to_numpy(np.float32)
                X_val = val[feature_cols].to_numpy(np.float32)
                y_train = train[spec.train_target].astype(str).to_numpy()
                y_val = val[spec.train_target].astype(str).to_numpy()
                inner_cv = StratifiedGroupKFold(
                    n_splits=inner_folds,
                    shuffle=True,
                    random_state=seed,
                )
                search = build_elastic_net_search(
                    inner_cv=inner_cv,
                    C_values=C_values,
                    l1_ratios=l1_ratios,
                    seed=seed,
                    max_iter=max_iter,
                    n_jobs=n_jobs,
                )
                started = perf_counter()
                search.fit(X_train, y_train, groups=train[group_col].to_numpy())
                elapsed = perf_counter() - started
                y_pred = search.predict(X_val)
                metrics, y_true_eval, y_pred_eval, original = _evaluate_target(
                    target,
                    y_val,
                    y_pred,
                    val[TARGET_EMOTION_QUADRANT].astype(str).to_numpy(),
                    spec,
                )
                best_classifier = search.best_estimator_.named_steps["classifier"]
                coefficients = np.mean(np.abs(best_classifier.coef_), axis=0)
                nonzero = np.mean(np.abs(best_classifier.coef_) > 1e-8, axis=0)
                total_importance = float(coefficients.sum()) or 1.0

                row = {
                    "representation": "egemaps",
                    "protocol": protocol,
                    "target": target,
                    "model": "logistic_regression",
                    "refinement": "elastic_net",
                    "fold": int(fold),
                    "seed": int(seed),
                    "result_type": "deterministic",
                    "n_input_features": len(feature_cols),
                    "n_features": int(np.sum(coefficients > 1e-8)),
                    "best_C": float(search.best_params_["classifier__C"]),
                    "best_l1_ratio": float(search.best_params_["classifier__l1_ratio"]),
                    "train_seconds": float(elapsed),
                    **metrics,
                }
                if original is not None:
                    row.update({f"original_{k}": v for k, v in original.items()})
                rows.append(row)
                predictions.append(
                    pd.DataFrame(
                        {
                            "file_id": val["file_id"].astype(str),
                            "protocol": protocol,
                            "target": target,
                            "model": "logistic_regression",
                            "refinement": "elastic_net",
                            "fold": int(fold),
                            "y_true": y_true_eval,
                            "y_pred": y_pred_eval,
                        }
                    )
                )

                feature_frame = pd.DataFrame(
                    {
                        "feature": feature_cols,
                        "family": [family_map[name] for name in feature_cols],
                        "importance": coefficients / total_importance,
                        "nonzero_fraction": nonzero,
                    }
                )
                for family, group in feature_frame.groupby("family", observed=True):
                    family_rows.append(
                        {
                            "protocol": protocol,
                            "target": target,
                            "fold": int(fold),
                            "family": family,
                            "family_label": FAMILY_LABELS[family],
                            "importance_normalized": float(group["importance"].sum()),
                            "importance_mean_per_feature": float(group["importance"].mean()),
                            "nonzero_fraction": float(group["nonzero_fraction"].mean()),
                            "n_features": int(len(group)),
                        }
                    )

    return {
        "fold_results": pd.DataFrame(rows),
        "predictions": pd.concat(predictions, ignore_index=True, sort=False),
        "family_importance": pd.DataFrame(family_rows),
    }


def save_attention_weights(
    attention_weights: Mapping[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Guarda pesos ragged mediante vector concatenado y offsets."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_ids = sorted(attention_weights)
    arrays = [np.asarray(attention_weights[file_id], dtype=np.float32) for file_id in file_ids]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(array) for array in arrays])
    values = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
    np.savez_compressed(
        path,
        file_ids=np.asarray(file_ids),
        offsets=offsets,
        values=values,
    )
    return path
