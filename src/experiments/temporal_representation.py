"""Experimentos de pooling temporal y selección de capas wav2vec.

El módulo reutiliza folds persistidos, métricas globales y loaders del proyecto.
No accede a los test finales. Los modelos aprendidos usan exclusivamente outer
train; early stopping se resuelve con una partición agrupada interna.
"""

from __future__ import annotations

import json
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
    PROTOCOL_INDEPENDENT,
    TARGET_EMOTION_ORIGINAL,
    TARGET_EMOTION_ORIGINAL_EVAL_QUADRANT,
    TARGET_EMOTION_QUADRANT,
)
from src.evaluation.metrics import compute_8_to_4_metrics, compute_metrics
from src.experiments.cross_validation import run_cv
from src.features.feature_store import prepare_model_table
from src.features.wav2vec_temporal import (
    LayerStatistics,
    MultiLayerSequenceDataset,
    build_flat_layer_representation,
    build_static_representation,
    collate_multilayer_sequences,
    validate_multilayer_store,
)
from src.models.attentive_statistics_pooling import (
    build_multilayer_attentive_statistics_classifier,
)
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


def _inner_split(
    table: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = table["actor_id"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    train_idx, val_idx = next(
        splitter.split(np.zeros(len(table)), y, groups=groups)
    )
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


def _torch_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(
        compute_metrics(y_true, y_pred, labels=list(range(n_classes)))["balanced_accuracy"]
    )


def _trainable_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


# ---------------------------------------------------------------------------
# Referencias estáticas y mezcla de estadísticas por capa
# ---------------------------------------------------------------------------

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
    import torch
    from torch import nn

    set_global_seed(seed)
    device = _device()
    model = model.to(device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = (
        None
        if X_val is None
        else torch.tensor(X_val, dtype=torch.float32, device=device)
    )

    loss_fn = nn.CrossEntropyLoss(
        weight=_class_weights(y_train, int(y_train.max()) + 1).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state = deepcopy(model.state_dict())
    best_score = -np.inf
    best_epoch = 1
    stale = 0
    epochs = fixed_epochs or max_epochs

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        if fixed_epochs is not None:
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            continue

        model.eval()
        with torch.inference_mode():
            val_pred = model(X_val_t).argmax(dim=1).cpu().numpy()
        score = _torch_balanced_accuracy(y_val, val_pred, int(y_train.max()) + 1)
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
                    pred["correct"] = pred["y_true"].eq(pred["y_pred"])
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
    poolings: Sequence[str] = ("mean_std",),
    protocols: Sequence[str] = (PROTOCOL_INDEPENDENT,),
    targets: Sequence[str] = (TARGET_EMOTION_ORIGINAL,),
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    max_epochs: int = 250,
    patience: int = 20,
    inner_folds: int = 3,
    n_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Reproduce la mezcla aprendida de estadísticas por capa."""
    all_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    weights_rows: list[dict[str, Any]] = []

    for pooling in poolings:
        refinement = f"learned_layers_{pooling}"
        representation = build_flat_layer_representation(stats, pooling)
        table, feature_cols = prepare_model_table(representation, metadata, splits)
        for protocol in protocols:
            fold_col = "fold_speaker_independent"
            folds = sorted(table[fold_col].unique())
            if n_folds is not None and len(folds) != n_folds:
                raise ValueError(f"Folds inesperados para {protocol}: {folds}")

            for target in targets:
                spec = _target_spec(table, target)
                encoder = LabelEncoder().fit(spec.train_labels)
                for fold in folds:
                    outer_train = table.loc[table[fold_col] != fold].reset_index(drop=True)
                    outer_val = table.loc[table[fold_col] == fold].reset_index(drop=True)
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
                    y_outer_train = encoder.transform(
                        outer_train[spec.train_target].astype(str)
                    )
                    y_outer_val_labels = outer_val[spec.train_target].astype(str).to_numpy()
                    inner_train_idx, inner_val_idx = _inner_split(
                        outer_train,
                        y_outer_train,
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
                        set_global_seed(seed)
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
                        set_global_seed(seed)
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
                        pred_labels = encoder.inverse_transform(
                            probabilities.argmax(axis=1)
                        )
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
                            "refinement": refinement,
                            "fold": int(fold),
                            "seed": int(seed),
                            "result_type": "seed",
                            "n_input_features": int(X_outer_train.shape[1] * X_outer_train.shape[2]),
                            "n_features": int(X_outer_train.shape[2]),
                            "trainable_params": _trainable_parameters(final_model),
                            "best_epoch": int(best_epoch),
                            "train_seconds": float(elapsed),
                            **metrics,
                        }
                        if original is not None:
                            row.update({f"original_{key}": value for key, value in original.items()})
                        all_rows.append(row)

                        for layer_idx, weight in enumerate(
                            final_model.layer_weights().cpu().numpy()
                        ):
                            weights_rows.append(
                                {
                                    "configuration": refinement,
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
                    ensemble_pred = encoder.inverse_transform(
                        ensemble_prob.argmax(axis=1)
                    )
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
                        "refinement": refinement,
                        "fold": int(fold),
                        "seed": -1,
                        "result_type": "ensemble",
                        "n_input_features": int(X_outer_train.shape[1] * X_outer_train.shape[2]),
                        "n_features": int(X_outer_train.shape[2]),
                        "trainable_params": int(stats.n_layers + X_outer_train.shape[2] * len(encoder.classes_) + len(encoder.classes_)),
                        "seed_macro_f1_std": float(np.std(seed_metrics, ddof=0)),
                        **metrics,
                    }
                    if original is not None:
                        ensemble_row.update({f"original_{key}": value for key, value in original.items()})
                    all_rows.append(ensemble_row)
                    predictions.append(
                        pd.DataFrame(
                            {
                                "file_id": outer_val["file_id"].astype(str),
                                "representation": "wav2vec_layer_statistics",
                                "protocol": protocol,
                                "target": target,
                                "model": "layer_mixture",
                                "refinement": refinement,
                                "fold": int(fold),
                                "y_true": y_true_eval,
                                "y_pred": y_pred_eval,
                                "probabilities": [
                                    json.dumps(row.tolist()) for row in ensemble_prob
                                ],
                                "correct": np.asarray(y_true_eval) == np.asarray(y_pred_eval),
                            }
                        )
                    )

    return {
        "fold_results": pd.DataFrame(all_rows),
        "predictions": pd.concat(predictions, ignore_index=True, sort=False),
        "layer_weights": pd.DataFrame(weights_rows),
    }


# ---------------------------------------------------------------------------
# Atención temporal sobre secuencias multicapa
# ---------------------------------------------------------------------------

def _make_multilayer_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collate_multilayer_sequences,
        num_workers=0,
    )


def _train_multilayer_attention_model(
    model,
    train_loader,
    val_loader,
    n_classes: int,
    class_weights,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[Any, int, float]:
    import torch
    from torch import nn

    set_global_seed(seed)
    device = _device()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state = deepcopy(model.state_dict())
    best_score = -np.inf
    best_epoch = 1
    stale = 0
    epochs = fixed_epochs or max_epochs

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            hidden_states = batch["hidden_states"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(hidden_states, mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )
            optimizer.step()

        if fixed_epochs is not None:
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            continue

        model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.inference_mode():
            for batch in val_loader:
                logits = model(
                    batch["hidden_states"].to(device),
                    batch["mask"].to(device),
                )
                y_true.extend(batch["labels"].numpy().tolist())
                y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        score = _torch_balanced_accuracy(
            np.asarray(y_true),
            np.asarray(y_pred),
            n_classes,
        )
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


def _predict_multilayer_attention(model, loader):
    import torch

    device = _device()
    model = model.to(device)
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    ordered_ids: list[str] = []
    attention: dict[str, np.ndarray] = {}

    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["hidden_states"].to(device),
                batch["mask"].to(device),
                return_attention=True,
                return_layer_weights=True,
            )
            probs = torch.softmax(output["logits"], dim=1).cpu().numpy()
            weights = output["attention_weights"].cpu().numpy()
            lengths = batch["lengths"].numpy()
            probabilities.append(probs)
            if "labels" in batch:
                labels.append(batch["labels"].numpy())
            ordered_ids.extend(batch["file_ids"])
            for file_id, row, length in zip(batch["file_ids"], weights, lengths):
                attention[str(file_id)] = row[: int(length)].astype(np.float32)

    return (
        np.concatenate(probabilities),
        np.concatenate(labels) if labels else None,
        ordered_ids,
        attention,
        model.layer_weights().detach().cpu().numpy().astype(np.float32),
    )


def _attention_diagnostics(weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(weights, dtype=np.float64)
    values = values / max(values.sum(), 1e-12)
    n_frames = len(values)
    entropy = float(-np.sum(values * np.log(np.clip(values, 1e-12, 1.0))))
    normalized_entropy = float(entropy / np.log(n_frames)) if n_frames > 1 else 0.0
    sorted_weights = np.sort(values)[::-1]
    frames_50 = int(np.searchsorted(np.cumsum(sorted_weights), 0.5) + 1)
    return {
        "attention_entropy": entropy,
        "attention_entropy_normalized": normalized_entropy,
        "max_attention": float(values.max()),
        "frames_for_50pct_mass": frames_50,
        "frames_fraction_50pct_mass": float(frames_50 / n_frames),
        "n_frames": int(n_frames),
    }


def run_multilayer_attention_cv(
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    multilayer_sequences_dir: str | Path,
    seeds: Sequence[int],
    layer_strategy: str,
    refinement: str,
    protocols: Sequence[str] = (PROTOCOL_INDEPENDENT,),
    targets: Sequence[str] = (TARGET_EMOTION_ORIGINAL,),
    n_layers: int = 13,
    input_dim: int = 768,
    attention_hidden_dim: int = 64,
    dropout: float = 0.50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 16,
    gradient_clip_norm: float = 1.0,
    inner_folds: int = 3,
    n_folds: int | None = None,
) -> dict[str, Any]:
    """Entrena atención temporal sobre promedio o mezcla aprendida de capas."""
    if layer_strategy not in {"uniform", "learned"}:
        raise ValueError("layer_strategy debe ser 'uniform' o 'learned'.")

    development = metadata.merge(splits, on="file_id", validate="one_to_one")
    development = development.loc[
        development["partition"].eq(PARTITION_DEVELOPMENT)
    ].copy()
    sequence_index = validate_multilayer_store(
        development["file_id"],
        multilayer_sequences_dir,
        expected_n_layers=n_layers,
        expected_hidden_size=input_dim,
    )
    table = development.merge(sequence_index, on="file_id", validate="one_to_one")

    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    layer_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    attention_records: dict[str, np.ndarray] = {}

    for protocol in protocols:
        fold_col = "fold_speaker_independent"
        folds = sorted(table[fold_col].unique())
        if n_folds is not None and len(folds) != n_folds:
            raise ValueError(f"Folds inesperados para {protocol}: {folds}")

        for target in targets:
            spec = _target_spec(table, target)
            encoder = LabelEncoder().fit(spec.train_labels)
            for fold in folds:
                outer_train = table.loc[table[fold_col] != fold].reset_index(drop=True)
                outer_val = table.loc[table[fold_col] == fold].reset_index(drop=True)
                y_outer_train = encoder.transform(
                    outer_train[spec.train_target].astype(str)
                )
                inner_train_idx, inner_val_idx = _inner_split(
                    outer_train,
                    y_outer_train,
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

                    train_dataset = MultiLayerSequenceDataset(
                        inner_train["file_id"],
                        inner_train["multilayer_sequence_path"],
                        y_inner_train,
                    )
                    val_dataset = MultiLayerSequenceDataset(
                        inner_val["file_id"],
                        inner_val["multilayer_sequence_path"],
                        y_inner_val,
                    )
                    train_loader = _make_multilayer_loader(
                        train_dataset,
                        batch_size,
                        True,
                        seed,
                    )
                    val_loader = _make_multilayer_loader(
                        val_dataset,
                        batch_size,
                        False,
                        seed,
                    )

                    set_global_seed(seed)
                    model = build_multilayer_attentive_statistics_classifier(
                        n_layers=n_layers,
                        input_dim=input_dim,
                        attention_hidden_dim=attention_hidden_dim,
                        n_classes=len(encoder.classes_),
                        layer_strategy=layer_strategy,
                        dropout=dropout,
                    )
                    model, best_epoch, _ = _train_multilayer_attention_model(
                        model,
                        train_loader,
                        val_loader,
                        len(encoder.classes_),
                        _class_weights(y_inner_train, len(encoder.classes_)),
                        learning_rate,
                        weight_decay,
                        max_epochs,
                        patience,
                        gradient_clip_norm,
                        seed,
                    )

                    outer_train_dataset = MultiLayerSequenceDataset(
                        outer_train["file_id"],
                        outer_train["multilayer_sequence_path"],
                        y_outer_train,
                    )
                    outer_val_encoded = encoder.transform(
                        outer_val[spec.train_target].astype(str)
                    )
                    outer_val_dataset = MultiLayerSequenceDataset(
                        outer_val["file_id"],
                        outer_val["multilayer_sequence_path"],
                        outer_val_encoded,
                    )
                    outer_train_loader = _make_multilayer_loader(
                        outer_train_dataset,
                        batch_size,
                        True,
                        seed,
                    )
                    outer_val_loader = _make_multilayer_loader(
                        outer_val_dataset,
                        batch_size,
                        False,
                        seed,
                    )

                    set_global_seed(seed)
                    final_model = build_multilayer_attentive_statistics_classifier(
                        n_layers=n_layers,
                        input_dim=input_dim,
                        attention_hidden_dim=attention_hidden_dim,
                        n_classes=len(encoder.classes_),
                        layer_strategy=layer_strategy,
                        dropout=dropout,
                    )
                    started = perf_counter()
                    final_model, _, _ = _train_multilayer_attention_model(
                        final_model,
                        outer_train_loader,
                        None,
                        len(encoder.classes_),
                        _class_weights(y_outer_train, len(encoder.classes_)),
                        learning_rate,
                        weight_decay,
                        max_epochs,
                        patience,
                        gradient_clip_norm,
                        seed,
                        fixed_epochs=best_epoch,
                    )
                    elapsed = perf_counter() - started
                    probs, _, ordered_ids, attentions, layer_weights = (
                        _predict_multilayer_attention(final_model, outer_val_loader)
                    )
                    order = pd.Index(ordered_ids).get_indexer(
                        outer_val["file_id"].astype(str)
                    )
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
                        "representation": "wav2vec_multilayer_sequence",
                        "layer_strategy": layer_strategy,
                        "pooling": "attentive_mean_std",
                        "protocol": protocol,
                        "target": target,
                        "model": "attentive_statistics",
                        "refinement": refinement,
                        "fold": int(fold),
                        "seed": int(seed),
                        "result_type": "seed",
                        "n_input_features": int(n_layers * input_dim),
                        "n_features": int(input_dim * 2),
                        "trainable_params": _trainable_parameters(final_model),
                        "best_epoch": int(best_epoch),
                        "train_seconds": float(elapsed),
                        **metrics,
                    }
                    if original is not None:
                        row.update({f"original_{key}": value for key, value in original.items()})
                    rows.append(row)

                    for layer_idx, weight in enumerate(layer_weights):
                        layer_rows.append(
                            {
                                "configuration": refinement,
                                "protocol": protocol,
                                "target": target,
                                "fold": int(fold),
                                "seed": int(seed),
                                "layer": int(layer_idx),
                                "weight": float(weight),
                            }
                        )
                    for file_id, weights in attentions.items():
                        diagnostics_rows.append(
                            {
                                "file_id": file_id,
                                "configuration": refinement,
                                "protocol": protocol,
                                "target": target,
                                "fold": int(fold),
                                "seed": int(seed),
                                **_attention_diagnostics(weights),
                            }
                        )

                ensemble_probs = np.mean(seed_probabilities, axis=0)
                ensemble_pred = encoder.inverse_transform(
                    ensemble_probs.argmax(axis=1)
                )
                y_true_labels = outer_val[spec.train_target].astype(str).to_numpy()
                metrics, y_true_eval, y_pred_eval, original = _evaluate_target(
                    target,
                    y_true_labels,
                    ensemble_pred,
                    outer_val[TARGET_EMOTION_QUADRANT].astype(str).to_numpy(),
                    spec,
                )
                ensemble_row = {
                    "representation": "wav2vec_multilayer_sequence",
                    "layer_strategy": layer_strategy,
                    "pooling": "attentive_mean_std",
                    "protocol": protocol,
                    "target": target,
                    "model": "attentive_statistics",
                    "refinement": refinement,
                    "fold": int(fold),
                    "seed": -1,
                    "result_type": "ensemble",
                    "n_input_features": int(n_layers * input_dim),
                    "n_features": int(input_dim * 2),
                    "trainable_params": int(
                        input_dim * attention_hidden_dim
                        + attention_hidden_dim
                        + attention_hidden_dim
                        + 1
                        + 2 * input_dim * 2
                        + input_dim * 2 * len(encoder.classes_)
                        + len(encoder.classes_)
                        + (n_layers if layer_strategy == "learned" else 0)
                    ),
                    "seed_macro_f1_std": float(np.std(seed_macro_f1, ddof=0)),
                    **metrics,
                }
                if original is not None:
                    ensemble_row.update({f"original_{key}": value for key, value in original.items()})
                rows.append(ensemble_row)

                predictions.append(
                    pd.DataFrame(
                        {
                            "file_id": outer_val["file_id"].astype(str),
                            "representation": "wav2vec_multilayer_sequence",
                            "protocol": protocol,
                            "target": target,
                            "model": "attentive_statistics",
                            "refinement": refinement,
                            "fold": int(fold),
                            "y_true": y_true_eval,
                            "y_pred": y_pred_eval,
                            "probabilities": [
                                json.dumps(row.tolist()) for row in ensemble_probs
                            ],
                            "correct": np.asarray(y_true_eval) == np.asarray(y_pred_eval),
                        }
                    )
                )

                for file_id in outer_val["file_id"].astype(str):
                    stacked = np.stack([item[file_id] for item in seed_attention])
                    averaged = stacked.mean(axis=0).astype(np.float32)
                    attention_records[f"{refinement}::{file_id}"] = averaged
                    diagnostics_rows.append(
                        {
                            "file_id": file_id,
                            "configuration": refinement,
                            "protocol": protocol,
                            "target": target,
                            "fold": int(fold),
                            "seed": -1,
                            **_attention_diagnostics(averaged),
                        }
                    )

    return {
        "fold_results": pd.DataFrame(rows),
        "predictions": pd.concat(predictions, ignore_index=True, sort=False),
        "layer_weights": pd.DataFrame(layer_rows),
        "attention_diagnostics": pd.DataFrame(diagnostics_rows),
        "attention_weights": attention_records,
    }


def run_average_attention_cv(**kwargs) -> dict[str, Any]:
    """Ejecuta ``average_attention_statistics``."""
    return run_multilayer_attention_cv(
        layer_strategy="uniform",
        refinement="average_attention_statistics",
        **kwargs,
    )


def run_learned_layer_attention_cv(**kwargs) -> dict[str, Any]:
    """Ejecuta ``learned_layers_attention_statistics``."""
    return run_multilayer_attention_cv(
        layer_strategy="learned",
        refinement="learned_layers_attention_statistics",
        **kwargs,
    )


def save_attention_weights(
    attention_weights: Mapping[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Guarda pesos ragged mediante vector concatenado y offsets."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(attention_weights)
    arrays = [np.asarray(attention_weights[key], dtype=np.float32) for key in keys]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(array) for array in arrays])
    values = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
    np.savez_compressed(
        path,
        keys=np.asarray(keys),
        offsets=offsets,
        values=values,
    )
    return path


def load_attention_weights(input_path: str | Path) -> dict[str, np.ndarray]:
    """Carga el store ragged de pesos de atención."""
    path = resolve_path(input_path)
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        key_name = "keys" if "keys" in data.files else "file_ids"
        keys = data[key_name].astype(str)
        offsets = data["offsets"]
        values = data["values"]
    return {
        key: values[offsets[index] : offsets[index + 1]].astype(np.float32)
        for index, key in enumerate(keys)
    }
