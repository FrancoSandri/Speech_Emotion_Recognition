"""Familias acústicas interpretables para las 88 features eGeMAPSv02."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


FAMILY_LABELS = {
    "prosody_f0": "F0 y prosodia",
    "energy_loudness": "Loudness y energía",
    "cepstral_mfcc": "MFCC",
    "formants": "Formantes",
    "spectral": "Espectro",
    "voice_quality": "Calidad vocal",
    "temporal_voicing": "Voicing y temporalidad",
}


def feature_family(feature_name: str) -> str:
    """Asigna una feature eGeMAPS a una única familia acústica."""
    if feature_name.startswith("F0semitoneFrom27.5Hz"):
        return "prosody_f0"
    if feature_name.startswith("loudness") or feature_name == "equivalentSoundLevel_dBp":
        return "energy_loudness"
    if feature_name.startswith("mfcc"):
        return "cepstral_mfcc"
    if feature_name.startswith(("F1", "F2", "F3")):
        return "formants"
    if feature_name.startswith(
        (
            "spectralFlux",
            "alphaRatio",
            "hammarbergIndex",
            "slopeV",
            "slopeUV",
        )
    ):
        return "spectral"
    if feature_name.startswith(
        ("jitter", "shimmer", "HNR", "logRelF0")
    ):
        return "voice_quality"
    if feature_name.startswith(
        (
            "VoicedSegments",
            "MeanVoicedSegment",
            "StddevVoicedSegment",
            "MeanUnvoicedSegment",
            "StddevUnvoicedSegment",
        )
    ):
        return "temporal_voicing"

    raise ValueError(f"Feature eGeMAPS sin familia: {feature_name!r}")


def build_family_mapping(feature_names: Iterable[str]) -> dict[str, str]:
    """Construye y valida el mapping feature → familia."""
    names = list(feature_names)
    if len(names) != len(set(names)):
        raise ValueError("feature_names contiene nombres duplicados.")
    mapping = {feature: feature_family(feature) for feature in names}
    validate_family_mapping(names, mapping)
    return mapping


def validate_family_mapping(
    feature_names: Iterable[str],
    mapping: dict[str, str] | None = None,
) -> None:
    """Comprueba cobertura exacta y familias conocidas."""
    names = list(feature_names)
    mapping = mapping or {feature: feature_family(feature) for feature in names}

    missing = set(names) - set(mapping)
    extra = set(mapping) - set(names)
    unknown_families = set(mapping.values()) - set(FAMILY_LABELS)
    if missing or extra or unknown_families:
        raise ValueError(
            "Mapping eGeMAPS inválido. "
            f"Faltantes={sorted(missing)}, extras={sorted(extra)}, "
            f"familias desconocidas={sorted(unknown_families)}."
        )


def family_feature_names(
    feature_names: Iterable[str],
    family: str,
) -> list[str]:
    """Devuelve las features pertenecientes a ``family``."""
    if family not in FAMILY_LABELS:
        raise ValueError(f"Familia desconocida: {family!r}")
    mapping = build_family_mapping(feature_names)
    selected = [feature for feature in feature_names if mapping[feature] == family]
    if not selected:
        raise ValueError(f"La familia {family!r} no contiene features.")
    return selected


def aggregate_family_importance(
    feature_results: pd.DataFrame,
    feature_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Agrega importancias individuales por familia, método y fold.

    Produce suma absoluta y media absoluta por feature. La media por feature
    es la medida principal para comparar familias con diferente tamaño.
    """
    required = {"feature", "importance", "method", "fold"}
    missing = required - set(feature_results.columns)
    if missing:
        raise KeyError(f"Faltan columnas de importancia: {sorted(missing)}")

    names = list(feature_names) if feature_names is not None else sorted(
        feature_results["feature"].dropna().unique()
    )
    mapping = build_family_mapping(names)

    data = feature_results.loc[feature_results["feature"].isin(mapping)].copy()
    data["family"] = data["feature"].map(mapping)
    data["family_label"] = data["family"].map(FAMILY_LABELS)
    data["importance_abs"] = data["importance"].abs()

    grouping = [
        column
        for column in ("protocol", "target", "model", "method", "fold", "family", "family_label")
        if column in data.columns
    ]
    aggregated = (
        data.groupby(grouping, observed=True, sort=False)
        .agg(
            importance_sum=("importance_abs", "sum"),
            importance_mean_per_feature=("importance_abs", "mean"),
            importance_signed_mean=("importance", "mean"),
            n_features=("feature", "nunique"),
        )
        .reset_index()
    )
    return aggregated


def normalize_family_importance(
    family_importance: pd.DataFrame,
    value_col: str = "importance_mean_per_feature",
) -> pd.DataFrame:
    """Normaliza las importancias dentro de cada método y fold a suma uno."""
    if family_importance.empty:
        return family_importance.copy()
    if value_col not in family_importance.columns:
        raise KeyError(f"Columna inexistente: {value_col!r}")

    grouping = [
        column
        for column in ("protocol", "target", "model", "method", "fold")
        if column in family_importance.columns
    ]
    data = family_importance.copy()
    denominator = data.groupby(grouping, observed=True)[value_col].transform("sum")
    data["importance_normalized"] = np.where(
        denominator > 0,
        data[value_col] / denominator,
        0.0,
    )
    return data
