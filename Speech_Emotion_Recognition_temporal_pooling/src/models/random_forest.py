"""Constructor del baseline Random Forest."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    params: Mapping[str, Any],
    seed: int,
) -> RandomForestClassifier:
    """Construye un Random Forest sin escalado previo."""
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=params.get("max_depth"),
        min_samples_leaf=int(params.get("min_samples_leaf", 2)),
        class_weight=params.get("class_weight", "balanced"),
        random_state=seed,
        n_jobs=-1,
    )
