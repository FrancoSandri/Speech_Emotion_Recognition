"""Selección y reducción de representaciones."""

from src.feature_selection.importance import (
    run_feature_importance,
    summarize_feature_importance,
)
from src.feature_selection.pca import (
    build_pca_linear_probe,
    get_pca_fold_metadata,
)
from src.feature_selection.rfecv import run_nested_rfecv
from src.feature_selection.supervised_projection import (
    build_lda_linear_probe,
    get_lda_fold_metadata,
)

__all__ = [
    "build_lda_linear_probe",
    "build_pca_linear_probe",
    "get_lda_fold_metadata",
    "get_pca_fold_metadata",
    "run_feature_importance",
    "run_nested_rfecv",
    "summarize_feature_importance",
]
