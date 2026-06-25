"""
src/utils/reproducibility.py
-----------------------------
Fija semillas globales para Python, NumPy y PyTorch (si está disponible).
Debe llamarse una sola vez al inicio de cada notebook o script.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Fija el seed en Python, NumPy y PyTorch (opcional).

    Parameters
    ----------
    seed : int
        Semilla global. Debe coincidir con config.yaml → seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch opcional en Sprint 1


def get_seed_from_config() -> int:
    """Devuelve el seed definido en config.yaml."""
    from src.utils.config import get_config
    return get_config().seed
