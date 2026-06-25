"""
src/utils/logging.py
--------------------
Logger estándar del proyecto. Todos los módulos deben usar get_logger()
en lugar de print() para mantener trazabilidad uniforme.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Devuelve un logger con formato consistente.

    Parameters
    ----------
    name : str
        Nombre del logger (usar __name__ en cada módulo).
    level : int
        Nivel de logging (logging.INFO por defecto).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Evitar handlers duplicados si se llama varias veces
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
