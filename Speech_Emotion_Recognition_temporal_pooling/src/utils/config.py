from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class _AttrDict(dict):
    """Dict que permite acceso por atributo."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(
                f"Config key not found: '{key}'"
            ) from None

        return _AttrDict(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


def get_project_root() -> Path:
    """
    Devuelve la raíz absoluta del proyecto.

    src/utils/config.py
    └── utils
        └── src
            └── project_root
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """
    Convierte un path relativo al proyecto en un path absoluto.

    Si el path ya es absoluto, lo conserva.

    Parameters
    ----------
    path : str | Path
        Path absoluto o relativo a la raíz del proyecto.

    Returns
    -------
    Path
        Path absoluto normalizado.
    """
    path = Path(path).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (get_project_root() / path).resolve()


def to_project_relative(path: str | Path) -> str:
    """
    Convierte un path en una ruta relativa a la raíz del proyecto.

    El resultado se guarda con separadores POSIX para que sea portable
    entre Windows, Linux y macOS.

    Parameters
    ----------
    path : str | Path
        Path absoluto o relativo al proyecto.

    Returns
    -------
    str
        Path relativo con separadores '/'.

    Raises
    ------
    ValueError
        Si el path se encuentra fuera de la raíz del proyecto.
    """
    project_root = get_project_root().resolve()
    absolute_path = resolve_path(path)

    try:
        relative_path = absolute_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            "El path debe encontrarse dentro de la raíz del proyecto.\n"
            f"Path recibido: {absolute_path}\n"
            f"Raíz del proyecto: {project_root}"
        ) from exc

    return relative_path.as_posix()


@lru_cache(maxsize=1)
def get_config(config_path: str | Path | None = None) -> _AttrDict:
    """
    Carga configs/config.yaml.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "config.yaml"
    else:
        config_path = resolve_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Working directory: {os.getcwd()}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    return _AttrDict(raw)