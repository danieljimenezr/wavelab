"""Registro por nombre punteado. Cinco costuras, un dict, cero magia.

Regla anti-astronauta, y está en el README: **si una costura necesita un segundo método significativo
para ser útil, está dibujada en el sitio equivocado.** Sin contenedor de inyección de dependencias,
sin descubrimiento por entry-points, sin framework de plugins. Un plugin de terceros es una ruta de
módulo en ``[plugins].modules``, importada al arrancar.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

__all__ = ["REGISTRY", "register", "resolve", "names", "load_plugin_modules", "SEAMS"]

#: Las cinco costuras. Añadir una sexta exige justificarlo por escrito en el README.
SEAMS = ("feed", "feature", "production", "signal_source", "veto")

REGISTRY: dict[str, Any] = {}


def register(name: str) -> Callable[[Any], Any]:
    """Registra bajo un nombre punteado ``costura.proveedor``, p. ej. ``feed.binance_ws``."""
    if "." not in name:
        raise ValueError(f"nombre de registro inválido {name!r}: se espera '<costura>.<nombre>'")
    seam = name.split(".", 1)[0]
    if seam not in SEAMS:
        raise ValueError(f"costura desconocida {seam!r}; las válidas son {SEAMS}")

    def deco(obj: Any) -> Any:
        if name in REGISTRY and REGISTRY[name] is not obj:
            raise ValueError(f"{name!r} ya está registrado por {REGISTRY[name]!r}")
        REGISTRY[name] = obj
        return obj

    return deco


def resolve(name: str) -> Any:
    try:
        return REGISTRY[name]
    except KeyError:
        near = [k for k in REGISTRY if k.split(".", 1)[0] == name.split(".", 1)[0]]
        raise KeyError(
            f"{name!r} no está registrado. Disponibles en esa costura: {sorted(near) or 'ninguno'}"
        ) from None


def names(seam: str | None = None) -> list[str]:
    if seam is None:
        return sorted(REGISTRY)
    return sorted(k for k in REGISTRY if k.split(".", 1)[0] == seam)


def load_plugin_modules(modules: list[str]) -> None:
    """Importa módulos de plugin para que sus decoradores ``@register`` se ejecuten."""
    for m in modules:
        importlib.import_module(m)
