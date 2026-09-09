"""Registry by dotted name. Five seams, one dict, zero magic.

The anti-astronaut rule, and it is in the README: **if a seam needs a second significant method to
be useful, it has been drawn in the wrong place.** No dependency-injection container, no
entry-point discovery, no plugin framework. A third-party plugin is a module path in
``[plugins].modules``, imported at startup.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

__all__ = ["REGISTRY", "SEAMS", "load_plugin_modules", "names", "register", "resolve"]

#: The five seams. Adding a sixth requires justifying it in writing in the README.
SEAMS = ("feed", "feature", "production", "signal_source", "veto")

REGISTRY: dict[str, Any] = {}


def register(name: str) -> Callable[[Any], Any]:
    """Register under a dotted name ``seam.provider``, e.g. ``feed.binance_ws``."""
    if "." not in name:
        raise ValueError(f"invalid registry name {name!r}: expected '<seam>.<name>'")
    seam = name.split(".", 1)[0]
    if seam not in SEAMS:
        raise ValueError(f"unknown seam {seam!r}; the valid ones are {SEAMS}")

    def deco(obj: Any) -> Any:
        if name in REGISTRY and REGISTRY[name] is not obj:
            raise ValueError(f"{name!r} is already registered by {REGISTRY[name]!r}")
        REGISTRY[name] = obj
        return obj

    return deco


def resolve(name: str) -> Any:
    try:
        return REGISTRY[name]
    except KeyError:
        near = [k for k in REGISTRY if k.split(".", 1)[0] == name.split(".", 1)[0]]
        raise KeyError(
            f"{name!r} is not registered. Available in that seam: {sorted(near) or 'none'}"
        ) from None


def names(seam: str | None = None) -> list[str]:
    if seam is None:
        return sorted(REGISTRY)
    return sorted(k for k in REGISTRY if k.split(".", 1)[0] == seam)


def load_plugin_modules(modules: list[str]) -> None:
    """Import plugin modules so that their ``@register`` decorators run."""
    for m in modules:
        importlib.import_module(m)
