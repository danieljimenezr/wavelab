"""Carga todas las familias de hipótesis registradas."""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from wavelab.hypotheses.base import REGISTRY, Hypothesis, Series, register  # noqa: F401

def load_all() -> dict[str, Hypothesis]:
    pkg = Path(__file__).parent
    for m in pkgutil.iter_modules([str(pkg)]):
        if m.name not in ("base",):
            importlib.import_module(f"wavelab.hypotheses.{m.name}")
    return REGISTRY
