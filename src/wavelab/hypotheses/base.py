"""Contrato de las hipótesis. Todas se REGISTRAN ANTES de mirar ningún resultado.

Por qué el registro previo. La diferencia entre ciencia y autoengaño no está en probar muchas cosas
—eso está bien— sino en decidir QUÉ cuenta como éxito antes de mirar. Si primero pruebas y luego
eliges, siempre encuentras algo, y no tienes forma de saber si es señal o el máximo de N sorteos.

Cada hipótesis declara:
  - `rationale`: POR QUÉ debería funcionar. Escrito antes de ver un solo número.
  - `prior`:     qué efecto se espera y en qué dirección.
  - `params`:    FIJOS. No se buscan. Cambiar uno es una hipótesis NUEVA y otro ensayo.

Y todas se cuentan para la corrección por contraste múltiple, incluidas las que fracasan. Ocultar
las fallidas es lo que convierte un estudio en un folleto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["REGISTRY", "Hypothesis", "Series", "register"]


@dataclass(frozen=True, slots=True)
class Series:
    """Las velas de un timeframe. Todo lo que una hipótesis puede ver."""
    tf: str
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return int(self.close.size)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """Una idea comprobable, declarada antes de comprobarla."""

    name: str
    family: str
    rationale: str
    prior: str
    fn: object                    # (Series) -> np.ndarray de {-1, 0, +1}
    params: dict = field(default_factory=dict)
    timeframes: tuple[str, ...] = ("15m", "1h", "4h", "1d")
    min_warmup: int = 200

    def signals(self, s: Series) -> np.ndarray:
        """+1 = largo, -1 = corto, 0 = fuera. Estrictamente causal: la posición i solo puede
        depender de datos hasta i incluido. Un desplazamiento mal hecho aquí inventa una ventaja
        que no existe, y es el error más común de todo el sector."""
        out = np.asarray(self.fn(s), dtype=np.int8)
        if out.size != len(s):
            raise ValueError(f"{self.name}: devolvió {out.size} señales para {len(s)} velas")
        out[: self.min_warmup] = 0
        return out


REGISTRY: dict[str, Hypothesis] = {}


def register(h: Hypothesis) -> Hypothesis:
    if h.name in REGISTRY:
        raise ValueError(f"hipótesis duplicada: {h.name}")
    if not h.rationale.strip() or not h.prior.strip():
        raise ValueError(f"{h.name}: toda hipótesis debe declarar rationale y prior ANTES de "
                         "ejecutarse. Sin eso no es una hipótesis, es una búsqueda.")
    REGISTRY[h.name] = h
    return h
