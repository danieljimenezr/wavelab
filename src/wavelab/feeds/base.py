"""La costura de los feeds: un Protocol con un método que importa.

Regla anti-astronauta: si esta costura necesitase un segundo método significativo para ser útil,
estaría dibujada en el sitio equivocado. ``fetch_aux`` no cuenta como segundo método porque devuelve
vacío por defecto y existe para que la capa de noticias de v2 sea una línea de configuración en vez
de un refactor del transporte.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from wavelab.core.timeframes import Timeframe
from wavelab.core.types import AuxEvent, Bar

__all__ = ["FeedAdapter", "FeedCaps", "Market"]


@dataclass(frozen=True, slots=True)
class FeedCaps:
    """Qué sabe hacer un feed. El motor consulta esto en vez de asumir.

    ``max_klines_per_request`` y ``weight_budget_per_min`` no son decoración: el gobernador de peso
    los usa para frenar antes del 418, que es un baneo de IP de hasta 3 días y, para una aplicación
    local, una caída total.
    """

    timeframes: frozenset[str] = frozenset()
    has_websocket: bool = False
    max_klines_per_request: int = 1000
    weight_budget_per_min: int = 6000
    deep_history_from_ms: int | None = None
    supports_aux: frozenset[str] = field(default_factory=frozenset)


@runtime_checkable
class FeedAdapter(Protocol):
    """Un origen de velas normalizadas."""

    name: str

    def caps(self) -> FeedCaps: ...

    async def fetch_klines(
        self, symbol: str, tf: Timeframe, start_ms: int, end_ms: int
    ) -> Sequence[Bar]: ...

    def stream(self, symbol: str, tf: Timeframe) -> AsyncIterator[Bar]:  # pragma: no cover
        """Velas en vivo. Un feed sin WebSocket puede no implementarlo; `caps()` lo declara."""
        raise NotImplementedError

    async def stream_events(self, symbol: str) -> AsyncIterator[AuxEvent]:  # pragma: no cover
        """Todo lo que no es una vela. Vacío por defecto: es el gancho que hace que la capa de
        noticias de v2 no obligue a redibujar la costura."""
        return
        yield  # type: ignore[unreachable]


class Market:
    SPOT = "spot"
    FUTURES_UM = "um"      # USD-M perpetuos
    FUTURES_CM = "cm"
