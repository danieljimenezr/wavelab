"""El reloj es una dependencia inyectada, nunca una llamada directa a ``time``.

Si el motor consultase la hora del sistema, el replay no sería reproducible y el arnés de
determinismo — la prueba que sostiene la afirmación de que backtest y live son el mismo código —
sería imposible de escribir.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "LiveClock", "SimClock"]


@runtime_checkable
class Clock(Protocol):
    def now_ms(self) -> int: ...


class LiveClock:
    """Reloj de pared. El único sitio del proyecto donde se llama a ``time``."""

    __slots__ = ()

    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000


class SimClock:
    """Reloj controlado, avanzado explícitamente por el arnés de replay.

    Es monótono a propósito: retroceder en el tiempo durante un replay significa que se ha alimentado
    una vela fuera de orden, y eso debe explotar, no corregirse en silencio.
    """

    __slots__ = ("_t",)

    def __init__(self, start_ms: int = 0) -> None:
        self._t = int(start_ms)

    def now_ms(self) -> int:
        return self._t

    def set(self, ts_ms: int) -> None:
        ts_ms = int(ts_ms)
        if ts_ms < self._t:
            raise ValueError(
                f"SimClock: intento de retroceder de {self._t} a {ts_ms}. "
                "Durante un replay eso significa una vela fuera de orden."
            )
        self._t = ts_ms

    def advance(self, ms: int) -> None:
        if ms < 0:
            raise ValueError("SimClock.advance no acepta valores negativos")
        self._t += int(ms)
