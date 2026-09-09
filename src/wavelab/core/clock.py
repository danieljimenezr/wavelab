"""The clock is an injected dependency, never a direct call to ``time``.

If the engine asked the system for the time, replay would not be reproducible and the determinism
harness — the test that backs the claim that backtest and live are the same code — would be
impossible to write.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "LiveClock", "SimClock"]


@runtime_checkable
class Clock(Protocol):
    def now_ms(self) -> int: ...


class LiveClock:
    """Wall clock. The only place in the project where ``time`` is called."""

    __slots__ = ()

    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000


class SimClock:
    """Controlled clock, advanced explicitly by the replay harness.

    It is monotonic on purpose: going backwards in time during a replay means a bar has been fed
    out of order, and that has to blow up, not be quietly patched over.
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
                f"SimClock: attempt to go backwards from {self._t} to {ts_ms}. "
                "During a replay that means a bar arrived out of order."
            )
        self._t = ts_ms

    def advance(self, ms: int) -> None:
        if ms < 0:
            raise ValueError("SimClock.advance does not accept negative values")
        self._t += int(ms)
