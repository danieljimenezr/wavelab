"""Circular buffer with timestamps, and the windows that come out of it.

The bug this module exists to prevent: a buffer indexed by POSITION on top of a store indexed by
TIME. Binance's 1m series have holes (halts, reconnections, incidents). With positional indexing,
``i-20`` for an ER(20) steps across a gap of hours **silently** and returns a safe, wrong number.
It does not raise, it does not warn, and the value is a perfectly valid float.

That is why the ring stores the timestamps ALONGSIDE the prices and every window **asserts the
step** instead of trusting position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelab.core.timeframes import Timeframe, close_time_for
from wavelab.core.types import Bar

__all__ = ["GapError", "ProvisionalWindow", "Ring", "Window"]


class GapError(ValueError):
    """The requested window contains a gap that invalidates the requested computation."""


@dataclass(frozen=True, slots=True)
class Window:
    """View over CLOSED bars, with the discontinuities made explicit.

    ``end_closed_ts_ms`` is set exclusively from the last closed bar, never from the ring's write
    cursor. That is the difference between a guard that works and one that steps around itself: if
    the end of the window were computed from the cursor, it would include the in-flight bar and the
    ATR would move intra-bar → the ZigZag threshold would move intra-bar → pivot confirmation would
    move intra-bar. And nothing would raise, because the values are float64 either way.
    """

    symbol: str
    tf: Timeframe
    ts: np.ndarray            # open_time_ms of each bar
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    n_source_bars: np.ndarray
    is_gap: np.ndarray        # True if the PREVIOUS bar is missing or its coverage is too poor
    end_closed_ts_ms: int

    def __len__(self) -> int:
        return int(self.ts.size)

    @property
    def complete(self) -> bool:
        """No discontinuities and no poorly covered bars anywhere in the window."""
        return not bool(self.is_gap.any())

    @property
    def n_gaps(self) -> int:
        return int(self.is_gap.sum())

    def require_complete(self, what: str) -> Window:
        """For computations that cannot tolerate gaps. Fails loudly instead of lying quietly."""
        if not self.complete:
            first = int(np.flatnonzero(self.is_gap)[0])
            raise GapError(
                f"{what}: the window of {len(self)} {self.tf} bars has {self.n_gaps} "
                f"discontinuity(ies); the first at ts={int(self.ts[first])}. "
                "Refusing to compute rather than return a confident, wrong number."
            )
        return self


@dataclass(frozen=True, slots=True)
class ProvisionalWindow:
    """The same shape as ``Window`` but a DIFFERENT type, and deliberately so.

    It includes the in-flight bar. ``@causal`` rejects it ALWAYS, without even looking at dates, so
    the provisional channel is structurally incapable of feeding causal features, signals, journal
    or statistics. All it can produce is tentative annotations: dashed stroke, hollow "?" label.
    """

    __wavelab_provisional__ = True

    symbol: str
    tf: Timeframe
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    n_source_bars: np.ndarray
    is_gap: np.ndarray
    last_ts_ms: int           # deliberately NOT called end_closed_ts_ms

    def __len__(self) -> int:
        return int(self.ts.size)


_FIELDS = ("open", "high", "low", "close", "volume")


class Ring:
    """Circular buffer of closed bars for one symbol and timeframe, plus the in-flight bar."""

    __slots__ = ("_cap", "_cols", "_gap", "_n", "_nsrc", "_prov", "_ts", "_w", "symbol", "tf")

    def __init__(self, symbol: str, tf: Timeframe, capacity: int = 8192) -> None:
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        self.symbol = symbol
        self.tf = tf
        self._cap = int(capacity)
        self._n = 0          # how many bars have been written in total
        self._w = 0          # write cursor
        self._ts = np.zeros(capacity, dtype=np.int64)
        self._cols = {f: np.zeros(capacity, dtype=np.float64) for f in _FIELDS}
        self._nsrc = np.zeros(capacity, dtype=np.int32)
        self._gap = np.zeros(capacity, dtype=bool)
        self._prov: Bar | None = None

    # ---------------------------------------------------------------------- writing

    def append(self, bar: Bar) -> None:
        """Append a CLOSED bar. Rejects in-flight, off-grid or out-of-order bars."""
        if not bar.is_closed:
            raise ValueError(
                f"Ring.append: {bar.symbol} {bar.tf} at {bar.open_time_ms} is not closed. "
                "Use set_provisional() for the in-flight bar."
            )
        if bar.tf is not self.tf or bar.symbol != self.symbol:
            raise ValueError(
                f"Ring.append: expected {self.symbol} {self.tf}, got {bar.symbol} {bar.tf}"
            )
        if self._n:
            last = int(self._ts[(self._w - 1) % self._cap])
            if bar.open_time_ms <= last:
                raise ValueError(
                    f"Ring.append: out-of-order or duplicate bar "
                    f"(open_time_ms={bar.open_time_ms} <= last={last}). "
                    "Deduplication is the store's job, not the ring's."
                )
            # Discontinuity: at least one bar is missing between the previous one and this one.
            discontinuous = (bar.open_time_ms - last) != self.tf.ms
        else:
            discontinuous = False

        i = self._w
        self._ts[i] = bar.open_time_ms
        self._cols["open"][i] = bar.open
        self._cols["high"][i] = bar.high
        self._cols["low"][i] = bar.low
        self._cols["close"][i] = bar.close
        self._cols["volume"][i] = bar.volume
        self._nsrc[i] = bar.n_source_bars
        self._gap[i] = discontinuous or bar.is_gap
        self._w = (i + 1) % self._cap
        self._n += 1
        self._prov = None  # the in-flight bar is absorbed by its closed version

    def set_provisional(self, bar: Bar) -> None:
        if bar.is_closed:
            raise ValueError("set_provisional expects the IN-FLIGHT bar (is_closed=False)")
        self._prov = bar

    # ---------------------------------------------------------------------- reading

    def __len__(self) -> int:
        return min(self._n, self._cap)

    @property
    def last_closed_ts_ms(self) -> int | None:
        if not self._n:
            return None
        return int(self._ts[(self._w - 1) % self._cap])

    def _take(self, n: int) -> np.ndarray:
        """Absolute indices of the last ``n`` bars, resolving the circular wrap-around."""
        avail = len(self)
        n = min(n, avail)
        start = (self._w - n) % self._cap
        return (start + np.arange(n)) % self._cap

    def window(self, n: int) -> Window:
        """Window over the last ``n`` CLOSED bars.

        Recomputes the gap mask from the real timestamps on every construction: it does not trust
        what was marked at write time, because the ring may have wrapped around since.
        """
        if not self._n:
            raise ValueError("Empty ring: there is not a single closed bar")
        idx = self._take(n)
        ts = self._ts[idx].copy()

        d = np.diff(ts)
        if (d <= 0).any():
            bad = int(np.flatnonzero(d <= 0)[0])
            raise ValueError(
                f"Ring.window: non-increasing timestamps at position {bad} "
                f"({int(ts[bad])} -> {int(ts[bad+1])}). The ring is corrupt."
            )
        gap = self._gap[idx].copy()
        gap[1:] |= d != self.tf.ms  # the REAL step overrides whatever was noted at write time

        end_ts = int(ts[-1])
        return Window(
            symbol=self.symbol, tf=self.tf, ts=ts,
            open=self._cols["open"][idx].copy(),
            high=self._cols["high"][idx].copy(),
            low=self._cols["low"][idx].copy(),
            close=self._cols["close"][idx].copy(),
            volume=self._cols["volume"][idx].copy(),
            n_source_bars=self._nsrc[idx].copy(),
            is_gap=gap,
            end_closed_ts_ms=close_time_for(end_ts, self.tf),
        )

    def provisional_window(self, n: int) -> ProvisionalWindow:
        """Window that INCLUDES the in-flight bar. For tentative annotation only."""
        if self._prov is None:
            raise ValueError("no in-flight bar: call set_provisional() first")
        w = self.window(max(0, n - 1))
        p = self._prov
        cat = lambda a, v: np.concatenate([a, np.array([v], dtype=a.dtype)])
        return ProvisionalWindow(
            symbol=self.symbol, tf=self.tf,
            ts=cat(w.ts, p.open_time_ms),
            open=cat(w.open, p.open), high=cat(w.high, p.high),
            low=cat(w.low, p.low), close=cat(w.close, p.close),
            volume=cat(w.volume, p.volume),
            n_source_bars=cat(w.n_source_bars, p.n_source_bars),
            is_gap=cat(w.is_gap, p.is_gap),
            last_ts_ms=p.open_time_ms,
        )
