"""Causal pivot detection: ATR-ZigZag with a LATCHED threshold.

THE MECHANIC ALMOST EVERYBODY SKIPS. A pivot is LOCATED at the bar τ where the extreme happened, but
it is only CONFIRMED at ``c(τ) = min{t > τ : |E_τ − price_t| >= thr}``. Until then a new high simply
moves it along: the pivot does not exist yet. These are two different timestamps, and ``idx`` is
where you DRAW it while ``confirmed_idx`` is the first bar on which you were ALLOWED to know it.

The lag between the two is a first-passage time to a barrier: a median of a few bars, a very heavy
right tail, hundreds of bars in a strong trend. **You can never assume a fixed lag**, which is why
"just shift it N bars" does not cut it.

THE THRESHOLD IS FROZEN AT THE EXTREME BAR — latched there, never recomputed. That is what makes
confirmation monotone, and four architectural claims hang off that monotonicity: the count history
is genuinely append-only, the "as it stood at bar t" snapshot comes for free, the replay harness is
O(n) instead of O(n²), and — the one that matters — a count the user has already seen cannot
disappear without an invalidation event.

If the threshold were recomputed with today's ATR, a pivot confirmed yesterday under low volatility
could stop satisfying the inequality tomorrow under expanded volatility. Nothing would fire. The
chart would simply change its mind about the past, which is exactly the dishonesty this design
exists to eliminate.

Measured cost: 1.19 ms over 5,000 bars in pure Python (M5, see docs/BENCHMARKS.md). numba is not
needed, and that is measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelab.core.types import Pivot, PivotKind
from wavelab.waves.store import PivotStore

__all__ = ["WilderATR", "ZigZag", "ZigZagConfig"]


class WilderATR:
    """Incremental Wilder ATR. Strictly causal: it only ever sees closed bars already delivered."""

    __slots__ = ("_atr", "_n", "_prev_close", "_sum", "period")

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._atr: float | None = None
        self._prev_close: float | None = None
        self._n = 0
        self._sum = 0.0

    @property
    def value(self) -> float | None:
        return self._atr

    @property
    def ready(self) -> bool:
        return self._atr is not None

    def update(self, high: float, low: float, close: float) -> float | None:
        # The FIRST bar has no previous close, so its "true range" is not true: it is just
        # high-low. TA-Lib drops it and starts accumulating on the second bar, so the first ATR(14)
        # lands at index 14 and not at 13. Including it skewed our ATR by ~2% permanently (the seed
        # carries forward), and with it the ZigZag threshold and therefore which pivots get
        # confirmed. Caught by the test that uses TA-Lib as an independent oracle.
        if self._prev_close is None:
            self._prev_close = close
            return None
        tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close
        if self._atr is None:
            self._n += 1
            self._sum += tr
            if self._n >= self.period:
                self._atr = self._sum / self.period   # first value is a plain mean, as Wilder does
        else:
            self._atr = (self._atr * (self.period - 1) + tr) / self.period
        return self._atr


@dataclass(frozen=True, slots=True)
class ZigZagConfig:
    k_atr: float = 1.5
    min_pct: float = 0.005
    atr_period: int = 14
    #: Confirm on the CLOSE rather than on the wick. Liquidation hunts are endemic in BTC and using
    #: wicks invalidates a huge fraction of otherwise valid structures. Exposed because it is a
    #: decision, not a universal constant — and identical on the live path and in replay.
    on_close: bool = False


class ZigZag:
    """Incremental detector. Feeds a ``PivotStore`` and exposes the provisional pivot.

    It is fed bar by bar and is NEVER called on the whole array. Calling it once over the entire
    history and then slicing the result is the bug that inflates every entry by roughly the whole
    threshold (1.2-2.5 ATR), which is larger than any real edge, and produces a magnificent backtest
    that trades appallingly.
    """

    __slots__ = (
        "_atr",
        "_confirmed_n",
        "_ext_i",
        "_ext_px",
        "_ext_thr",
        "_ext_ts",
        "_i",
        "_up",
        "cfg",
        "store",
    )

    def __init__(self, cfg: ZigZagConfig | None = None, store: PivotStore | None = None) -> None:
        self.cfg = cfg or ZigZagConfig()
        self.store = store or PivotStore()
        self._atr = WilderATR(self.cfg.atr_period)
        self._i = -1
        self._up: bool | None = None      # None = no direction yet
        self._ext_i = 0
        self._ext_ts = 0
        self._ext_px = 0.0
        self._ext_thr = 0.0
        self._confirmed_n = 0

    # ------------------------------------------------------------------ internals

    def _thr(self, close: float) -> float:
        """Absolute, volatility-scaled threshold.

        Being absolute is what makes `k` dimensionless: the same k=1.5 means the same thing on BTC,
        on EURUSD and on AAPL. That is the multi-asset requirement made real instead of
        aspirational.
        """
        atr = self._atr.value or 0.0
        return max(self.cfg.k_atr * atr, self.cfg.min_pct * close)

    def _set_extreme(self, i: int, ts: int, px: float, close: float) -> None:
        self._ext_i, self._ext_ts, self._ext_px = i, ts, px
        # LATCHED here. Never touched again until there is a new extreme.
        self._ext_thr = self._thr(close)

    # ------------------------------------------------------------------ public

    def update(self, ts_ms: int, high: float, low: float, close: float) -> Pivot | None:
        """Process a CLOSED bar. Returns the pivot just confirmed, if there is one."""
        self._i += 1
        self._atr.update(high, low, close)
        if not self._atr.ready:
            return None

        i, ts = self._i, ts_ms
        up_px = close if self.cfg.on_close else high
        down_px = close if self.cfg.on_close else low

        if self._up is None:
            self._up = True
            self._set_extreme(i, ts, up_px, close)
            self._publish_provisional()
            return None

        confirmed: Pivot | None = None

        if self._up:
            if up_px > self._ext_px:
                self._set_extreme(i, ts, up_px, close)
            elif self._ext_px - down_px >= self._ext_thr:
                confirmed = Pivot(self._ext_i, self._ext_ts, self._ext_px, PivotKind.HIGH,
                                  self._ext_thr).confirmed_at(i, ts)
                self.store.append_confirmed(confirmed)
                self._confirmed_n += 1
                self._up = False
                self._set_extreme(i, ts, down_px, close)
        else:
            if down_px < self._ext_px:
                self._set_extreme(i, ts, down_px, close)
            elif up_px - self._ext_px >= self._ext_thr:
                confirmed = Pivot(self._ext_i, self._ext_ts, self._ext_px, PivotKind.LOW,
                                  self._ext_thr).confirmed_at(i, ts)
                self.store.append_confirmed(confirmed)
                self._confirmed_n += 1
                self._up = True
                self._set_extreme(i, ts, up_px, close)

        self._publish_provisional()
        return confirmed

    def _publish_provisional(self) -> None:
        if self._up is None:
            return
        self.store.set_provisional(Pivot(
            self._ext_i, self._ext_ts, self._ext_px,
            PivotKind.HIGH if self._up else PivotKind.LOW, self._ext_thr,
        ))

    # ------------------------------------------------------------------ reads

    @property
    def n_confirmed(self) -> int:
        return self._confirmed_n

    @property
    def atr(self) -> float | None:
        return self._atr.value

    def confirm_price(self) -> float | None:
        """The price at which the provisional pivot would become confirmed.

        ★ This is the grey line on the chart: "the count confirms below 108,240". Almost nobody
        implements it, and it turns Elliott's repainting weakness into the most actionable line on
        the screen: the user stops seeing "this might be the top" and starts seeing the exact price
        at which it stops being a maybe.
        """
        if self._up is None:
            return None
        return (self._ext_px - self._ext_thr) if self._up else (self._ext_px + self._ext_thr)

    def legs_as_of(self, now_ms: int, include_provisional: bool = True,
                   since_ms: int | None = None) -> list[dict]:
        """Legs to draw. Confirmed ones go SOLID; the provisional one, DASHED.

        The visual separation is not cosmetic: the provisional leg legitimately changes as price
        moves, and presenting it exactly like a confirmed one asserts a certainty nobody has.
        """
        pivs = self.store.as_of(now_ms)
        if since_ms is not None:
            # One extra leg backwards: without it the first visible leg would dangle, starting
            # from nowhere at the left edge of the chart.
            i = next((j for j, p in enumerate(pivs) if p.ts_ms >= since_ms), len(pivs))
            pivs = pivs[max(0, i - 1):]
        out = [{"ts": p.ts_ms, "price": p.price, "kind": int(p.kind),
                "confirmed_ts": p.confirmed_ts_ms, "tentative": False} for p in pivs]
        if include_provisional:
            prov = self.store.provisional_as_of(now_ms)
            if prov is not None and (not out or prov.ts_ms > out[-1]["ts"]):
                out.append({"ts": prov.ts_ms, "price": prov.price, "kind": int(prov.kind),
                            "confirmed_ts": None, "tentative": True})
        return out


def detect_batch(ts: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 cfg: ZigZagConfig | None = None) -> ZigZag:
    """Replay an array bar by bar. This is NOT a vectorised path: it is the very same loop.

    It exists for warm-up and for the tests, not to go faster. If there were a separate vectorised
    version, its divergence from the live path would silently reintroduce lookahead.
    """
    z = ZigZag(cfg)
    for i in range(ts.size):
        z.update(int(ts[i]), float(high[i]), float(low[i]), float(close[i]))
    return z
