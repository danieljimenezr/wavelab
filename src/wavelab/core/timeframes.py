"""Timeframes, their roles, and resampling from the single source series (1m).

Two hard invariants live here:

**Separation of roles.** 1d/4h are a directional VETO that never emits a timestamped signal; 1h
classifies the regime; 15m is the ONLY timeframe allowed to draw entries and exits. Without this,
two timeframes can emit signals that contradict each other and the user has no way of knowing which
one to follow.

**A single source series.** 1m is stored and everything else is resampled locally. If every
timeframe were requested separately from the API, their boundaries and their gaps would not line
up, and the cross-timeframe agreement logic would be measuring alignment noise instead of market
structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas only at the I/O boundary; never inside a loop
    import pandas as pd

__all__ = [
    "BY_NAME",
    "MIN_SOURCE_COVERAGE",
    "TF_1D",
    "TF_1H",
    "TF_1M",
    "TF_4H",
    "TF_5M",
    "TF_15M",
    "TfRole",
    "Timeframe",
    "close_time_for",
    "resample_from_1m",
]

MINUTE_MS = 60_000

#: Minimum coverage of 1m bars for a resampled bar to be allowed to emit a signal.
#: A 1h bar built out of 43 minutes is not a 1h bar; it is a lie shaped like a bar.
MIN_SOURCE_COVERAGE = 0.9


class TfRole(StrEnum):
    SOURCE = "source"    # 1m: only stored, never analysed directly
    TRIGGER = "trigger"  # the only one that draws entries/exits
    REGIME = "regime"    # classifies the market regime
    GATE = "gate"        # directional veto; NEVER emits a timestamped signal


@dataclass(frozen=True, slots=True, order=True)
class Timeframe:
    ms: int
    name: str
    role: TfRole

    @property
    def minutes(self) -> int:
        return self.ms // MINUTE_MS

    @property
    def expected_source_bars(self) -> int:
        """How many 1m bars a complete bar of this timeframe should contain."""
        return self.ms // MINUTE_MS

    @property
    def can_emit_signals(self) -> bool:
        """Only the trigger timeframe draws entries. The rest inform or veto."""
        return self.role is TfRole.TRIGGER

    def floor_ms(self, ts_ms: int) -> int:
        """Start of this timeframe's bar that contains ``ts_ms`` (UTC grid)."""
        return ts_ms - (ts_ms % self.ms)

    def __str__(self) -> str:
        return self.name


TF_1M = Timeframe(1 * MINUTE_MS, "1m", TfRole.SOURCE)
TF_5M = Timeframe(5 * MINUTE_MS, "5m", TfRole.SOURCE)
TF_15M = Timeframe(15 * MINUTE_MS, "15m", TfRole.TRIGGER)
TF_1H = Timeframe(60 * MINUTE_MS, "1h", TfRole.REGIME)
TF_4H = Timeframe(240 * MINUTE_MS, "4h", TfRole.GATE)
TF_1D = Timeframe(1440 * MINUTE_MS, "1d", TfRole.GATE)

BY_NAME: dict[str, Timeframe] = {tf.name: tf for tf in (TF_1M, TF_5M, TF_15M, TF_1H, TF_4H, TF_1D)}


def close_time_for(open_time_ms: int, tf: Timeframe) -> int:
    """The `close_time` contract, in exactly one place.

    Binance returns ``open + duration - 1 ms``; OKX, Bybit and Coinbase return only ``open_time``.
    If each adapter normalised in its own way, a resampling that works today thanks to a 1 ms
    accident would break the day somebody rounded up. Here it is always DERIVED from ``open_time``,
    which is unambiguous, and adapters are forbidden from inventing another convention.
    """
    return open_time_ms + tf.ms - 1


def resample_from_1m(df_1m: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    """Resample 1m to the requested timeframe, counting the source bars that were really there.

    ``df_1m`` must be indexed by ``open_time_ms`` (int64, UTC), free of duplicates and sorted. It
    does NOT have to be complete: gaps are normal, and counting them is precisely what has to be
    propagated.

    The resampling is done on **open_time**, not on close_time. The original plan said "indexed by
    close_time", but that only works because Binance's ``-1 ms`` falls inside the right interval;
    with a provider that rounded differently, every bar would be silently assigned to the following
    period. On open_time no ambiguity is possible.

    Returns the OHLCV columns plus:
      ``n_source_bars``  1m bars that made up the bar
      ``is_gap``         True if coverage is too poor to trade on it
    """
    import pandas as pd

    if df_1m.empty:
        return df_1m.iloc[0:0].assign(n_source_bars=pd.Series(dtype="int32"),
                                      is_gap=pd.Series(dtype="bool"))

    idx = df_1m.index.to_numpy()
    if (idx[1:] <= idx[:-1]).any():
        raise ValueError("resample_from_1m: the index must be strictly increasing, no duplicates")

    bucket = idx - (idx % tf.ms)
    g = df_1m.groupby(bucket, sort=True)

    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }
    for optional in ("quote_volume", "trades", "taker_buy_base", "taker_buy_quote"):
        if optional in df_1m.columns:
            agg[optional] = "sum"

    out = g.agg(agg)
    out["n_source_bars"] = g.size().astype("int32")
    out.index.name = "open_time_ms"

    expected = tf.expected_source_bars
    out["is_gap"] = out["n_source_bars"] < int(MIN_SOURCE_COVERAGE * expected)
    out["close_time_ms"] = out.index.to_numpy() + tf.ms - 1
    return out
