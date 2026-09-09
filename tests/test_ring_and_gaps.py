"""Gaps: the bug that renders without any error.

A 1m series with holes in it plus a buffer indexed by POSITION means that `i-20` for an ER(20)
reaches across a gap of hours and returns a perfectly valid, completely wrong float. Nothing
raises. That is why the step is asserted from the timestamps on every window construction.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wavelab.core.ring import GapError, Ring
from wavelab.core.timeframes import MIN_SOURCE_COVERAGE, TF_1H, TF_1M, TF_15M, resample_from_1m
from wavelab.core.types import Bar

from .conftest import SYMBOL, make_bars


class TestRing:
    def test_a_continuous_window_is_complete(self):
        r = Ring(SYMBOL, TF_15M, capacity=64)
        for b in make_bars(20):
            r.append(b)
        w = r.window(20)
        assert w.complete and w.n_gaps == 0
        assert w.end_closed_ts_ms == w.ts[-1] + TF_15M.ms - 1

    def test_it_detects_the_gap_from_the_timestamps(self):
        r = Ring(SYMBOL, TF_15M, capacity=64)
        for b in make_bars(20, drop={7, 8, 9}):
            r.append(b)
        w = r.window(20)
        assert not w.complete
        assert w.n_gaps == 1        # one discontinuity, even though three bars are missing
        with pytest.raises(GapError, match="Refusing to compute"):
            w.require_complete("ER(20)")

    def test_the_gap_survives_wrapping_around_the_ring(self):
        """The mask is RECOMPUTED from the timestamps, not inherited from the write."""
        r = Ring(SYMBOL, TF_15M, capacity=8)
        for b in make_bars(20, drop={17}):
            r.append(b)
        w = r.window(8)
        assert len(w) == 8 and not w.complete

    def test_it_rejects_duplicates_and_out_of_order_bars(self):
        r = Ring(SYMBOL, TF_15M, capacity=16)
        bars = make_bars(5)
        for b in bars:
            r.append(b)
        with pytest.raises(ValueError, match="out-of-order or duplicate"):
            r.append(bars[-1])

    def test_it_rejects_unclosed_bars(self):
        r = Ring(SYMBOL, TF_15M, capacity=16)
        b = make_bars(1)[0]
        unclosed = Bar(symbol=b.symbol, tf=b.tf, open_time_ms=b.open_time_ms, open=b.open,
                       high=b.high, low=b.low, close=b.close, volume=b.volume, is_closed=False)
        with pytest.raises(ValueError, match="is not closed"):
            r.append(unclosed)

    def test_the_in_flight_bar_does_not_touch_the_closed_window(self):
        r = Ring(SYMBOL, TF_15M, capacity=16)
        bars = make_bars(5)
        for b in bars:
            r.append(b)
        before = r.window(5)
        r.set_provisional(Bar(symbol=SYMBOL, tf=TF_15M,
                              open_time_ms=bars[-1].open_time_ms + TF_15M.ms,
                              open=1, high=99999, low=0.1, close=50000, volume=1, is_closed=False))
        after = r.window(5)
        assert (before.high == after.high).all(), (
            "the in-flight bar has contaminated the closed window: the ATR would move intra-bar "
            "and with it the ZigZag threshold and the confirmation of pivots"
        )
        assert len(r.provisional_window(5)) == 5


class TestResampling:
    def _df(self, n: int, drop: set[int] | None = None) -> pd.DataFrame:
        bars = make_bars(n, tf=TF_1M, drop=drop)
        return pd.DataFrame(
            {"open": [b.open for b in bars], "high": [b.high for b in bars],
             "low": [b.low for b in bars], "close": [b.close for b in bars],
             "volume": [b.volume for b in bars]},
            index=pd.Index([b.open_time_ms for b in bars], name="open_time_ms"),
        )

    def test_it_counts_the_source_bars(self):
        out = resample_from_1m(self._df(120), TF_1H)
        assert len(out) == 2
        assert (out["n_source_bars"] == 60).all()
        assert not out["is_gap"].any()
        assert (out["close_time_ms"] == out.index + TF_1H.ms - 1).all()

    def test_it_marks_a_badly_covered_bar_as_a_gap(self):
        """A 1h bar built out of 43 minutes is not a 1h bar."""
        missing = set(range(20))         # 40 of 60 minutes -> 0.67 < 0.9
        out = resample_from_1m(self._df(120, drop=missing), TF_1H)
        assert out.iloc[0]["n_source_bars"] == 40
        assert bool(out.iloc[0]["is_gap"]) is True
        assert bool(out.iloc[1]["is_gap"]) is False

    def test_the_coverage_threshold_is_the_declared_one(self):
        exactly = set(range(6))          # 54/60 = 0.90 -> passes
        out = resample_from_1m(self._df(60, drop=exactly), TF_1H)
        assert out.iloc[0]["n_source_bars"] == 54
        assert bool(out.iloc[0]["is_gap"]) is (54 < int(MIN_SOURCE_COVERAGE * 60))

    def test_it_rejects_a_non_increasing_index(self):
        df = self._df(10)
        df.index = df.index[::-1]
        with pytest.raises(ValueError, match="strictly increasing"):
            resample_from_1m(df, TF_1H)
