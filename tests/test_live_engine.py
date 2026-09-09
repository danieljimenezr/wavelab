"""The live engine must produce EXACTLY the same thing as the offline path.

If live resampling and backtest resampling differed even in the last decimal, the whole promise of
«one single function» would be false: the backtest would be measuring one series and the user would
be looking at another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelab.core.timeframes import TF_1H, TF_1M, TF_15M, resample_from_1m
from wavelab.engine.live import LiveEngine, Mode

from .conftest import SYMBOL, make_bars


def _engine(tfs=("1m", "15m", "1h")) -> LiveEngine:
    return LiveEngine(SYMBOL, list(tfs), ring_capacity=4096, trigger_tf="15m")


class TestLiveResampling:
    def test_it_matches_the_offline_path_bar_for_bar(self):
        bars = make_bars(600, tf=TF_1M)
        e = _engine()
        for b in bars:
            e.on_bar_1m(b)

        df = pd.DataFrame(
            {"open": [b.open for b in bars], "high": [b.high for b in bars],
             "low": [b.low for b in bars], "close": [b.close for b in bars],
             "volume": [b.volume for b in bars]},
            index=pd.Index([b.open_time_ms for b in bars], name="open_time_ms"),
        )
        for tf in (TF_15M, TF_1H):
            offline = resample_from_1m(df, tf)
            offline = offline[offline["n_source_bars"] == tf.expected_source_bars]
            ring = e.state.rings[tf.name]
            w = ring.window(len(ring))
            common = offline.index.intersection(w.ts)
            assert len(common) >= 4, f"too few {tf.name} bars to compare"
            for ts in common:
                i = int(np.flatnonzero(w.ts == ts)[0])
                o = offline.loc[ts]
                assert w.open[i] == pytest.approx(o["open"]), f"{tf.name} open at {ts}"
                assert w.high[i] == pytest.approx(o["high"]), f"{tf.name} high at {ts}"
                assert w.low[i] == pytest.approx(o["low"]), f"{tf.name} low at {ts}"
                assert w.close[i] == pytest.approx(o["close"]), f"{tf.name} close at {ts}"
                assert w.volume[i] == pytest.approx(o["volume"]), f"{tf.name} volume at {ts}"

    def test_it_counts_the_source_bars(self):
        e = _engine()
        for b in make_bars(120, tf=TF_1M):
            e.on_bar_1m(b)
        w = e.state.rings["1h"].window(2)
        assert list(w.n_source_bars) == [60, 60]

    def test_it_marks_a_badly_covered_bar_as_a_gap(self):
        """A 1h bar built out of 20 minutes is not a 1h bar."""
        e = _engine()
        for b in make_bars(120, tf=TF_1M, drop=set(range(40))):
            e.on_bar_1m(b)
        w = e.state.rings["1h"].window(1)
        assert int(w.n_source_bars[0]) == 60, "the 2nd hour is complete"
        assert not bool(w.is_gap[0])

    def test_the_in_flight_bar_closes_nothing(self):
        e = _engine()
        bars = make_bars(60, tf=TF_1M)
        for b in bars[:-1]:
            e.on_bar_1m(b)
        before = len(e.state.rings["1h"])
        last = bars[-1]
        unclosed = type(last)(
            symbol=last.symbol, tf=last.tf, open_time_ms=last.open_time_ms,
            open=last.open, high=last.high, low=last.low, close=last.close,
            volume=last.volume, is_closed=False)
        assert e.on_bar_1m(unclosed) == []
        assert len(e.state.rings["1h"]) == before
        assert e.state.provisional is unclosed

    def test_duplicates_from_the_healing_pass_are_ignored(self):
        e = _engine()
        bars = make_bars(30, tf=TF_1M)
        for b in bars:
            e.on_bar_1m(b)
        n = len(e.state.rings["1m"])
        for b in bars[-10:]:
            assert e.on_bar_1m(b) == []
        assert len(e.state.rings["1m"]) == n


class TestCatchUpMode:
    def test_it_starts_warming_up_and_emits_nothing(self):
        e = _engine()
        assert e.state.health.mode is Mode.WARMUP
        assert not e.emitting

    def test_a_large_lag_prevents_emitting(self):
        e = _engine()
        for b in make_bars(120, tf=TF_1M, start_ms=1_600_000_000_000 - (1_600_000_000_000 % TF_1M.ms)):
            e.on_bar_1m(b)
        # The bars are from 2020: the lag is measured in years.
        e.update_health(connected=True, reconnects=0, healed=0, silent_seconds=0.0)
        assert e.state.health.mode is Mode.CATCH_UP
        assert not e.emitting, (
            "emitting while catching up describes a price that has already been and gone; it is "
            "the likeliest way to lose the user's trust in the first week"
        )
