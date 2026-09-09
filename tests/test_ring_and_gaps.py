"""Gaps: the bug that renders without any error.

A 1m series with holes in it plus a buffer indexed by POSITION means that `i-20` for an ER(20)
reaches across a gap of hours and returns a perfectly valid, completely wrong float. Nothing
raises. That is why the step is asserted from the timestamps on every window construction.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from wavelab.core.ring import GapError, Ring
from wavelab.core.timeframes import (
    MIN_SOURCE_COVERAGE,
    TF_1D,
    TF_1H,
    TF_1M,
    TF_4H,
    TF_5M,
    TF_15M,
    TfRole,
    Timeframe,
    resample_from_1m,
)
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
        assert w.require_complete("ER(20)") is w, (
            "require_complete must hand the window back so it can be used inline: "
            "`ring.window(20).require_complete('ER(20)').close` is the intended call site, and "
            "returning anything else turns the guard into an AttributeError at the caller"
        )

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
        """A hole is still reported once the buffer has wrapped over the bar that made it.

        NOTE ON WHAT THIS DOES *NOT* PROVE. `Ring.window` recomputes the gap mask from the real
        timestamps (`gap[1:] |= d != tf.ms`) rather than trusting the flag written by `append`.
        This test cannot tell the two apart, and no test can: `append` derives the write-time flag
        from those same timestamps, so the recomputation agrees with it in every state reachable
        through the public API (checked directly over ~1,300 windows across capacities, holes and
        carried flags). The line is defensive redundancy against a future writer that sets the flag
        some other way — deleting it is invisible to this suite by construction, not by omission.
        """
        r = Ring(SYMBOL, TF_15M, capacity=8)
        for b in make_bars(20, drop={17}):
            r.append(b)
        w = r.window(8)
        assert len(w) == 8 and not w.complete

    def test_a_badly_covered_bar_stays_a_gap_all_the_way_through(self):
        """The timestamps cannot see this one: only the flag the bar carries can.

        A 1h bar resampled out of 43 minutes lands on the grid with a flawless step. Nothing about
        its timestamp is wrong, so the step assertion — the ring's main defence — has literally
        nothing to look at. The single thing standing between that bar and an ER(20) computed on a
        lie is the `is_gap` it arrives with, so the ring must store it and the window must report
        it alongside the discontinuities it works out for itself.
        """
        r = Ring(SYMBOL, TF_1H, capacity=16)
        bars = make_bars(6, tf=TF_1H)
        holed = 3
        for i, b in enumerate(bars):
            r.append(replace(b, n_source_bars=43, is_gap=True) if i == holed else b)

        w = r.window(6)
        assert list(w.ts) == [b.open_time_ms for b in bars], (
            "the premise of this test: the timestamps are perfectly contiguous, so a check that "
            "only looks at the step has nothing to find here"
        )
        assert bool(w.is_gap[holed]), (
            "the 43-minute bar arrived flagged and the flag was dropped between append() and "
            "window(): a bar that is 72% of an hour is now indistinguishable from a real one"
        )
        assert w.n_gaps == 1 and not w.complete
        with pytest.raises(GapError, match="Refusing to compute"):
            w.require_complete("ER(20)")

    def test_a_window_that_starts_just_after_a_hole_does_not_claim_to_be_complete(self):
        """The step assertion is blind to the hole that falls before the window's first bar.

        `np.diff` over n bars yields n-1 steps: the discontinuity in front of position 0 is
        outside its reach for ever. That case is not exotic — it is exactly what the first windows
        after a reconnection look like, when the engine is catching up and most needs to be told
        that it is standing on the far side of an outage. The only record of it is the flag the
        ring writes when the bar arrives.
        """
        r = Ring(SYMBOL, TF_15M, capacity=64)
        for b in make_bars(20, drop={16, 17}):
            r.append(b)

        w = r.window(2)          # bars 18 and 19: contiguous with EACH OTHER
        assert list(w.ts) == [b.open_time_ms for b in make_bars(20)[18:]], "sanity: the last two"
        assert bool(w.is_gap[0]), (
            "the window opens on the bar right after a 30-minute hole and reports no gap: the "
            "engine would resume after an outage believing it had continuous history"
        )
        assert not w.complete
        with pytest.raises(GapError, match="Refusing to compute"):
            w.require_complete("ER(20)")

    def test_it_rejects_duplicates_and_out_of_order_bars(self):
        r = Ring(SYMBOL, TF_15M, capacity=16)
        bars = make_bars(5)
        for b in bars:
            r.append(b)
        with pytest.raises(ValueError, match="out-of-order or duplicate"):
            r.append(bars[-1])

    def test_it_rejects_a_bar_from_another_symbol_or_timeframe(self):
        """A ring is one symbol and one timeframe; a mis-routed bar has no signature.

        The live feed multiplexes several streams over one socket. Route an ETH bar into the BTC
        ring and every price in the window is a perfectly valid float from the wrong asset; route
        a 1h bar into the 15m ring and it lands on the 15m grid quite legally, because every 1h
        boundary IS a 15m boundary. Neither mistake can be spotted downstream, so it has to be
        refused at the door.
        """
        r = Ring(SYMBOL, TF_15M, capacity=16)
        for b in make_bars(3):
            r.append(b)
        nxt = make_bars(4)[3]

        foreign_symbol = replace(nxt, symbol="ETHUSDT")
        with pytest.raises(ValueError, match="got ETHUSDT"):
            r.append(foreign_symbol)

        other_tf = Bar(symbol=SYMBOL, tf=TF_1H, open_time_ms=TF_1H.floor_ms(nxt.open_time_ms),
                       open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0,
                       n_source_bars=TF_1H.expected_source_bars)
        assert other_tf.open_time_ms % TF_15M.ms == 0, (
            "the point of this test: the 1h bar sits on the 15m grid, so the grid check cannot "
            "catch it and only the timeframe check can"
        )
        with pytest.raises(ValueError, match="got BTCUSDT 1h"):
            r.append(other_tf)

        assert len(r) == 3, "a rejected bar must not have been written to the ring"

    def test_it_rejects_unclosed_bars(self):
        r = Ring(SYMBOL, TF_15M, capacity=16)
        b = make_bars(1)[0]
        unclosed = Bar(symbol=b.symbol, tf=b.tf, open_time_ms=b.open_time_ms, open=b.open,
                       high=b.high, low=b.low, close=b.close, volume=b.volume, is_closed=False)
        with pytest.raises(ValueError, match="is not closed"):
            r.append(unclosed)

    def test_the_provisional_slot_only_ever_holds_the_in_flight_bar(self):
        """The two channels are kept apart by type, and this is the door.

        Everything downstream trusts that whatever sits in the provisional slot is still forming:
        it is drawn dashed, it is labelled "?", and `last_ts_ms` is what the staleness check
        compares against the clock. Let an already-closed bar in and the last confirmed candle is
        drawn a second time as tentative, while a feed that died minutes ago goes on reporting a
        fresh `last_ts_ms`.
        """
        r = Ring(SYMBOL, TF_15M, capacity=16)
        bars = make_bars(5)
        for b in bars:
            r.append(b)
        with pytest.raises(ValueError, match="IN-FLIGHT"):
            r.set_provisional(bars[-1])
        with pytest.raises(ValueError, match="no in-flight bar"):
            r.provisional_window(3)

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
        """Both sides of the 90% line, against numbers written out by hand.

        Deriving the expectation from `MIN_SOURCE_COVERAGE` — `is_gap is (54 < int(MIN * 60))` —
        moves the goalposts with the constant: drop the threshold to 0.5 and the assertion quietly
        re-aims and still passes. The declared value is part of the contract, so it is spelled out
        here, and the pass/fail pair is placed on the two adjacent minutes either side of it.
        """
        assert MIN_SOURCE_COVERAGE == 0.9, (
            "an hour needs 54 of its 60 minutes to be tradable; changing that is a decision about "
            "what counts as a real bar, not an implementation detail"
        )
        floor = int(MIN_SOURCE_COVERAGE * 60)
        assert floor == 54, "sanity: the 90% line falls at 54 minutes"

        ok = resample_from_1m(self._df(60, drop=set(range(6))), TF_1H)      # 54/60, exactly on it
        assert ok.iloc[0]["n_source_bars"] == 54
        assert bool(ok.iloc[0]["is_gap"]) is False, (
            "54 minutes is exactly the declared threshold, and landing ON it is enough"
        )

        bad = resample_from_1m(self._df(60, drop=set(range(7))), TF_1H)     # 53/60, one short
        assert bad.iloc[0]["n_source_bars"] == 53
        assert bool(bad.iloc[0]["is_gap"]) is True, (
            "53 minutes is one short of the threshold and must be flagged: an hour built from "
            "less than 90% of itself is a lie shaped like a bar"
        )

    def test_it_rejects_a_minute_delivered_twice(self):
        """Out of order is the obvious corruption; REPEATED is the one that actually happens.

        A websocket reconnection replays the last klines it already sent, in perfect ascending
        order. If equal timestamps slipped past the index check, those minutes would be aggregated
        twice: volume inflated, `n_source_bars` climbing past the 60 minutes an hour physically
        has, and an hour that is really half empty sailing through the coverage check. Strictly
        increasing means equal fails as loudly as decreasing.
        """
        df = self._df(120)
        replayed = pd.concat([df, df.iloc[[70]]]).sort_index()   # one minute delivered twice
        assert replayed.index.has_duplicates, "the premise: the same open_time_ms appears twice"
        assert replayed.index.is_monotonic_increasing, (
            "and it arrives in ascending order, so a check that only looks for a step backwards "
            "sees nothing wrong with it"
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            resample_from_1m(replayed, TF_1H)

    def test_it_rejects_a_non_increasing_index(self):
        df = self._df(10)
        df.index = df.index[::-1]
        with pytest.raises(ValueError, match="strictly increasing"):
            resample_from_1m(df, TF_1H)


class TestOnlyTheTriggerDrawsEntries:
    """`can_emit_signals` is the separation-of-roles invariant, and it was asserted nowhere.

    The module docstring calls it a hard invariant: 1d/4h are a directional VETO that never emits a
    timestamped signal, 1h classifies the regime, and 15m is the ONLY timeframe allowed to draw
    entries and exits. Without it two timeframes can emit signals that contradict each other and
    the user has no way to know which to follow.

    It is one `is` comparison. Widened to `is not TfRole.SOURCE` — the obvious slip, since it reads
    as "anything we actually analyse" — the gates and the regime timeframe start emitting entries
    and the whole suite stays green.
    """

    def test_exactly_one_timeframe_can_emit(self):
        emitting = [tf for tf in (TF_1M, TF_5M, TF_15M, TF_1H, TF_4H, TF_1D) if tf.can_emit_signals]
        assert emitting == [TF_15M], (
            f"exactly one timeframe draws entries and it is 15m; these claim they can: "
            f"{[str(tf) for tf in emitting]}"
        )

    def test_the_gates_and_the_regime_never_emit(self):
        for tf in (TF_1H, TF_4H, TF_1D):
            assert not tf.can_emit_signals, (
                f"{tf} is a {tf.role}, not the trigger: a timestamped entry drawn on it can "
                f"contradict the 15m entry with nothing to arbitrate between them"
            )

    def test_the_source_series_is_never_analysed_directly(self):
        for tf in (TF_1M, TF_5M):
            assert not tf.can_emit_signals, f"{tf} is the stored source series, not a signal path"

    def test_emitting_follows_the_role_and_not_the_name(self):
        """The property is a statement about the ROLE, so it has to hold for a timeframe that
        never appears in the table above."""
        assert Timeframe(3 * 60_000, "3m", TfRole.TRIGGER).can_emit_signals
        assert not Timeframe(3 * 60_000, "3m", TfRole.REGIME).can_emit_signals
        assert not Timeframe(3 * 60_000, "3m", TfRole.GATE).can_emit_signals
        assert not Timeframe(3 * 60_000, "3m", TfRole.SOURCE).can_emit_signals
