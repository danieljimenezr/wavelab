"""Gaps: the bug that renders without any error.

A 1m series with holes in it plus a buffer indexed by POSITION means that `i-20` for an ER(20)
reaches across a gap of hours and returns a perfectly valid, completely wrong float. Nothing
raises. That is why the step is asserted from the timestamps on every window construction.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
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

from .conftest import SYMBOL, T0, make_bars


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

    def test_the_gap_error_points_at_the_first_discontinuity(self):
        """The timestamp in the message is what a person actually acts on at three in the morning.

        `require_complete` refuses to compute either way, so the arithmetic is safe whichever
        discontinuity it names — but the message is the whole of the diagnosis. Pointed at the
        last hole instead of the first, it sends whoever is reading it to the wrong outage, in the
        wrong hour of the archive, on the one code path that only ever runs when something has
        already gone wrong.
        """
        r = Ring(SYMBOL, TF_15M, capacity=64)
        for b in make_bars(20, drop={5, 12}):
            r.append(b)
        w = r.window(20)
        assert w.n_gaps == 2, "the premise: two separate holes, so first and last differ"

        whole = make_bars(20)
        first, last = whole[6].open_time_ms, whole[13].open_time_ms
        with pytest.raises(GapError) as excinfo:
            w.require_complete("ER(20)")
        assert f"ts={first}" in str(excinfo.value), (
            f"the error says `the first at` and names ts={last} instead of ts={first}: it points "
            "at the last discontinuity in the window, so the hunt starts at the wrong outage"
        )

    def test_len_is_what_the_ring_still_holds_once_it_has_wrapped(self):
        """`window(len(ring))` is how every chart endpoint reads the ring, at six call sites.

        Once more bars have arrived than the buffer can hold, `len` has to report what is
        RETAINED, not how many were appended. A `len` that keeps counting arrivals asks `window`
        for more bars than exist; `_take` clamps to that inflated number, walks the buffer round
        two or three times and hands back repeated, non-monotonic timestamps. On an 8192-slot 1m
        ring that lands about six days after a deploy — every chart 500s at once, with nothing in
        the diff to point at.
        """
        r = Ring(SYMBOL, TF_15M, capacity=8)
        bars = make_bars(20)
        for b in bars:
            r.append(b)

        assert len(r) == 8, (
            f"the ring keeps 8 bars and len() says {len(r)} after 20 appends: it is counting "
            "arrivals rather than what survived the wrap, so window(len(ring)) over-reads"
        )
        retained = [b.open_time_ms for b in bars[-8:]]
        assert list(r.window(len(r)).ts) == retained, (
            "window(len(ring)) after a wrap is not the last 8 bars in order: the read walked past "
            "the start of the buffer and returned bars it had already returned"
        )
        assert list(r.window(min(1500, len(r))).ts) == retained, (
            "the server's own idiom, window(min(N, len(ring))), disagrees with window(len(ring))"
        )

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

    def test_a_closed_bar_absorbs_its_own_in_flight_version(self):
        """The in-flight slot has to be EMPTIED by the bar that closes it, not merely overwritten.

        The live engine parks every forming bar in the provisional slot; when its closed version
        arrives, `append` is the only thing that drops the tentative copy. Leave it there and the
        provisional window carries the newest candle twice — once solid, once as the dashed "?" —
        so the user counts one more bar than exists. It also freezes `last_ts_ms`, which is what
        the staleness check compares against the clock, one bar in the past's favour: a feed that
        died goes on reporting itself fresh.
        """
        r = Ring(SYMBOL, TF_15M, capacity=16)
        bars = make_bars(7)
        for b in bars[:5]:
            r.append(b)
        r.set_provisional(replace(bars[5], is_closed=False))

        r.append(bars[5])
        with pytest.raises(ValueError, match="no in-flight bar"):
            r.provisional_window(3)

        # And with a genuinely newer in-flight bar parked, the window is one bar per slot on the
        # grid, stamped with that bar's OPEN time.
        in_flight = replace(bars[6], is_closed=False)
        r.set_provisional(in_flight)
        w = r.provisional_window(4)
        assert list(np.diff(w.ts)) == [TF_15M.ms] * 3, (
            f"the provisional window's timestamps are {list(w.ts)}: they repeat or skip, so the "
            "same candle is drawn twice, once confirmed and once as a maybe"
        )
        assert w.last_ts_ms == in_flight.open_time_ms, (
            "last_ts_ms is not the in-flight bar's OPEN time: taken from its close, the staleness "
            "check reports the feed a whole bar fresher than it is and an outage stays hidden"
        )

    def test_asking_for_just_the_current_candle_returns_just_the_current_candle(self):
        """`n` counts the in-flight bar, so `n=1` is that bar alone and `n=0` is a miscount.

        This is the first boundary a chart endpoint asking for "the current candle" hits, and it
        was broken in both directions: `n=1` built a zero-length closed window and died inside
        NumPy on `int(ts[-1])` with a message about an axis, while `n=0` and `n=-5` silently
        returned the same one-bar window instead of refusing. Off-by-one here is not cosmetic —
        a window that is quietly one bar short of what was asked for is the positional-indexing
        bug this whole module exists to prevent, arriving through the front door.
        """
        r = Ring(SYMBOL, TF_15M, capacity=16)
        bars = make_bars(7)
        for b in bars[:5]:
            r.append(b)
        in_flight = replace(bars[5], is_closed=False)
        r.set_provisional(in_flight)

        w1 = r.provisional_window(1)
        assert len(w1) == 1, (
            f"provisional_window(1) returned {len(w1)} bars: `n` counts the in-flight bar, so "
            "one means one — any other length is a silent off-by-one on the newest candle"
        )
        assert int(w1.ts[0]) == in_flight.open_time_ms
        assert w1.close[0] == in_flight.close
        assert w1.last_ts_ms == in_flight.open_time_ms

        # Each further bar reaches exactly one bar further back into the closed history.
        for n in (2, 3, 4):
            w = r.provisional_window(n)
            assert len(w) == n, f"provisional_window({n}) returned {len(w)} bars"
            assert int(w.ts[-1]) == in_flight.open_time_ms
            assert int(w.ts[0]) == in_flight.open_time_ms - (n - 1) * TF_15M.ms

        for bad in (0, -1, -5):
            with pytest.raises(ValueError, match="must be >= 1"):
                r.provisional_window(bad)


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
class TestTheUtcGridIsNotNegotiable:
    """Every index in the system is an open_time. An instant off the grid is in no bar at all."""

    #: DO NOT TRIM THIS LIST. `off=1` is the only value that kills a check weakened from
    #: `% tf.ms != 0` to `% tf.ms > 1`: −1, 7 and `ms - 1` all satisfy `> 1` and sail through.
    #: The parametrisation states the general claim; the kill is single-point, so removing the
    #: "redundant-looking" first case reopens the hole in silence.
    @pytest.mark.parametrize("off", [1, -1, 7, TF_15M.ms - 1])
    def test_a_bar_that_misses_the_grid_is_refused_however_narrowly(self, off):
        """One integer comparison, guarding what `Bar`'s own docstring calls the costliest
        mistake in the project: an adapter that invents its own close_time convention.

        A millisecond is the dangerous offset, not a wild one. Shift every bar by +1 ms and
        `resample_from_1m` still buckets them correctly and the ring's step check still sees a
        clean `tf.ms` between two equally-shifted neighbours, so the corruption travels the whole
        way with nothing raising; shift ONE bar and the ring reports two discontinuities that do
        not exist. Construction is the only place it can be caught, so it has to be caught for
        every remainder rather than only for large ones.
        """
        with pytest.raises(ValueError, match="does not land on"):
            Bar(symbol=SYMBOL, tf=TF_15M, open_time_ms=T0 + off,
                open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0)

    def test_an_aligned_bar_covers_its_own_period_and_abuts_the_next(self):
        """The accepting case has to pin what the grid MEANS, not echo the constructor.

        `assert b.open_time_ms == T0` is what this test used to say, and a frozen dataclass makes
        that true for every implementation that does not raise — including one with the grid check
        deleted, or `close_time_ms` off by a period. The content is the open/close convention that
        `Bar`'s own docstring calls the costliest mistake in the project: a bar owns the last
        millisecond of its period and shares none with its successor. Off by one in either
        direction and two adjacent bars either overlap on an instant or leave one in no bar at all.
        """
        b = Bar(symbol=SYMBOL, tf=TF_15M, open_time_ms=T0,
                open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0)
        assert b.close_time_ms == T0 + TF_15M.ms - 1, (
            f"close_time_ms is {b.close_time_ms}, not the last millisecond of the period "
            f"({T0 + TF_15M.ms - 1}): the adapter has invented its own close convention"
        )
        nxt = Bar(symbol=SYMBOL, tf=TF_15M, open_time_ms=T0 + TF_15M.ms,
                  open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0)
        assert nxt.open_time_ms - b.close_time_ms == 1, (
            "two adjacent bars must abut with exactly one millisecond between close and open: "
            "any other gap means an instant belongs to two bars or to none"
        )

    def test_coverage_is_measured_against_the_period_the_bar_claims_to_be(self):
        """`coverage` divides by the number of 1m bars the period SHOULD contain.

        Off by one in the denominator and a 59-of-60 hour reports 100% complete — the number whose
        entire job is to say "this bar is missing data" says the opposite, and it says it in the
        direction that manufactures confidence. Nothing in `src/` reads this property today, which
        is exactly why it needs a test: it will be wired to a completeness badge by someone who
        trusts it, not by someone who re-derives it.
        """
        h = Bar(symbol=SYMBOL, tf=TF_1H, open_time_ms=T0, open=1.0, high=2.0, low=0.5,
                close=1.5, volume=1.0, n_source_bars=59)
        assert h.coverage == pytest.approx(59 / 60), (
            f"an hour built from 59 of its 60 minutes reports coverage {h.coverage}, not "
            f"{59/60}: the denominator is not the period's own expected bar count"
        )
        full = replace(h, n_source_bars=60)
        assert full.coverage == pytest.approx(1.0)
        # A 1m bar is its own source: the ratio is degenerate and the answer is the fixed 1.0.
        m = Bar(symbol=SYMBOL, tf=TF_1M, open_time_ms=T0, open=1.0, high=2.0, low=0.5,
                close=1.5, volume=1.0, n_source_bars=1)
        assert m.coverage == 1.0

    def test_floor_ms_lands_on_the_bar_that_contains_the_instant(self):
        """`floor_ms` is the one method whose job is mapping an arbitrary instant onto the grid,
        and the direction it rounds is the whole of its meaning.

        Rounded the other way, an instant is attributed to the bar AFTER the one it happened in —
        a fact filed one bar into the future, which is this project's one fatal class of bug. The
        error is invisible to any caller that merely checks the result is a multiple of `tf.ms`,
        which is why the assertions here are on the value and include the fixed point.
        """
        assert TF_1H.floor_ms(T0) == T0, (
            "an instant already on the grid is a fixed point; moving it means floor_ms rounds up"
        )
        assert TF_1H.floor_ms(T0 + TF_1H.ms - 1) == T0, (
            "the last millisecond of an hour belongs to that hour, not to the next one"
        )
        assert TF_1H.floor_ms(T0 - 1) == T0 - TF_1H.ms
