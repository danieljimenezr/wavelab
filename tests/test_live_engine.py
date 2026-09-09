"""The live engine must produce EXACTLY the same thing as the offline path.

If live resampling and backtest resampling differed even in the last decimal, the whole promise of
«one single function» would be false: the backtest would be measuring one series and the user would
be looking at another.

Hanging off that same path are the two things the promise is FOR: the labels the backtest learns
from (triple barrier) and the numbers it hands back. A label whose geometry is not the geometry of
the plan on screen, or a statistic that quietly drops the fees, breaks the promise just as
thoroughly as a mis-resampled bar — and it breaks it upwards, which is worse, because the result
looks better rather than broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelab.backtest.run import BacktestResult, Signal, run_backtest
from wavelab.core.timeframes import BY_NAME, TF_1H, TF_1M, TF_15M, resample_from_1m
from wavelab.core.types import Direction, MaturityLevel, Verdict
from wavelab.engine import live as live_module
from wavelab.engine.live import LiveEngine, Mode
from wavelab.labeling.barriers import Outcome, resolve_triple_barrier

from .conftest import SYMBOL, T0, make_bars

#: 20 days of 1m bars: long enough for real wave structure to form, resolve and be counted, and
#: still under a second to build. Shorter than this and the backtest tests below become vacuous.
TRIGGER_TF = "1h"
TIMEFRAMES = ("1m", "15m", "1h")
MAX_BARS_HOLD = 12
COOLDOWN_BARS = 3


def _engine(tfs=("1m", "15m", "1h")) -> LiveEngine:
    return LiveEngine(SYMBOL, list(tfs), ring_capacity=4096, trigger_tf="15m")


@pytest.fixture(scope="module")
def bars_20d() -> list:
    return make_bars(60 * 24 * 20, tf=TF_1M, seed=3)


@pytest.fixture(scope="module")
def backtest_20d(bars_20d) -> BacktestResult:
    return run_backtest(bars_20d, trigger_tf=TRIGGER_TF, timeframes=TIMEFRAMES,
                        max_bars_hold=MAX_BARS_HOLD, cooldown_bars=COOLDOWN_BARS)


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

    def test_a_thin_hour_travels_labelled_as_a_gap(self):
        """An hour built out of half its minutes is not an hour, and it has to SAY so.

        The neighbouring test only proves that a COMPLETE bar is not flagged — which an engine
        that never flags anything also passes. This is the other half: if `is_gap` stops being
        computed, a bar made of 30 minutes enters the ring indistinguishable from a real one, and
        every ATR, every pivot and every backtest statistic built on top of it describes a market
        that was never there. Nothing downstream can recover the difference afterwards.
        """
        e = _engine()
        closed: list = []
        # The 2nd hour loses 30 of its 60 minutes. Its closing minute survives, so it still closes.
        for b in make_bars(120, tf=TF_1M, drop=set(range(70, 100))):
            closed += e.on_bar_1m(b)

        thin = [b for b in closed if b.tf is TF_1H][-1]
        assert thin.n_source_bars == 30, (
            f"the 2nd hour should have been built from 30 minutes, not {thin.n_source_bars}: "
            "the fixture no longer produces a thin bar and this test proves nothing"
        )
        assert thin.is_gap, (
            "an hour built out of 30 of its 60 minutes was handed on as a complete bar: is_gap is "
            "not being computed, so nothing downstream can tell a real hour from half of one"
        )
        w = e.state.rings["1h"].window(1)
        assert bool(w.is_gap[0]), "the thin hour was stored in the ring without its gap flag"

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

    def test_reloading_history_goes_back_to_warming_up(self, monkeypatch):
        """History is replayed through the very same `on_bar` as the live path.

        That is the point — one function, no second implementation — and it is also why loading
        history has to put the badge back to WARMUP first. Re-hydrating an engine that is already
        LIVE (after a long outage, on a symbol change) walks a week of bars through the emitting
        path: anything asking the engine what it thinks meanwhile is answered with a decision card
        drawn from a half-loaded ring, at a price from last Tuesday, labelled live.
        """
        now = {"ms": T0 + 600 * TF_1M.ms}
        monkeypatch.setattr(live_module.time, "time", lambda: now["ms"] / 1000.0)

        e = _engine()
        e.warmup(make_bars(600, tf=TF_1M, start_ms=T0))
        e.update_health(connected=True, reconnects=0, healed=0, silent_seconds=0.0)
        assert e.emitting, "precondition: the engine is up to date and emitting"

        e.warmup(make_bars(600, tf=TF_1M, start_ms=T0))
        assert e.state.health.mode is Mode.WARMUP, (
            f"a second warm-up left the engine in {e.state.health.mode}: history is being replayed "
            "with the badge still claiming the data is live"
        )
        assert not e.emitting, "nothing may be emitted while history is loading"

    def test_an_outage_stops_emission_and_only_catching_up_restarts_it(self, monkeypatch):
        """The whole round trip: live → the laptop sleeps → live again.

        `on_bar` is deliberately the same code live and in replay, so on waking the engine will
        cheerfully compute an entry zone around a price that went past two hours ago and present it
        as current. The three transitions each guard a different way of getting that wrong: not
        noticing the outage at all, leaving CATCH_UP before the data has caught up, and never
        leaving it once the hole is filled (at which point the tool is silent for good and the user
        assumes it is broken).
        """
        now = {"ms": T0 + 600 * TF_1M.ms}
        monkeypatch.setattr(live_module.time, "time", lambda: now["ms"] / 1000.0)

        e = _engine()                                   # trigger timeframe: 15m
        e.warmup(make_bars(600, tf=TF_1M, start_ms=T0))  # the last minute closes exactly at `now`
        e.update_health(connected=True, reconnects=0, healed=0, silent_seconds=0.0)
        assert e.state.health.mode is Mode.LIVE and e.emitting, (
            "up to date and connected, the engine must be LIVE and emitting"
        )
        assert e.check_clock() is Mode.LIVE, (
            "an ordinary tick with no clock jump was mistaken for an outage: the engine would stop "
            "emitting during normal operation"
        )

        # The process is suspended for two hours: eight trigger bars go by with no data at all.
        now["ms"] += 8 * TF_15M.ms
        assert e.check_clock() is Mode.CATCH_UP, (
            "a two-hour wall-clock jump went unnoticed; the engine still believes it is live and "
            "will publish an entry zone at a price that is two hours stale"
        )
        h = e.update_health(connected=True, reconnects=1, healed=0, silent_seconds=7200.0)
        assert h.lag_bars > 1.0, f"the lag should be about 8 trigger bars, it reads {h.lag_bars}"
        assert e.state.health.mode is Mode.CATCH_UP and not e.emitting, (
            f"the engine went back to emitting {h.lag_bars:.1f} trigger bars behind the market"
        )

        # The healing pass delivers the 120 minutes that were missed.
        for b in make_bars(120, tf=TF_1M, start_ms=T0 + 600 * TF_1M.ms):
            e.on_bar_1m(b)
        h = e.update_health(connected=True, reconnects=1, healed=120, silent_seconds=0.0)
        assert h.lag_bars <= 1.0, f"the hole is filled, the lag should be tiny: {h.lag_bars}"
        assert e.state.health.mode is Mode.LIVE and e.emitting, (
            "the hole is filled and the engine is up to date, but it never left CATCH_UP: it would "
            "stay mute for the rest of the session"
        )


class TestTheDecisionCard:
    def test_the_verdict_never_goes_above_watch_while_the_maturity_is_prior(self, bars_20d):
        """PRIOR means not one trade of ours has resolved yet.

        The expectancies behind the card are a hand-written expert table; marking anything
        ACTIONABLE on that basis tells the user to risk money on a number nobody has measured.
        `Decision`'s constructor refuses the combination structurally, but that is the last line of
        defence and it only fires if a `Decision` is ever built: `decide()` returns a plain dict
        straight to the interface, so this is the line that actually holds.
        """
        e = LiveEngine(SYMBOL, list(TIMEFRAMES), 8192, TRIGGER_TF)
        in_zone = 0
        for b in bars_20d:
            for htf in e.on_bar_1m(b):
                if htf.tf is not TF_1H:
                    continue
                d = e.decide(TRIGGER_TF, htf.close)
                assert d["maturity"] == int(MaturityLevel.PRIOR), (
                    f"maturity {d['maturity']} at {htf.open_time_ms}: nothing has been validated, "
                    "the card cannot claim a level above PRIOR"
                )
                assert d["verdict"] != Verdict.ACTIONABLE.value, (
                    f"decide() returned ACTIONABLE at {htf.open_time_ms} while the maturity is "
                    "PRIOR: the user is being told to take a trade whose expectancy comes from a "
                    "hand-written table"
                )
                in_zone += sum(1 for h in d["hypotheses"] if h["viable"] and h["in_zone"])

        assert in_zone > 0, (
            "not one hypothesis was ever viable AND inside its entry zone over 20 days, so the "
            "cap above was never actually exercised: this test proves nothing as it stands"
        )


class TestTheTripleBarrier:
    """The labels are what everything downstream learns from, and they fail upward.

    A label that says «won» where the trade lost does not raise: it produces a prettier hit rate.
    Nothing later in the pipeline can detect it, because every expectancy it is compared against
    was computed from the same labels.
    """

    def test_a_bar_that_touches_both_barriers_is_a_loss_and_says_it_was_ambiguous(self):
        """Inside one bar there is no way to know whether the target or the stop came first.

        Resolving the coin-flip as a win is how a backtest gets a hit rate nobody can reproduce
        with real money. It is resolved pessimistically, and — just as important — the ambiguity is
        RECORDED, because the rate of it is the alarm that says the barriers are too close together
        and the labels have stopped describing real trades.
        """
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,
            np.array([112.0]), np.array([94.0]), np.array([100.0]),
            max_bars=5,
        )
        assert out.barrier == "sl", (
            f"a bar that touched target AND stop was resolved as '{out.barrier}': with no way to "
            "know which came first, the optimistic reading inflates every statistic downstream"
        )
        assert out.r == -1.0, f"an unresolvable bar must cost a full R, it booked {out.r}"
        assert out.ambiguous is True, (
            "the tie was resolved pessimistically but not flagged: the ambiguity rate is what "
            "tells us the labels have stopped describing real trades, and it would read 0%"
        )

    @pytest.mark.parametrize(
        ("fine_high", "fine_low", "barrier", "r"),
        [
            ([111.0, 96.0], [99.0, 94.0], "tp", 2.0),   # the target came first
            ([101.0, 112.0], [94.0, 99.0], "sl", -1.0),  # the stop came first
        ],
    )
    def test_the_1m_series_breaks_the_tie_and_the_label_stops_being_ambiguous(
        self, fine_high, fine_low, barrier, r
    ):
        """When the minute bars inside the hour DO know the order, the coin-flip is not needed.

        This is the reason 1m is the single stored series: dropping down to it turns a pessimistic
        guess into a fact. If the fine data stops being consulted, every one of these becomes a
        loss and the strategy is measured as worse than it is — and the ambiguity rate stops being
        a signal about the barriers and becomes a constant.
        """
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,
            np.array([112.0]), np.array([94.0]), np.array([100.0]),
            max_bars=5,
            fine_highs=[np.array(fine_high)], fine_lows=[np.array(fine_low)],
        )
        assert out.barrier == barrier, (
            f"the 1m bars say the {barrier} came first; the label reads '{out.barrier}'"
        )
        assert out.r == pytest.approx(r), f"expected {r}R, got {out.r}R"
        assert out.bars == 1, (
            f"the trade resolved inside the first bar, the label says it was held {out.bars}: "
            "holding time is what the vertical barrier and every duration statistic are built on"
        )
        assert out.ambiguous is False, (
            "the minute bars resolved the order, so this label is a fact and must not be counted "
            "towards the ambiguity rate that decides whether the labels are usable at all"
        )

    def test_a_minute_that_is_itself_ambiguous_stays_pessimistic(self):
        """1m is as fine as the data goes. When even a minute touched both, we still do not know.

        Guessing at that point is the same optimism as guessing at the hour, just better hidden.
        """
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,
            np.array([112.0]), np.array([94.0]), np.array([100.0]),
            max_bars=5,
            fine_highs=[np.array([112.0])], fine_lows=[np.array([94.0])],
        )
        assert (out.barrier, out.r, out.ambiguous) == ("sl", -1.0, True), (
            f"a minute bar that touched both barriers was resolved as {out.barrier} r={out.r} "
            f"ambiguous={out.ambiguous}: the order is still unknown, so the label must stay "
            "pessimistic and stay flagged"
        )

    @pytest.mark.parametrize(
        ("highs", "lows", "closes", "barrier", "held"),
        [
            ([103.0, 111.0], [96.0, 99.0], [102.0, 110.0], "tp", 2),
            ([103.0, 101.0], [96.0, 94.0], [102.0, 95.0], "sl", 2),
            ([103.0, 104.0], [96.0, 97.0], [102.0, 103.0], "vertical", 2),
        ],
    )
    def test_a_short_is_the_exact_mirror_image_of_the_long(
        self, highs, lows, closes, barrier, held
    ):
        """The interface only shows longs; the engine labels BOTH directions from day one.

        So half the evidence the project will ever accumulate is short evidence, and it is checked
        by nothing that looks at the screen. Every comparison in the short branch is a separate
        `if long else`, and each one is an opportunity to compare against the wrong side of the
        bar: a short's target is BELOW the entry and its stop ABOVE it. Reflecting the whole series
        around the entry price has to produce the identical outcome, R for R and excursion for
        excursion — if it does not, the short half of the book is measuring something else.
        """
        entry = 100.0
        mirror = 2 * entry                        # p -> 200 - p reflects price around the entry

        long_out = resolve_triple_barrier(
            entry, 95.0, 110.0,
            np.array(highs), np.array(lows), np.array(closes),
            max_bars=2, long=True,
        )
        short_out = resolve_triple_barrier(
            entry, mirror - 95.0, mirror - 110.0,
            np.array([mirror - x for x in lows]),     # the mirror of a low is a high
            np.array([mirror - x for x in highs]),
            np.array([mirror - x for x in closes]),
            max_bars=2, long=False,
        )
        assert (long_out.barrier, long_out.bars) == (barrier, held), (
            f"the fixture no longer exercises the case it names: expected {barrier} after {held} "
            f"bars, got {long_out.barrier} after {long_out.bars}"
        )
        assert short_out == long_out, (
            f"the mirrored short resolved differently from the long: {short_out} != {long_out}. "
            "One of the direction branches is comparing against the wrong side of the bar"
        )

    def test_the_time_barrier_closes_the_trade_at_market(self):
        """The third barrier is the one people forget, and it decides roughly half the trades.

        A trade that neither wins nor loses within its holding window is CLOSED at market — that is
        what the user would actually do. If the time barrier stops being applied, positions are
        held until the data runs out: the R booked belongs to a trade nobody would have sat
        through, and the tail of the distribution the sizing is built on becomes fiction.
        """
        highs = np.full(50, 101.0)
        lows = np.full(50, 99.0)
        closes = np.full(50, 100.5)
        closes[9] = 102.0                       # the close at the moment the window expires
        out = resolve_triple_barrier(100.0, 95.0, 110.0, highs, lows, closes, max_bars=10)

        assert out.barrier == "vertical", f"nothing was touched, expected 'vertical': {out}"
        assert out.bars == 10, (
            f"the trade was held for {out.bars} bars with a 10-bar window: the time barrier is not "
            "being applied and the label describes a trade that was never taken"
        )
        assert out.r == pytest.approx((102.0 - 100.0) / 5.0), (
            f"the exit must be booked at the close of the last bar held, got {out.r}R"
        )

    def test_a_stop_out_costs_exactly_one_r(self):
        """R is DEFINED as the distance from entry to stop, so a stop-out is -1R by definition.

        Book it as anything else and every number downstream is quietly rescaled: expectancy,
        profit factor and the required hit rate all move, and none of them looks wrong.
        """
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,
            np.array([101.0, 99.0]), np.array([98.0, 94.0]), np.array([100.0, 95.0]),
            max_bars=5,
        )
        assert out.barrier == "sl", f"the second bar traded through the stop: {out}"
        assert out.r == -1.0, f"a stop-out must cost exactly -1R, it booked {out.r}R"
        assert out.bars == 2, f"the stop was hit on the 2nd bar, the label says {out.bars}"

    def test_the_two_excursions_measure_opposite_directions(self):
        """MAE is what the trade cost you before it worked; MFE is what it offered before it did.

        They are the input to any future rule about tightening stops or taking partials. Swapped,
        such a rule would be fitted to the best case while believing it was the worst — the single
        most dangerous way to be wrong about risk.
        """
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,
            np.array([101.0, 108.0]), np.array([97.0, 100.0]), np.array([100.0, 107.0]),
            max_bars=2,
        )
        assert out.mae_r == pytest.approx(0.6), (
            f"price went 3 points against a 5-point risk: MAE is 0.6R, the label says {out.mae_r}"
        )
        assert out.mfe_r == pytest.approx(1.6), (
            f"price went 8 points in favour of a 5-point risk: MFE is 1.6R, the label says "
            f"{out.mfe_r}"
        )

    def test_the_excursions_are_not_interchangeable_on_a_resolved_trade(self):
        """The test above only ever reaches the VERTICAL exit, so it pins the excursions on one of
        the three return paths and leaves the other two free.

        `resolve_triple_barrier` builds an `Outcome` at four separate places, each passing `mae`
        and `mfe` positionally into adjacent fields of the same type. Swapping that pair on the
        `tp` or `sl` return is invisible to every other assertion in this file — and a stop-tighten
        rule fitted to the swapped pair would be fitted to the best case while believing it was
        the worst.

        Both cases below are deliberately asymmetric (MAE != MFE), so the swap cannot hide.
        """
        # --- resolved at the TARGET, after first going against the trade -------------------
        tp = resolve_triple_barrier(
            100.0, 95.0, 110.0,                       # risk = 5
            np.array([101.0, 112.0]), np.array([97.0, 100.0]), np.array([100.0, 111.0]),
            max_bars=5,
        )
        assert tp.barrier == "tp", f"bar 2 traded through the target: {tp}"
        assert tp.mae_r == pytest.approx(0.6), (
            f"the low of 97 is 3 points against a 5-point risk: MAE is 0.6R, not {tp.mae_r}"
        )
        assert tp.mfe_r == pytest.approx(2.4), (
            f"the high of 112 is 12 points in favour of a 5-point risk: MFE is 2.4R, not "
            f"{tp.mfe_r}"
        )

        # --- resolved at the STOP, after first going in favour -----------------------------
        sl = resolve_triple_barrier(
            100.0, 95.0, 120.0,                       # risk = 5, target far away
            np.array([104.0, 101.0]), np.array([98.0, 94.0]), np.array([103.0, 95.0]),
            max_bars=5,
        )
        assert sl.barrier == "sl", f"bar 2 traded through the stop: {sl}"
        assert sl.mae_r == pytest.approx(1.2), (
            f"the low of 94 is 6 points against a 5-point risk: MAE is 1.2R, not {sl.mae_r}"
        )
        assert sl.mfe_r == pytest.approx(0.8), (
            f"the high of 104 is 4 points in favour of a 5-point risk: MFE is 0.8R, not "
            f"{sl.mfe_r}"
        )

    def test_it_refuses_a_trade_whose_stop_is_its_entry(self):
        """Every label is a ratio to |entry - stop|. A zero there poisons silently.

        The division would yield inf or nan, those flow into the expectancy, and the backtest comes
        back with a number that is not wrong so much as meaningless. Refusing loudly is the only
        way anyone finds out.
        """
        with pytest.raises(ValueError, match="zero risk"):
            resolve_triple_barrier(
                100.0, 100.0, 110.0,
                np.array([101.0]), np.array([99.0]), np.array([100.0]),
                max_bars=5,
            )


class TestTheBacktest:
    """What the backtest measures has to be the trade the user would have taken. Nothing else."""

    def test_a_trade_is_resolved_only_with_bars_that_come_after_the_decision(
        self, bars_20d, backtest_20d
    ):
        """The decision is causal; the RESOLUTION is allowed to look forward — but not that far.

        Entry is the close of the bar that produced the signal, so that bar's own high and low
        happened BEFORE the entry existed. Resolving against them books targets and stops at prices
        the trade could never have touched, and the error fails upward: on a 4h bar whose range
        spans both barriers, roughly half those phantom fills are wins. Here the outcome of every
        signal is recomputed from the offline resampler — a different code path from the engine's —
        using strictly the bars that came after the decision.
        """
        df = pd.DataFrame(
            {"open": [b.open for b in bars_20d], "high": [b.high for b in bars_20d],
             "low": [b.low for b in bars_20d], "close": [b.close for b in bars_20d],
             "volume": [b.volume for b in bars_20d]},
            index=pd.Index([b.open_time_ms for b in bars_20d], name="open_time_ms"),
        )
        offline = resample_from_1m(df, BY_NAME[TRIGGER_TF])
        high = offline["high"].to_numpy()
        low = offline["low"].to_numpy()
        close = offline["close"].to_numpy()
        at = {ts: i for i, ts in enumerate(offline.index)}

        resolved = [s for s in backtest_20d.signals if s.outcome is not None]
        assert len(resolved) >= 20, (
            f"only {len(resolved)} resolved signals: too few for this comparison to mean anything"
        )
        for s in resolved:
            i = at[s.ts_ms]
            expected = resolve_triple_barrier(
                s.entry, s.stop, s.target, high[i + 1:], low[i + 1:], close[i + 1:],
                max_bars=MAX_BARS_HOLD, long=s.direction is Direction.LONG,
            )
            assert (s.outcome.barrier, s.outcome.bars) == (expected.barrier, expected.bars), (
                f"the signal at {s.ts_ms} was booked as {s.outcome.barrier} after "
                f"{s.outcome.bars} bars; resolved against the bars that came after the decision it "
                f"is {expected.barrier} after {expected.bars}. The entry bar is being resolved "
                "against itself"
            )
            assert s.outcome.r == pytest.approx(expected.r), (
                f"the signal at {s.ts_ms} booked {s.outcome.r}R, the bars after it give "
                f"{expected.r}R"
            )

    def test_every_signal_is_a_trade_you_could_take(self, backtest_20d):
        """Stop on the losing side, target on the winning side, and a risk that is not zero.

        A signal with the stop on the wrong side of the entry is not a bad trade, it is not a trade
        at all: it is stopped out the instant it opens, and the label it produces is noise mixed
        into the statistics of every other signal that shares its archetype.
        """
        assert backtest_20d.signals, "no signals: this test is checking an empty list"
        longs = sum(1 for s in backtest_20d.signals if s.direction is Direction.LONG)
        assert 0 < longs < len(backtest_20d.signals), (
            "both directions have to appear, or the short branch is never checked here"
        )
        for s in backtest_20d.signals:
            if s.direction is Direction.LONG:
                assert s.stop < s.entry < s.target, (
                    f"long signal at {s.ts_ms}: stop {s.stop}, entry {s.entry}, target {s.target} "
                    "— a long's stop goes below the entry and its target above it"
                )
            else:
                assert s.target < s.entry < s.stop, (
                    f"short signal at {s.ts_ms}: stop {s.stop}, entry {s.entry}, target "
                    f"{s.target} — a short's stop goes above the entry and its target below it"
                )

    def test_the_fees_are_subtracted_and_paying_more_can_only_make_it_worse(self, backtest_20d):
        """A backtest that reports gross R is a brochure. The user pays the taker fee twice.

        `cost_r` is that fee expressed in R, and it is the difference between an edge and a story:
        at a 0.5% stop it eats 0.30R, which is most of what any honest expectancy is worth. The
        direction of the sign is the whole point — turn it around and the tighter the stop, the
        better the strategy looks.
        """
        priced = [s for s in backtest_20d.signals if s.outcome and s.cost_r > 0]
        assert priced, "no signal carried a cost: the fee arithmetic is not being exercised"
        for s in priced:
            assert s.net_r == pytest.approx(s.outcome.r - s.cost_r), (
                f"the signal at {s.ts_ms} gross {s.outcome.r}R with {s.cost_r}R of fees reports a "
                f"net of {s.net_r}R: the fee is not being taken off"
            )

        cheap = backtest_20d.stats(cost_mult=1.0)
        dear = backtest_20d.stats(cost_mult=3.0)
        assert dear["expectancy_r"] < cheap["expectancy_r"], (
            f"tripling the fees left the expectancy at {dear['expectancy_r']} against "
            f"{cheap['expectancy_r']}: the cost multiplier is not reaching the statistics, so a "
            "fee-sensitivity check would always come back reassuring"
        )
        per_trade = cheap["expectancy_r"] - dear["expectancy_r"]
        assert per_trade == pytest.approx(
            2 * float(np.mean([s.cost_r for s in backtest_20d.signals if s.outcome]))
        ), "tripling the fee has to subtract exactly two more fees from the average trade"

    def test_the_null_arm_is_a_different_trade_from_the_signal(self, backtest_20d):
        """The null arm is the control, and a control that copies the treatment is not one.

        Every signal spawns a twin at a random nearby instant with the same geometry in R. That
        pairing is what makes the result interpretable long before the absolute hit rate is: it
        answers «does the count add anything?» instead of «did BTC go up?». If the twin were
        entered at the same instant it would BE the signal, `edge_vs_null` would be exactly zero
        by construction, and the number that decides whether the whole Elliott premise is worth
        anything would be a tautology reading "no edge" for ever.
        """
        twinned = [s for s in backtest_20d.signals if s.outcome and s.null_outcome]
        assert len(twinned) >= 20, f"only {len(twinned)} twinned signals: too few to compare"

        same = sum(1 for s in twinned if s.null_outcome == s.outcome)
        assert same < len(twinned) // 2, (
            f"{same} of {len(twinned)} null twins resolved identically to their signal: the "
            "control is entering at the same instant as the trade it is supposed to control for"
        )
        st = backtest_20d.stats()
        assert st["edge_vs_null"] is not None and st["edge_vs_null"] != 0.0, (
            f"edge_vs_null came back {st['edge_vs_null']}: an exact zero means the null arm is the "
            "signal itself"
        )

    def test_one_count_does_not_become_twenty_trades(self, backtest_20d):
        """The debounce, and it is a statistics question rather than a tidiness one.

        The same wave count stays valid and in its zone for bar after bar. Taken on each of them it
        produces twenty near-identical trades out of ONE independent observation, and every measure
        of significance downstream treats them as twenty: the sample looks four times bigger than
        it is and the confidence interval shrinks around a number that never earned it. That is
        pseudo-replication, and it is how a backtest talks itself into a strategy.
        """
        tf_ms = BY_NAME[TRIGGER_TF].ms
        last: dict[str, int] = {}
        for s in sorted(backtest_20d.signals, key=lambda s: s.ts_ms):
            previous = last.get(s.archetype)
            if previous is not None:
                apart = (s.ts_ms - previous) // tf_ms
                assert apart >= COOLDOWN_BARS, (
                    f"two {s.archetype} signals {apart} trigger bars apart with a cooldown of "
                    f"{COOLDOWN_BARS}: the same count is being counted as several independent "
                    "trades"
                )
            last[s.archetype] = s.ts_ms
        assert len(last) > 1, "only one archetype fired; the debounce was barely exercised"

    def test_the_score_filter_actually_filters(self, bars_20d, backtest_20d):
        """`min_score` is the operator saying «only the counts that fit well».

        A filter that is read and then ignored is worse than no filter: the report comes back
        looking like the selective run that was asked for, and the trades it was meant to exclude
        are in there raising the sample size and lowering the quality of every archetype's stats.
        """
        assert backtest_20d.signals, "no unfiltered signals to compare against"
        strict = run_backtest(bars_20d, trigger_tf=TRIGGER_TF, timeframes=TIMEFRAMES,
                              max_bars_hold=MAX_BARS_HOLD, cooldown_bars=COOLDOWN_BARS,
                              min_score=1.01)      # the score is a weighted mean: it cannot pass 1
        assert not strict.signals, (
            f"a score floor no hypothesis can clear still let {len(strict.signals)} signals "
            "through: min_score is not being applied"
        )

    def test_the_same_seed_replays_the_same_backtest(self, bars_20d):
        """Two runs of the same code over the same bars have to agree, or nothing can be compared.

        The null arm draws its entry at random. Unseeded, the answer to «did this change help?»
        moves on its own between runs, and every comparison — a config change, a new rule, a
        refactor — is measuring the generator instead of the change.
        """
        kw = {"trigger_tf": TRIGGER_TF, "timeframes": TIMEFRAMES,
              "max_bars_hold": MAX_BARS_HOLD, "cooldown_bars": COOLDOWN_BARS}

        def fingerprint(r: BacktestResult) -> list[tuple]:
            return [(s.ts_ms, s.entry, s.stop, s.target, s.outcome, s.null_outcome)
                    for s in r.signals]

        a = run_backtest(bars_20d, seed=17, **kw)
        b = run_backtest(bars_20d, seed=17, **kw)
        assert fingerprint(a) == fingerprint(b), (
            "two runs with the same seed produced different backtests: the result is not "
            "reproducible and no comparison between two runs means anything"
        )

        c = run_backtest(bars_20d, seed=99, **kw)
        assert [s.outcome for s in a.signals] == [s.outcome for s in c.signals], (
            "changing the seed changed the SIGNALS: the trades themselves must not depend on the "
            "random draw, only their null twins do"
        )
        assert [s.null_outcome for s in a.signals] != [s.null_outcome for s in c.signals], (
            "changing the seed left the null arm untouched: the draw is not seeded from the "
            "argument, so `seed` is not actually controlling anything"
        )


class TestTheBacktestStatistics:
    """The report card. Every one of these numbers ends up in a decision about real money."""

    @staticmethod
    def _result(*rs: float, cost_r: float = 0.0) -> BacktestResult:
        res = BacktestResult()
        for i, r in enumerate(rs):
            s = Signal(ts_ms=i, archetype="w2_long", direction=Direction.LONG, entry=100.0,
                       stop=95.0, target=110.0, rr=2.0, score=1.0, cost_r=cost_r,
                       size_factor=1.0, invalidation=94.0)
            s.outcome = Outcome("tp" if r > 0 else "sl", r, 0.0, 0.0, 1)
            res.signals.append(s)
        return res

    def test_a_breakeven_trade_is_not_a_win(self):
        """The hit rate is the number the user reads first and the one they overweight.

        A trade that closes exactly where it opened won nothing. Counted as a win it lifts the
        stated hit rate above the one the expectancy was computed from, and the two numbers on the
        same card then quietly contradict each other.
        """
        st = self._result(2.0, -1.0, 0.0).stats()
        assert st["n"] == 3
        assert st["hit_rate"] == pytest.approx(1 / 3), (
            f"one win in three trades is a hit rate of 0.33, the card says {st['hit_rate']}: a "
            "breakeven trade is being counted as a winner"
        )

    def test_the_profit_factor_is_what_was_won_over_what_was_lost(self):
        """Profit factor below 1 means the strategy loses money. It has to be that way round.

        Inverted, a losing system reports 2.0 and a winning one 0.5 — and because the figure is
        read as a quality score rather than checked, it would be believed.
        """
        st = self._result(3.0, 1.0, -2.0).stats()
        assert st["profit_factor"] == pytest.approx(2.0), (
            f"4R won against 2R lost is a profit factor of 2.0, the card says "
            f"{st['profit_factor']}"
        )
        losing = self._result(1.0, -2.0, -2.0).stats()
        assert losing["profit_factor"] < 1.0, (
            f"a system that lost 4R to win 1R reports a profit factor of "
            f"{losing['profit_factor']}: above 1 it reads as profitable"
        )

    def test_the_cost_multiplier_charges_every_trade_again(self):
        """The fee-sensitivity knob: «and if the spread were three times worse?».

        It is the cheapest test of whether an edge is real or is really a rebate. If the multiplier
        does not reach the arithmetic, the answer is always «it still works», which is precisely
        the answer nobody should trust.
        """
        res = self._result(2.0, -1.0, 0.0, cost_r=0.1)
        assert res.stats(cost_mult=1.0)["expectancy_r"] == pytest.approx(1 / 3 - 0.1)
        assert res.stats(cost_mult=2.0)["expectancy_r"] == pytest.approx(1 / 3 - 0.2), (
            "doubling the multiplier has to charge each trade one more fee"
        )
        assert res.stats(cost_mult=0.0)["expectancy_r"] == pytest.approx(1 / 3), (
            "at zero cost the expectancy is the gross one"
        )
