"""Causality has to be impossible to violate, not merely unlikely to be violated."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from wavelab.core.causality import AsOf, CausalityError, causal, is_causal
from wavelab.core.clock import LiveClock, SimClock
from wavelab.core.ring import Ring
from wavelab.core.timeframes import TF_15M
from wavelab.core.types import (
    AuxEvent,
    Bar,
    Decision,
    Direction,
    MaturityLevel,
    Pivot,
    PivotKind,
    Stat,
    TradePlan,
    Verdict,
)
from wavelab.waves.store import PivotStore

from .conftest import SYMBOL, T0, make_bars


class TestAsOf:
    def test_reading_before_it_was_available_raises(self):
        v = AsOf(42.0, available_at_ms=1000)
        assert v.get(1000) == 42.0
        assert v.get(5000) == 42.0
        with pytest.raises(CausalityError, match="acausal"):
            v.get(999)

    def test_known_at_does_not_raise(self):
        v = AsOf("x", 1000)
        assert not v.known_at(999)
        assert v.known_at(1000)


class TestPivotStore:
    """`as_of` is the ONLY reachable path. Everything else raises."""

    @pytest.fixture
    def store(self) -> PivotStore:
        s = PivotStore()
        s.append_confirmed(Pivot(10, 1000, 100.0, PivotKind.HIGH, 5.0).confirmed_at(20, 2000))
        s.append_confirmed(Pivot(30, 3000, 90.0, PivotKind.LOW, 5.0).confirmed_at(40, 4000))
        return s

    @pytest.mark.parametrize("op", [lambda s: s[0], lambda s: list(s), lambda s: len(s)])
    def test_non_causal_accessors_raise(self, store, op):
        with pytest.raises(CausalityError):
            op(store)

    def test_as_of_returns_only_what_was_confirmed(self, store):
        assert store.n_as_of(1999) == 0   # the extreme had already happened, but was not confirmed
        assert store.n_as_of(2000) == 1
        assert store.n_as_of(9999) == 2

    def test_as_of_is_a_strict_prefix(self, store):
        full = store.as_of(10_000)
        for t in (0, 1500, 2000, 3500, 4000, 9999):
            partial = store.as_of(t)
            assert partial == full[: len(partial)], (
                "as_of must return a PREFIX: otherwise the history of counts is not append-only "
                "and a count already shown to the user could vanish without an invalidation."
            )

    def test_reconfirming_raises(self):
        p = Pivot(1, 100, 10.0, PivotKind.LOW, 1.0).confirmed_at(2, 200)
        with pytest.raises(ValueError, match="write-once"):
            p.confirmed_at(3, 300)

    def test_out_of_order_confirmation_raises(self, store):
        late = Pivot(50, 5000, 80.0, PivotKind.HIGH, 5.0).confirmed_at(60, 1000)
        with pytest.raises(ValueError, match="monotone"):
            store.append_confirmed(late)

    def test_last_as_of_never_reaches_past_now(self, store):
        """`last_as_of` is what the count reads on every bar, so it is where a peek would hide.

        The last element of the list is the pivot confirmed most RECENTLY, which from the point of
        view of an earlier bar is exactly the one that has not happened yet. Hand that back and
        tomorrow's swing enters today's wave count; the backtest comes out better, not worse, and
        nobody investigates a result that improved.
        """
        assert store.last_as_of(1999) is None, (
            "last_as_of returned a pivot at 1999, before the first confirmation at 2000: the "
            "count on that bar is built on a swing nobody could know about yet"
        )
        assert store.last_as_of(2000).idx == 10
        assert store.last_as_of(3999).idx == 10, (
            "last_as_of skipped ahead to the pivot confirmed at 4000 while standing on 3999: "
            "this is the lookahead the whole module exists to make unwritable"
        )
        assert store.last_as_of(4000).idx == 30
        assert store.last_as_of(10**15).idx == 30

    def test_two_pivots_confirmed_on_the_same_bar_are_both_kept(self, store):
        """Confirmation order is MONOTONE, not strictly increasing, and the difference is real.

        One close can carry price past two pending thresholds at once — routine while a gap is
        being replayed after an outage. Refusing the equal case would abort the engine on exactly
        the bar it is catching up on, and refusing it quietly (by dropping the second pivot) would
        lose a wave from the count for good.
        """
        same_bar = Pivot(35, 3500, 110.0, PivotKind.HIGH, 5.0).confirmed_at(41, 4000)
        store.append_confirmed(same_bar)
        assert store.n_as_of(3999) == 1
        assert store.n_as_of(4000) == 3, (
            "two pivots confirmed on the same bar close and the store kept fewer than both: "
            "an outage replay silently loses waves"
        )
        assert [p.idx for p in store.as_of(4000)] == [10, 30, 35]

    def test_a_pivot_that_goes_backwards_in_position_is_refused(self, store):
        """The beam is monotone in POSITION too, and this check is the only thing that says so.

        A pivot whose `idx` sits behind the last one still passes the confirmation-TIME check, so
        `as_of` goes on returning a tidy prefix and nothing looks wrong from the outside.
        Downstream every wave is measured as `idx_b - idx_a`: one pivot out of order and a wave has
        a negative length, the Elliott rules are evaluated against a leg that runs backwards, and
        the count is nonsense that nothing flags.
        """
        backwards = Pivot(20, 2500, 95.0, PivotKind.HIGH, 5.0).confirmed_at(45, 5000)
        with pytest.raises(ValueError, match="positional order"):
            store.append_confirmed(backwards)
        assert store.n_as_of(10**15) == 2, "a rejected pivot must not have been appended anyway"

    def test_the_provisional_pivot_is_invisible_before_its_own_extreme(self, store):
        """The tentative channel is still an as-of read, not a free pass.

        The provisional pivot is written once and read on every bar of the replay. Hand it back
        regardless of `now_ms` and the chart draws a dashed "?" on a bar that has not printed yet;
        worse, the annotation then depends on how far the run has got rather than on the bar being
        drawn, so the same replay stops being reproducible.
        """
        prov = Pivot(60, 6000, 120.0, PivotKind.HIGH, 5.0)
        store.set_provisional(prov)
        assert store.provisional_as_of(5999) is None, (
            "the provisional pivot was handed back before the bar of its extreme had even closed"
        )
        assert store.provisional_as_of(6000) is prov, (
            "and it must appear on the very bar of its extreme, not one later"
        )
        assert store.provisional_as_of(10**15) is prov

    def test_a_confirmed_pivot_cannot_be_parked_in_the_provisional_slot(self, store):
        """Confirmed and tentative are two channels, and one pivot cannot be in both at once.

        What comes out of `provisional_as_of` is drawn dashed and hollow and never enters a
        statistic. Park a confirmed pivot there and the same swing is drawn twice — once solid,
        once as a maybe — so the user counts one more wave than the engine has.
        """
        confirmed = Pivot(60, 6000, 120.0, PivotKind.HIGH, 5.0).confirmed_at(70, 7000)
        with pytest.raises(ValueError, match="already-confirmed"):
            store.set_provisional(confirmed)
        assert store.provisional_as_of(10**15) is None, (
            "the refused pivot was stored anyway: the tentative channel now holds a confirmed one"
        )


class TestCausalDecorator:
    def test_it_demands_now_ms(self):
        with pytest.raises(TypeError, match="now_ms"):
            @causal
            def without_a_clock(x: int) -> int:
                return x

    def test_it_marks_the_function(self):
        @causal
        def f(now_ms: int) -> int:
            return now_ms
        assert is_causal(f)
        assert not is_causal(lambda: None)

    def test_it_rejects_an_asof_from_the_future(self):
        @causal
        def f(now_ms: int, datum: AsOf[float]) -> float:
            return datum.get(now_ms)
        assert f(2000, AsOf(1.0, 1000)) == 1.0
        with pytest.raises(CausalityError, match="into the future"):
            f(500, AsOf(1.0, 1000))

    def test_the_asof_boundary_is_the_millisecond_itself(self):
        """`2000` and `500` against an availability of `1000` leave the boundary unpinned.

        The guard is an inequality, and the only interesting instants for an inequality are
        `available_at - 1`, `available_at` and `available_at + 1`. A guard that is one millisecond
        slack passes every test written with round numbers far from the edge, and one millisecond
        is an entire bar's worth of lookahead on a series that ticks in milliseconds.
        """
        @causal
        def f(now_ms: int, datum: AsOf[float]) -> float:
            return datum.value

        v = AsOf(1.0, available_at_ms=1000)
        assert f(1000, v) == 1.0, "at exactly the instant it became available, it is readable"
        assert f(1001, v) == 1.0
        with pytest.raises(CausalityError, match="into the future"):
            f(999, v)          # one ms early is still early

    def test_the_window_end_boundary_is_the_millisecond_itself(self):
        """The same edge for the `Window` branch, which no other test in this class reaches.

        `test_it_always_rejects_a_provisional_window` exercises the PASSING side of this branch
        only, so the comparison could be slack by a millisecond — or by an hour — and stay green.
        """
        @causal
        def f(now_ms: int, w) -> int:
            return len(w)

        ring = Ring(SYMBOL, TF_15M, capacity=16)
        for b in make_bars(5):
            ring.append(b)
        w = ring.window(3)
        end = w.end_closed_ts_ms

        assert f(end, w) == 3, "at the instant the window's last bar closed, it is readable"
        with pytest.raises(CausalityError, match="after now_ms"):
            f(end - 1, w)      # one ms before that close the window does not exist yet

    def test_it_rejects_an_unclosed_bar(self):
        @causal
        def f(now_ms: int, bar: Bar) -> float:
            return bar.close
        b = make_bars(1)[0]
        unclosed = Bar(**{**{k: getattr(b, k) for k in
                           ("symbol", "tf", "open_time_ms", "open", "high", "low", "close", "volume")},
                        "is_closed": False})
        with pytest.raises(CausalityError, match="UNCLOSED"):
            f(b.close_time_ms, unclosed)

    def test_a_bar_like_that_never_declares_is_closed_is_guarded_all_the_same(self):
        """The module checks by ATTRIBUTE so that, in its own words, "any future type exposing the
        same contract is protected without touching this file". `Bar` alone cannot prove that.

        `Bar` is the only shipped type with a `close_time_ms` and it always carries `is_closed`, so
        the `getattr(v, "is_closed", True)` default is dead today and alive the first time someone
        wraps a raw exchange kline, a replay row or a store-backed view. Two separate things have
        to hold for such a wrapper to be safe and neither is exercised by `Bar`: the close-time
        comparison has to apply to it at all, and the optional flag has to default the way the
        module intends. It defaults OPEN — a bar-like that says nothing is taken to be closed and
        let through — which is the permissive direction, so this test is the record of the choice:
        change the default and it goes red, and the change gets made deliberately instead of by a
        `getattr` nobody re-reads.
        """
        class DuckBar:
            """A bar-like from some future adapter: the two timestamps, and no `is_closed`."""
            open_time_ms = T0
            close_time_ms = T0 + TF_15M.ms - 1

        @causal
        def f(now_ms: int, bar) -> int:
            return bar.close_time_ms

        assert f(DuckBar.close_time_ms, DuckBar()) == DuckBar.close_time_ms, (
            "a bar-like carrying no `is_closed` was refused at the very instant it closed: every "
            "bar-like but `Bar` is treated as permanently in flight, so the duck-typed contract "
            "the module advertises protects nothing and blocks everything"
        )
        with pytest.raises(CausalityError, match="after now_ms"):
            f(DuckBar.open_time_ms, DuckBar())   # and the close-time guard still applies to it

    def test_it_rejects_a_bar_that_closes_later(self):
        @causal
        def f(now_ms: int, bar: Bar) -> float:
            return bar.close
        b = make_bars(1)[0]
        with pytest.raises(CausalityError, match="after now_ms"):
            f(b.open_time_ms, b)

    def test_it_always_rejects_a_provisional_window(self):
        """The provisional channel is structurally incapable of feeding the causal path."""
        @causal
        def f(now_ms: int, w) -> int:
            return len(w)

        ring = Ring(SYMBOL, TF_15M, capacity=16)
        bars = make_bars(5)
        for b in bars:
            ring.append(b)
        in_progress = Bar(
            symbol=SYMBOL, tf=TF_15M, open_time_ms=bars[-1].open_time_ms + TF_15M.ms,
            open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0, is_closed=False,
        )
        ring.set_provisional(in_progress)

        # The closed window goes through...
        assert f(bars[-1].close_time_ms, ring.window(3)) == 3
        # ...and the provisional one does NOT, not even with a now_ms long after it.
        with pytest.raises(CausalityError, match="PROVISIONAL"):
            f(in_progress.open_time_ms + 10**9, ring.provisional_window(3))

    def test_it_walks_into_sequences(self):
        @causal
        def f(now_ms: int, pivots: tuple) -> int:
            return len(pivots)
        future = Pivot(1, 100, 10.0, PivotKind.LOW, 1.0).confirmed_at(2, 9_000)
        with pytest.raises(CausalityError, match=r"pivots\[0\]"):
            f(1000, (future,))

    def test_it_walks_into_lists_and_into_nesting(self):
        """A guard that only understands tuples is switched off by one refactor.

        The beam is a tuple today, but anything that builds it incrementally hands over a list,
        and a per-timeframe grouping hands over a list of tuples. If the walk stopped at the
        outer container, every pivot inside would enter the causal path unchecked and NOTHING
        would raise — the backtest would simply come out better.
        """
        @causal
        def f(now_ms: int, pivots) -> int:
            return len(pivots)
        future = Pivot(1, 100, 10.0, PivotKind.LOW, 1.0).confirmed_at(2, 9_000)
        assert f(9_000, [future]) == 1, "a pivot confirmed at now_ms is knowable inside a list too"
        for container in ([future], [[future]], ([future],), [(future,)]):
            with pytest.raises(CausalityError, match=r"pivots\[0\]"):
                f(1000, container)

    def test_it_checks_arguments_passed_by_keyword(self):
        """`bar=b` must be as guarded as `b`, or the guard is opt-out by typing.

        Providers are called both ways across the registry and the replay harness. A wrapper that
        only inspects *args leaves a hole nobody can see at the call site: the exact same call,
        written with a keyword, stops being checked.
        """
        @causal
        def f(now_ms: int, bar: Bar) -> float:
            return bar.close
        b = make_bars(1)[0]
        with pytest.raises(CausalityError, match="after now_ms"):
            f(b.open_time_ms, bar=b)
        with pytest.raises(CausalityError, match="after now_ms"):
            f(now_ms=b.open_time_ms, bar=b)

    def test_it_accepts_the_bar_that_has_just_closed(self):
        """The bar closing exactly AT now_ms is the bar the analysis exists to run on.

        `now_ms` is taken from a bar close, so the boundary has to be inclusive. Were it
        exclusive, the engine could never look at the bar whose close woke it up: every feature
        would be computed one bar late, for ever, and no test of "does it repaint?" would notice
        because lagging is the safe direction.
        """
        @causal
        def f(now_ms: int, bar: Bar) -> float:
            return bar.close
        b = make_bars(1)[0]
        assert f(b.close_time_ms, b) == b.close, (
            "the bar that closed exactly at now_ms was rejected: the engine can never analyse "
            "the bar that triggered it, so every feature is shifted one bar late"
        )
        with pytest.raises(CausalityError, match="after now_ms"):
            f(b.close_time_ms - 1, b)

    @pytest.mark.parametrize("attr", ["confirmed_ts_ms", "available_at_ms", "ts_event_ms"])
    def test_every_declared_timestamp_attribute_is_honoured(self, attr):
        """The guard checks by ATTRIBUTE, not by type — and that list is the contract.

        `confirmed_ts_ms` is a pivot, `ts_event_ms` a news item or a signal, `available_at_ms`
        anything hand-stamped. Drop one name from the list and that entire family walks into the
        causal path unchecked; the boundary must be inclusive so that a fact stamped at this very
        millisecond stays usable on this bar.
        """
        @causal
        def f(now_ms: int, datum) -> int:
            return 1
        stamped = SimpleNamespace(**{attr: 1000})
        assert f(1000, stamped) == 1, (
            f"an object whose `{attr}` is exactly now_ms was refused: facts stamped on this bar "
            "close would be unusable until the next one"
        )
        with pytest.raises(CausalityError, match=attr):
            f(999, stamped)

    def test_it_refuses_to_guess_the_instant(self):
        """A causal function called without `now_ms` must fail loudly, never assume one.

        `now_ms: int | None = None` is easy to write and impossible to spot in review. If the
        wrapper filled in an instant of its own, every check downstream would be answering a
        question about a moment nobody chose — and a check that guesses the instant is not a
        check.
        """
        @causal
        def f(bar: Bar, now_ms: int | None = None) -> float:
            return bar.close
        b = make_bars(1)[0]
        with pytest.raises(TypeError, match="now_ms"):
            f(b)

    def test_it_demands_now_ms_in_integer_milliseconds(self):
        """Seconds, floats and `datetime`s all compare against int ms without complaining.

        `time.time()` returns seconds as a float; compared against millisecond timestamps every
        fact in the system looks like it is from the far future, and a float of milliseconds is a
        rounding accident sitting exactly on a bar boundary. Refuse the type instead of answering
        a question about the wrong instant.
        """
        @causal
        def f(now_ms: int) -> int:
            return now_ms
        with pytest.raises(TypeError, match="now_ms"):
            f(1_600_000_000.0)

    @pytest.mark.parametrize("n", [1, 2, 600])
    def test_the_walk_does_not_depend_on_how_long_the_sequence_is(self, n):
        """A guard that inspects short sequences and waves long ones through is no guard.

        The module advertises its own per-call cost, which is exactly the invitation to add a fast
        path for the one argument that is actually long — the pivot beam. Every other sequence
        test in this class builds a container holding a single pivot, so any predicate keyed on
        length passes them all; the beam is the only thing that ever gets big, and it is the thing
        that must not skip the check. The offender sits in the MIDDLE of the beam here, so a walk
        that only looks at the ends does not pass either.
        """
        @causal
        def f(now_ms: int, pivots) -> int:
            return len(pivots)

        now = 1_000_000
        knowable = [Pivot(i, i * 1000, 10.0, PivotKind.LOW, 1.0).confirmed_at(i, i * 1000)
                    for i in range(n)]
        assert f(now, knowable) == n, "sanity: a beam of pivots all confirmed by now_ms goes through"

        bad = n // 2
        beam = list(knowable)
        beam[bad] = Pivot(bad, bad * 1000, 10.0, PivotKind.LOW, 1.0).confirmed_at(bad, 9_000_000)
        with pytest.raises(CausalityError, match=rf"pivots\[{bad}\]"):
            f(now, beam)


def _plan() -> TradePlan:
    """A minimal, self-consistent long plan. The Decision guards care only that one exists."""
    return TradePlan(
        archetype="w2_long", direction=Direction.LONG,
        entry_lo=100.0, entry_hi=102.0, stop=98.0,
        invalidation_price=98.0, invalidation_rule="R1",
        targets=(107.0,), exit_template_id="2R_48b",
    )


def _decision(**kw) -> Decision:
    base = {"ts_ms": T0, "symbol": SYMBOL, "tf": TF_15M, "verdict": Verdict.ACTIONABLE,
            "maturity": MaturityLevel.HISTORICAL, "plan": _plan()}
    return Decision(**{**base, **kw})


class TestAnActionableDecisionHasToBeEarned:
    """`Decision` is the unit that gets journalled, and its three refusals are load-bearing.

    Each of them can be deleted or weakened without a single test going red today, because
    `engine/live.py` still returns plain dicts and the only `Decision` built anywhere in the repo
    is a NO_TRADE. That means these guards will first fire on the day the engine is wired to the
    type it was designed around — which is precisely the day nobody re-reads them.
    """

    def test_an_actionable_decision_without_a_plan_is_refused(self):
        """ACTIONABLE means "here is the trade". With no plan there is no trade, only a claim.

        Journalled, such a row says the engine told the user to act while recording nothing it
        told them to do: no entry, no stop, no invalidation. The decision can never be scored
        afterwards, so the outcome history quietly acquires holes exactly where it was most
        confident.
        """
        with pytest.raises(ValueError, match="nothing to trade"):
            _decision(plan=None)
        assert _decision().actionable, "the same decision WITH a plan must construct"

    def test_an_actionable_decision_while_catching_up_is_refused(self):
        """Replaying an outage, the entry zone describes a price that has already gone.

        The bars are arriving as fast as they can be fetched and the "current" price is minutes or
        hours old; an entry drawn on it is unfillable, and a journal entry written from it is
        scored against a fill nobody could have got. `stale` is a different flag about a different
        thing and must not be able to excuse this one — a decision that is catching up is refused
        whether or not anything else is wrong with it.
        """
        with pytest.raises(ValueError, match="CATCH_UP"):
            _decision(catching_up=True, stale=False)
        assert _decision(catching_up=False, stale=True).actionable, (
            "the refusal is about catching up on a gap, not about staleness: a stale-but-live "
            "decision is still allowed to be actionable and to say so on the card"
        )

    def test_an_actionable_decision_at_prior_maturity_is_refused(self):
        """PRIOR means the number on the card comes from a hand-written table and nothing else.

        The maturity ladder is the promise that the interface never shows more confidence than it
        has evidence for. At PRIOR the verdict is capped at WATCH; let ACTIONABLE through and a
        guess is presented with the same weight as nine years of purged walk-forward, which is the
        one thing the ladder exists to make impossible.
        """
        with pytest.raises(ValueError, match="PRIOR"):
            _decision(maturity=MaturityLevel.PRIOR)
        assert _decision(maturity=MaturityLevel.HISTORICAL).actionable, (
            "the cap is at PRIOR: a decision backed by historical evidence must still construct, "
            "or the ladder refuses every level rather than the bottom one"
        )

    def test_the_refusals_attach_to_actionable_and_to_nothing_else(self):
        """WATCH and NO_TRADE are how the engine says "not yet", and they must survive exactly the
        conditions that forbid ACTIONABLE.

        Widen the guards to every verdict and the engine can no longer report a planless, stale,
        unbacked state at all: an outage replay would abort the run at the constructor instead of
        degrading to a card that says so.
        """
        d = Decision(ts_ms=T0, symbol=SYMBOL, tf=TF_15M, verdict=Verdict.WATCH,
                     maturity=MaturityLevel.PRIOR, plan=None, catching_up=True, stale=True)
        assert not d.actionable


class TestRiskIsMeasuredTheSameInBothDirections:
    def test_a_short_reports_the_same_risk_and_reward_as_its_mirrored_long(self):
        """A short's stop sits ABOVE its entry, and `abs()` is the whole of the arithmetic.

        Without it the risk of every short is negative, `rr_at` falls into its `risk > 0`
        fallback and returns a flat 0.0 for every target. Nothing raises and nothing looks empty:
        the journal simply reports zero R:R on one side of the book, and the shorts appear to have
        no reward for months while the longs look fine. This project has already paid for that bug
        once.
        """
        long_ = TradePlan(archetype="w2_long", direction=Direction.LONG,
                          entry_lo=100.0, entry_hi=102.0, stop=98.0,
                          invalidation_price=98.0, invalidation_rule="R1",
                          targets=(107.0,), exit_template_id="2R_48b")
        short = TradePlan(archetype="w2_short", direction=Direction.SHORT,
                          entry_lo=100.0, entry_hi=102.0, stop=104.0,
                          invalidation_price=104.0, invalidation_rule="R1",
                          targets=(95.0,), exit_template_id="2R_48b")

        assert long_.risk_per_unit == pytest.approx(3.0)
        assert short.risk_per_unit == pytest.approx(long_.risk_per_unit), (
            f"the mirrored short reports risk {short.risk_per_unit} against the long's "
            f"{long_.risk_per_unit}: a stop above the entry is measured as negative distance"
        )
        assert short.rr_at(95.0) == pytest.approx(long_.rr_at(107.0)), (
            "the short's R:R does not mirror the long's; with non-positive risk `rr_at` returns "
            "its 0.0 fallback and every short is journalled as a trade with no reward"
        )
        assert short.rr_at(95.0) > 0.0


class TestADirectionCarriesItsOwnSign:
    """`Direction.sign` is what turns a side into a multiplier, and it has no caller yet.

    That is what makes it dangerous rather than harmless: the first `decide()`, sizing path or
    P&L attribution to need a sign will reach for it precisely BECAUSE it looks like the tested
    way to get one. Inverted, it raises nothing and empties nothing — every long is booked with a
    short's sign, the backtest becomes its own mirror image, and the equity curve is wrong in the
    one direction nobody investigates, because a result that improved looks like a discovery.
    """

    def test_long_is_positive_short_is_negative_and_flat_is_neither(self):
        assert Direction.LONG.sign == 1, (
            f"LONG signs itself {Direction.LONG.sign}: a long profits when price rises, so every "
            "winning long would be booked as a loss"
        )
        assert Direction.SHORT.sign == -1, (
            f"SHORT signs itself {Direction.SHORT.sign}: a short's P&L is carrying a long's sign, "
            "so a losing short is recorded as a winner"
        )
        assert Direction.FLAT.sign == 0, "FLAT takes no side: it multiplies any move to nothing"

    def test_the_two_sides_book_the_same_move_against_each_other(self):
        """The property, not the three constants: the sides are opposites, whatever the encoding.

        A helper that returned +1 for both sides would satisfy "LONG is positive" and still make
        the book directionless — every hedge would add instead of cancel.
        """
        rise = 10.0
        assert Direction.LONG.sign * rise > 0 > Direction.SHORT.sign * rise, (
            "a rise does not pay the long and cost the short: the two sides are not opposites"
        )


class TestAnEventKnowsWhenItHappenedAndWhenItWasLearned:
    """`AuxEvent` carries two timestamps for the same reason a `Pivot` does, and `latency_ms` is
    the only arithmetic in the type with a direction to get wrong.

    Its own docstring is explicit that the gap between "when it happened" and "when we found out"
    CANNOT be reconstructed after the fact — either it is recorded from the start or the news
    history is born useless for calibrating latency. `latency_ms` is the single accessor that
    turns those two fields into that quantity. Reversed, every funding, liquidation and news
    latency ever measured comes out negative; nothing raises, the rows look fine, and the person
    who eventually finds it will be debugging why the v2 lag model is inverted, long after the
    history it was fitted on was written.
    """

    def test_latency_runs_from_the_event_to_the_moment_it_was_learned(self):
        ev = AuxEvent(ts_event_ms=1_000_000, ts_ingest_ms=1_000_350, kind="funding")
        assert ev.latency_ms == 350, (
            f"an event learned 350 ms after it happened reports a latency of {ev.latency_ms}: the "
            "two timestamps are subtracted the wrong way round, so every lag in the history is "
            "the negative of the truth"
        )
        assert ev.latency_ms > 0, "nothing is ever learned before it happens"

    def test_an_event_learned_the_instant_it_happened_has_no_latency(self):
        """The zero is the fixed point of the subtraction and the only value a sign flip survives,
        so it is here to say that zero means simultaneous — not as the proof of the direction."""
        t = 1_000_000
        assert AuxEvent(ts_event_ms=t, ts_ingest_ms=t, kind="liquidation").latency_ms == 0


class TestTheClockOnlyEverMovesForward:
    def test_the_sim_clock_refuses_to_go_back_and_allows_standing_still(self):
        """`SimClock` is what makes a replay reproducible, and every one of its edges matters.

        Backwards has to blow up: during a replay it means a bar arrived out of order, and quietly
        patching that over is how a run stops being a run of the data it claims. Equality has to be
        allowed: two bars close on the same millisecond every hour (a 15m close IS an hourly
        close), so refusing it would abort a replay on the boundaries it exists to cross. And
        `advance` has to ADD — turned into a `set`, `advance(500)` after an epoch timestamp rewinds
        the clock by half a century, which the backwards guard never sees because it is not the
        one being called.
        """
        c = SimClock(1000)

        c.set(1000)
        assert c.now_ms() == 1000, (
            "re-setting the clock to the instant it already reads was refused or moved it: two "
            "bars closing on the same millisecond would abort a replay"
        )
        c.set(2000)
        assert c.now_ms() == 2000
        with pytest.raises(ValueError, match="backwards"):
            c.set(1999)
        assert c.now_ms() == 2000, "a refused set must not have moved the clock"

        c.advance(500)
        assert c.now_ms() == 2500, (
            f"advance(500) left the clock at {c.now_ms()} instead of 2500: it is assigning the "
            "argument rather than adding it, so any advance silently rewinds the clock"
        )
        c.advance(0)
        assert c.now_ms() == 2500, "advancing by zero is standing still, not an error"
        with pytest.raises(ValueError):
            c.advance(-1)

    def test_the_live_clock_reports_unix_epoch_milliseconds(self):
        """The unit slip class, in the class that claims to be the only caller of `time`.

        Everything in the system compares against integer milliseconds. Return microseconds and
        every fact ever recorded looks a thousandfold older than it is: the staleness check reports
        a healthy feed as weeks dead, and the health badge is wrong in the direction that gets the
        engine restarted rather than the direction that gets noticed.
        """
        t = LiveClock().now_ms()
        assert isinstance(t, int)
        assert abs(t - int(time.time() * 1000)) < 1000, (
            f"LiveClock().now_ms() returned {t}, which is not the wall clock in ms "
            f"(time.time() says {int(time.time() * 1000)})"
        )
        assert 1_600_000_000_000 < t < 4_000_000_000_000, (
            f"{t} is not a plausible epoch in MILLISECONDS: seconds, microseconds and nanoseconds "
            "all compare against bar timestamps without complaining"
        )


class TestAStatisticRefusesToSpeakBelowItsOwnMinimum:
    """`Stat` exists so that a number never appears without the n it came from.

    Nothing in `src/` constructs one yet, and that is the reason to pin the boundary now rather
    than later: the class is a contract waiting for its first caller, and the first caller will
    read `computable` and trust it. Both directions of the comparison matter and they fail
    differently — one refuses a statistic that had exactly enough data (a blank panel, annoying),
    the other publishes one that did not (a reassuring number out of insufficient data, which is
    the failure this class was written to make impossible).
    """

    def test_exactly_enough_data_is_enough(self):
        s = Stat(name="hit rate", value=0.55, n=10, n_required=10)
        assert s.computable is True, (
            "a statistic with exactly its required n is computable: `>=` has become `>` and the "
            "panel goes blank at the moment it should light up"
        )
        assert s.missing == 0
        assert "n=10" in s.render() and "not computable" not in s.render()

    def test_one_observation_short_is_not_enough(self):
        s = Stat(name="hit rate", value=0.55, n=9, n_required=10)
        assert s.computable is False, (
            "a statistic one observation short reports itself computable: the bar has moved off "
            "n_required and a number is being published out of insufficient data"
        )
        assert s.missing == 1
        assert "not computable" in s.render()

    def test_a_value_that_was_never_computed_is_never_computable(self):
        """Enough rows and still no number is a different failure from too few rows, and the
        property has to catch it or a `None` reaches a format string as the word 'None'."""
        s = Stat(name="hit rate", value=None, n=10_000, n_required=10)
        assert s.computable is False
        assert "not computable" in s.render()
