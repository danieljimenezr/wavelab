"""Causality has to be impossible to violate, not merely unlikely to be violated."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wavelab.core.causality import AsOf, CausalityError, causal, is_causal
from wavelab.core.ring import Ring
from wavelab.core.timeframes import TF_15M
from wavelab.core.types import Bar, Pivot, PivotKind
from wavelab.waves.store import PivotStore

from .conftest import SYMBOL, make_bars


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
