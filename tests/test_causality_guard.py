"""Causality has to be impossible to violate, not merely unlikely to be violated."""

from __future__ import annotations

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
