"""The property four architectural claims hang from.

`as_of(t)` must be a STRICT PREFIX of `as_of(t+1)`, always. If it were not:
  - the history of counts would stop being append-only,
  - the «as it stood at bar t» snapshot would stop being free,
  - the replay harness would go from O(n) to O(n²),
  - and a count the user had already seen would vanish with no invalidation event.

That last consequence is the serious one: it is exactly the dishonesty the design exists to
eliminate.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wavelab.core.types import Pivot, PivotKind
from wavelab.waves.store import PivotStore


@st.composite
def pivot_history(draw, max_n: int = 40):
    """Valid pivots: alternating, with monotone idx and confirmation, threshold frozen."""
    n = draw(st.integers(min_value=1, max_value=max_n))
    idx, conf_ts, out = 0, 0, []
    for i in range(n):
        idx += draw(st.integers(min_value=1, max_value=50))
        lag = draw(st.integers(min_value=1, max_value=200))
        conf_ts += draw(st.integers(min_value=1, max_value=10_000))
        price = draw(st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False))
        thr = draw(st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False))
        kind = PivotKind.HIGH if i % 2 == 0 else PivotKind.LOW
        out.append(Pivot(idx, conf_ts, price, kind, thr).confirmed_at(idx + lag, conf_ts))
    return out


@given(pivots=pivot_history(), ts=st.lists(st.integers(0, 500_000), min_size=1, max_size=30))
@settings(max_examples=200, deadline=None)
def test_as_of_always_returns_a_prefix(pivots, ts):
    s = PivotStore()
    for p in pivots:
        s.append_confirmed(p)
    full = s.as_of(10**12)
    for t in sorted(ts):
        partial = s.as_of(t)
        assert partial == full[: len(partial)]


@given(pivots=pivot_history())
@settings(max_examples=200, deadline=None)
def test_as_of_is_monotone_non_decreasing(pivots):
    s = PivotStore()
    for p in pivots:
        s.append_confirmed(p)
    stamps = sorted({p.confirmed_ts_ms for p in pivots} | {0})
    previous = ()
    for t in stamps:
        current = s.as_of(t)
        assert len(current) >= len(previous)
        assert current[: len(previous)] == previous, "the confirmed history can only GROW"
        previous = current


@given(pivots=pivot_history())
@settings(max_examples=100, deadline=None)
def test_no_pivot_is_visible_before_it_is_confirmed(pivots):
    s = PivotStore()
    for p in pivots:
        s.append_confirmed(p)
    for p in pivots:
        visible = s.as_of(p.confirmed_ts_ms - 1)
        assert p not in visible, (
            f"pivot located at ts={p.ts_ms} is visible before its confirmation "
            f"at {p.confirmed_ts_ms}: that is exactly what repainting is"
        )
        assert p in s.as_of(p.confirmed_ts_ms)


class TestTheWritesThatKeepItAPrefix:
    """The prefix property above is only ever as strong as what `append_confirmed` refuses.

    Every property in this file is stated over histories that were built one legal write at a time.
    The three guards below are what make "legal" mean something.

    The confirmation-TIME guard was believed to be pinned from both sides and was not: what the
    suite held was the equal case and a value below EVERY confirmation on record. The region where
    the guard is actually decided — above the first confirmation and below the last, which is what
    a replay catching up after an outage produces — had no test, and an index slip there is
    invisible from outside because the store still hands back a tidy prefix. It just contains the
    future.
    """

    @staticmethod
    def _store() -> PivotStore:
        s = PivotStore()
        s.append_confirmed(Pivot(10, 1000, 100.0, PivotKind.HIGH, 5.0).confirmed_at(20, 2000))
        s.append_confirmed(Pivot(30, 3000, 90.0, PivotKind.LOW, 5.0).confirmed_at(40, 4000))
        return s

    def test_a_second_pivot_at_the_same_position_is_refused(self):
        """`idx` is STRICTLY increasing, and the equal case is the one the existing test misses.

        A repeat of the last `idx` passes the confirmation-time check, so `as_of` goes on returning
        a tidy prefix and nothing looks wrong from outside. Downstream every wave is measured as
        `idx_b - idx_a`: two pivots sharing an index give a leg of zero length, so R2's "wave 3 is
        never the shortest" is decided against a wave with no extent and the guideline ratios
        divide by nothing at all.
        """
        store = self._store()
        same_idx = Pivot(30, 4500, 95.0, PivotKind.HIGH, 5.0).confirmed_at(50, 5000)
        with pytest.raises(ValueError, match="positional order"):
            store.append_confirmed(same_idx)
        assert store.n_as_of(10**15) == 2, (
            "a pivot repeating the last idx was accepted: the beam now contains a leg of zero "
            "length and every ratio measured across it is a division by zero away"
        )

    def test_a_confirmation_interleaved_into_the_history_is_refused(self):
        """The confirmation instant is compared against the LAST one on record, not the first.

        `_conf_ts` is the array `as_of` bisects, so its only job is to stay sorted. Comparing
        against `_conf_ts[0]` — a one-character slip on a parallel list — still refuses anything
        older than the whole history, which is the only case the existing tests reach, and quietly
        accepts a confirmation that lands in the MIDDLE of it. A `bisect_right` over an unsorted
        array is undefined, and the store goes on returning prefixes, so nothing looks wrong.

        The write that produces this is not exotic: a replay catching up after a feed outage
        appends confirmations interleaved with the ones already stored. `idx=50` here is ahead of
        everything on record on purpose, so the positional guard cannot fire and mask the result —
        this test has to be decided by the confirmation-time guard alone.
        """
        store = self._store()                       # confirmed at 2000 and 4000
        interleaved = Pivot(50, 3500, 95.0, PivotKind.HIGH, 5.0).confirmed_at(45, 3000)
        with pytest.raises(ValueError, match="monotone"):
            store.append_confirmed(interleaved)
        assert store.n_as_of(10**15) == 2, (
            "a confirmation older than the last one on record was accepted: `_conf_ts` is no "
            "longer sorted, and every `as_of` from here on is a bisect over an unsorted array"
        )

    def test_the_store_never_hands_out_a_confirmation_from_the_future(self):
        """What the guard above actually buys, said as a consequence rather than as a refusal.

        With `_conf_ts` out of order, `bisect_right` does not merely lose a pivot — it hands back a
        prefix containing a pivot confirmed LATER than the instant that was asked for. Here that is
        seven seconds of pure lookahead served as settled history, and it passes every prefix
        property in this file, because it IS a prefix. A backtest reading it would be trading a
        swing hours before it existed and would simply look profitable.
        """
        store = PivotStore()
        store.append_confirmed(Pivot(10, 500, 100.0, PivotKind.HIGH, 5.0).confirmed_at(15, 1000))
        store.append_confirmed(Pivot(20, 8000, 90.0, PivotKind.LOW, 5.0).confirmed_at(25, 9000))
        for idx, conf_ts in ((30, 2000), (40, 3000)):
            with pytest.raises(ValueError, match="monotone"):
                store.append_confirmed(
                    Pivot(idx, conf_ts - 500, 95.0, PivotKind.HIGH, 5.0).confirmed_at(idx, conf_ts)
                )

        assert [p.confirmed_ts_ms for p in store.as_of(2000)] == [1000], (
            "asked what was knowable at 2000, the store answered with a pivot confirmed at 9000: "
            "that is the future served as settled history, and every prefix assertion in this "
            "file still passes while it happens"
        )

    def test_an_unconfirmed_pivot_cannot_be_appended_to_the_confirmed_history(self):
        """The mirror of the guard on `set_provisional`, and the only one of the pair untested.

        `append_confirmed` and `set_provisional` are two channels and a pivot belongs to exactly
        one of them. A caller that crossed them would be writing a pivot with no confirmation
        instant into a list that `as_of` slices by confirmation instant. The refusal has to be
        LOUD: the quiet alternative — dropping it — is the worst available outcome, because the
        wave simply never appears in the count and no error is raised anywhere to say so.
        """
        store = self._store()
        tentative = Pivot(60, 6000, 120.0, PivotKind.HIGH, 5.0)
        assert not tentative.is_confirmed
        with pytest.raises(ValueError, match="unconfirmed"):
            store.append_confirmed(tentative)
        assert store.n_as_of(10**15) == 2, (
            "the unconfirmed pivot was swallowed instead of refused: a caller mixing the two "
            "channels loses a wave from the count with nothing raised to say it happened"
        )
