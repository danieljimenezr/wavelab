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
