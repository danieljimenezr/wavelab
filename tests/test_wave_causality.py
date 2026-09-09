"""The M3 tests. If these fail, everything built on top of them is a lie.

Repainting **fails upward**: a detector that peeks at the future produces a prettier backtest, not
an error. No suite of expected values catches it, because the expected values were computed with
the same lookahead. The only thing that catches it is comparing what the system said AT THE TIME
with what it says afterwards.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wavelab.core.timeframes import TF_4H
from wavelab.core.types import PivotKind
from wavelab.waves.pivots import WilderATR, ZigZag, ZigZagConfig, detect_batch

from .conftest import make_bars


def _series(n: int, seed: int = 3, tf=TF_4H):
    bars = make_bars(n, tf=tf, seed=seed)
    return (np.array([b.open_time_ms for b in bars], dtype=np.int64),
            np.array([b.high for b in bars]),
            np.array([b.low for b in bars]),
            np.array([b.close for b in bars]))


class TestCausality:
    """The property everything else hangs from."""

    def test_as_of_is_identical_to_recomputing_from_the_past_alone(self):
        """★ THE M3 test.

        `pivots_as_of(t)` computed at instant t must be IDENTICAL to the one obtained by replaying
        only the bars up to t. If they differed, the detector would be using information from after
        t to decide what was there at t.
        """
        ts, h, l, c = _series(400)
        full = detect_batch(ts, h, l, c)

        for k in (60, 120, 200, 300, 399):
            partial = detect_batch(ts[:k + 1], h[:k + 1], l[:k + 1], c[:k + 1])
            a = partial.store.as_of(int(ts[k]))
            b = full.store.as_of(int(ts[k]))
            assert a == b, (
                f"at bar {k} the detector sees {len(a)} pivots replaying only the past "
                f"and {len(b)} knowing the future: it is looking ahead"
            )

    def test_the_history_can_only_grow(self):
        """A count the user has already seen cannot vanish without an invalidation event."""
        ts, h, l, c = _series(500, seed=11)
        z = ZigZag()
        previous: tuple = ()
        for i in range(ts.size):
            z.update(int(ts[i]), float(h[i]), float(l[i]), float(c[i]))
            current = z.store.as_of(int(ts[i]))
            assert current[: len(previous)] == previous, (
                f"at bar {i} the confirmed history CHANGED instead of growing"
            )
            assert len(current) >= len(previous)
            previous = current

    def test_no_pivot_is_visible_before_it_is_confirmed(self):
        ts, h, l, c = _series(400, seed=5)
        z = detect_batch(ts, h, l, c)
        for p in z.store.as_of(10**15):
            assert p not in z.store.as_of(p.confirmed_ts_ms - 1), (
                f"the pivot located at {p.ts_ms} is visible before being confirmed "
                f"at {p.confirmed_ts_ms}: that IS repainting"
            )
            assert p in z.store.as_of(p.confirmed_ts_ms)

    def test_the_confirmation_lag_is_variable_and_sometimes_enormous(self):
        """Documents why «shift by N bars» does not work as a mitigation."""
        ts, h, l, c = _series(1500, seed=2)
        z = detect_batch(ts, h, l, c)
        pivs = z.store.as_of(10**15)
        lags = [(p.confirmed_ts_ms - p.ts_ms) // TF_4H.ms for p in pivs]
        assert len(set(lags)) > 3, "the lag should vary a lot, not be nearly constant"
        assert max(lags) > 3 * (sorted(lags)[len(lags) // 2] or 1), (
            "there should be a heavy right tail: some pivots take an age to confirm"
        )


class TestFrozenThreshold:
    """The threshold is frozen at the extreme's bar. That is where the monotonicity comes from."""

    def test_the_threshold_is_recorded_in_the_pivot(self):
        ts, h, l, c = _series(300)
        z = detect_batch(ts, h, l, c)
        for p in z.store.as_of(10**15):
            assert p.thr_at_extreme > 0
            expected = (p.price - p.thr_at_extreme if p.kind is PivotKind.HIGH
                        else p.price + p.thr_at_extreme)
            assert p.confirm_price == pytest.approx(expected)

    def test_a_volatility_explosion_unconfirms_nothing(self):
        """Without freezing the threshold, an ATR that blows up could invalidate already-confirmed
        pivots and the chart would change its mind about the past, without raising anything."""
        ts, h, l, c = _series(300, seed=9)
        z = ZigZag()
        for i in range(200):
            z.update(int(ts[i]), float(h[i]), float(l[i]), float(c[i]))
        before = z.store.as_of(int(ts[199]))

        # A hundred bars of brutal volatility: the ATR multiplies.
        base = float(c[199])
        for j in range(100):
            t = int(ts[199]) + (j + 1) * TF_4H.ms
            z.update(t, base * 1.30, base * 0.70, base)
        after = z.store.as_of(int(ts[199]))
        assert after == before, (
            "a volatility explosion has rewritten the past: the threshold was not frozen"
        )


class TestConfirmationPrice:
    def test_it_is_on_the_far_side_of_the_extreme(self):
        ts, h, l, c = _series(300)
        z = detect_batch(ts, h, l, c)
        prov = z.store.provisional_as_of(10**15)
        cp = z.confirm_price()
        assert prov is not None and cp is not None
        if prov.kind is PivotKind.HIGH:
            assert cp < prov.price, "a provisional high confirms by falling BELOW it"
        else:
            assert cp > prov.price, "a provisional low confirms by rising ABOVE it"

    def test_there_is_at_most_one_provisional(self):
        ts, h, l, c = _series(200)
        z = detect_batch(ts, h, l, c)
        legs = z.legs_as_of(10**15)
        assert sum(1 for x in legs if x["tentative"]) <= 1


class TestATR:
    def test_it_matches_talib(self):
        """An independent oracle: if our ATR drifts, the whole threshold drifts with it."""
        talib = pytest.importorskip("talib")
        ts, h, l, c = _series(300)
        ref = talib.ATR(h, l, c, timeperiod=14)
        a = WilderATR(14)
        ours = [a.update(float(h[i]), float(l[i]), float(c[i])) for i in range(ts.size)]
        for i in range(20, ts.size):
            if not np.isnan(ref[i]):
                assert ours[i] == pytest.approx(ref[i], rel=1e-9), f"divergence at {i}"

    def test_it_is_not_ready_before_the_period(self):
        """First ATR(14) at index 14, not at 13: the opening bar contributes no true range
        because it has no previous close. That is what TA-Lib does."""
        a = WilderATR(14)
        for i in range(14):
            assert a.update(10 + i, 9 + i, 9.5 + i) is None, f"ready too early at {i}"
        assert a.update(24, 23, 23.5) is not None


class TestScaleInvariance:
    """The threshold is absolute and scaled by volatility, so k is dimensionless: the same k=1.5
    means the same thing on BTC, EURUSD and AAPL. That is multi-asset made real."""

    @given(factor=st.floats(min_value=0.01, max_value=100.0, allow_nan=False))
    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_scaling_the_prices_does_not_change_the_pivots(self, factor):
        ts, h, l, c = _series(250, seed=4)
        a = detect_batch(ts, h, l, c, ZigZagConfig(min_pct=0.0))  # min_pct=0 -> pure ATR
        b = detect_batch(ts, h * factor, l * factor, c * factor, ZigZagConfig(min_pct=0.0))
        pa = [(p.idx, int(p.kind)) for p in a.store.as_of(10**15)]
        pb = [(p.idx, int(p.kind)) for p in b.store.as_of(10**15)]
        assert pa == pb, f"scaling by x{factor} changed the pivots: k is not dimensionless"
