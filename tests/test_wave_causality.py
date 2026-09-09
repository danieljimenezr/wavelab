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

#: A 4h-grid timestamp to hang hand-built bars off. Only the spacing matters to the detector.
_T0 = 1_600_000_000_000 - (1_600_000_000_000 % TF_4H.ms)


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

    def test_confirmation_is_stamped_at_the_confirming_bar_not_at_the_extreme(self):
        """★ The stamp itself has to be right, and `as_of` cannot be the one to tell us.

        `test_no_pivot_is_visible_before_it_is_confirmed` reads `confirmed_ts_ms` off the pivot and
        then asks `as_of` whether the pivot is visible before it. `as_of` filters ON that very
        field, so the assertion holds for ANY value the detector chooses to stamp — including
        `confirmed_ts_ms = ts_ms`, which is repainting in its purest form. That test constrains
        `as_of`; it cannot constrain the detector, and it passes unchanged when the detector stamps
        every pivot at its own extreme.

        The stamp has to be pinned against something outside the pivot. Two independent facts do
        it: an extreme is never knowable on the bar it happens (the price still has to travel
        `thr_at_extreme` away from it), and the bar where that travel completes is recomputable
        from the series alone.
        """
        ts, h, l, c = _series(400, seed=5)
        pivs = detect_batch(ts, h, l, c).store.as_of(10**15)
        assert len(pivs) >= 5, "sanity: this series has to produce a few pivots to say anything"

        for p in pivs:
            assert p.confirmed_ts_ms > p.ts_ms, (
                f"the {p.kind} at ts={p.ts_ms} claims it was knowable at {p.confirmed_ts_ms}, "
                "its own bar. An extreme is only an extreme once price has moved away from it: "
                "stamping it at its own bar backdates every count that rests on it"
            )
            assert p.confirmed_idx > p.idx, (
                f"confirmed_idx={p.confirmed_idx} is not after idx={p.idx}"
            )

        # And the instant is not merely "later": it is the FIRST bar on which price had travelled
        # `thr_at_extreme` back from the extreme. Recomputed here from the bars alone.
        by_ts = {int(t): i for i, t in enumerate(ts)}
        for p in pivs:
            i0 = by_ts[p.ts_ms]
            expected = next(
                (j for j in range(i0 + 1, ts.size)
                 if ((p.price - l[j] >= p.thr_at_extreme) if p.kind is PivotKind.HIGH
                     else (h[j] - p.price >= p.thr_at_extreme))),
                None,
            )
            assert expected is not None, "the pivot confirmed, so the travel must exist"
            assert p.confirmed_ts_ms == int(ts[expected]), (
                f"the {p.kind} at ts={p.ts_ms} (thr={p.thr_at_extreme:.2f}) first saw price "
                f"travel its threshold on bar {expected} (ts={int(ts[expected])}), but it is "
                f"stamped confirmed at {p.confirmed_ts_ms}"
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


class TestTheGreyLineIsAPromise:
    """★ "The count confirms below 108,240" is the most actionable line on the screen, and the only
    reason it is worth anything is that it is EXACT.

    `confirm_price()` is published before the bar; the confirmation is decided during it. Those two
    numbers come from different places in the code — the published line from the latched
    ``thr_at_extreme``, the decision from the comparison inside ``update`` — so nothing but a test
    stops them drifting apart.

    Drift is silent and it is the whole ballgame. Recompute the threshold with today's ATR instead
    of the latched one and the line the user was shown stops being the line that fires: price cuts
    clean through it and no count invalidates, or a count invalidates while the line is still
    untouched. Either way the promise the product is sold on has quietly stopped being true, and the
    chart still looks perfect.
    """

    def test_a_confirmation_happens_exactly_when_the_published_line_is_reached(self):
        """Not one bar sooner, not one bar later, over a whole series.

        This is what pins the threshold to the bar of the extreme: any threshold recomputed later
        makes the gate disagree with the line that was already on the screen.
        """
        ts, h, l, c = _series(600, seed=8)
        z = ZigZag()
        for i in range(ts.size):
            prov = z.store.provisional_as_of(10**15)
            line = z.confirm_price()
            if prov is not None:
                assert line == pytest.approx(prov.confirm_price), (
                    f"at bar {i} the detector publishes {line} while the pivot it belongs to says "
                    f"{prov.confirm_price}: the chart and the count are showing different lines"
                )
            confirmed = z.update(int(ts[i]), float(h[i]), float(l[i]), float(c[i]))
            if prov is None:
                continue
            up = prov.kind is PivotKind.HIGH
            hi, lo = float(h[i]), float(l[i])
            # A bar that first makes a NEW extreme moves the pivot along; it never confirms it,
            # however far the other side of the same bar travelled.
            moved_on = (hi > prov.price) if up else (lo < prov.price)
            reached = (lo <= line) if up else (hi >= line)
            assert (confirmed is not None) is (reached and not moved_on), (
                f"at bar {i} the line published was {line} and the bar ran "
                f"[{float(l[i])}, {float(h[i])}] (provisional {prov.kind.name} at {prov.price}); "
                f"the detector {'confirmed' if confirmed else 'did NOT confirm'}. The price that "
                f"fires is no longer the price the user was shown"
            )

    def test_touching_the_line_is_enough(self):
        """The line is drawn AT a price, and the user is told the count confirms there.

        A flat warm-up puts the threshold on the percentage floor, so the line lands on an exact
        binary value and the bar can touch it to the tick — which is the one place a `>=` quietly
        turned into a `>` would hide, and the one place a real bar loves to land, because the level
        is where everyone's orders are.
        """
        z = ZigZag()
        for i in range(15):
            z.update(_T0 + i * TF_4H.ms, 100.0, 100.0, 100.0)
        line = z.confirm_price()
        assert line == 99.5, f"the flat warm-up should put the line on the 0.5% floor, got {line}"

        confirmed = z.update(_T0 + 15 * TF_4H.ms, 100.0, line, (100.0 + line) / 2)
        assert confirmed is not None, (
            f"the bar came down to exactly {line}, the price the grey line was drawn at, and the "
            "count did not confirm: the published line is off by a tick"
        )

    def test_touching_the_line_is_enough_on_the_LOW_side_too(self):
        """The same promise, made about a low. Half of the confirmations in the product are these.

        The gate is written twice, once per direction, so it can be tightened on one side only —
        and a one-sided tightening is invisible: a low that confirms by price rising EXACTLY to the
        published line simply stops confirming, the leg is never drawn, and nothing anywhere
        reports a count that was not made. The test above pins the high side; without this one the
        low side has no tick-exactness guard at all.

        The arithmetic is exact in binary on purpose: the flat warm-up puts the threshold on the
        0.5% floor of 100.0, so the low's published line lands on 100.0 to the bit and a bar can
        touch it without a rounding error deciding the outcome.
        """
        z = ZigZag()
        for i in range(15):
            z.update(_T0 + i * TF_4H.ms, 100.0, 100.0, 100.0)
        # Down through the high's line, closing back at 100.0 so the LOW's own threshold is the
        # same 0.5 floor and its line lands on an exact binary value.
        assert z.update(_T0 + 15 * TF_4H.ms, 100.0, 99.5, 100.0) is not None, (
            "sanity: the bar came down through the high's line and the high did not confirm"
        )
        prov = z.store.provisional_as_of(10**15)
        assert prov is not None and prov.kind is PivotKind.LOW and prov.price == 99.5
        line = z.confirm_price()
        assert line == 100.0, f"the low's line should sit on the 0.5 floor above 99.5, got {line}"

        confirmed = z.update(_T0 + 16 * TF_4H.ms, line, 99.5, 99.75)
        assert confirmed is not None, (
            f"the bar rose to exactly {line}, the price the grey line was drawn at, and the low "
            "did not confirm: the promise is kept on highs and broken on lows"
        )
        assert confirmed.kind is PivotKind.LOW


class TestAPivotIsPricedAtTheExtremeItMarks:
    """★ A pivot's price is the price the market actually printed at its own bar.

    Everything downstream is a distance measured from that number: the leg lengths R2 compares, the
    Fibonacci ratios the guideline score ranks on, the ``confirm_price`` line, and the invalidation
    the card is sold on. Get the number wrong and the count is still perfectly well-formed — the
    legs join up, the rules pass, the chart looks right — it is simply anchored to a price that
    never happened.

    The place this goes wrong is the direction flip. ``update`` decides ``up_px``/``down_px`` once
    at the top, from ``on_close``; the two lines that seed the NEW extreme after a confirmation sit
    forty lines further down, and re-deriving the price there from ``close`` reads as a tidy-up. It
    is not observable through the confirmation machinery, because the wrong price is then used
    consistently by both the published line and the gate that fires on it.
    """

    def test_a_flip_down_seeds_the_new_low_from_the_bar_s_wick(self):
        z = ZigZag()
        for i in range(15):
            z.update(_T0 + i * TF_4H.ms, 100.0, 100.0, 100.0)
        # The bar that confirms the high: it trades down to 95 and closes back up at 99.4, the
        # ordinary shape of a flush. The low of that bar is the new pivot, not its close.
        z.update(_T0 + 15 * TF_4H.ms, 100.0, 95.0, 99.4)

        prov = z.store.provisional_as_of(10**15)
        assert prov is not None and prov.kind is PivotKind.LOW
        assert prov.price == 95.0, (
            f"the flip bar traded down to 95.0 and closed at 99.4, and the new low was placed at "
            f"{prov.price}: a 4.6% error in the pivot's own price, on the bar the count turns. "
            "Every leg measured from here is wrong by that much"
        )
        assert prov.idx == 15 and prov.ts_ms == _T0 + 15 * TF_4H.ms

    def test_a_flip_up_seeds_the_new_high_from_the_bar_s_wick(self):
        """The mirror. A fault seeded on one branch only would leave the count biased in one
        direction, which is the hardest kind to see on a chart."""
        z = ZigZag()
        for i in range(15):
            z.update(_T0 + i * TF_4H.ms, 100.0, 100.0, 100.0)
        z.update(_T0 + 15 * TF_4H.ms, 100.0, 95.0, 99.4)          # high confirms, now hunting a low
        assert z.update(_T0 + 16 * TF_4H.ms, 105.0, 95.0, 95.6) is not None, (
            "sanity: the low should have confirmed on a bar that ran 10 points above it"
        )

        prov = z.store.provisional_as_of(10**15)
        assert prov is not None and prov.kind is PivotKind.HIGH
        assert prov.price == 105.0, (
            f"the flip bar ran up to 105.0 and closed at 95.6, and the new high was placed at "
            f"{prov.price}"
        )

    @pytest.mark.parametrize("on_close", [False, True])
    def test_every_flip_seeds_the_new_extreme_by_the_rule_the_knob_selected(self, on_close):
        """The general statement — checked at the moment the extreme is SEEDED, on every flip.

        This test used to inspect only the pivots that survived to the end of the run, and that is
        almost no test at all: a wrongly-seeded extreme is overwritten by the very next bar that
        makes a real one, so a mispriced seed reaches a finished pivot only when the flip bar
        itself turns out to be the pivot AND nothing between the flip and the confirmation trades
        past its close. Measured over 50 seeds of this fixture, the survivor-only form caught a
        seeding fault on 5 of them — one seed in ten, and one bad pivot out of ~57 when it fired.
        It read as the strongest test in the class and was by an order of magnitude the weakest.
        Checking the provisional at each confirmation exercises every flip in the series instead
        of the one or two that happen to survive.

        The ``on_close`` parametrisation is the other half, and it is the half that was missing.
        ``update`` picks ``up_px``/``down_px`` from the knob once at the top; the two lines that
        seed the new extreme after a confirmation sit forty lines below, and re-deriving the price
        there from the raw wick reads as a tidy-up. With the default ``on_close=False`` the wick
        IS the selected price, so nothing in the class could see the difference. Switch the knob
        on — the operator's explicit instruction that wicks are noise on this asset — and a
        detector that ignores it anchors a third of its pivots to the very prices they excluded,
        while the badge on the panel goes on saying close-confirmed.
        """
        ts, h, l, c = _series(400, seed=6)
        z = ZigZag(ZigZagConfig(on_close=on_close))
        flips = 0
        for i in range(ts.size):
            got = z.update(int(ts[i]), float(h[i]), float(l[i]), float(c[i]))
            if got is None:
                continue
            flips += 1
            prov = z.store.provisional_as_of(int(ts[i]))
            assert prov is not None and prov.idx == i and prov.ts_ms == int(ts[i]), (
                f"the flip on bar {i} did not seed a new provisional pivot on that same bar"
            )
            if on_close:
                expected = c[i]
            else:
                expected = l[i] if prov.kind is PivotKind.LOW else h[i]
            assert prov.price == expected, (
                f"on_close={on_close}: the flip on bar {i} seeded its new {prov.kind.name} at "
                f"{prov.price}, but that bar ran [{l[i]}, {h[i]}] and closed at {c[i]}, so the "
                f"price selected by the configuration is {expected}. Every leg, ratio and "
                "invalidation measured from this pivot is anchored to a price the operator's "
                "own setting says to ignore"
            )
        assert flips >= 20, (
            f"only {flips} flips in 400 bars: the fixture stopped exercising the property and "
            "this test now proves almost nothing"
        )

    def test_every_pivot_sits_on_its_own_bar_s_high_or_low(self):
        """The cheap end-state sweep, kept as a second and weaker net.

        It says nothing the per-flip test above does not say better; it is here because it is the
        statement a reader of the chart would make ("no pivot is drawn at a price its bar did not
        trade"), and it would catch a fault introduced anywhere other than the seeding lines.
        """
        ts, h, l, c = _series(400, seed=6)
        z = detect_batch(ts, h, l, c)
        pivs = z.store.as_of(10**15)
        assert len(pivs) >= 5, "sanity: the series has to produce pivots to say anything"

        prov = z.store.provisional_as_of(10**15)
        for p in (*pivs, prov):
            if p is None:
                continue
            expected = h[p.idx] if p.kind is PivotKind.HIGH else l[p.idx]
            assert p.price == expected, (
                f"the {p.kind.name} at bar {p.idx} is priced {p.price}, but that bar ran "
                f"[{l[p.idx]}, {h[p.idx]}] and closed at {c[p.idx]}: the pivot is drawn at a price "
                "the market did not make on the bar it is drawn on"
            )


class TestTheThresholdIsWorthWhatItSays:
    """The threshold is what separates structure from noise, and both of its terms earn their keep.

    ``max(k * ATR, min_pct * close)``: the ATR term makes ``k`` dimensionless so the same setting
    means the same thing on BTC, EURUSD and AAPL; the percentage term is the floor that stops a
    market which has not moved from manufacturing structure out of its own spread.
    """

    @staticmethod
    def _flat_market(n: int = 300) -> tuple:
        """A quiet range: ±0.05% wiggles around 30,000, an order of magnitude under the floor."""
        close = 30_000.0 * (1 + 0.0005 * np.sin(np.arange(n) / 3.0))
        ts = np.array([_T0 + i * TF_4H.ms for i in range(n)], dtype=np.int64)
        return ts, close * 1.0002, close * 0.9998, close

    def test_a_market_that_has_not_moved_produces_no_structure(self):
        """Without the percentage floor a quiet tape confirms a pivot every few bars, and the count
        becomes an elaborate reading of the spread. Every downstream number — the archetype, the
        invalidation, the hit rate — is then computed over noise, and none of it looks wrong."""
        ts, h, l, c = self._flat_market()
        z = detect_batch(ts, h, l, c)
        assert z.n_confirmed == 0, (
            f"{z.n_confirmed} pivots confirmed in a range that never moved 0.1%, on a detector "
            f"whose floor is 0.5% of price: the noise floor is not holding"
        )

    def test_but_a_real_move_still_gets_through(self):
        """The floor must silence noise, not the market. Without this the test above would pass
        just as happily on a detector that never confirms anything at all."""
        _, _, _, c = self._flat_market()
        leg_up = c[-1] * np.linspace(1.0, 1.08, 150)
        leg_down = leg_up[-1] * np.linspace(1.0, 0.92, 150)
        c = np.concatenate([c, leg_up, leg_down])
        ts = np.array([_T0 + i * TF_4H.ms for i in range(c.size)], dtype=np.int64)
        z = detect_batch(ts, c * 1.0002, c * 0.9998, c)
        assert z.n_confirmed >= 1, (
            "an 8% rally followed by an 8% selloff confirmed no pivot at all: the threshold is not "
            "a floor any more, it is a wall"
        )

    def test_the_floor_is_a_percentage_of_the_CLOSE_not_of_the_wick(self):
        """``min_pct * close`` — the close, on the bar of the extreme.

        The two candidates differ by the wick-to-close distance, which is small, and that is why
        the difference is worth pinning rather than shrugging at: it is small and it is
        SYSTEMATICALLY DIRECTIONAL. A high latches off a price above its close and a low off a
        price below it, so in a range-bound market lows would confirm a shade more readily than
        highs, the count would acquire a downward bias, and no reading of the chart would ever
        attribute it to this line.

        The warm-up here is flat with a deliberately wide wick, so the ATR term (0.06) is nowhere
        near the floor and the latched value is exactly 0.5% of one of two prices: 100.0 (the
        close, correct) or 100.02 (the high).
        """
        z = ZigZag()
        for i in range(15):
            z.update(_T0 + i * TF_4H.ms, 100.02, 99.98, 100.0)

        prov = z.store.provisional_as_of(10**15)
        assert prov is not None and prov.price == 100.02, "sanity: the extreme is the bar's high"
        assert prov.thr_at_extreme == 0.5, (
            f"the frozen threshold came out {prov.thr_at_extreme!r}; on a tape this quiet the "
            "floor binds and it is 0.5% of the CLOSE (100.0 -> 0.5), not of the extreme price "
            "the wick reached (100.02 -> 0.5001)"
        )
        assert z.confirm_price() == 100.02 - 0.5

    def test_k_is_a_knob_and_turning_it_changes_something(self):
        """``k`` is the one setting that gets tuned per asset. If the sensitivity it is supposed to
        control is really driven by something else, every asset silently runs at one sensitivity and
        the multi-asset claim is decoration."""
        ts, h, l, c = _series(800, seed=3)
        counts = [detect_batch(ts, h, l, c, ZigZagConfig(k_atr=k)).n_confirmed
                  for k in (0.5, 1.5, 4.0)]
        assert counts[0] > counts[1] > counts[2], (
            f"k=0.5/1.5/4.0 confirmed {counts} pivots: a bigger threshold has to mean strictly "
            f"fewer pivots on the same series, or k is not the knob it is documented to be"
        )


class TestConfirmOnCloseIsAPerAssetDecision:
    """``on_close`` is wired to per-asset configuration (``r3_on``), so it has to actually do
    something. Liquidation hunts are endemic in BTC: a wick spears a level, everything stops out,
    and price closes back where it was. Counting those wicks invalidates a large fraction of
    otherwise sound structures — which is exactly why the knob exists.

    A knob that reads the config and then ignores it is worse than no knob: the operator believes
    the asset is configured one way and the engine is running the other.
    """

    @staticmethod
    def _hunt(on_close: bool) -> list[bool]:
        """Flat warm-up, then: a bar speared by a 10% lower wick that closes flat; a bar that
        actually closes down through the line; a bar speared by an upper wick that closes flat."""
        z = ZigZag(ZigZagConfig(on_close=on_close))
        for i in range(15):
            z.update(_T0 + i * TF_4H.ms, 100.0, 100.0, 100.0)
        return [z.update(_T0 + 15 * TF_4H.ms, 100.0, 90.0, 99.9) is not None,
                z.update(_T0 + 16 * TF_4H.ms, 100.0, 99.9, 99.4) is not None,
                z.update(_T0 + 17 * TF_4H.ms, 110.0, 99.4, 99.5) is not None]

    def test_on_close_ignores_the_wick_and_obeys_the_close(self):
        speared_down, closed_down, speared_up = self._hunt(on_close=True)
        assert not speared_down, (
            "a 10% lower wick that closed back at the highs confirmed the high anyway: with "
            "on_close the count must be decided by the CLOSE, which is the whole point of the knob"
        )
        assert closed_down, (
            "price closed through the line and nothing confirmed: on_close must still confirm on "
            "closes, otherwise the asset simply never produces a count"
        )
        assert not speared_up, (
            "an upper wick confirmed the low: on_close is being ignored on the up side even though "
            "it is respected on the down side"
        )

    def test_with_wicks_the_same_hunt_does_confirm(self):
        """The counterpart: the bars are chosen so the two settings genuinely disagree. Without
        this the test above would pass on a detector that had simply stopped confirming."""
        speared_down, _, _ = self._hunt(on_close=False)
        assert speared_down, (
            "on the wick setting a 10% spear through the line did not confirm: the two settings no "
            "longer differ, so the test above is proving nothing"
        )


class TestTheExtremeBelongsToTheBarWhereItHappened:
    """A tie is not a new extreme, and the threshold is latched "until there is a new extreme".

    A run of equal highs is a double top — an ordinary, frequent shape. If an equal high moved the
    pivot, it would be drawn at the LAST touch instead of the first, every leg and every timestamp
    downstream would shift with it, and — the part that actually breaks the architecture — the
    frozen threshold would be silently re-latched against a newer ATR on a bar where the extreme
    never changed. The freeze is what makes confirmation monotone; anything that quietly re-latches
    it takes the monotonicity with it.
    """

    def test_a_run_of_equal_highs_keeps_the_first_one(self):
        z = ZigZag()
        for i in range(15):
            z.update(_T0 + i * TF_4H.ms, 100.0, 100.0, 100.0)
        first = z.store.provisional_as_of(10**15)
        assert first is not None and first.idx == 14

        for i in range(15, 19):                      # four more bars at exactly the same high
            z.update(_T0 + i * TF_4H.ms, 100.0, 100.0, 100.0)
        tied = z.store.provisional_as_of(10**15)
        assert tied is not None
        assert (tied.idx, tied.ts_ms) == (first.idx, first.ts_ms), (
            f"four bars tied the high and the pivot walked from bar {first.idx} to bar {tied.idx}: "
            "a double top is drawn where the high FIRST happened"
        )
        assert tied.thr_at_extreme == first.thr_at_extreme, (
            f"the frozen threshold was re-latched ({first.thr_at_extreme} -> "
            f"{tied.thr_at_extreme}) on bars where the extreme never moved: the freeze is what "
            "makes confirmation monotone"
        )

        confirmed = z.update(_T0 + 19 * TF_4H.ms, 100.0, 99.4, 99.4)
        assert confirmed is not None and confirmed.idx == 14, (
            f"the confirmed high was recorded at bar {confirmed and confirmed.idx} instead of 14, "
            "where the high actually happened"
        )

    def test_a_run_of_equal_lows_keeps_the_first_one(self):
        z = ZigZag()
        for i in range(15):
            z.update(_T0 + i * TF_4H.ms, 100.0, 100.0, 100.0)
        z.update(_T0 + 15 * TF_4H.ms, 100.0, 99.4, 99.4)     # confirms the high, turns down
        low = z.store.provisional_as_of(10**15)
        assert low is not None and low.kind is PivotKind.LOW

        for i in range(16, 20):                              # four bars tying the low
            z.update(_T0 + i * TF_4H.ms, 99.4, 99.4, 99.4)
        tied = z.store.provisional_as_of(10**15)
        assert tied is not None
        assert (tied.idx, tied.ts_ms, tied.thr_at_extreme) == (low.idx, low.ts_ms,
                                                               low.thr_at_extreme), (
            f"the low walked from bar {low.idx} to bar {tied.idx} across bars that only TIED it "
            "(and re-latched its frozen threshold on the way)"
        )


class TestWhatIsDrawnSaysWhatItKnows:
    """The provisional leg legitimately moves as price moves. Drawing it like a confirmed one
    asserts a certainty nobody has, and that is the exact dishonesty this design exists to remove.

    ``legs_as_of`` is what the server hands the chart, so these are the wire contract, not cosmetics.
    """

    def test_only_the_tentative_leg_lacks_a_confirmation_time(self):
        ts, h, l, c = _series(400, seed=6)
        z = detect_batch(ts, h, l, c)
        legs = z.legs_as_of(10**15)
        confirmed_ts = {p.ts_ms for p in z.store.as_of(10**15)}

        tentative = [x for x in legs if x["tentative"]]
        assert len(tentative) == 1, f"expected exactly one tentative leg, got {len(tentative)}"
        assert legs[-1] is tentative[0], "the tentative leg has to be the LAST one on the chart"

        for leg in legs:
            assert leg["tentative"] is (leg["ts"] not in confirmed_ts), (
                f"the leg at ts={leg['ts']} is marked tentative={leg['tentative']} but the store "
                f"{'has' if leg['ts'] in confirmed_ts else 'does not have'} it confirmed: the "
                "dashed/solid distinction is what tells the user which parts can still move"
            )
            assert (leg["confirmed_ts"] is None) is leg["tentative"], (
                f"the leg at ts={leg['ts']} carries confirmed_ts={leg['confirmed_ts']} while "
                f"tentative={leg['tentative']}: a leg that can still move must not carry a "
                "confirmation time, or it will be read as settled"
            )

    def test_the_confirmation_time_on_the_wire_is_the_CONFIRMATION_not_the_extreme(self):
        """★ The test above checks only whether ``confirmed_ts`` is None. Any non-None value
        satisfies it — including the pivot's own extreme timestamp, which is the located/confirmed
        conflation this whole module exists to prevent, reintroduced at the single point where the
        data leaves the process.

        ``legs_as_of`` is what the server hands the chart, so this field is the wire contract. A
        consumer replaying "what was knowable at t" off a backdated ``confirmed_ts`` would place
        every leg at its extreme instead of at the bar it became knowable — a median of a few bars
        of free lookahead, hundreds in a trend, and it makes the replay look better rather than
        raising anything.
        """
        ts, h, l, c = _series(400, seed=6)
        z = detect_batch(ts, h, l, c)
        by_ts = {p.ts_ms: p for p in z.store.as_of(10**15)}
        legs = z.legs_as_of(10**15)
        checked = 0

        for leg in legs:
            if leg["tentative"]:
                continue
            p = by_ts[leg["ts"]]
            assert leg["confirmed_ts"] == p.confirmed_ts_ms, (
                f"the leg at ts={leg['ts']} goes out on the wire stamped "
                f"{leg['confirmed_ts']} while the store says it became knowable at "
                f"{p.confirmed_ts_ms}: the chart and the count disagree about WHEN"
            )
            assert leg["confirmed_ts"] > leg["ts"], (
                f"the leg at ts={leg['ts']} claims it was confirmed at {leg['confirmed_ts']}, its "
                "own extreme or earlier. An extreme is never knowable on its own bar, so on the "
                "wire the confirmation time is always strictly later than the vertex"
            )
            checked += 1

        assert checked >= 5, f"only {checked} confirmed legs: the fixture proves too little"

    def test_a_consumer_that_asks_for_settled_legs_only_is_given_settled_legs_only(self):
        """``include_provisional=False`` is a public parameter with no caller in ``src/`` today.

        That is precisely why it needs pinning now: an argument nothing exercises is an argument
        nothing protects, and the first caller will be a route sending a confirmed-only view to a
        consumer that asked for settled data. Ignore the flag there and the tentative leg goes out
        stripped of the one field — ``tentative`` — that would have let the consumer notice, so it
        is read as a settled vertex that will then silently move.
        """
        ts, h, l, c = _series(400, seed=6)
        z = detect_batch(ts, h, l, c)
        full = z.legs_as_of(10**15)
        settled = z.legs_as_of(10**15, include_provisional=False)

        assert [x for x in full if x["tentative"]], "sanity: the fixture must produce a live leg"
        assert not [x for x in settled if x["tentative"]], (
            "a caller that asked to suppress the provisional leg was sent it anyway, and it is "
            "the leg that moves"
        )
        assert settled == full[:-1], (
            "suppressing the provisional leg changed something other than dropping the last one: "
            f"{len(settled)} legs against {len(full)}"
        )

    def test_a_windowed_chart_still_anchors_its_first_leg(self):
        """``since_ms`` is how the server sends only the visible window. A leg needs two vertices,
        so the window has to reach one pivot further back than it shows: without that the first leg
        on the user's screen starts from nowhere at the left edge."""
        ts, h, l, c = _series(400, seed=6)
        z = detect_batch(ts, h, l, c)
        pivots = z.store.as_of(10**15)
        assert len(pivots) >= 4, "need a few pivots for this to mean anything"

        since = pivots[2].ts_ms
        legs = z.legs_as_of(10**15, since_ms=since)
        assert legs[0]["ts"] < since, (
            f"the window starting at {since} begins at {legs[0]['ts']}, inside itself: the first "
            "visible leg has no origin to be drawn from"
        )
        assert {p.ts_ms for p in pivots if p.ts_ms >= since} <= {x["ts"] for x in legs}, (
            "the window dropped pivots that fall inside it"
        )


class TestReplayIsTheSameCodeAsLive:
    """``detect_batch`` exists for warm-up and for tests, never to go faster.

    The moment replay and the live path differ by so much as one bar of ATR, every backtest is
    measuring a detector nobody trades, and the difference shows up as a BETTER backtest, not as an
    error. That is why this is asserted rather than assumed.
    """

    def test_replaying_an_array_equals_feeding_the_bars_one_by_one(self):
        ts, h, l, c = _series(500, seed=12)
        batch = detect_batch(ts, h, l, c)

        live = ZigZag()
        for i in range(ts.size):
            live.update(int(ts[i]), float(h[i]), float(l[i]), float(c[i]))

        assert batch.store.as_of(10**15) == live.store.as_of(10**15), (
            "warm-up by replay and the live loop disagree about the confirmed pivots: the backtest "
            "is measuring a detector that is not the one running"
        )
        assert batch.store.provisional_as_of(10**15) == live.store.provisional_as_of(10**15)
        assert batch.confirm_price() == live.confirm_price()
        assert batch.atr == live.atr, "replay and live disagree about the ATR itself"
        assert batch.n_confirmed == len(batch.store.as_of(10**15)) == live.n_confirmed, (
            f"the counter says {batch.n_confirmed} pivots and the store holds "
            f"{len(batch.store.as_of(10**15))}: the number on the status line is not the count"
        )


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

    def test_the_true_range_includes_the_GAP_from_the_previous_close(self):
        """True range is a range measured from the previous CLOSE, not the bar's own high-low.

        The TA-Lib oracle above cannot see this: its fixture is a random walk where every bar opens
        at the previous close, so high-low and the true range are the same number on every single
        bar. The market is not like that. After an exchange outage — and this project keeps a
        ``is_gap`` flag precisely because they happen — trading resumes far from where it stopped,
        and a bar can sit entirely above the previous close.

        Ignoring the gap understates volatility at the exact moment volatility was highest. The
        ZigZag threshold shrinks with it, so the detector manufactures pivots in the most violent
        part of the tape and then reports them as ordinary structure.
        """
        a = WilderATR(1)                    # period 1: the first value IS the true range, exactly
        assert a.update(100.0, 99.0, 99.5) is None, "the opening bar has no previous close"
        tr = a.update(120.0, 119.0, 119.5)  # the whole bar sits 19.5 above the previous close
        assert tr == pytest.approx(20.5), (
            f"true range came out {tr}; the bar is only 1.0 tall but it gapped 19.5 away from the "
            "previous close, so the range that was actually traded through is 20.5"
        )

    def test_the_gap_is_measured_in_BOTH_directions(self):
        """The test above gaps UP, and an up-gap is carried entirely by the ``high - prev_close``
        term. Drop the absolute value from the OTHER term and it stays green, because that term is
        never the one that decides an up-gap.

        A down-gap is where the low-side term is the one that matters, and a down-gap is not the
        exotic case: it is the crash, the liquidation cascade and the bar that prints when a feed
        comes back after an outage. Without the absolute value the true range there collapses to
        the bar's own height, so the ATR — and with it the ZigZag threshold — is understated
        precisely and only when the market falls. The detector then manufactures structure in the
        most violent part of the tape and reports it as ordinary, and it does so in one direction
        only, which is the shape of bias nobody spots on a chart.
        """
        a = WilderATR(1)                    # period 1: the first value IS the true range, exactly
        assert a.update(100.0, 99.0, 99.5) is None
        tr = a.update(80.5, 79.5, 80.0)     # the whole bar sits 19.0 BELOW the previous close
        assert tr == pytest.approx(20.0), (
            f"true range came out {tr}; the bar is 1.0 tall and gapped DOWN 19.0 from the previous "
            "close of 99.5, so the range actually traded through is 20.0. A down-gap has to count "
            "for exactly as much as an up-gap"
        )

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
