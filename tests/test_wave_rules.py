"""The hard rules against hand-labelled fixtures over real BTC.

An engine that does not REJECT is as useless as one that does not accept. Six of the ten cases are
negative, and four of those are real market sequences that look like perfect five-wave impulses to
the eye and are not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from wavelab.core.types import Direction
from wavelab.waves.rules import (
    ImpulseState,
    check_impulse,
    fib_projection,
    fib_retracement,
    invalidation_for,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "impulses_btc.json").read_text())
CASES = FIXTURES["cases"]
DIR = {"long": Direction.LONG, "short": Direction.SHORT}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_the_fixtures_give_the_labelled_verdict(case):
    r = check_impulse(case["points"], DIR[case["direction"]])
    assert r.valid is case["expect_valid"], (
        f"{case['id']}: expected valid={case['expect_valid']} and got {r.valid} "
        f"(broken: {r.broken}). Note from the label: {case.get('notes','')[:120]}"
    )
    if not case["expect_valid"]:
        assert set(case["breaks"]) == set(r.broken), (
            f"{case['id']}: expected it to break {case['breaks']} and it broke {list(r.broken)}"
        )
    if "state" in case:
        assert r.state.value == case["state"]
    if case.get("expect_truncated"):
        assert r.truncated, "a truncated fifth must be FLAGGED, never rejected"


class TestPartialNonTerminals:
    """★ 100% of entries live inside INCOMPLETE impulses. An engine that only evaluates finished
    structures is an annotation toy."""

    #: ClassVar, not an instance attribute: these are hand-labelled vertices, shared read-only by
    #: every test in the class. Do not move them into __init__ and do not touch a price — the whole
    #: point is that the same six numbers produce a different state at each prefix length.
    P: ClassVar[list[float]] = [100.0, 130.0, 115.0, 160.0, 140.0, 155.0]

    @pytest.mark.parametrize("n,state,archetype", [
        (3, ImpulseState.AT_2, "w2"),
        (4, ImpulseState.AT_3, None),
        (5, ImpulseState.AT_4, "w4"),
        (6, ImpulseState.COMPLETE, "abc"),
    ])
    def test_each_state_offers_its_archetype(self, n, state, archetype):
        r = check_impulse(self.P[:n])
        assert r.state is state
        assert r.archetype == archetype, (
            "you do not enter inside wave 3, you manage it: AT_3 must offer no archetype"
        )

    @pytest.mark.parametrize("n,evaluable", [
        (3, {"R1"}),
        (4, {"R1", "R2b"}),
        (5, {"R1", "R2b", "R3"}),
        (6, {"R1", "R2b", "R3", "R2"}),
    ])
    def test_only_the_rules_that_can_be_evaluated_are(self, n, evaluable):
        r = check_impulse(self.P[:n])
        assert {v.rule for v in r.verdicts if v.evaluable} == evaluable

    def test_r2_is_only_guidance_while_w5_is_missing(self):
        """"Wave 3 is never the shortest" cannot be asserted without w5. Faking a verdict about
        structure that does not exist yet is inventing certainty."""
        # w1=40, w3=25 (short, but R2b and R3 hold: w3 exceeds w1 and w4 does not invade it).
        partial = check_impulse([100.0, 140.0, 120.0, 145.0, 142.0])
        r2 = next(v for v in partial.verdicts if v.rule == "R2")
        assert not r2.evaluable
        assert partial.valid, (
            "at impulse@4 every evaluable rule holds; a non-evaluable R2 does not invalidate"
        )

        # With w5=58, wave 3 (25) becomes the SHORTEST and R2 does invalidate.
        complete = check_impulse([100.0, 140.0, 120.0, 145.0, 142.0, 200.0])
        r2 = next(v for v in complete.verdicts if v.rule == "R2")
        assert r2.evaluable and not r2.ok
        assert not complete.valid, "with w5 present, R2 does invalidate"

    def test_the_invalidation_moves_from_P0_to_P1(self):
        """Inside w3 the stop is P0 by R1; from w4 onwards it moves to P1, which is tighter.
        That is why the wave-4 entry has a worse R:R even though it intuitively feels safer."""
        p0, _ = invalidation_for(ImpulseState.AT_2, self.P[:3])
        p1, _ = invalidation_for(ImpulseState.AT_4, self.P[:5])
        assert p0 == self.P[0] and p1 == self.P[1]
        assert abs(self.P[4] - p1) < abs(self.P[2] - p0), "the w4 stop must be tighter"

    def test_the_invalidation_carries_the_name_of_its_rule(self):
        for n in (3, 4, 5, 6):
            r = check_impulse(self.P[:n])
            assert r.invalidation_rule.startswith("R"), (
                "the biggest number on the card must carry the rule that produces it"
            )


class TestOverlap:
    """Prechter's exception: in futures and commodities wave 4 is allowed to overlap."""

    # w4=125 < w1=130 -> it overlaps
    P: ClassVar[list[float]] = [100.0, 130.0, 115.0, 160.0, 125.0, 175.0]

    def test_it_is_rejected_by_default(self):
        assert not check_impulse(self.P).valid

    def test_with_allow_overlap_it_is_accepted(self):
        r = check_impulse(self.P, allow_overlap=True)
        assert r.valid
        assert "Prechter" in next(v for v in r.verdicts if v.rule == "R3").detail


class TestSymmetry:
    """The rules are written once with s=+1/-1. A bear is a bull reflected."""

    @given(pts=st.lists(st.floats(50, 500, allow_nan=False), min_size=6, max_size=6, unique=True))
    @settings(max_examples=150, deadline=None)
    def test_reflecting_the_prices_gives_the_same_verdict(self, pts):
        mirror = [1000.0 - p for p in pts]
        # `1000.0 - p` is NOT an exact operation, and that is a property of binary floating point,
        # not of the rules. One ULP at 50 is 7.1e-15 but at 950 it is 1.1e-13, so two prices a few
        # ULPs apart down at the bottom of the range collapse onto a single float once mirrored:
        # 50.0 and 50.00000000000001 both come back as exactly 950.0. When that happens `mirror`
        # is a DIFFERENT six-point structure rather than a reflection of this one, and the two
        # verdicts disagreeing says nothing about whether the rules are symmetric — which is the
        # only thing this test is for. So require the mirror to preserve every pairwise ordering
        # before comparing verdicts. Real prices sit on a tick grid where this never arises; the
        # `unique=True` above is not enough, because uniqueness is not preserved by the mirror.
        assume(all((pts[j] > pts[i]) == (mirror[j] < mirror[i])
                   for i in range(len(pts)) for j in range(i + 1, len(pts))))
        a = check_impulse(pts, Direction.LONG)
        b = check_impulse(mirror, Direction.SHORT)
        assert a.valid == b.valid
        assert set(a.broken) == set(b.broken)

    @given(f=st.floats(0.1, 50.0, allow_nan=False))
    @settings(max_examples=50, deadline=None)
    def test_scaling_does_not_change_the_verdict(self, f):
        base = [100.0, 130.0, 115.0, 160.0, 140.0, 155.0]
        assert check_impulse(base).valid == check_impulse([p * f for p in base]).valid


class TestFibonacci:
    def test_the_retracement_falls_between_the_extremes(self):
        levels = fib_retracement(100.0, 200.0, (0.382, 0.5, 0.618, 0.786))
        assert all(100.0 < n < 200.0 for n in levels)
        assert levels == sorted(levels, reverse=True)

    def test_the_golden_pocket_is_where_it_should_be(self):
        """The 0.618-0.65 golden pocket of a 100->200 leg falls at 135-138.2, not at 161.8.
        A retracement is measured FROM the end of the leg, backwards."""
        lo, hi = fib_retracement(100.0, 200.0, (0.65, 0.618))
        assert lo == pytest.approx(135.0)
        assert hi == pytest.approx(138.2)
        assert 130.0 < lo < hi < 140.0

    def test_the_extensions_project_forward(self):
        targets = fib_projection(150.0, 100.0, 200.0)
        assert targets[0] == pytest.approx(250.0)     # 1.000
        assert targets[2] == pytest.approx(311.8)     # 1.618


class TestGuards:
    @pytest.mark.parametrize("n", [0, 1, 2, 7])
    def test_invalid_number_of_vertices(self, n):
        with pytest.raises(ValueError):
            check_impulse([100.0 + i for i in range(n)])


class TestCostFilter:
    """A signal whose stop sits too close is rejected BEFORE anything gets modelled: if fees eat a
    fifth of the risk, no plausible statistical edge survives."""

    def _hyp(self, points):
        from wavelab.core.types import Pivot, PivotKind
        from wavelab.waves.matcher import Hypothesis
        from wavelab.waves.rules import ImpulseState, check_impulse
        r = check_impulse(list(points), Direction.LONG)
        pivs = tuple(Pivot(i, 1_600_000_000_000 + i * 14_400_000, p,
                           PivotKind.LOW if i % 2 == 0 else PivotKind.HIGH, 1.0)
                     .confirmed_at(i + 1, 1_600_000_000_000 + (i + 1) * 14_400_000)
                     for i, p in enumerate(points))
        return Hypothesis(ImpulseState.AT_2, Direction.LONG, pivs, tuple(points), 0.8, {},
                          r.invalidation_price, r.invalidation_rule, "w2")

    def test_it_rejects_a_stop_that_is_too_close(self):
        from wavelab.waves.projection import PlanConfig, build_plan
        # Low volatility (ATR 100 on 79,000 = 0.13%): the stop is roomy in ATR terms —2.2— but in
        # absolute terms it sits so close that fees take half an R.
        h = self._hyp([79000.0, 79400.0, 79200.0])
        r = build_plan(h, 79150.0, 100.0, PlanConfig())
        assert not r.viable
        assert "fees" in r.reasons[0]
        assert r.cost_r > 0.20

    def test_it_accepts_a_roomy_stop(self):
        from wavelab.waves.projection import PlanConfig, build_plan
        h = self._hyp([70000.0, 80000.0, 74000.0])
        r = build_plan(h, 75000.0, 900.0, PlanConfig())
        assert r.viable, r.reasons
        assert r.cost_r < 0.05

    def test_the_cost_is_shown_even_when_it_is_rejected(self):
        """A 'no' without numbers is an opinion; with numbers it is an argument you can dispute."""
        from wavelab.waves.projection import PlanConfig, build_plan
        # Rejected by the NOISE FLOOR, not by cost: the arithmetic has to come along all the same.
        h = self._hyp([79000.0, 79300.0, 79100.0])
        r = build_plan(h, 79150.0, 400.0, PlanConfig())
        assert not r.viable and "noise floor" in r.reasons[0]
        assert r.cost_r > 0 and r.rr_t2 > 0 and r.p_required > 0, (
            "a rejection has to bring its numbers too"
        )
