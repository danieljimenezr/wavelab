"""The hard rules against hand-labelled fixtures over real BTC.

An engine that does not REJECT is as useless as one that does not accept. Six of the ten cases are
negative, and four of those are real market sequences that look like perfect five-wave impulses to
the eye and are not.
"""

from __future__ import annotations

import json
from itertools import permutations, product
from pathlib import Path
from typing import ClassVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wavelab.core.types import Direction, RuleVerdict
from wavelab.waves.rules import (
    ImpulseState,
    RuleSet,
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


class TestTheHardRulesAreStrict:
    """Elliott's three hard rules are all inequalities, and every one of them is STRICT.

    w2 must stay above the origin, w3 must go BEYOND the end of w1, w4 must STAY OUT of w1's
    territory. Landing exactly on the limit is a break, not a pass — "equal" is not "beyond".

    Loosening any of them to `>=` costs real money in a way that is invisible in a backtest built
    with the same loose rule. A w2 accepted at exactly 100% retracement puts the invalidation at the
    entry price, so the card offers a trade whose R is zero. A w4 that comes to rest exactly on P1
    has overlapped, and overlap is the commonest reason a five-wave reading has to be thrown away:
    accepting it keeps a dead count on the screen, still emitting entries.

    Equality is also the one place random floats never land, which is why these are written out by
    hand instead of drawn.
    """

    #: Mirror point for the SHORT reflection. Every value here is exact in binary, so a
    #: disagreement between the two directions is the rules disagreeing, never the arithmetic.
    MIRROR: ClassVar[float] = 300.0

    @pytest.mark.parametrize("rule,pts", [
        ("R1",  [100.0, 130.0, 100.0]),                 # w2 retraces EXACTLY 100% of w1
        ("R2b", [100.0, 130.0, 115.0, 130.0]),          # w3 stops EXACTLY at the end of w1
        ("R3",  [100.0, 130.0, 115.0, 160.0, 130.0]),   # w4 comes to rest EXACTLY on P1
    ])
    @pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
    def test_landing_exactly_on_the_limit_already_breaks_the_rule(self, rule, pts, direction):
        p = pts if direction is Direction.LONG else [self.MIRROR - x for x in pts]
        r = check_impulse(p, direction)
        assert r.broken == (rule,), (
            f"{direction.name} {p} sits EXACTLY on {rule}'s limit and the engine broke "
            f"{list(r.broken)} instead of just ({rule},): the hard rules are strict, "
            f"'equal' is not 'beyond'"
        )
        assert not r.valid, f"{rule} accepted a count that only TOUCHES its limit"

    @pytest.mark.parametrize("name,pts", [
        # w1 = w3 = 30, w5 = 20
        ("w3 ties with w1", [100.0, 130.0, 115.0, 145.0, 135.0, 155.0]),
        # w3 = w5 = 25, w1 = 30
        ("w3 ties with w5", [100.0, 130.0, 115.0, 140.0, 132.0, 157.0]),
    ])
    @pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
    def test_a_wave_3_that_TIES_for_shortest_is_not_the_shortest(self, name, pts, direction):
        """R2 forbids w3 being THE shortest of 1/3/5. A tie is not that.

        Tightening the comparison to `<=` rejects perfectly legal counts, and a rejection here is
        silent: the structure simply stops being offered and nobody sees the trade that was not
        taken. Equal lengths are common in the real market precisely because Fibonacci relations
        between waves are what the whole method is built on.
        """
        p = pts if direction is Direction.LONG else [self.MIRROR - x for x in pts]
        r = check_impulse(p, direction)
        assert r.valid, (
            f"{name} ({direction.name}) was rejected, breaking {list(r.broken)}: R2 says w3 is "
            f"never THE SHORTEST, and a wave tied with another is not the shortest"
        )

    @pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
    def test_a_fifth_that_only_REACHES_the_third_is_already_truncated(self, direction):
        """Truncation is w5 failing to EXCEED the end of w3, and merely reaching it is failing.

        The flag is the whole point of the case: a truncated fifth is exhaustion, so it is legal
        (rejecting it throws away the counts you most want near a top) but it must be labelled. An
        unflagged truncation reads on the card as an ordinary impulse with the trend intact, at the
        exact moment the trend is ending.
        """
        pts = [100.0, 130.0, 115.0, 160.0, 140.0, 160.0]   # P5 lands exactly on P3
        p = pts if direction is Direction.LONG else [self.MIRROR - x for x in pts]
        r = check_impulse(p, direction)
        assert r.valid, f"a truncated fifth is LEGAL and was rejected: broke {list(r.broken)}"
        assert r.truncated, (
            f"{direction.name} w5 ended exactly ON w3's extreme and was not flagged as truncated: "
            "reaching is not exceeding, and this is the shape of a top"
        )


class TestTheInvalidationIsWhereTheCountActuallyDies:
    """★ The card's biggest number has to be the price at which THIS structure becomes false.

    The design's best property is that the invalidation comes out of the rule engine instead of
    being assembled downstream, so it cannot drift away from the rule that justifies it. This
    checks the thing that property is FOR, at every partial state: the invalidation price is
    exactly the frontier for the last vertex — one tick on the live side and the count still
    stands, one tick past it (or exactly on it) and an evaluable rule breaks.

    Drift in either direction is expensive and silent. A stop set looser than the frontier holds a
    position that the engine already knows is dead. A stop set tighter than it is stopped out of a
    count that is still valid, and the card's R:R and position size were computed from that number.
    """

    #: Prefixes of one hand-checked long impulse. COMPLETE is excluded on purpose: once w5 exists
    #: the structure is finished, and its invalidation governs the move that follows rather than
    #: the placement of the last vertex.
    CASES: ClassVar[dict[ImpulseState, list[float]]] = {
        ImpulseState.AT_2: [100.0, 130.0, 115.0],
        ImpulseState.AT_3: [100.0, 130.0, 115.0, 160.0],
        ImpulseState.AT_4: [100.0, 130.0, 115.0, 160.0, 140.0],
    }

    @pytest.mark.parametrize("state", list(CASES))
    @pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
    def test_the_last_vertex_is_alive_on_one_side_of_it_and_dead_on_the_other(self, state,
                                                                             direction):
        pts = self.CASES[state]
        if direction is Direction.SHORT:
            pts = [300.0 - x for x in pts]
        sign = 1 if direction is Direction.LONG else -1
        tick = 0.5

        r = check_impulse(pts, direction)
        assert r.valid, f"the {state.value} fixture must start out valid; broke {list(r.broken)}"
        assert r.state is state
        price, rule = invalidation_for(state, pts, direction)
        assert price == r.invalidation_price, (
            f"{state.value}: check_impulse says the count dies at {r.invalidation_price} and "
            f"invalidation_for says {price} — the stop and its reason have drifted apart"
        )

        alive = [*pts[:-1], price + sign * tick]
        assert check_impulse(alive, direction).valid, (
            f"{state.value} {direction.name}: the last vertex a tick on the LIVE side of the "
            f"invalidation ({price}) was rejected, breaking "
            f"{list(check_impulse(alive, direction).broken)}. The stop is tighter than the rules: "
            f"it would close a count that is still standing"
        )

        for label, last in (("exactly on", price), ("a tick past", price - sign * tick)):
            dead = check_impulse([*pts[:-1], last], direction)
            assert not dead.valid, (
                f"{state.value} {direction.name}: the last vertex {label} the invalidation "
                f"({price}, from {rule}) and every rule still holds. The card would keep a dead "
                f"count on screen with a stop that never triggers"
            )

    def test_a_rule_nobody_can_evaluate_yet_invalidates_nothing(self):
        """R2 degrades to a guideline while w5 is missing, and 100% of entries are taken before w5
        exists. If an unevaluable verdict could veto, every partial count would be rejected and the
        engine would offer no entries at all — the annotation toy this design exists not to be."""
        rs = RuleSet(
            state=ImpulseState.AT_2,
            direction=Direction.LONG,
            verdicts=(RuleVerdict("R1", True, 100.0, "holds"),
                      RuleVerdict("R2", False, 130.0, "cannot be known yet", evaluable=False)),
            invalidation_price=100.0,
            invalidation_rule="R1",
            archetype="w2",
        )
        assert rs.valid, "a NON-EVALUABLE verdict vetoed the count: it may not invalidate anything"
        assert rs.broken == (), (
            f"a non-evaluable rule was reported as broken ({list(rs.broken)}): the card would name "
            "a rule that was never actually tested"
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
    """The rules are written once with s=+1/-1. A bear is a bull reflected.

    This used to be a Hypothesis test over random floats, and it was a much weaker guard than it
    looked: a deliberately introduced asymmetry in R1 showed up in 0.17% of draws and needed some
    4,000 examples to surface, while the test ran 150. It passed because it rarely looked at
    anything interesting, not because the rules were symmetric.

    Random floats are the wrong instrument here. What the rules actually compare is the ORDERING of
    six points plus three lengths, so the structure space is small and can be walked exhaustively.
    And a strict-vs-non-strict slip — `<` where `<=` belongs — is invisible everywhere EXCEPT at
    equality, which is precisely where independent random floats never land.

    The mirror is exact by construction: every value below is representable in binary and so is
    `M - x`, so a disagreement is the rules disagreeing and never the arithmetic.
    """

    @staticmethod
    def _asymmetries(structures, mirror_about):
        out = []
        for pts in structures:
            pts = list(pts)
            a = check_impulse(pts, Direction.LONG)
            b = check_impulse([mirror_about - x for x in pts], Direction.SHORT)
            broken_a = sorted(v.rule for v in a.verdicts if not v.ok)
            broken_b = sorted(v.rule for v in b.verdicts if not v.ok)
            if a.valid != b.valid or broken_a != broken_b:
                out.append(f"{pts}: long valid={a.valid} broke {broken_a} | "
                           f"short valid={b.valid} broke {broken_b}")
        return out

    def test_every_ordering_of_six_points_mirrors(self):
        """All 720 permutations: exhaustive over the strict-ordering patterns the rules can see."""
        bad = self._asymmetries(permutations([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]), 70.0)
        assert not bad, "the rules are not direction-symmetric:\n" + "\n".join(bad[:10])

    @pytest.mark.parametrize("name,pts", [
        ("R1  P2 == P0",     [100.0, 130.0, 100.0, 160.0, 140.0, 175.0]),
        ("R2b P3 == P1",     [100.0, 130.0, 115.0, 130.0, 125.0, 145.0]),
        ("R3  P4 == P1",     [100.0, 130.0, 115.0, 160.0, 130.0, 175.0]),
        ("R2  len3 == len1", [100.0, 130.0, 115.0, 145.0, 135.0, 190.0]),
        ("R2  len3 == len5", [100.0, 160.0, 130.0, 160.0, 140.0, 170.0]),
        ("all rules tied",   [100.0, 130.0, 100.0, 130.0, 130.0, 160.0]),
        ("degenerate flat",  [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
    ])
    def test_the_rule_boundaries_mirror(self, name, pts):
        """Each rule sitting exactly ON its own boundary. This is where a `<`/`<=` slip lives, and
        it is the one place independent random floats will never put you."""
        bad = self._asymmetries([pts], 200.0)
        assert not bad, f"{name} is not direction-symmetric:\n" + "\n".join(bad)

    def test_ties_everywhere_mirror(self):
        """4,096 structures over a four-value grid, so equal points are the norm rather than a
        once-in-a-million draw. Exhaustive, deterministic, and about 20x more boundary contact than
        the Hypothesis version managed in 150 tries."""
        bad = self._asymmetries(product([100.0, 110.0, 120.0, 130.0], repeat=6), 230.0)
        assert not bad, (f"{len(bad)} of 4096 tied structures are not direction-symmetric:\n"
                         + "\n".join(bad[:10]))

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
