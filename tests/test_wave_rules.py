"""The hard rules against hand-labelled fixtures over real BTC.

An engine that does not REJECT is as useless as one that does not accept. Six of the ten cases are
negative, and four of those are real market sequences that look like perfect five-wave impulses to
the eye and are not.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from itertools import permutations, product
from pathlib import Path
from typing import ClassVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wavelab.core.types import Direction, Pivot, PivotKind, RuleVerdict
from wavelab.waves.matcher import (
    Hypothesis,
    MatcherConfig,
    match_impulses,
    score_guidelines,
)
from wavelab.waves.projection import PlanConfig, build_plan
from wavelab.waves.rules import (
    ARCHETYPES,
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

#: A 4h-grid instant to hang hand-built pivots off. Only the spacing matters to the matcher.
_T0 = 1_600_000_000_000
_STEP = 14_400_000
_STATES = {3: ImpulseState.AT_2, 4: ImpulseState.AT_3,
           5: ImpulseState.AT_4, 6: ImpulseState.COMPLETE}


def _pivots(prices, direction: Direction = Direction.LONG) -> tuple[Pivot, ...]:
    """Confirmed, alternating pivots at these prices. A bullish impulse starts from a low."""
    first_is_low = direction is Direction.LONG
    return tuple(
        Pivot(i * 10, _T0 + i * 10 * _STEP, px,
              PivotKind.LOW if (i % 2 == 0) is first_is_low else PivotKind.HIGH, 1.0)
        .confirmed_at(i * 10 + 1, _T0 + (i * 10 + 1) * _STEP)
        for i, px in enumerate(prices)
    )


def _hypothesis(points, direction: Direction = Direction.LONG) -> Hypothesis:
    """The hypothesis the matcher would build from these vertices, with the rule engine's own
    invalidation. Built here rather than through ``match_impulses`` so a plan can be asked for by
    hand, without a pivot sequence that happens to put the structure near the edge."""
    r = check_impulse(list(points), direction)
    assert r.valid, f"fixture {points} is not a legal count: {list(r.broken)}"
    state = _STATES[len(points)]
    return Hypothesis(state, direction, _pivots(points, direction), tuple(points), 0.8, {},
                      r.invalidation_price, r.invalidation_rule, ARCHETYPES[state])


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


class TestEachRuleCarriesTheVertexItIsWrittenAgainst:
    """The module's stated reason for its own shape: every rule emits ITS OWN invalidation price.

    The design note at the top of ``rules.py`` says the biggest number on the card comes out of the
    rule engine rather than being assembled downstream, so it cannot drift away from the rule that
    justifies it. That claim is about ``RuleVerdict.invalidation_price``, one per rule — and the
    suite only ever reads the aggregate. Each verdict's price can be given a different vertex and
    every existing test stays green, which means the property the design is built around is
    currently prose.

    The correspondence is not arbitrary: R1 is an inequality about P2 against P0, so its frontier
    is P0; R2b and R3 are both written against P1, which is why the stop tightens once wave 4
    exists. A verdict carrying some other vertex would name a price at which its own rule does not
    change its mind.
    """

    P: ClassVar[list[float]] = [100.0, 130.0, 115.0, 160.0, 140.0, 155.0]

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    @pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
    def test_each_verdict_names_its_own_frontier(self, n, direction):
        pts = self.P[:n] if direction is Direction.LONG else [300.0 - x for x in self.P[:n]]
        r = check_impulse(pts, direction)
        price = {v.rule: v.invalidation_price for v in r.verdicts}

        assert price["R1"] == pts[0], (
            f"R1 says wave 2 cannot retrace 100% of wave 1, which is the statement 'P2 stays past "
            f"P0'. Its verdict carries {price['R1']} instead of P0={pts[0]}"
        )
        for rule in ("R2b", "R3", "R2"):
            assert price[rule] == pts[1], (
                f"{rule} is written against the end of wave 1, so its frontier is P1={pts[1]}; "
                f"its verdict carries {price[rule]}"
            )

    def test_the_percentage_a_verdict_quotes_is_the_retracement_it_is_about(self):
        """The class above pins every verdict's PRICE and never looks at what it SAYS.

        `detail` is the per-rule explanation — the half of the card that turns a number into an
        argument the user can push back on. Measured from the wrong vertex, R1's line reads "w2
        retraces 75.0% of w1" for a retracement of 25%: not a rounding error but a different
        quantity, and the one the operator uses to judge whether the count is shallow or deep.
        Nothing reads `RuleVerdict.detail` in `src/` today, which is why the fault survived every
        other test in this file; it becomes a wrong number on the screen the day anyone renders
        the per-rule explanations, which is the only reason the field exists.

        The vertices are chosen so the two candidate measurements cannot coincide: |P2-P1| is 25
        and |P2-P0| is 75, against a wave 1 of 100.
        """
        r = check_impulse([100.0, 200.0, 175.0])
        r1 = next(v for v in r.verdicts if v.rule == "R1")
        assert "25.0%" in r1.detail, (
            f"R1 explains itself as {r1.detail!r}. Wave 1 ran 100 points and wave 2 gave back 25 "
            "of them, so the retracement is 25.0%; any other number is measured off a leg that is "
            "not wave 2"
        )
        assert r1.ok and "past 100%" not in r1.detail

        # And the mirror at the other end of the range, so a sign or an `abs()` cannot hide here:
        # wave 2 gives back 95 of the same 100 points. Still legal, and deep.
        deep = next(v for v in check_impulse([100.0, 200.0, 105.0]).verdicts if v.rule == "R1")
        assert "95.0%" in deep.detail, f"R1 explains a 95% retracement as {deep.detail!r}"

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_the_headline_invalidation_is_the_one_its_own_rule_published(self, n):
        """The aggregate is not computed a second time somewhere else: it IS one of the verdicts.

        `invalidation_rule` is the name printed next to the stop on the card. If the price beside
        it came from anywhere but that rule's own verdict, the card would be naming a reason that
        does not produce the number it is standing next to — which is precisely the drift this
        design exists to make impossible.
        """
        r = check_impulse(self.P[:n])
        named = r.invalidation_rule.split()[0]
        by_rule = {v.rule: v.invalidation_price for v in r.verdicts}
        assert named in by_rule, f"the card names a rule that emitted no verdict: {named!r}"
        assert r.invalidation_price == by_rule[named], (
            f"the card would show 'invalidated at {r.invalidation_price}, by {named}' while "
            f"{named} itself says it dies at {by_rule[named]}"
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

    def test_every_level_it_promises_is_a_level_it_returns(self):
        """The test above reads two of the four levels, so the other two could be anything.

        ``EXTENSIONS`` is the published contract of this function: four ratios, in order. A level
        that quietly changes value moves a target on the card, and a target is the numerator of the
        R:R the user sizes on — the one number in the whole chain nobody double-checks by eye.
        """
        assert fib_projection(150.0, 100.0, 200.0) == pytest.approx([250.0, 277.2, 311.8, 411.8])

    def test_a_leg_that_points_DOWN_projects_DOWNWARDS(self):
        """A projection is SIGNED: it continues the leg it was measured from.

        Every existing case here calls it with ``leg_a < leg_b``, so the whole short side is
        unexercised — and taking the length in absolute value is the natural-looking edit, because
        "length" sounds like it ought to be positive. It would put every short's targets ABOVE its
        entry: a card that tells the user to sell and then take profit higher up.
        """
        up = fib_projection(150.0, 100.0, 200.0)
        down = fib_projection(150.0, 200.0, 100.0)
        assert down == pytest.approx([300.0 - x for x in up]), (
            f"the down leg projected {down}, which is not the reflection of the up leg {up} about "
            "their shared origin: the sign of the leg has been thrown away"
        )
        assert all(x < 150.0 for x in down), (
            f"projecting a leg that fell from 200 to 100 produced targets above the origin: {down}"
        )


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

    def test_the_gate_is_the_cost_CEILING_and_not_some_other_constant(self):
        """Which config field the gate reads is untestable from the two cases above.

        They sit at ``cost_r`` ≈ 0.53 and ≈ 0.029, so nothing in the suite lives anywhere near the
        0.20 ceiling — or near ``ev_min_r`` (0.15), the neighbouring constant a hand could slip
        onto. Sweeping the threshold past a fixed scenario pins the gate to ``max_cost_r`` itself
        rather than to one lucky number, and it is worth pinning because the rejection sentence
        interpolates ``max_cost_r``: a gate reading a different field would tell the user the
        maximum is 20% while refusing at 15%.
        """
        from wavelab.waves.projection import PlanConfig, build_plan
        h = self._hyp([70000.0, 80000.0, 74000.0])
        base = build_plan(h, 75000.0, 900.0, PlanConfig())
        assert base.viable and 0 < base.cost_r < PlanConfig().ev_min_r, "sanity: a viable, cheap trade"

        roomy = build_plan(h, 75000.0, 900.0, PlanConfig(max_cost_r=base.cost_r * 1.5))
        assert roomy.viable, (
            f"cost {base.cost_r:.4f} of R was refused under a ceiling of "
            f"{base.cost_r * 1.5:.4f}: the gate is not reading max_cost_r"
        )
        tight = build_plan(h, 75000.0, 900.0, PlanConfig(max_cost_r=base.cost_r * 0.5))
        assert not tight.viable, (
            f"cost {base.cost_r:.4f} of R was accepted under a ceiling of "
            f"{base.cost_r * 0.5:.4f}: lowering max_cost_r changed nothing, so the gate is "
            "deciding on some other constant and the rejection message quotes a limit that is "
            "not the one being applied"
        )
        assert "fees" in tight.reasons[0]

    def test_a_stop_that_lands_on_the_entry_is_refused_and_does_not_take_the_engine_down(self):
        """★ Zero risk is a division by zero one line later, and it had no test of its own.

        The mutation-audit column recorded this guard as covered. It was not: the only thing that
        went red when the guard was weakened was a translation-harvesting fixture in
        `test_server_i18n.py` that happens to build a zero-risk plan while collecting rejection
        strings — three ERRORS at fixture setup, and not one failure in this file. Anyone tidying
        that fixture, a reasonable thing to do, would have freed the guard silently while the
        audit went on saying it was covered. So it gets a test where the behaviour lives.

        The scenario is reachable, not contrived: `stop = invalidation - cushion`, and a rule whose
        invalidation lands exactly one cushion above the middle of the entry zone puts the stop on
        the entry. There is no risk to divide by, no R to quote, and the honest answer is a refusal
        carrying the reason — never a `ZeroDivisionError` out of a function whose whole contract is
        that it explains itself instead of raising.
        """
        from wavelab.waves.projection import PlanConfig, build_plan
        cfg = PlanConfig()
        points = (100.0, 110.0, 105.0)
        lo, hi = 110.0 - 10.0 * 0.786, 110.0 - 10.0 * 0.500
        entry = (lo + hi) / 2.0
        cushion = max(cfg.stop_buffer_atr * 1.0, 2 * cfg.tick_size, cfg.stop_buffer_pct * 106.0)

        h = self._hyp(list(points))
        # The invalidation the rule engine would have to publish for the stop to land on the entry.
        h = replace(h, invalidation_price=entry + cushion)
        r = build_plan(h, 106.0, 1.0, cfg)

        assert not r.viable, (
            f"a plan was returned with its stop on its own entry: risk {abs(entry - entry)}, "
            f"reasons {r.reasons}"
        )
        assert "far side of the stop" in r.reasons[0], (
            f"the refusal reads {r.reasons[0]!r}: the zero-risk case fell through to a later gate, "
            "so the arithmetic downstream ran on a risk of zero"
        )

        # The gate is exactly `risk <= 0`, and `risk` is an ABSOLUTE distance, so this is the only
        # input it ever refuses. A stop the same distance on the WRONG side of the entry — a long
        # stopped out above where it buys — measures as the same positive risk and is planned
        # normally. That is a real gap and it is recorded in docs/TEST_COVERAGE.md rather than
        # asserted away here; this line pins the boundary as it actually stands, so that closing
        # the gap has to come past a test that says what changed.
        wrong_side = build_plan(replace(h, invalidation_price=entry + cushion + 5.0), 106.0, 1.0, cfg)
        assert wrong_side.stop_atr == pytest.approx(5.0), (
            "the stop is 5.0 above the entry and the risk is measured as its absolute distance"
        )

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


class TestTheGuidelineScore:
    """The score decides the ORDER of the hypotheses, and `min_score` decides whether the top one
    is tradable at all. Nothing in the suite ever asserts a fit ratio or a score against a computed
    number, so every leg the score is a ratio of can be swapped for its neighbour silently.

    A ratio measured across the wrong pair of vertices does not look wrong: it is still a plausible
    number between 0 and 3, it still ranks, and the card still shows a count. It just ranks a
    different structure first, and the archetype of the top card is what decides whether an entry
    zone is offered at all.
    """

    #: w1 = 100, w2 retraces 25, w3 extends 150, w4 retraces 50. Every ratio exact in binary, and
    #: deliberately NOT symmetric: with a wave 2 that retraces exactly half, |P2-P1| and |P2-P0|
    #: are the same number, so a ratio taken across the wrong pair of vertices comes out right by
    #: accident and the fixture proves nothing.
    P: ClassVar[tuple[float, ...]] = (100.0, 200.0, 175.0, 325.0, 275.0)

    def test_each_fit_is_the_ratio_of_the_legs_it_is_named_after(self):
        _, fit = score_guidelines(self.P, MatcherConfig())
        assert fit["retr_w2"] == pytest.approx(0.25), (
            f"retr_w2={fit['retr_w2']}: wave 2 is the P1->P2 leg (25) measured against wave 1, "
            "P0->P1 (100) — not the distance from P0, which is 75"
        )
        assert fit["ext_w3"] == pytest.approx(1.5), (
            f"ext_w3={fit['ext_w3']}: wave 3 is P2->P3 (150) against wave 1 (100)"
        )
        assert fit["retr_w4"] == pytest.approx(1 / 3), (
            f"retr_w4={fit['retr_w4']}: wave 4 is P3->P4 (50) against WAVE 3 (150), not against "
            "wave 1 — alternation is the whole reason it is measured against its own wave"
        )
        assert fit["alternation"] == pytest.approx(1 / 12), (
            "alternation is the distance between the two retracements, 0.25 and 1/3"
        )

    def test_a_partial_structure_is_scored_only_on_what_it_can_show(self):
        """At impulse@2 there is one guideline to judge — the wave 2 retracement — so a structure
        sitting exactly on the golden ratio is a perfect 1.0.

        Averaging over the guidelines that do not exist yet instead would deflate every partial
        state by a fixed factor, and partial states are where 100% of the entries live. They would
        all sink below `min_score` together, which reads as "no setups today" rather than as a bug.
        """
        score, fit = score_guidelines((100.0, 200.0, 138.2), MatcherConfig())
        assert fit["retr_w2"] == pytest.approx(0.618)
        assert score == pytest.approx(1.0), (
            f"a wave 2 landing exactly on 0.618 scored {score}: at impulse@2 it is the only "
            "guideline that can be judged, so it is the whole score"
        )

    def test_no_single_guideline_can_run_away_with_the_score(self):
        """The score is compared against a threshold and against other scores, so it has to stay on
        one scale: a weighted average of terms in [0,1].

        Alternation is the one that can blow up, because it is a raw distance divided by 0.30
        rather than a closeness. Uncapped, a structure with a wildly deep wave 2 and a flat wave 4
        would out-score every well-formed count in the book on that term alone."""
        wild = (100.0, 200.0, 500.0, 500.5, 500.0)     # retr_w2 = 3.0, retr_w4 ~ 0
        for pts in (wild, self.P, (100.0, 200.0, 138.2)):
            score, _ = score_guidelines(pts, MatcherConfig())
            assert 0.0 <= score <= 1.0, f"{pts} scored {score}, outside [0, 1]"

    @pytest.mark.parametrize("pts", [
        (100.0, 100.0, 150.0, 300.0, 250.0),   # P0 == P1: wave 1 has no length
        (100.0, 200.0, 200.0, 300.0, 250.0),   # P1 == P2
        (100.0, 200.0, 150.0, 150.0, 250.0),   # P2 == P3: wave 3 has no length
        (100.0, 200.0, 150.0, 300.0, 300.0),   # P3 == P4
        (100.0, 100.0, 100.0, 100.0, 100.0),   # everything flat
    ])
    def test_two_pivots_at_the_same_price_do_not_take_the_engine_down(self, pts):
        """Each ratio divides by a leg, and each division is guarded — by a guard that has to name
        that leg's OWN pair of vertices. The guards sit four lines apart and read almost
        identically, which is exactly how one ends up copy-pasted with the wrong indices.

        A zero-length leg is not exotic: it is a flat top, a double bottom, or any structure priced
        to the tick on a quiet asset. The wrong guard turns it into a ZeroDivisionError, and this
        function is called inside `LiveEngine.decide` on every confirmed pivot.
        """
        score, fit = score_guidelines(pts, MatcherConfig())
        assert 0.0 <= score <= 1.0, f"{pts} scored {score}"
        assert all(math.isfinite(v) for v in fit.values()), f"non-finite fit {fit}"


class TestOnlyStructureNearTheEdgeIsTradable:
    """`max_from_edge` is the matcher's second design decision and it has no test.

    Both directions of the error are expensive and neither raises. Widen the window and the engine
    offers counts over structure that finished weeks ago — the module's own docstring says those
    "produce no decision" — and they compete for the four slots on the card. Narrow it and the
    counts that end on the newest pivot, which are precisely the ones still tradable, stop being
    found at all: no error, no signal, nothing.
    """

    #: One hand-checked impulse@4, plus filler pivots to push it away from the edge.
    IMPULSE: ClassVar[tuple[float, ...]] = (100.0, 200.0, 150.0, 300.0, 250.0)
    TRAIL: ClassVar[tuple[float, ...]] = (260.0, 255.0, 265.0, 258.0, 270.0)

    @staticmethod
    def _found(distance: int, cfg: MatcherConfig) -> bool:
        prices = TestOnlyStructureNearTheEdgeIsTradable.IMPULSE + \
            TestOnlyStructureNearTheEdgeIsTradable.TRAIL[:distance]
        out = match_impulses(_pivots(prices), cfg)
        return any(h.points == TestOnlyStructureNearTheEdgeIsTradable.IMPULSE
                   and h.state is ImpulseState.AT_4 for h in out)

    #: Written out rather than read from `MatcherConfig`: a bound taken from the very config the
    #: test is checking moves along with any change to it, so it would pin nothing at all. Three is
    #: the shipped window, and widening it is a product decision, not a tidy-up.
    WINDOW: ClassVar[int] = 3

    @pytest.mark.parametrize("distance", [0, 1, 2, 3])
    def test_a_count_within_the_window_is_offered(self, distance):
        assert self._found(distance, MatcherConfig(top_n=64)), (
            f"the impulse ends {distance} pivots from the edge, inside the shipped window of "
            f"{self.WINDOW}, and the matcher did not return it. A count that ends on or near the "
            "newest pivot is the only kind you can still trade; when it stops being found there "
            "is no error, there is simply no signal"
        )

    def test_a_count_four_pivots_from_the_edge_is_not(self):
        assert not self._found(self.WINDOW + 1, MatcherConfig(top_n=64)), (
            f"the impulse ends {self.WINDOW + 1} pivots from the edge, past the shipped window of "
            f"{self.WINDOW}, and was offered anyway. Structure that finished that long ago "
            "produces no decision, and it is competing for the slots on the card with the counts "
            "that do"
        )

    def test_the_window_is_the_configured_one(self):
        """`max_from_edge` is a knob, so moving it has to move the window. A default hard-coded
        somewhere else would leave the operator configuring a number nothing reads."""
        assert self._found(4, MatcherConfig(max_from_edge=4, top_n=64))
        assert not self._found(2, MatcherConfig(max_from_edge=1, top_n=64))


class TestTheHypothesesComeOutBestFirst:
    """The card shows the top few readings, so the ORDER is the product.

    On a tie the more complete structure wins: it asserts more, so it is more falsifiable, and it
    is the one whose archetype offers a plan. Reverse that and the top card flips between an
    `abc` (no entry) and a `w4` (a full plan with an entry zone) with nothing to show for it.
    """

    #: Six alternating pivots. Every prefix window is a legal count, so all four states appear.
    PRICES: ClassVar[tuple[float, ...]] = (100.0, 200.0, 150.0, 300.0, 250.0, 400.0)

    def test_with_the_guidelines_switched_off_completeness_decides(self):
        """Zero weights make every score exactly 0.0, so the tie-break is the only thing left
        deciding, and a genuine tie is what the tie-break is for. Ties are not hypothetical: the
        Fibonacci relations the score is built on are exactly what makes real structures agree."""
        flat = MatcherConfig(w_fib2=0.0, w_fib3=0.0, w_fib4=0.0, w_alt=0.0, top_n=64)
        out = match_impulses(_pivots(self.PRICES), flat)
        assert len(out) >= 4, f"the fixture must produce a few hypotheses, got {len(out)}"
        assert {h.score for h in out} == {0.0}, "sanity: with no weights every score ties at zero"

        sizes = [h.state.n_points for h in out]
        assert sizes == sorted(sizes, reverse=True), (
            f"tied hypotheses came back ordered {[h.state.value for h in out]}: on a tie the more "
            "complete structure has to come first, because it is the one that asserts the most"
        )
        assert out[0].state is ImpulseState.COMPLETE, (
            f"the top tied hypothesis is {out[0].state.value}, not the complete five"
        )

    def test_never_more_cards_than_the_configuration_asks_for(self):
        cfg = MatcherConfig()
        out = match_impulses(_pivots(self.PRICES), cfg)
        assert 0 < len(out) <= cfg.top_n, (
            f"top_n={cfg.top_n} and the matcher returned {len(out)} hypotheses: the card shows a "
            "reading the operator did not ask to see"
        )

    #: Five alternating pivots holding two three-vertex windows of deliberately different quality.
    #: (P0,P1,P2) retraces 61.8% of wave 1 — the golden pocket, dead centre of the guideline.
    #: (P2,P3,P4) retraces 95% of an identically sized wave 1 — legal under R1, and as far from a
    #: Fibonacci relation as a valid count gets. Both are AT_2, so completeness cannot separate
    #: them and the guideline score is the only thing that can.
    GOLDEN: ClassVar[tuple[float, ...]] = (100.0, 200.0, 138.2, 238.2, 143.2)

    def test_the_better_formed_reading_is_the_one_at_the_top_of_the_card(self):
        """★ The ordering claim that is not free.

        This test used to be `assert scores == sorted(scores, reverse=True)`, which is a property
        of `list.sort` and holds for every scoring function that could ever be written — including
        `score = 0.5`, which I verified leaves it green. It contributed no unique coverage: the
        only source change that broke it was deleting the sort call, and its sibling above already
        catches that.

        What the operator relies on is the mapping from structure to RANK: the reading at the top
        of the card is the one that fits the guidelines best. So the anchors here are the Fibonacci
        ratios themselves, written into the fixture as literals rather than read back out of the
        code — 61.8% against 95% of the same wave 1 — and the assertion is that the count built on
        the ratio outranks the one that is not, in a pair the tie-break cannot separate.
        """
        out = match_impulses(_pivots(self.GOLDEN), MatcherConfig(top_n=64))
        by_points = {h.points: h for h in out}
        golden = by_points.get((100.0, 200.0, 138.2))
        ragged = by_points.get((138.2, 238.2, 143.2))
        assert golden is not None and ragged is not None, (
            f"the fixture stopped producing both windows: got {[h.points for h in out]}"
        )
        assert golden.state is ragged.state, "sanity: only the score may separate these two"

        assert golden.fit["retr_w2"] == pytest.approx(0.618, abs=1e-9), (
            f"the window built to retrace exactly 61.8% of wave 1 reports "
            f"{golden.fit['retr_w2']}: the guideline is being measured off the wrong two legs, so "
            "every ratio the card shows describes a structure other than the one drawn"
        )
        assert ragged.fit["retr_w2"] == pytest.approx(0.95, abs=1e-9)

        assert golden.score > ragged.score, (
            f"the golden-pocket count scores {golden.score:.4f} and the 95% retracement scores "
            f"{ragged.score:.4f}: the score no longer tracks how well the structure fits the "
            "guidelines, so the number on the card is not a measure of anything"
        )
        assert out.index(golden) < out.index(ragged), (
            f"the card ranks the 95% retracement above the golden pocket "
            f"({[h.points for h in out]}): the operator reads the top row first, and it is the "
            "worse reading of the two"
        )
        scores = [h.score for h in out]
        assert scores == sorted(scores, reverse=True), f"unordered: {scores}"


#: (name, vertices, direction, live price, ATR). Hand-checked legal counts, one per archetype and
#: direction, plus the wave-4 case whose entry zone actually reaches into wave 1's territory.
_PLANS: list[tuple] = [
    ("w2_long", (100.0, 200.0, 150.0), Direction.LONG, 135.0, 5.0),
    ("w2_long_btc", (70000.0, 80000.0, 74000.0), Direction.LONG, 75000.0, 900.0),
    ("w2_short_btc", (80000.0, 70000.0, 76000.0), Direction.SHORT, 75000.0, 900.0),
    ("w4_long", (100.0, 200.0, 150.0, 300.0, 250.0), Direction.LONG, 235.0, 3.0),
    ("w4_short", (300.0, 200.0, 250.0, 100.0, 150.0), Direction.SHORT, 165.0, 3.0),
    ("w4_long_tight_zone", (100.0, 210.0, 150.0, 260.0, 215.0), Direction.LONG, 214.0, 3.0),
]
_PLAN_IDS = [c[0] for c in _PLANS]


class TestThePlanNumbersDescribeThePlan:
    """★ Every number on the signal card except the invalidation price is unasserted.

    `TestCostFilter` checks `cost_r`, `rr_t2` and `p_required` against loose one-sided bounds — is
    it above 0.20, is it under 0.05, is it positive — and loose bounds survive almost any error in
    the arithmetic behind them. The R:R can be quoted against the wrong target, the fees can be
    counted one side of a round trip instead of two, the whole block can be measured from the edge
    of the zone rather than from the entry the user is told to take, and the suite stays green.

    The oracle here is the plan the function itself returned, so it cannot drift with the inputs:
    whatever entry, stop and targets came back, the ratios have to be the ratios OF THOSE. A card
    that advertises 7R on a target sitting at 4.4R, and a hit rate of 16% where the arithmetic says
    23%, is not a rounding problem — it is the user sizing a different trade from the one on screen.
    """

    @pytest.mark.parametrize("name,pts,direction,price,atr", _PLANS, ids=_PLAN_IDS)
    def test_the_ratios_are_the_ratios_of_the_plan_it_returned(self, name, pts, direction, price,
                                                               atr):
        cfg = PlanConfig()
        r = build_plan(_hypothesis(pts, direction), price, atr, cfg)
        assert r.viable, f"{name}: expected a plan, got {r.reasons}"

        entry = r.plan.entry_mid
        risk = abs(entry - r.plan.stop)
        assert risk > 0

        assert r.rr_t2 == pytest.approx(abs(r.plan.targets[1] - entry) / risk), (
            f"{name}: the card advertises R:R {r.rr_t2:.3f} while T2 ({r.plan.targets[1]}) sits "
            f"{abs(r.plan.targets[1] - entry) / risk:.3f}R from the entry ({entry}). The number "
            "the user sizes on is not the trade the card is offering"
        )
        assert r.cost_r == pytest.approx((2 * cfg.fee_bps_taker / 10_000) * entry / risk), (
            f"{name}: cost_r={r.cost_r:.5f} is not the round trip ({cfg.fee_bps_taker} bps twice) "
            f"on a notional of {entry} against a risk of {risk}. This is the gate that rejects a "
            "trade before anything is modelled, so a factor of two here moves the gate itself"
        )
        assert r.p_required == pytest.approx(
            (cfg.expected_loss_r + r.cost_r + cfg.ev_min_r) / (r.rr_t2 + cfg.expected_loss_r)), (
            f"{name}: p_required={r.p_required:.4f} is not (L + cost + EV_min) / (RR + L) for the "
            f"cost and R:R on this very card. It is the one number the user reads as a plain-"
            "English claim about how often they have to be right"
        )

    @pytest.mark.parametrize("name,pts,direction,price,atr", _PLANS, ids=_PLAN_IDS)
    def test_the_entry_is_never_on_the_wrong_side_of_its_own_invalidation(self, name, pts,
                                                                         direction, price, atr):
        """The zone is built from Fibonacci levels and the stop from the count; nothing makes them
        agree except the truncation that pushes the zone clear of wave 1's territory.

        Get that truncation wrong and the result is not a bad trade, it is an impossible one: a
        long whose stop sits above its entry. The R:R and the position size are then computed from
        a negative risk taken in absolute value, so every number on the card still looks ordinary.
        """
        r = build_plan(_hypothesis(pts, direction), price, atr, PlanConfig())
        assert r.viable, r.reasons
        plan = r.plan
        if direction is Direction.LONG:
            assert plan.stop < plan.entry_lo <= plan.entry_hi, f"{name}: {plan}"
            assert plan.entry_lo > plan.invalidation_price, (
                f"{name}: the entry zone starts at {plan.entry_lo}, at or below the price "
                f"{plan.invalidation_price} at which this count is already dead"
            )
            assert plan.targets[0] > plan.entry_hi, f"{name}: a long taking profit below its entry"
        else:
            assert plan.stop > plan.entry_hi >= plan.entry_lo, f"{name}: {plan}"
            assert plan.entry_hi < plan.invalidation_price, (
                f"{name}: the entry zone reaches {plan.entry_hi}, at or above the price "
                f"{plan.invalidation_price} at which this count is already dead"
            )
            assert plan.targets[0] < plan.entry_lo, f"{name}: a short taking profit above its entry"

    def test_the_edge_of_the_zone_is_inside_the_zone(self):
        """`in_zone` drives the verdict, and the edges of the zone are COMPUTED Fibonacci levels —
        not two independent floats. A price landing exactly on one is not the measure-zero event it
        would otherwise be: it is a limit order resting at a published level.

        Feeding the boundary back from the plan keeps it exact, so the comparison is tested at the
        only place a `<=` quietly turned into a `<` would show.
        """
        h = _hypothesis((70000.0, 80000.0, 74000.0))
        cfg = PlanConfig()
        base = build_plan(h, 75000.0, 900.0, cfg)
        lo, hi = base.plan.entry_lo, base.plan.entry_hi
        assert base.in_zone

        for edge in (lo, hi):
            assert build_plan(h, edge, 900.0, cfg).in_zone, (
                f"price exactly at {edge}, an edge of the entry zone ({lo}, {hi}), was reported "
                "outside it: the zone is drawn at a price the engine will not honour"
            )
        for outside in (lo - 0.01, hi + 0.01):
            assert not build_plan(h, outside, 900.0, cfg).in_zone, (
                f"price {outside} is outside the zone ({lo}, {hi}) and was reported inside it"
            )



class TestTheSizeIsCutAndTheStopIsNot:
    """★ The rule the module names as the one that "looks minor and is not".

    The stop is where the count becomes false, plus a cushion. When that distance is large the
    SIZE is cut and the stop is left exactly where it was, because tightening it severs the link
    between the stop and the invalidation and leaves a stop that means nothing.

    Neither half of that is asserted anywhere. The branch can be disabled outright, its comparison
    can be turned round so it fires for near stops and never for far ones, and the factor can be
    inverted — which produces a size factor ABOVE one, an instruction to take MORE risk precisely
    because the stop is too far away. All three keep the suite green, and `size_factor` goes out on
    the live payload.
    """

    PTS: ClassVar[tuple[float, ...]] = (70000.0, 80000.0, 74000.0)
    PRICE: ClassVar[float] = 75000.0

    @pytest.mark.parametrize("atr,expect_cut", [(100.0, True), (900.0, True), (1400.0, False)])
    def test_a_far_stop_costs_size_and_only_size(self, atr, expect_cut):
        cfg = PlanConfig()
        h = _hypothesis(self.PTS)
        r = build_plan(h, self.PRICE, atr, cfg)
        assert r.viable, r.reasons
        assert r.stop_atr == pytest.approx(abs(r.plan.entry_mid - r.plan.stop) / atr)
        assert 0.0 < r.size_factor <= 1.0, (
            f"size_factor={r.size_factor} at {r.stop_atr:.2f} ATR. A factor above 1 is an "
            "instruction to size UP on the trades whose stop is furthest away"
        )

        if expect_cut:
            assert r.stop_atr > cfg.max_stop_atr, "fixture check"
            assert r.size_factor == pytest.approx(cfg.max_stop_atr / r.stop_atr), (
                f"the stop is {r.stop_atr:.2f} ATR away, over the {cfg.max_stop_atr} limit, and "
                f"the size came back at {r.size_factor:.0%} instead of "
                f"{cfg.max_stop_atr / r.stop_atr:.0%}"
            )
            assert any("size cut" in x for x in r.reasons), (
                f"the size was cut to {r.size_factor:.0%} and the card says nothing: {r.reasons}"
            )
        else:
            assert cfg.min_stop_atr <= r.stop_atr <= cfg.max_stop_atr, "fixture check"
            assert r.size_factor == 1.0, (
                f"the stop is {r.stop_atr:.2f} ATR away, inside the {cfg.max_stop_atr} limit, and "
                f"the size was cut to {r.size_factor:.0%} anyway"
            )

        expected_stop = h.invalidation_price - max(
            cfg.stop_buffer_atr * atr, 2 * cfg.tick_size, cfg.stop_buffer_pct * self.PRICE)
        assert r.plan.stop == pytest.approx(expected_stop), (
            f"the stop moved to {r.plan.stop} instead of staying at {expected_stop}. Whatever "
            "happens to the size, the stop stays a cushion away from the invalidation: that link "
            "is the whole reason the count is worth having"
        )

    @pytest.mark.parametrize("term,pts,price,atr", [
        ("volatility", (70000.0, 80000.0, 74000.0), 75000.0, 1400.0),   # 0.25 ATR = 350
        ("percentage", (70000.0, 80000.0, 74000.0), 75000.0, 100.0),    # 0.1% of price = 75
        ("ticks", (10.0, 20.0, 14.0), 15.0, 0.05),                      # 2 ticks = 0.02
    ])
    def test_the_cushion_is_the_LARGEST_of_its_three_floors(self, term, pts, price, atr):
        """Three floors, and each one is there for a market the other two do not cover: a wick hunt
        on a volatile asset, a spread-sized stop on a quiet one, and an instrument priced in single
        digits where a percentage of price is smaller than a tick.

        Taking the smallest instead of the largest collapses the cushion to whichever floor is
        least relevant — on BTC that is two cents — and parks the stop on the invalidation itself,
        which is exactly the wick hunt the cushion exists to survive. Each case below is chosen so
        a different term is the one that binds, so no single scenario can stand in for the rule.
        """
        cfg = PlanConfig()
        terms = {"volatility": cfg.stop_buffer_atr * atr,
                 "percentage": cfg.stop_buffer_pct * price,
                 "ticks": 2 * cfg.tick_size}
        assert max(terms, key=terms.get) == term, f"fixture check: {terms}"

        h = _hypothesis(pts)
        r = build_plan(h, price, atr, cfg)
        assert r.viable, r.reasons
        cushion = h.invalidation_price - r.plan.stop
        assert cushion == pytest.approx(terms[term]), (
            f"the {term} floor is the binding one at {terms[term]}, and the stop was placed "
            f"{cushion} below the invalidation ({min(terms.values())} is the smallest floor). A "
            "cushion that collapses to the smallest floor puts the stop on the invalidation itself"
        )


class TestTheTargetsAreProjectedFromTheVertexTheWaveStartsAt:
    """Where a target comes from is geometry, and the geometry is not asserted anywhere.

    Every vertex here is exact in binary, so the expected numbers are the arithmetic and not a
    tolerance. Anchoring a projection one vertex away moves every target by a whole wave and still
    produces an ordered, plausible-looking set of three.
    """

    def test_a_wave_two_entry_is_measured_from_the_end_of_wave_two(self):
        """Wave 3 starts at P2, so that is where its extensions are measured from. Projecting from
        P1 instead puts every target a full wave-2 leg too far away, and R:R with it."""
        r = build_plan(_hypothesis((100.0, 200.0, 150.0)), 135.0, 5.0, PlanConfig())
        assert r.viable, r.reasons
        assert (r.plan.entry_lo, r.plan.entry_hi) == pytest.approx((121.4, 150.0)), (
            "the wave-2 zone is the 0.500-0.786 retracement of wave 1 (100 -> 200)"
        )
        assert r.plan.targets == pytest.approx((250.0, 311.8, 411.8)), (
            f"targets {r.plan.targets}: wave 1 is 100 long and wave 3 is projected from P2 = 150, "
            "so 1.000/1.618/2.618 land at 250/311.8/411.8"
        )

    def test_a_wave_four_entry_exists_and_is_drawn_on_the_wave_three_leg(self):
        """The whole w4 archetype is untested: it can be struck off the list of archetypes that
        offer an entry, and the card silently degrades to "watch only" with no test complaining.

        Its zone is the shallow 0.382-0.500 retracement of WAVE 3 — wave 4 is the correction of
        wave 3, not of itself — and its middle target is projected across P0->P3, the whole move so
        far. Drawn on the wrong leg the zone lands above the price it should be buying at and the
        R:R falls by two thirds, all of it still looking like an ordinary card.
        """
        h = _hypothesis((100.0, 200.0, 150.0, 300.0, 250.0))
        assert h.archetype == "w4"
        r = build_plan(h, 235.0, 3.0, PlanConfig())
        assert r.viable, f"the wave-4 archetype offered no plan at all: {r.reasons}"
        assert (r.plan.entry_lo, r.plan.entry_hi) == pytest.approx((225.0, 242.7)), (
            f"zone {r.plan.entry_lo, r.plan.entry_hi}: the 0.382-0.500 retracement of wave 3 "
            "(150 -> 300) is 225 to 242.7"
        )
        assert r.plan.targets == pytest.approx((350.0, 373.6, 411.8)), (
            f"targets {r.plan.targets}: wave 5 is projected from P4 = 250 — 1.000 of wave 1 (100), "
            "0.618 of the whole P0->P3 move (200), and 1.618 of wave 1"
        )
        assert r.plan.archetype == "w4_long"

    def test_a_wave_four_zone_is_truncated_against_the_end_of_wave_one(self):
        """The truncation exists so the entry can never sit on the wrong side of its own stop, and
        it has to be measured against P1 — the vertex R3 makes the invalidation — not against P0.

        Here the raw retracement zone starts at 205, below the invalidation at 210. Truncated
        against P1 the zone is pushed clear of it; truncated against P0 (100) the truncation does
        nothing at all and the plan comes back offering a long entry BELOW the price at which the
        count is already dead, with the stop above the entry.
        """
        h = _hypothesis((100.0, 210.0, 150.0, 260.0, 215.0))
        assert h.invalidation_price == 210.0
        r = build_plan(h, 214.0, 3.0, PlanConfig())
        assert r.viable, r.reasons
        assert r.plan.entry_lo == pytest.approx(210.105), (
            f"the zone starts at {r.plan.entry_lo}: the raw 0.500 retracement is 205, inside wave "
            "1's territory, and it has to be pushed clear of P1 = 210"
        )
        assert r.plan.entry_lo > h.invalidation_price > r.plan.stop

    def test_a_wave_four_zone_that_truncates_to_nothing_is_refused_with_its_reason(self):
        """When wave 4 has retraced so far that no legal entry is left above the invalidation, the
        answer is "no entry", not a plan on the remaining sliver."""
        h = _hypothesis((100.0, 220.0, 150.0, 240.0, 225.0))
        r = build_plan(h, 225.0, 3.0, PlanConfig())
        assert not r.viable
        assert "entry zone is empty" in r.reasons[0], r.reasons
