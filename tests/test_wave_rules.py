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
    RETRACEMENTS,
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
        invalidation lands exactly one cushion above the fill puts the stop on the entry. There is
        no risk to divide by, no R to quote, and the honest answer is a refusal carrying the reason
        — never a `ZeroDivisionError` out of a function whose whole contract is that it explains
        itself instead of raising.

        The entry is now `price`, the close of the bar that produced the plan, so the invalidation
        is solved against that rather than against the middle of the zone. The gate itself did not
        move; what it is measuring did.
        """
        from wavelab.waves.projection import PlanConfig, build_plan
        cfg = PlanConfig()
        points = (100.0, 110.0, 105.0)
        price = 106.0
        cushion = max(cfg.stop_buffer_atr * 1.0, 2 * cfg.tick_size, cfg.stop_buffer_pct * price)

        h = self._hyp(list(points))
        # The invalidation the rule engine would have to publish for the stop to land on the fill.
        h = replace(h, invalidation_price=price + cushion)
        r = build_plan(h, price, 1.0, cfg)

        assert not r.viable, (
            f"a plan was returned with its stop on its own entry: risk {abs(price - price)}, "
            f"reasons {r.reasons}"
        )
        assert "far side of the stop" in r.reasons[0], (
            f"the refusal reads {r.reasons[0]!r}: the zero-risk case fell through to a later gate, "
            "so the arithmetic downstream ran on a risk of zero"
        )

        # `risk` is an ABSOLUTE distance, so this gate fires only where the two sides coincide. A
        # stop the same distance on the WRONG side of the entry — a long stopped out above where it
        # buys — measures as the same positive risk and used to be planned normally; it is now
        # refused by a gate of its own, on the sign rather than on the magnitude. Both directions,
        # and why the case only became reachable here, are in
        # `TestAnEntryPastItsOwnStopIsRefusedRatherThanSized`. This line pins that the two gates do
        # not overlap: five dollars on the wrong side is a live, positive risk, and it is still a no.
        wrong_side = build_plan(replace(h, invalidation_price=price + cushion + 5.0), price, 1.0, cfg)
        assert wrong_side.stop_atr == 0.0 and not wrong_side.viable, (
            f"a stop 5.0 ABOVE a long's entry came back with stop_atr={wrong_side.stop_atr} and "
            f"viable={wrong_side.viable}: the absolute distance was taken for ordinary risk and "
            f"the trade was sized. Reasons: {wrong_side.reasons}"
        )
        assert "past the stop" in wrong_side.reasons[0], (
            f"the refusal reads {wrong_side.reasons[0]!r}: it is not the wrong-side gate speaking"
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


class TestTheGatesDecideAtTheirOwnBoundary:
    """★ Three gates in `build_plan` were pinned everywhere except at the value they compare to.

    `cost_r > max_cost_r`, `stop_atr < min_stop_atr` and `lo >= hi` each survive being loosened or
    tightened by one character, because every scenario in the suite sits well clear of the
    threshold. The audit left them open on the grounds that a test would have to solve for an input
    whose arithmetic lands exactly on 0.20 or 0.75 and would then be pinned to that solution
    instead of to the rule.

    It does not have to. Two of the three thresholds are CONFIGURATION: run the plan once, read the
    `cost_r` or `stop_atr` the function itself computed, and hand the same scenario back under a
    config whose threshold IS that number, bit for bit. The boundary is derived from the code at
    runtime, so it moves with the arithmetic instead of pinning today's value of it, and the only
    thing left for the assertion to turn on is whether the comparison is strict — which is exactly
    what the mutants flip. It is the idiom `test_the_edge_of_the_zone_is_inside_the_zone` already
    uses on `in_zone`, applied to the two gates that reject.

    What is asserted AT equality is a rule, not a measurement, and the source states the rule twice
    over in each case. `max_cost_r` is documented as the maximum tolerable cost — "Above it the
    trade is REJECTED" — and the refusal sentence quotes it back to the user as "maximum 20%", so a
    cost landing exactly on 20% has to be tradable or the tool is refusing at the number it
    publishes as allowed. `min_stop_atr` is refused with "it sits inside the noise floor (<0.75
    ATR)", and a stop exactly at the floor is not inside it. The empty-zone gate is the one that
    goes the other way: `lo == hi` is a zone with nowhere to enter, and it has to be refused.
    """

    PTS: ClassVar[tuple[float, ...]] = (70000.0, 80000.0, 74000.0)
    PRICE: ClassVar[float] = 75000.0
    #: Chosen so the plan comes back clean: the risk is measured from the FILL at 75,000 down to a
    #: stop a 0.25-ATR cushion under the invalidation at 70,000, which is 2.75 ATR — inside
    #: `min_stop_atr` and `max_stop_atr` both, so neither gate under test is standing behind the
    #: other one, and the size is not cut. Measured from the zone midpoint instead this same
    #: scenario is 2.04 ATR, which is why the number here moved when the entry did.
    ATR: ClassVar[float] = 2000.0

    def _accepted_baseline(self):
        h = _hypothesis(self.PTS)
        base = build_plan(h, self.PRICE, self.ATR, PlanConfig())
        assert base.viable and base.size_factor == 1.0 and not base.reasons, (
            f"the baseline has to be an unremarkable accepted plan before a threshold is moved "
            f"onto one of its own numbers: viable={base.viable}, size={base.size_factor}, "
            f"reasons {base.reasons}"
        )
        return h, base

    def test_a_cost_landing_exactly_on_the_ceiling_is_still_tradable(self):
        """`max_cost_r` is a MAXIMUM, and the refusal sentence tells the user so.

        Loosen the gate to `cost_r >= cfg.max_cost_r` and the engine rejects a trade whose fees
        take exactly 20% of R while printing "(maximum 20%)" as the reason — a refusal that
        contradicts itself on its own line. The reverse error is worse in money: a gate that reads
        `>=` on a tightened config rejects the whole tail of trades sitting on the ceiling, and
        because the ceiling is the FEE gate, that tail is every trade a fee change has just pushed
        onto the limit.
        """
        h, base = self._accepted_baseline()

        at_ceiling = replace(PlanConfig(), max_cost_r=base.cost_r)
        r = build_plan(h, self.PRICE, self.ATR, at_ceiling)
        assert r.viable, (
            f"a cost of exactly {base.cost_r!r} of R was refused under a max_cost_r of exactly "
            f"that same float: the gate rejects AT its ceiling rather than above it, so the "
            f"maximum the config documents and the refusal sentence quotes is not the maximum "
            f"being applied. Reasons: {r.reasons}"
        )

        # The other half of the pincer. Without it the assertion above would also pass on a gate
        # deleted outright, which is the failure mode this file has been caught by before.
        # One ULP under the computed cost is the largest ceiling this trade genuinely exceeds.
        just_under = replace(PlanConfig(), max_cost_r=math.nextafter(base.cost_r, 0.0))
        r_under = build_plan(h, self.PRICE, self.ATR, just_under)
        assert not r_under.viable and "fees" in r_under.reasons[0], (
            f"cost {base.cost_r!r} of R was accepted under a ceiling of "
            f"{just_under.max_cost_r!r} — one ULP lower, and genuinely exceeded. The acceptance "
            f"one ULP higher is therefore not the gate deciding. Reasons: {r_under.reasons}"
        )

    def test_a_stop_landing_exactly_on_the_noise_floor_is_still_tradable(self):
        """`min_stop_atr` is the floor of what is tradable, not the first value refused.

        Loosen the gate to `stop_atr <= cfg.min_stop_atr` and a stop measuring exactly 0.75 ATR is
        thrown out as sitting "inside the noise floor (<0.75 ATR)" — a sentence that says, in the
        same breath, that it is not inside it. The stop here comes from the invalidation, so this
        gate is the only thing standing between the user and a structurally correct stop being
        discarded on a rounding of the volatility estimate.
        """
        h, base = self._accepted_baseline()

        at_floor = replace(PlanConfig(), min_stop_atr=base.stop_atr)
        r = build_plan(h, self.PRICE, self.ATR, at_floor)
        assert r.viable, (
            f"a stop of exactly {base.stop_atr!r} ATR was refused under a min_stop_atr of exactly "
            f"that same float: the gate refuses AT the floor rather than below it, so the floor "
            f"the refusal sentence prints as '<{base.stop_atr}' is not the floor being applied. "
            f"Reasons: {r.reasons}"
        )

        just_over = replace(PlanConfig(), min_stop_atr=math.nextafter(base.stop_atr, math.inf))
        r_over = build_plan(h, self.PRICE, self.ATR, just_over)
        assert not r_over.viable and "noise floor" in r_over.reasons[0], (
            f"a stop of {base.stop_atr!r} ATR was accepted under a floor of "
            f"{just_over.min_stop_atr!r} — one ULP higher, so it is genuinely below it. The "
            f"acceptance one ULP lower is therefore not the gate deciding. "
            f"Reasons: {r_over.reasons}"
        )

    def test_an_entry_zone_of_zero_width_is_refused_as_an_empty_zone(self):
        """`lo >= hi` is the only one of the three that has to refuse at equality: a zone whose two
        edges are the same price is not a narrow zone, it is no zone at all.

        Zero width is NOT reachable from `match_impulses`, and that is worth saying rather than
        implying: the matcher only emits counts `check_impulse` calls valid, and every route to
        `lo == hi` needs a leg of zero length — R1 on the w2 path, and on the w4 path a wave 2
        finishing ABOVE wave 1's top, since otherwise the truncation against `P1 * 1.0005` pushes
        `lo` past `hi` and the gate fires on the strict `>` either way. So the hypothesis is built
        here by hand. `build_plan` is exported and takes any `Hypothesis`; a flat top, a double
        bottom or a series priced to the tick is exactly the input
        `test_two_pivots_at_the_same_price_do_not_take_the_engine_down` already treats as ordinary,
        and this is the gate that decides what happens when one reaches the planner.

        Tighten it to `lo > hi` and the single surviving price is priced as if it were a zone: the
        entry lands on the invalidation, the risk collapses to the cushion, and the user is refused
        by the FEE gate instead — told the round trip costs 150% of R, for a zone that does not
        exist. A rejection in this tool is meant to be an argument the user can push back on, and
        that one is an argument about the wrong thing.
        """
        flat = (79000.0, 79000.0, 79000.0)
        v = check_impulse(list(flat), Direction.LONG)
        assert not v.valid and "R1" in v.broken, (
            f"a wave 1 of zero length is expected to break R1 and be filtered out of the matcher; "
            f"it now reports valid={v.valid}, broken={list(v.broken)}. If this count has become "
            "legal, the zero-width zone is reachable from real pivots and this test is no longer "
            "the hand-built case its docstring describes"
        )
        h = Hypothesis(ImpulseState.AT_2, Direction.LONG, _pivots(flat), flat, 0.8, {},
                       v.invalidation_price, v.invalidation_rule, ARCHETYPES[ImpulseState.AT_2])

        r = build_plan(h, 79000.0, 100.0, PlanConfig())
        assert r.plan is None, (
            f"a plan was returned for an entry zone of zero width: "
            f"{r.plan.entry_lo}-{r.plan.entry_hi}. There is one price in it, `in_zone` is true at "
            f"that float and nowhere else, and the card offers it as a range"
        )
        assert "entry zone is empty" in r.reasons[0], (
            f"the zone has zero width and the refusal reads {r.reasons[0]!r}: it fell through the "
            "empty-zone gate and was caught by a later one, so the user is given a reason about "
            "fees or volatility for a zone that has nowhere to enter at all"
        )


class TestTheZoneIsReadFromTheRetracementTable:
    """★ `RETRACEMENTS` was dead configuration and is now the single definition of the published
    entry zone — `build_plan` reads both bounds out of it instead of open-coding 0.500/0.786 and
    0.382/0.500. That change is only worth something while it stays true, and nothing held it down:
    re-typing the four literals back into `projection.py` changes no behaviour, so every other test
    in this file stays green while the table quietly becomes an orphan again. Measured, not
    assumed — the open-coded version passes the whole suite.

    So this is the test of the WIRING, and it is the only one. Move the table and the zone the card
    draws has to move with it.

    Neither case names a Fibonacci number. Both state what a retracement ratio MEANS on the leg it
    is measured against — 0.0 is the end of the leg, 1.0 is its start, 0.5 is halfway back — so the
    expected zone is written in terms of the vertices and never as a float solved for in advance.
    If `_fib_zone`'s arithmetic is ever rewritten, these follow it.

    What they cannot see: `_fib_zone` sorts its two outputs, so reading the table's two bounds in
    the wrong order produces the identical zone. That swap is an equivalent mutant here, not a hole.
    """

    #: Wave 1 runs 100 → 200 and wave 3 runs 150 → 300, so every zone below is a round fraction of
    #: a leg whose ends are in the fixture rather than in the assertion.
    W2_PTS: ClassVar[tuple[float, ...]] = (100.0, 200.0, 150.0)
    W4_PTS: ClassVar[tuple[float, ...]] = (100.0, 200.0, 150.0, 300.0, 250.0)

    def test_the_wave_two_zone_follows_the_w2_row(self, monkeypatch):
        """Both bounds, not one. The first case moves them together and the second moves only the
        far one, so a `build_plan` that read slot 0 twice — or read the table for one edge and kept
        a literal for the other — fails the second case while passing the first."""
        h = _hypothesis(self.W2_PTS)
        start, end = self.W2_PTS[0], self.W2_PTS[1]        # wave 1: 100 → 200

        monkeypatch.setitem(RETRACEMENTS, "w2", (0.0, 1.0, 0.618, 0.650))
        r = build_plan(h, 135.0, 5.0, PlanConfig())
        assert r.viable, r.reasons
        assert (r.plan.entry_lo, r.plan.entry_hi) == pytest.approx((start, end)), (
            f"with the w2 row set to retrace between 0.0 and 1.0 of wave 1, the zone is the whole "
            f"of wave 1 — {start} to {end} — and the plan drew {r.plan.entry_lo}-{r.plan.entry_hi}. "
            "The zone is not being measured from this table, so `RETRACEMENTS` is describing a "
            "level the product does not use"
        )

        monkeypatch.setitem(RETRACEMENTS, "w2", (0.0, 0.5, 0.618, 0.650))
        r = build_plan(h, 135.0, 5.0, PlanConfig())
        assert r.viable, r.reasons
        assert (r.plan.entry_lo, r.plan.entry_hi) == pytest.approx(((start + end) / 2, end)), (
            f"only the far bound of the w2 row moved (1.0 → 0.5) and the zone should have pulled "
            f"back to halfway down wave 1, {(start + end) / 2} to {end}; it drew "
            f"{r.plan.entry_lo}-{r.plan.entry_hi}. One of the two bounds is not coming from the row"
        )

    def test_the_wave_four_zone_follows_the_w4_row(self, monkeypatch):
        """The second reading site. `build_plan` looks the table up twice, and a wiring that
        covered only the w2 branch would leave the shallower, lower-confidence archetype still
        open-coded — the half nobody looks at."""
        h = _hypothesis(self.W4_PTS)
        assert h.archetype == "w4"
        start, end = self.W4_PTS[2], self.W4_PTS[3]        # wave 3: 150 → 300

        monkeypatch.setitem(RETRACEMENTS, "w4", (0.0, 0.5, 0.382, 0.450))
        r = build_plan(h, 235.0, 3.0, PlanConfig())
        assert r.viable, r.reasons
        assert (r.plan.entry_lo, r.plan.entry_hi) == pytest.approx(((start + end) / 2, end)), (
            f"the w4 row was set to give back between 0.0 and 0.5 of wave 3, so the zone is the "
            f"top half of wave 3, {(start + end) / 2} to {end}; the plan drew "
            f"{r.plan.entry_lo}-{r.plan.entry_hi}. The w4 branch is not reading the table"
        )
        # Deliberately clear of the P1 * 1.0005 truncation (200.1 here): this test is about where
        # the zone is read from, and letting the guard bite would hide that behind a clamp.
        assert r.plan.entry_lo > self.W4_PTS[1] * 1.0005, (
            "the fixture is meant to keep the zone above wave 1's top so the truncation never "
            "fires; if it now binds, this test is measuring the clamp and not the table"
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

    WHICH entry is the whole question, and this class used to answer it with the middle of the
    published zone. That is the price a limit order gets IF it fills, and a limit order may never
    fill; the backtest, meanwhile, books at the close of the bar that produced the plan. So the
    headline the card quoted and the R:R the results measured described two different trades, and
    the card's was the flattering one — over six months of 4h bars, 649 of 925 plans quoted a
    better ratio than was reachable, median +1.65R.

    So the oracle moved to `price`, the close of the bar, which is the price you can always get
    because you can always buy at market. The zone figure did not disappear: it travels beside the
    headline as `rr_in_zone`, the better case, and it is asserted here too. The failure message
    below — "the number the user sizes on is not the trade the card is offering" — is the defect
    this class was written for; it is now pointed at the side that can actually be filled.
    """

    @pytest.mark.parametrize("name,pts,direction,price,atr", _PLANS, ids=_PLAN_IDS)
    def test_the_ratios_are_the_ratios_of_the_plan_it_returned(self, name, pts, direction, price,
                                                               atr):
        cfg = PlanConfig()
        r = build_plan(_hypothesis(pts, direction), price, atr, cfg)
        assert r.viable, f"{name}: expected a plan, got {r.reasons}"

        # The headline is quoted at the FILL: the price passed in, which is the bar close the
        # backtest books at. Not `r.plan.entry_mid` — that is the limit-order case below.
        entry = price
        risk = abs(entry - r.plan.stop)
        assert risk > 0

        assert r.rr_t2 == pytest.approx(abs(r.plan.targets[1] - entry) / risk), (
            f"{name}: the card advertises R:R {r.rr_t2:.3f} while T2 ({r.plan.targets[1]}) sits "
            f"{abs(r.plan.targets[1] - entry) / risk:.3f}R from the fill ({entry}). The number "
            "the user sizes on is not the trade the card is offering"
        )
        assert r.cost_r == pytest.approx((2 * cfg.fee_bps_taker / 10_000) * entry / risk), (
            f"{name}: cost_r={r.cost_r:.5f} is not the round trip ({cfg.fee_bps_taker} bps twice) "
            f"on a notional of {entry} against a risk of {risk}. This is the gate that rejects a "
            "trade before anything is modelled, so a factor of two here moves the gate itself"
        )
        assert r.stop_atr == pytest.approx(risk / atr), (
            f"{name}: stop_atr={r.stop_atr:.3f} is not the distance from the fill ({entry}) to the "
            f"stop ({r.plan.stop}) in ATRs of {atr}. This is what the size is cut on, and the two "
            "volatility gates decide on it"
        )

        # The upside case, travelling beside the headline: the ratio a limit order resting in the
        # zone gets IF it fills. It is measured at the zone midpoint, and it is the number the card
        # used to lead with.
        mid = r.plan.entry_mid
        risk_mid = abs(mid - r.plan.stop)
        assert risk_mid > 0
        assert r.rr_in_zone == pytest.approx(abs(r.plan.targets[1] - mid) / risk_mid), (
            f"{name}: rr_in_zone={r.rr_in_zone:.3f} while T2 ({r.plan.targets[1]}) sits "
            f"{abs(r.plan.targets[1] - mid) / risk_mid:.3f}R from the middle of the published zone "
            f"({r.plan.entry_lo}-{r.plan.entry_hi}). The better case has to be the better case OF "
            "THIS PLAN, or it is a second unexplained number on the card"
        )

        assert r.p_required == pytest.approx(
            (cfg.expected_loss_r + r.cost_r + cfg.ev_min_r) / (r.rr_t2 + cfg.expected_loss_r)), (
            f"{name}: p_required={r.p_required:.4f} is not (L + cost + EV_min) / (RR + L) for the "
            f"cost and R:R on this very card. It is the one number the user reads as a plain-"
            "English claim about how often they have to be right, so it is computed off the "
            "reachable R:R and never off the limit-order one"
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


class TestTheHeadlineCanNeverBeTheFlatteringNumber:
    """★ The property the whole change exists to hold: the card may not quote an R:R better than
    the one the reader can actually reach.

    `rr_t2` is measured at the fill and `rr_in_zone` at the middle of the published zone. The zone
    is a retracement you are WAITING for, so when the market has not come back to it the fill is
    the worse price — further from the target and further from the stop, both of which push the
    ratio down. The headline therefore has to sit strictly below the zone figure whenever the price
    is on the far side of the zone; equal means the card is quoting the zone twice over, which is
    the arrangement this replaced, and larger means it is advertising a number that depends on a
    limit order filling — the optimistic bias the product exists to catch.

    This is stated as a property over the whole interval between the zone and the target rather
    than as a pair of examples, because the direction of the inequality is what matters and a
    handful of prices cannot show that it never turns over. The interval is the honest domain, and
    it is the FAR side only — the name of the test says so, because "outside the zone" would be a
    claim about two sides and only one of them holds:

    - inside the zone the two fall either way, depending on which side of the midpoint the price
      is on, and the second test measures how far apart they get (a median 0.78R over six months
      of 4h bars, up to 2.95R — which is why the card prints both in the zone as well);
    - on the NEAR side — a long below the zone it wanted to buy in — the fill is the cheaper entry
      and rr_t2 is legitimately the LARGER number. 179 of 1,532 viable plans measured over the
      same window sit there. That is not the card flattering anyone: it is the market offering a
      better price than the one the zone figure is quoted at, and it is reachable at market. The
      third test below leans on exactly such a case;
    - past T2 you are buying above the target you are aiming at, which is not a trade the card
      offers.

    So the invariant this class is named for is "the headline is the ratio at a price the reader
    can actually get", and the strict inequality below is what that becomes on the one side where
    the old arithmetic flattered.

    Both signs are here for one reason — the arithmetic is written once with `abs()`, so a short
    is the only place a dropped sign can hide.

    The third test carries the same claim into the gates, which is where it costs money rather than
    credibility: a gate reading the flattering number ACCEPTS trades the honest arithmetic refuses,
    and each one reaches the user wearing the tool's approval.
    """

    @pytest.mark.parametrize("name,pts,direction,price,atr", _PLANS, ids=_PLAN_IDS)
    @given(f=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=60, deadline=None)
    def test_the_headline_never_beats_the_zone_figure_from_the_far_side_of_the_zone(
        self, name, pts, direction, price, atr, f
    ):
        cfg = PlanConfig()
        h = _hypothesis(pts, direction)
        base = build_plan(h, price, atr, cfg)
        assert base.viable, f"{name}: the fixture must plan before it can be swept: {base.reasons}"

        # The far side of the zone, up to the target being quoted: a long buying ABOVE the zone it
        # wanted to buy in, a short selling BELOW it. `f` walks that interval.
        edge = base.plan.entry_hi if direction is Direction.LONG else base.plan.entry_lo
        target = base.plan.targets[1]
        swept = edge + (target - edge) * f

        r = build_plan(h, swept, atr, cfg)
        assert r.rr_in_zone > 0, (
            f"{name}: no zone figure came back at price {swept} ({r.reasons}), so there is nothing "
            "for the headline to be compared against"
        )
        # STRICTLY worse, not merely no better. The whole interval starts at the far EDGE of the
        # zone, which is already past the midpoint the zone figure is quoted at, so there is no
        # price in it where the two may legitimately coincide. `<=` would also be satisfied by the
        # two numbers being the same number, which is exactly what quoting the zone twice looks
        # like — the arrangement this change replaced.
        side = "above" if direction is Direction.LONG else "below"
        assert r.rr_t2 < r.rr_in_zone, (
            f"{name} {direction.name}: at {swept} — {side} the zone "
            f"{base.plan.entry_lo}-{base.plan.entry_hi} — the card quotes {r.rr_t2:.4f}R at the "
            f"fill against {r.rr_in_zone:.4f}R in the zone. The market has not come back to the "
            "zone, so the fill is the worse price and the headline has to say so; if the two are "
            "equal the headline is the zone figure again, and if it is larger the card is "
            "advertising a limit order that may never fill as the trade on offer"
        )

    @pytest.mark.parametrize("name,pts,direction,price,atr", _PLANS, ids=_PLAN_IDS)
    def test_inside_the_zone_the_two_numbers_agree_to_the_width_of_the_zone(
        self, name, pts, direction, price, atr
    ):
        """HOW close, and WHY — stated as the law rather than as a reassuring adjective.

        WHY: the two ratios are the same function of the entry price, read at two points. For a
        stop S and a target T,

            |rr(e1) - rr(e2)| = |T - S| * |e1 - e2| / (|e1 - S| * |e2 - S|)

        which rearranges into the form the reader can actually use:

            gap = (rr_in_zone + 1) * |price - midpoint| / risk

        HOW CLOSE: exactly zero when the price sits on the middle of the zone — there the fill IS
        the zone entry and there is only one number — and growing monotonically with the distance
        from that midpoint, fastest on the side that faces the stop, where the risk in the
        denominator is smallest.

        And then the part worth writing down instead of rounding off: inside the zone that gap is
        bounded by NOTHING except the zone's own half-width. The shipped wave-2 zone is the
        0.500-0.786 retracement, 28.6% of wave 1, and against these fixtures' risk that buys a
        disagreement of up to 76% of the zone figure at the edges (3.6R on `w2_long`, 15.2R on
        `w4_long_tight_zone`, whose near edge sits less than one R from the stop). So "the two
        nearly agree in the zone" is true at the middle and false at the rim, and the honest
        summary is that they agree in proportion to how narrow the zone is against the risk.

        That is the remaining daylight between the headline and the upside figure, and it is why
        both travel on the card rather than one standing in for the other. Outside the zone the gap
        is not bounded at all: at T2 the headline is 0R against a zone figure of several R, and
        that is the case the class docstring above is about.
        """
        cfg = PlanConfig()
        h = _hypothesis(pts, direction)
        base = build_plan(h, price, atr, cfg)
        assert base.viable, base.reasons
        lo, hi, mid = base.plan.entry_lo, base.plan.entry_hi, base.plan.entry_mid

        sweep, samples = [], 0
        for i in range(9):
            swept = lo + (hi - lo) * i / 8
            r = build_plan(h, swept, atr, cfg)
            if r.plan is None:
                # The edge of the zone that FACES the stop can collapse the risk far enough for a
                # volatility gate to refuse before any plan exists. The two ratios still come back
                # — a refusal always brings its numbers — but the stop they were measured against
                # does not, and that is what the closed form needs. Skipped, not asserted away:
                # the sample count below keeps the sweep from quietly emptying out.
                continue
            samples += 1
            assert r.in_zone, f"{name}: {swept} was swept out of the zone {lo}-{hi}"
            assert r.rr_in_zone > 0, f"{name}: no zone figure at {swept}: {r.reasons}"

            risk, risk_mid = abs(swept - r.plan.stop), abs(mid - r.plan.stop)
            closed_form = (abs(r.plan.targets[1] - r.plan.stop) * abs(swept - mid)
                           / (risk * risk_mid))
            gap = abs(r.rr_t2 - r.rr_in_zone)
            assert gap == pytest.approx(closed_form, rel=1e-9, abs=1e-9), (
                f"{name}: at {swept} the two ratios differ by {gap:.6f} where the geometry says "
                f"{closed_form:.6f}. They are no longer the same ratio read at two entry prices, "
                "so one of them is measured against a different stop or a different target"
            )
            # The same law in the form the reader can check off the card: the disagreement is the
            # distance from the zone's middle, expressed in R, scaled by the zone figure plus one.
            assert gap == pytest.approx((r.rr_in_zone + 1) * abs(swept - mid) / risk,
                                        rel=1e-9, abs=1e-9), (
                f"{name}: at {swept} the gap of {gap:.6f} is not (rr_in_zone + 1) x "
                f"{abs(swept - mid)} / {risk}. The two ratios have stopped sharing a stop"
            )
            sweep.append((swept - mid, gap))

        assert samples >= 5, (
            f"{name}: only {samples} of 9 prices across the zone {lo}-{hi} produced a plan, so "
            "this sweep is no longer looking at the zone it claims to"
        )

        at_mid = build_plan(h, mid, atr, cfg)
        assert at_mid.rr_t2 == pytest.approx(at_mid.rr_in_zone, rel=1e-9), (
            f"{name}: with the price sitting exactly on the middle of the zone the fill IS the "
            f"zone entry, and the card still shows two different numbers: {at_mid.rr_t2} against "
            f"{at_mid.rr_in_zone}"
        )

        # Monotone in the distance from the midpoint, on each side separately: that is what makes
        # the zone's half-width the ceiling on the disagreement, and it is the only sense in which
        # being in the zone bounds anything. Sorting by distance and checking the gaps come out
        # sorted too is the claim; a gap that peaked in the middle of a side would mean the two
        # ratios are not the same curve read at two points.
        for side in (+1, -1):
            same_side = sorted((abs(d), g) for d, g in sweep
                               if d != 0 and (d > 0) is (side > 0))
            gaps = [g for _, g in same_side]
            assert gaps == sorted(gaps), (
                f"{name}: walking away from the middle of the zone the gap went "
                f"{[f'{g:.4f}' for g in gaps]} — it has to widen the whole way, because the price "
                "is moving steadily further from the entry the zone figure is quoted at"
            )

    def test_the_gates_decide_on_the_number_the_user_can_actually_get(self):
        """The same claim in the accept/reject channel, where it is worth more than on the card.

        A number that only misleads is bad; a GATE reading the optimistic number admits trades the
        conservative arithmetic would have thrown out, and every one of them arrives at the user
        wearing the tool's approval. `max_cost_r`, `min_stop_atr` and `max_stop_atr` therefore
        decide on `cost_r` and `stop_atr` as measured at the fill.

        Both cases below are chosen so the two sides of the arithmetic give OPPOSITE verdicts, which
        is the only kind of case that can tell them apart:

        - the fee gate: a long at 70,100, below its zone and close above its stop. The risk from
          there is 170 points, so the round trip eats 62% of R and the trade is refused. Measured
          from the middle of the zone the risk is 3,640 points, the fees are 3% of R, and the same
          input is a perfectly ordinary accepted plan.
        - the size gate: a long at 76,500, above its zone. The stop is 3.5 ATR away and the size is
          cut; from the zone midpoint it is 2.0 ATR and nothing is cut at all.
        """
        cfg = PlanConfig()
        h = _hypothesis((70000.0, 80000.0, 74000.0))

        refused = build_plan(h, 70100.0, 100.0, cfg)
        assert not refused.viable, (
            f"a long bought at 70,100 — a hundred points above the stop, with the entry zone "
            f"still 2,000 above it — was accepted with {refused.cost_r:.0%} of R going to fees. "
            f"The fee gate is reading the cost at an entry the market has not reached: "
            f"{refused.plan}"
        )
        assert "fees" in refused.reasons[0], refused.reasons
        assert refused.cost_r > cfg.max_cost_r, (
            f"cost_r={refused.cost_r:.4f} is not above the {cfg.max_cost_r} ceiling, so this "
            "fixture is no longer the case its docstring describes"
        )
        # The other half of the disagreement, computed here rather than taken on trust: from the
        # middle of the zone this same trade costs 3% of R and sails through the gate it was just
        # refused by. `plan` is None on a refusal, so the stop is rebuilt the way the source does.
        cushion = max(cfg.stop_buffer_atr * 100.0, 2 * cfg.tick_size,
                      cfg.stop_buffer_pct * 70100.0)
        stop = h.invalidation_price - cushion
        mid = (72140.0 + 75000.0) / 2.0
        cost_from_zone = (2 * cfg.fee_bps_taker / 10_000) * mid / abs(mid - stop)
        assert cost_from_zone < cfg.max_cost_r < refused.cost_r, (
            f"the fixture no longer splits the two sides: from the zone the fees are "
            f"{cost_from_zone:.1%} of R and from the fill {refused.cost_r:.1%}, against a ceiling "
            f"of {cfg.max_cost_r:.0%}. Both sides now agree, so this case cannot tell which one "
            "the gate is reading"
        )

        # Worth naming, because it is the reason the R:R property test stays on the FAR side of the
        # zone: below the zone the fill flatters the ratio rather than punishing it — the stop is a
        # hundred points away, so the trade measures 118R — and the only thing telling the truth
        # about it is the cost. A gate reading the zone would wave it through on both counts.
        assert refused.rr_t2 > refused.rr_in_zone, (
            f"sanity: a long bought a hundred points above its stop should measure a huge R:R at "
            f"the fill; it reports {refused.rr_t2:.2f} against {refused.rr_in_zone:.2f} in zone"
        )

        cut = build_plan(h, 76500.0, 2000.0, cfg)
        assert cut.viable, cut.reasons
        assert cut.stop_atr > cfg.max_stop_atr and cut.size_factor < 1.0, (
            f"a long bought at 76,500 has its stop {cut.stop_atr:.2f} ATR away, past the "
            f"{cfg.max_stop_atr} limit, and came back at full size ({cut.size_factor:.0%}). "
            "Measured from the zone the same stop is 2.0 ATR and nothing is cut, so the size is "
            "being set on a distance the user is not taking"
        )
        assert cut.size_factor == pytest.approx(cfg.max_stop_atr / cut.stop_atr)
        assert any("size cut" in x for x in cut.reasons), cut.reasons


class TestTheZoneFigureIsGuardedLikeTheHeadline:
    """★ The same `abs()` trap as the entry gate, one line down, and it survived the pass that
    fixed the other one.

    `rr_in_zone` divides by the distance from the zone midpoint to the stop. Written with `abs()`
    it reports a midpoint on the DEAD side of the stop as ordinary positive risk and hands back a
    perfectly plausible ratio — "and 4.1R if your limit fills in the zone" — for a limit order that
    would be underwater the moment it filled. The number the reader is invited to wait for is then
    the most attractive thing on the card and the most impossible.

    Not reachable through `match_impulses`: 1,532 real plans, zero crossings, nearest approach
    0.058R. But `build_plan` is exported and takes the invalidation it is handed, and the entry
    gate's own history is that "unreachable today" is exactly how it got here — it was unreachable
    too, right up until the arithmetic moved to the fill price and it became 71 plans in 925.

    The geometry is solved for rather than written down, so the fixture cannot quietly stop being
    the case it describes if a cushion or a retracement ever changes.
    """

    @pytest.mark.parametrize("name,pts,direction,atr", [
        ("long", (70000.0, 80000.0, 74000.0), Direction.LONG, 900.0),
        ("short", (80000.0, 70000.0, 76000.0), Direction.SHORT, 900.0),
    ], ids=["long", "short"])
    def test_a_zone_midpoint_on_the_dead_side_of_the_stop_quotes_no_reward(
            self, name, pts, direction, atr):
        cfg = PlanConfig()
        s = 1 if direction is Direction.LONG else -1
        h = _hypothesis(pts, direction)

        # Where the zone actually is, asked of the code rather than recomputed here.
        base = build_plan(h, h.points[2], atr, cfg)
        assert base.plan is not None, base.reasons
        zone_mid = base.plan.entry_mid

        # Move the invalidation until the stop sits just past the zone midpoint, on the side that
        # kills it. `stop = invalidation - s * cushion`, and the cushion's percentage floor moves
        # with the invalidation, so solve, then step clear of the wobble.
        cushion = max(cfg.stop_buffer_atr * atr, 2 * cfg.tick_size,
                      cfg.stop_buffer_pct * zone_mid)
        broken = replace(h, invalidation_price=zone_mid + s * (cushion + 400.0))
        r = build_plan(broken, zone_mid + s * 4000.0, atr, cfg)

        # The fill is far on the live side, so the entry gate cannot be what refuses this: whatever
        # happens here is the zone figure's own doing.
        assert r.rr_t2 > 0, (
            f"{name}: the fill was rejected too, so this no longer isolates the zone figure — "
            f"the plan came back with {r.reasons}"
        )
        assert r.rr_in_zone == 0.0, (
            f"{name}: the card offers {r.rr_in_zone:.2f}R for a limit filling at {zone_mid:,.0f}, "
            f"which is on the dead side of its own stop. There is no reward to quote on an entry "
            f"that is already a loss, and quoting one puts the most attractive number on the card "
            f"next to the least reachable trade"
        )


class TestAnEntryPastItsOwnStopIsRefusedRatherThanSized:
    """★ The refusal that only became reachable when the arithmetic moved to the fill price.

    A long whose market sits BELOW the stop, or a short whose market sits above it, is not a poor
    trade — it is a purchase that is a loss on the first tick. The count is already over.

    `risk = abs(entry - stop)` reports that as ordinary positive risk, so before this gate existed
    the plan came back looking entirely normal: a size, a cost, a required hit rate, all computed
    off a distance whose sign had been thrown away. It was unreachable while the card quoted the
    zone, because the zone midpoint is on the correct side of the stop by construction — the w4
    zone is truncated against P1 and the w2 zone is a retracement of a leg that ends above P0. The
    fill has no such guarantee: it is wherever the market is. Over six months of 4h bars, 71 of 925
    plans land here.

    Both directions are built deliberately, by solving for a price on the dead side of the stop
    rather than by hunting for one, so the fixture cannot quietly stop being the case it describes.
    """

    #: (name, vertices, direction, ATR). The invalidation is P0 in both — R1, the wave-2 stop.
    CASES: ClassVar[list[tuple]] = [
        ("long", (70000.0, 80000.0, 74000.0), Direction.LONG, 900.0),
        ("short", (80000.0, 70000.0, 76000.0), Direction.SHORT, 900.0),
    ]

    @pytest.mark.parametrize("name,pts,direction,atr", CASES, ids=[c[0] for c in CASES])
    def test_a_price_on_the_dead_side_of_the_stop_is_refused(self, name, pts, direction, atr):
        cfg = PlanConfig()
        h = _hypothesis(pts, direction)
        s = 1 if direction is Direction.LONG else -1

        # Solve for the stop the way the source does, then step a clear 500 past it — onto the side
        # where the count is finished. The cushion's percentage floor moves with the price, so the
        # step has to be wider than the cushion can shift, and 500 is far wider.
        cushion = max(cfg.stop_buffer_atr * atr, 2 * cfg.tick_size,
                      cfg.stop_buffer_pct * h.invalidation_price)
        stop = h.invalidation_price - s * cushion
        dead = stop - s * 500.0

        r = build_plan(h, dead, atr, cfg)
        assert not r.viable, (
            f"{name}: a {direction.name} was planned at {dead} with its stop at about {stop} — on "
            f"the far side of it. The plan came back sized at {r.size_factor:.0%} with "
            f"{r.rr_t2:.2f}R quoted: {r.plan}"
        )
        assert "past the stop" in r.reasons[0], (
            f"{name}: the refusal reads {r.reasons[0]!r}. The entry is past its own stop and it "
            "was turned down for some other reason, so the user is being given an argument about "
            "fees or volatility for a count that is simply over"
        )
        assert r.rr_t2 == 0 and r.size_factor == 1.0, (
            f"{name}: the refusal carries {r.rr_t2:.2f}R and a size of {r.size_factor:.0%}. There "
            "is no R to quote on a trade that loses on the first tick, and quoting one invites the "
            "reader to take it anyway"
        )

    @pytest.mark.parametrize("name,pts,direction,atr", CASES, ids=[c[0] for c in CASES])
    def test_the_zone_midpoint_could_never_have_reached_this_gate(self, name, pts, direction, atr):
        """The other half of the story, and the reason the refusal is NEW rather than newly
        noticed: the zone midpoint is on the live side of the stop by construction, so no price
        the market could print would have tripped this gate while the card quoted the zone."""
        cfg = PlanConfig()
        h = _hypothesis(pts, direction)
        s = 1 if direction is Direction.LONG else -1
        base = build_plan(h, h.points[2], atr, cfg)
        assert base.plan is not None, base.reasons
        assert s * (base.plan.entry_mid - base.plan.stop) > 0, (
            f"{name}: the middle of the published zone ({base.plan.entry_mid}) is on the dead side "
            f"of the stop ({base.plan.stop}). If that is now possible, the zone is being drawn "
            "across its own invalidation and the truncation has stopped working"
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

    #: 2000 rather than 1400 for the uncut case: the risk is now measured from the FILL at 75,000
    #: down to the stop, not from the middle of the zone, so the same scenario measures further in
    #: ATRs and 1400 lands at 3.82 — over the limit, and no longer the "inside the band" fixture
    #: this case is here to be.
    @pytest.mark.parametrize("atr,expect_cut", [(100.0, True), (900.0, True), (2000.0, False)])
    def test_a_far_stop_costs_size_and_only_size(self, atr, expect_cut):
        cfg = PlanConfig()
        h = _hypothesis(self.PTS)
        r = build_plan(h, self.PRICE, atr, cfg)
        assert r.viable, r.reasons
        assert r.stop_atr == pytest.approx(abs(self.PRICE - r.plan.stop) / atr), (
            f"stop_atr={r.stop_atr:.3f} is not the distance from the fill ({self.PRICE}) to the "
            f"stop ({r.plan.stop}) in ATRs. The size is cut on this number, so measuring it from "
            "the zone the market has not reached sizes the trade the user is not taking"
        )
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
