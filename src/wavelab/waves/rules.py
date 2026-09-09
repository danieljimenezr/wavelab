"""Elliott's hard rules. Boolean, and each one emits ITS OWN invalidation price.

This is the best property of the design: the biggest number on the signal card is produced by the
rule engine, not assembled downstream. That way it cannot drift away from the rule that justifies
it — if the stop and its reason were computed in different modules, sooner or later they would say
different things and nobody would notice.

★ PARTIAL NON-TERMINALS. An engine that only evaluates COMPLETE impulses is an annotation toy: 100%
of the entries live in incomplete impulses (end of wave 2, end of wave 4). That is why every partial
state has its own SUBSET of evaluable rules and its own invalidation:

    Impulse@2  (w1,w2 done, inside w3)     R1          invalidation P0   -> archetype w2_long
    Impulse@3  (inside w4)                 R1, R2b     invalidation P1   -> watch only
    Impulse@4  (inside w5)                 R1,R2b,R3   invalidation P1   -> archetype w4_long
    Impulse    (complete)                  + R2        invalidation P1

R2 ("wave 3 is never the shortest") is NOT evaluable before w5 exists: it degrades to a GUIDELINE
while the impulse is incomplete. Flagging it as a hard rule and evaluating it anyway would be making
a verdict up.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from wavelab.core.types import Direction, RuleVerdict

__all__ = [
    "ARCHETYPES",
    "ImpulseState",
    "RuleSet",
    "check_impulse",
    "fib_projection",
    "fib_retracement",
    "invalidation_for",
]


class ImpulseState(StrEnum):
    """How far the impulse is built. Determines what can be asserted."""
    AT_2 = "impulse@2"     # P0,P1,P2 known; inside wave 3
    AT_3 = "impulse@3"     # + P3; inside wave 4
    AT_4 = "impulse@4"     # + P4; inside wave 5
    COMPLETE = "impulse"   # + P5

    @property
    def n_points(self) -> int:
        return {"impulse@2": 3, "impulse@3": 4, "impulse@4": 5, "impulse": 6}[self.value]


#: Which trade archetype each state offers. `None` = offers no entry, watch only.
ARCHETYPES: dict[ImpulseState, str | None] = {
    ImpulseState.AT_2: "w2",   # Elliott's highest-quality entry
    ImpulseState.AT_3: None,   # inside w3: you no longer enter, you manage
    ImpulseState.AT_4: "w4",   # lower confidence; watch for truncation and ending diagonals
    ImpulseState.COMPLETE: "abc",  # after w5 a correction is expected
}


@dataclass(frozen=True, slots=True)
class RuleSet:
    """A candidate's verdicts, with its invalidation and its archetype."""

    state: ImpulseState
    direction: Direction
    verdicts: tuple[RuleVerdict, ...]
    invalidation_price: float
    invalidation_rule: str
    archetype: str | None
    truncated: bool = False

    @property
    def valid(self) -> bool:
        """Only EVALUABLE rules count. A rule that cannot be evaluated invalidates nothing."""
        return all(v.ok for v in self.verdicts if v.evaluable)

    @property
    def broken(self) -> tuple[str, ...]:
        return tuple(v.rule for v in self.verdicts if v.evaluable and not v.ok)


def _s(direction: Direction) -> int:
    return 1 if direction is Direction.LONG else -1


def check_impulse(
    points: list[float],
    direction: Direction = Direction.LONG,
    *,
    allow_overlap: bool = False,
) -> RuleSet:
    """Evaluate an impulse from its vertices ``[P0, P1, P2, ...]``.

    3 points = Impulse@2, 4 = @3, 5 = @4, 6 = complete. With fewer than 3 there is nothing to
    assert.

    ``allow_overlap`` is per pattern AND per asset: always true for diagonals, and true for impulses
    only in futures and commodities (Prechter's exception). One line of configuration, which is the
    cheapest possible implementation of the multi-asset hook.
    """
    n = len(points)
    if n < 3:
        raise ValueError(f"at least 3 vertices are needed (P0,P1,P2); got {n}")
    if n > 6:
        raise ValueError(f"an impulse has at most 6 vertices; got {n}")

    state = {3: ImpulseState.AT_2, 4: ImpulseState.AT_3,
             5: ImpulseState.AT_4, 6: ImpulseState.COMPLETE}[n]
    s = _s(direction)
    P = points
    v: list[RuleVerdict] = []

    # --- R1: wave 2 never retraces 100% of wave 1. Invalidation at P0. -----------------------
    r1_ok = s * P[2] > s * P[0]
    retr2 = abs(P[2] - P[1]) / abs(P[1] - P[0]) if P[1] != P[0] else float("inf")
    v.append(RuleVerdict(
        "R1", r1_ok, P[0],
        f"w2 retraces {retr2:.1%} of w1" + ("" if r1_ok else " — past 100%")))

    # --- R2b: wave 3 goes beyond the end of wave 1. Invalidation at P1. ----------------------
    if n >= 4:
        ok = s * P[3] > s * P[1]
        v.append(RuleVerdict(
            "R2b", ok, P[1],
            "w3 clears the end of w1" if ok else "w3 does NOT clear the end of w1"))
    else:
        v.append(RuleVerdict("R2b", True, P[1], "still inside w3: not evaluable", evaluable=False))

    # --- R3: wave 4 does not enter wave 1's territory. Invalidation at P1. -------------------
    if n >= 5:
        ok = (s * P[4] > s * P[1]) or allow_overlap
        note = "w4 respects w1's territory"
        if not ok:
            note = "w4 ENTERS w1's territory"
        elif allow_overlap and s * P[4] <= s * P[1]:
            note = "w4 overlaps, allowed on this asset/pattern (Prechter's exception)"
        v.append(RuleVerdict("R3", ok, P[1], note))
    else:
        v.append(RuleVerdict("R3", True, P[1], "still inside w4: not evaluable", evaluable=False))

    # --- R2: wave 3 is never the shortest of 1/3/5. ------------------------------------------
    # NOT evaluable without w5. Degrading it to a guideline instead of faking a verdict is the
    # difference between an honest engine and one that invents certainties about structure that
    # does not exist yet.
    if n == 6:
        l1, l3, l5 = abs(P[1] - P[0]), abs(P[3] - P[2]), abs(P[5] - P[4])
        ok = not (l3 < l1 and l3 < l5)
        v.append(RuleVerdict("R2", ok, P[1],
                             f"lengths w1={l1:.0f} w3={l3:.0f} w5={l5:.0f}"
                             + ("" if ok else " — w3 is the SHORTEST")))
    else:
        v.append(RuleVerdict("R2", True, P[1],
                             "w5 does not exist yet: R2 is a guideline, not a rule",
                             evaluable=False))

    # --- Truncation: LEGAL. No rule requires w5 to go beyond w3. -----------------------------
    # Rejecting it would throw away exactly the counts you want most near a top.
    truncated = n == 6 and (s * P[5] <= s * P[3])

    price, rule = invalidation_for(state, points, direction)
    return RuleSet(state, direction, tuple(v), price, rule, ARCHETYPES[state], truncated)


def invalidation_for(state: ImpulseState, points: list[float],
                     direction: Direction = Direction.LONG) -> tuple[float, str]:
    """The price that falsifies this structure, and the NAME of the rule that produces it.

    Inside wave 3 the invalidation is P0, by R1. From wave 4 onwards it moves to P1, because both
    R2b and R3 lean on it: a tighter stop, and the reason the wave 4 entry has a worse R:R even
    though it feels safer.
    """
    if state is ImpulseState.AT_2:
        return points[0], "R1 (w2 cannot retrace 100% of w1)"
    return points[1], "R3 (w4 cannot enter w1's territory)"


# ---------------------------------------------------------------------------- Fibonacci

#: Typical retracements per wave. The "golden pocket" 0.618-0.65 is the core of the w2 zone.
RETRACEMENTS = {
    "w2": (0.500, 0.786, 0.618, 0.650),   # (min, max, core_lo, core_hi)
    "w4": (0.382, 0.500, 0.382, 0.450),
}

#: Extensions used to project targets.
EXTENSIONS = (1.000, 1.272, 1.618, 2.618)


def fib_retracement(a: float, b: float, ratios: tuple[float, ...]) -> list[float]:
    """Retracements of the a→b leg. In LINEAR space, because a retracement is a fraction of the
    leg and not a price ratio: logs add nothing here and only confuse."""
    return [b - (b - a) * r for r in ratios]


def fib_projection(origin: float, leg_a: float, leg_b: float,
                   ratios: tuple[float, ...] = EXTENSIONS) -> list[float]:
    """Project targets from ``origin`` using the length of the ``leg_a→leg_b`` leg."""
    length = leg_b - leg_a
    return [origin + length * r for r in ratios]
