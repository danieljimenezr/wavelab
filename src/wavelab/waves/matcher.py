"""Template matcher: finds candidate structures in the pivot sequence.

THREE DECISIONS THAT MAKE IT VIABLE:

1. **CONSECUTIVE pivots only** at a given degree. Allowing gaps turns the problem into O(k⁵) tuples
   and is what sinks the implementations out there. Higher degrees do not come from skipping pivots:
   they come from the recursion of the grammar.

2. **RIGHT-ANCHORED**: any candidate whose last vertex sits more than ``max_from_edge`` pivots from
   the edge is discarded. History you cannot trade does not deserve CPU, and a count over structure
   that finished months ago produces no decision.

3. **PARTIAL states included**. A matcher that only looks for complete fives would never find an
   entry, because the entries live in half-built impulses.

Cost: ``max_from_edge + 1`` starts × 4 states × 2 directions — 32 checks per update at the shipped
``max_from_edge = 3`` — and it is only recomputed when a new pivot is CONFIRMED (once every 6-20
bars). Cheaper than the indicator battery. This line used to say "~13 starts ≈ 100 checks", which
was the arithmetic of a second knob, ``max_starts``, that never once bound: right-anchoring makes
the number of starting points identical to the number of end points, so ``max_from_edge`` was
always the tighter of the two and the estimate was 3× the real cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wavelab.core.types import Direction, Pivot, PivotKind
from wavelab.waves.rules import ARCHETYPES, ImpulseState, check_impulse

__all__ = ["Hypothesis", "MatcherConfig", "match_impulses", "score_guidelines"]


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One possible reading of the current structure. NEVER produced on its own.

    The product is an ORDERED set of hypotheses: a real count is not unique, and presenting a single
    one asserts a certainty that does not exist.
    """

    state: ImpulseState
    direction: Direction
    pivots: tuple[Pivot, ...]
    points: tuple[float, ...]
    score: float
    fit: dict[str, float] = field(default_factory=dict)
    invalidation_price: float = 0.0
    invalidation_rule: str = ""
    archetype: str | None = None
    truncated: bool = False

    @property
    def id(self) -> str:
        return f"{self.state.value}:{self.direction.name}:{self.pivots[0].ts_ms}"

    @property
    def terminal_label(self) -> str:
        """Which wave we are in RIGHT NOW. This is what goes on the chart label."""
        return {ImpulseState.AT_2: "w2 complete → inside w3",
                ImpulseState.AT_3: "w3 complete → inside w4",
                ImpulseState.AT_4: "w4 complete → inside w5",
                ImpulseState.COMPLETE: "five complete → ABC expected"}[self.state]


@dataclass(frozen=True, slots=True)
class MatcherConfig:
    #: How many pivots from the edge the last vertex may sit. Beyond that the structure has already
    #: finished and produces no decision. Because every candidate is right-anchored and has a fixed
    #: length, this ALSO fixes how many starting points are tried: exactly one per end point, so
    #: ``max_from_edge + 1`` of them. There was a second knob here, ``max_starts = 13``, that
    #: claimed to control that count and could not: each candidate's start is `end - k`, so the
    #: earliest start the loop ever reaches is `n - max_from_edge - k`, which is never below
    #: `n - max_starts - k` for any `max_starts >= max_from_edge`. It was an operator-facing dial
    #: wired to nothing — turn it and the matcher searched exactly the same pivots — so it is gone
    #: rather than fixed: this field is the axis it was pretending to be a second opinion on.
    max_from_edge: int = 3
    #: Guideline weights. These are HAND-PICKED CONSTANTS: changing one is an experiment and goes to
    #: trials.sqlite, because tuning by eye without recording it is what makes the effective N lie.
    w_fib2: float = 0.30
    w_fib3: float = 0.30
    w_fib4: float = 0.20
    w_alt: float = 0.20
    top_n: int = 4


def _gauss(x: float, mu: float, sigma: float) -> float:
    """Closeness to a Fibonacci ratio, in [0,1]. Unnormalised: only ever compared with itself."""
    return float(pow(2.718281828459045, -0.5 * ((x - mu) / sigma) ** 2))


def score_guidelines(points: tuple[float, ...], cfg: MatcherConfig) -> tuple[float, dict[str, float]]:
    """Scores the GUIDELINES (which do not invalidate) against the hard rules (which do).

    Kept apart deliberately: a broken guideline lowers confidence, a broken rule kills the count.
    Mixing them would turn "unusual" into "impossible", and the atypical structures are exactly the
    ones carrying the most information.
    """
    fit: dict[str, float] = {}
    n = len(points)

    # w2: typical retracement 0.5-0.786, with the golden pocket at 0.618.
    if n >= 3 and points[1] != points[0]:
        r2 = abs(points[2] - points[1]) / abs(points[1] - points[0])
        fit["retr_w2"] = r2
        s2 = _gauss(r2, 0.618, 0.18)
    else:
        s2 = 0.0

    # w3: typical extension 1.618 of w1. An extended wave 3 is the signature of a healthy impulse.
    if n >= 4 and points[1] != points[0]:
        e3 = abs(points[3] - points[2]) / abs(points[1] - points[0])
        fit["ext_w3"] = e3
        s3 = max(_gauss(e3, 1.618, 0.45), _gauss(e3, 2.618, 0.55))
    else:
        s3 = 0.0

    # w4: typical retracement 0.382 of w3, far shallower than w2.
    if n >= 5 and points[3] != points[2]:
        r4 = abs(points[4] - points[3]) / abs(points[3] - points[2])
        fit["retr_w4"] = r4
        s4 = _gauss(r4, 0.382, 0.15)
    else:
        s4 = 0.0

    # Alternation: if w2 is deep, w4 is flat, and the other way round. Elliott's most reliable
    # guideline.
    if "retr_w2" in fit and "retr_w4" in fit:
        alt = abs(fit["retr_w2"] - fit["retr_w4"])
        fit["alternation"] = alt
        sa = min(1.0, alt / 0.30)
    else:
        sa = 0.0

    weights = [(cfg.w_fib2, s2, n >= 3), (cfg.w_fib3, s3, n >= 4),
               (cfg.w_fib4, s4, n >= 5), (cfg.w_alt, sa, n >= 5)]
    active = [(w, s) for w, s, ok in weights if ok]
    total_w = sum(w for w, _ in active) or 1.0
    score = sum(w * s for w, s in active) / total_w
    return score, fit


def match_impulses(
    pivots: tuple[Pivot, ...],
    cfg: MatcherConfig | None = None,
    *,
    allow_overlap: bool = False,
    directions: tuple[Direction, ...] = (Direction.LONG, Direction.SHORT),
) -> list[Hypothesis]:
    """Returns the viable hypotheses, ORDERED best to worst.

    Only confirmed pivots are considered: the tentative one may annotate the chart, but it can never
    produce a hypothesis anyone trades.
    """
    cfg = cfg or MatcherConfig()
    if len(pivots) < 3:
        return []

    n = len(pivots)
    out: list[Hypothesis] = []
    states = ((3, ImpulseState.AT_2), (4, ImpulseState.AT_3),
              (5, ImpulseState.AT_4), (6, ImpulseState.COMPLETE))

    for k, state in states:
        # Right-anchored: the candidate's end has to be near the edge.
        for end in range(n, max(n - cfg.max_from_edge - 1, k - 1), -1):
            start = end - k
            if start < 0:      # a slice guard: a negative start would wrap silently, not raise
                continue
            seq = pivots[start:end]
            pts = tuple(p.price for p in seq)

            for d in directions:
                # The first leg fixes the direction: a bullish impulse starts from a low.
                expected = PivotKind.LOW if d is Direction.LONG else PivotKind.HIGH
                if seq[0].kind is not expected:
                    continue
                r = check_impulse(list(pts), d, allow_overlap=allow_overlap)
                if not r.valid:
                    continue
                score, fit = score_guidelines(pts, cfg)
                out.append(Hypothesis(
                    state=state, direction=d, pivots=seq, points=pts,
                    score=score, fit=fit,
                    invalidation_price=r.invalidation_price,
                    invalidation_rule=r.invalidation_rule,
                    archetype=ARCHETYPES[state], truncated=r.truncated,
                ))

    # Tie-break: guideline score first; on a tie, the MORE COMPLETE structure, which asserts more
    # and is therefore more falsifiable.
    out.sort(key=lambda h: (h.score, h.state.n_points), reverse=True)
    return out[: cfg.top_n]
