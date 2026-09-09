"""From hypothesis to plan: entry zone, stop, targets and the arithmetic of the rejection.

THE STOP COMES FROM THE INVALIDATION, not from an ATR multiple picked to taste. That is Elliott's
real practical advantage, whether or not its predictive power holds up: the count produces an
OBJECTIVE price at which the structure is false. The ATR only supplies a cushion so a wick hunt does
not kill a still-valid count.

A RULE THAT LOOKS MINOR AND IS NOT: if the distance to the stop exceeds ``max_stop_atr``, THE SIZE
IS CUT — the stop is never tightened. Tightening it severs the logical link between the stop and the
invalidation, which is the whole point of the framework: you would be left with a stop that means
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from wavelab.core.types import Direction, TradePlan
from wavelab.waves.matcher import Hypothesis

__all__ = ["PlanConfig", "PlanResult", "build_plan", "required_hit_rate"]


@dataclass(frozen=True, slots=True)
class PlanConfig:
    stop_buffer_atr: float = 0.25
    stop_buffer_pct: float = 0.001
    tick_size: float = 0.01
    max_stop_atr: float = 3.0
    min_stop_atr: float = 0.75
    fee_bps_taker: float = 7.5
    #: Maximum tolerable cost as a fraction of R. Above it the trade is REJECTED before modelling
    #: anything: if fees eat a fifth of the risk, no plausible statistical edge survives. At 7.5 bps
    #: per side, a 0.5% stop costs 0.30R and a 1.5% one costs 0.10R.
    max_cost_r: float = 0.20
    ev_min_r: float = 0.15
    expected_loss_r: float = 1.10
    exit_template_id: str = "std_2r_48b"


@dataclass(frozen=True, slots=True)
class PlanResult:
    """The plan plus the arithmetic that accepts or rejects it, always visible."""

    plan: TradePlan | None
    in_zone: bool
    rr_t2: float
    cost_r: float
    p_required: float
    stop_atr: float
    size_factor: float           # 1.0 normally; <1 when the stop is far away
    reasons: tuple[str, ...]     # why NOT, when there is no plan

    @property
    def viable(self) -> bool:
        return self.plan is not None


def required_hit_rate(rr: float, cost_r: float, cfg: PlanConfig) -> float:
    """``p = (L + c + EV_min) / (RR + L)``.

    ALWAYS shown, even when the tool rejects. It is the intuitive framing — "this R:R needs you to
    be right 36% of the time" — while the machine decides on the distribution of R.
    """
    L = cfg.expected_loss_r
    return (L + cost_r + cfg.ev_min_r) / (rr + L)


def _fib_zone(a: float, b: float, lo_r: float, hi_r: float) -> tuple[float, float]:
    """Retracement zone of the a→b leg between two ratios. Returned in order."""
    x, y = b - (b - a) * lo_r, b - (b - a) * hi_r
    return (min(x, y), max(x, y))


def build_plan(h: Hypothesis, price: float, atr: float, cfg: PlanConfig | None = None) -> PlanResult:
    """Build the plan for a hypothesis, or explain why there is none."""
    cfg = cfg or PlanConfig()
    s = 1 if h.direction is Direction.LONG else -1
    P = h.points
    reasons: list[str] = []

    if h.archetype not in ("w2", "w4"):
        return PlanResult(None, False, 0, 0, 0, 0, 1.0,
                          ((f"state '{h.terminal_label}' offers no entry; "
                            "inside wave 3 you manage, you do not enter."),))

    if h.archetype == "w2":
        # Wave 2 complete, wave 3 beginning: Elliott's highest-quality entry.
        # Zone = 0.5-0.786 retracement of wave 1. Core: golden pocket 0.618-0.65.
        lo, hi = _fib_zone(P[0], P[1], 0.500, 0.786)
        base = P[2]
        w1 = P[1] - P[0]
        targets = (base + w1 * 1.000, base + w1 * 1.618, base + w1 * 2.618)
    else:
        # Wave 4 complete, wave 5 beginning: lower confidence, and the stop is TIGHTER (P1 rather
        # than P0), so the R:R is usually worse even though it intuitively feels like the safer
        # entry.
        lo, hi = _fib_zone(P[2], P[3], 0.382, 0.500)
        # Truncate the zone so it can never invade wave 1's territory: an entry in there would be
        # an entry on the wrong side of its own invalidation.
        if h.direction is Direction.LONG:
            lo = max(lo, P[1] * 1.0005)
        else:
            hi = min(hi, P[1] * 0.9995)
        base = P[4]
        w1 = P[1] - P[0]
        targets = (base + w1 * 1.000, base + (P[3] - P[0]) * 0.618, base + w1 * 1.618)

    if lo >= hi:
        return PlanResult(None, False, 0, 0, 0, 0, 1.0,
                          (("the entry zone is empty once truncated against the invalidation: "
                            "there is nowhere left to enter above the stop."),))

    entry = (lo + hi) / 2.0

    # --- stop: invalidation plus a cushion ----------------------------------------------------
    buffer = max(cfg.stop_buffer_atr * atr, 2 * cfg.tick_size, cfg.stop_buffer_pct * price)
    stop = h.invalidation_price - s * buffer
    risk = abs(entry - stop)
    if risk <= 0:
        return PlanResult(None, False, 0, 0, 0, 0, 1.0,
                          ("the entry zone is on the far side of the stop.",))

    stop_atr = risk / atr if atr else float("inf")

    # The arithmetic is computed BEFORE any rejection, and always travels in the result.
    # A "no" with no numbers is an opinion; with numbers it is an argument the user can push back
    # on, and one that teaches them why that setup was not worth taking.
    # Funding is not computed on spot: it belongs to the perpetual, and here it would be an
    # invented cost.
    cost_r = (2 * cfg.fee_bps_taker / 10_000) * entry / risk
    rr_t2 = abs(targets[1] - entry) / risk
    p_req = required_hit_rate(rr_t2, cost_r, cfg)

    size = 1.0
    if stop_atr > cfg.max_stop_atr:
        # CUT THE SIZE, never tighten the stop: tightening it severs the link to the invalidation.
        size = cfg.max_stop_atr / stop_atr
        reasons.append(f"stop at {stop_atr:.1f} ATR (>{cfg.max_stop_atr}): size cut to "
                       f"{size:.0%}, the stop is NOT tightened")
    if stop_atr < cfg.min_stop_atr:
        return PlanResult(None, False, rr_t2, cost_r, p_req, stop_atr, 1.0,
                          ((f"stop at {stop_atr:.2f} ATR: it sits inside the noise floor "
                            f"(<{cfg.min_stop_atr} ATR) and any wick would sweep it."),))

    if cost_r > cfg.max_cost_r:
        return PlanResult(None, False, rr_t2, cost_r, p_req, stop_atr, size,
                          ((f"the round trip in fees takes {cost_r:.0%} of R "
                            f"(maximum {cfg.max_cost_r:.0%}): the stop is too close for any "
                            "plausible edge to survive the fees."),))

    in_zone = lo <= price <= hi
    if not in_zone:
        reasons.append(f"price ${price:,.0f} outside the zone ${lo:,.0f}-${hi:,.0f}")

    plan = TradePlan(
        archetype=f"{h.archetype}_{'long' if s > 0 else 'short'}",
        direction=h.direction, entry_lo=lo, entry_hi=hi, stop=stop,
        invalidation_price=h.invalidation_price, invalidation_rule=h.invalidation_rule,
        targets=targets, exit_template_id=cfg.exit_template_id, count_id=h.id,
    )
    return PlanResult(plan, in_zone, rr_t2, cost_r, p_req, stop_atr, size, tuple(reasons))
