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
from wavelab.waves.rules import RETRACEMENTS

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

    def __post_init__(self) -> None:
        # A floor above its own ceiling refuses every stop that exists: too tight for the noise
        # floor and too wide for the size cut, at the same time, for any distance. Nothing crashes
        # — build_plan simply turns everything down, with a reason that blames the market. The
        # operator reads "the stop sits inside the noise floor" on a setup after setup and has no
        # way to learn that the knob they turned made it unanswerable.
        if self.min_stop_atr > self.max_stop_atr:
            raise ValueError(
                f"min_stop_atr ({self.min_stop_atr}) is above max_stop_atr ({self.max_stop_atr}): "
                "no stop distance can satisfy both, so every plan would be refused and the reason "
                "given would be about the market rather than about this setting.")
        if self.max_cost_r <= 0 or self.fee_bps_taker < 0 or self.tick_size <= 0:
            raise ValueError(
                f"max_cost_r ({self.max_cost_r}) and tick_size ({self.tick_size}) must be positive "
                f"and fee_bps_taker ({self.fee_bps_taker}) cannot be negative.")


@dataclass(frozen=True, slots=True)
class PlanResult:
    """The plan plus the arithmetic that accepts or rejects it, always visible."""

    plan: TradePlan | None
    in_zone: bool
    rr_t2: float                 # quoted at the FILL price: what you get buying now, at market
    rr_in_zone: float            # what you would get if a limit inside the zone fills instead
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
        return PlanResult(None, False, 0, 0, 0, 0, 0, 1.0,
                          ((f"state '{h.terminal_label}' offers no entry; "
                            "inside wave 3 you manage, you do not enter."),))

    if h.archetype == "w2":
        # Wave 2 complete, wave 3 beginning: Elliott's highest-quality entry.
        # Zone = 0.5-0.786 retracement of wave 1. Core: golden pocket 0.618-0.65.
        # The two ratios come from `rules.RETRACEMENTS`, which is the one place they are defined.
        # They used to be open-coded here as well, which is two definitions of one published level:
        # the zone drawn on the card would have followed this literal and the table would have gone
        # on claiming something else, with nothing to notice the difference.
        zone_lo, zone_hi = RETRACEMENTS["w2"][:2]
        lo, hi = _fib_zone(P[0], P[1], zone_lo, zone_hi)
        base = P[2]
        w1 = P[1] - P[0]
        targets = (base + w1 * 1.000, base + w1 * 1.618, base + w1 * 2.618)
    else:
        # Wave 4 complete, wave 5 beginning: lower confidence, and the stop is TIGHTER (P1 rather
        # than P0), so the R:R is usually worse even though it intuitively feels like the safer
        # entry.
        zone_lo, zone_hi = RETRACEMENTS["w4"][:2]
        lo, hi = _fib_zone(P[2], P[3], zone_lo, zone_hi)
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
        return PlanResult(None, False, 0, 0, 0, 0, 0, 1.0,
                          (("the entry zone is empty once truncated against the invalidation: "
                            "there is nowhere left to enter above the stop."),))

    # THE QUOTED ARITHMETIC IS COMPUTED AT THE PRICE YOU CAN ACTUALLY GET, which is `price` — the
    # close of the bar that produced this plan, and the same price the backtest books at.
    #
    # It used to be computed at the middle of the entry zone, and the two disagreed on 131 of 132
    # signals, the worst by 2.29R: the card showed 6.32 while the trade it was measured against
    # returned 4.03. The zone figure is the better one, because the zone is a retracement you are
    # waiting for — so the card was quoting a number the measurement did not deliver, in a product
    # whose whole argument is that it does not flatter you.
    #
    # A limit order resting in the zone may never fill. Quoting it as the headline assumes a price
    # you might not get, which is the optimistic bias this project exists to catch; quoting the
    # close assumes only that you can buy at market, which is always true. So the conservative
    # number leads and decides the gates, and the zone figure travels beside it as upside.
    #
    # Two things that were claimed for this change and do not survive being measured, kept here
    # because the next reader will otherwise reach for them again:
    #
    # It is NOT true that the two numbers are nearly the same inside the zone. They are the same
    # ratio read at two entry prices, and the gap is (rr_in_zone + 1) * |price - mid| / risk — zero
    # only at the midpoint, and widening the whole way to either rim. Over six months of 4h bars,
    # 273 viable in-zone plans disagreed by a median 0.78R (19.9% of the zone figure) and by up to
    # 2.95R; 99.6% of them by more than the two decimals the card prints. Both numbers therefore
    # travel on the card in the zone as well as outside it.
    #
    # And the headline is not always the smaller of the two. Below the zone a long is buying
    # CHEAPER than the entry the zone figure is quoted at, so the fill is the better trade and
    # rr_t2 is legitimately the larger number: 179 of 1,532 viable plans. The invariant is that
    # the headline is the ratio at a price the reader can actually get — not that it is the lower
    # figure. On the far side of the zone, which is where the old arithmetic flattered, it is both.
    entry = price
    entry_zone = (lo + hi) / 2.0

    # --- stop: invalidation plus a cushion ----------------------------------------------------
    buffer = max(cfg.stop_buffer_atr * atr, 2 * cfg.tick_size, cfg.stop_buffer_pct * price)
    stop = h.invalidation_price - s * buffer
    risk = abs(entry - stop)
    if risk <= 0:
        return PlanResult(None, False, 0, 0, 0, 0, 0, 1.0,
                          ("the entry zone is on the far side of the stop.",))

    # The price is already PAST its own stop: a long whose market is below where it would give up,
    # a short whose market is above. `abs()` above reports that as ordinary positive risk, so the
    # plan would come out looking normal — with a size, a cost and a required hit rate — while
    # describing a purchase that is a loss the instant it is made.
    #
    # Only reachable since the arithmetic moved to the fill price: the zone midpoint is on the
    # correct side of the stop by construction, so this could not happen while the card quoted the
    # zone. Measured over six months of 4h bars, 71 of 925 plans land here.
    if s * (entry - stop) <= 0:
        return PlanResult(None, False, 0, 0, 0, 0, 0, 1.0,
                          ((f"price ${entry:,.0f} is already past the stop ${stop:,.0f}: this "
                            "count is over, and buying here is a loss on the first tick."),))

    if atr <= 0:
        # A market with no range. Every number below is measured in ATRs, so with no ATR there is
        # no noise floor to clear and no size to cut: `risk / 0` took the `inf` branch, `inf` was
        # above max_stop_atr, and the size came out as 3.0/inf = 0.0. The card then published a
        # viable plan — entry zone, stop, three targets — instructing the reader to take 0% of a
        # position. A "yes" nobody can act on, from a tool whose whole claim is that its refusals
        # come with arithmetic. WilderATR returns a real 0.0 over a flat series, not None, so the
        # route above does not catch this either.
        return PlanResult(None, False, 0, 0, 0, 0, 0, 1.0,
                          (("the market has no range at all (ATR is zero): there is no volatility "
                            "to measure a stop against, so there is no plan to make here."),))

    stop_atr = risk / atr

    # The arithmetic is computed BEFORE any rejection, and always travels in the result.
    # A "no" with no numbers is an opinion; with numbers it is an argument the user can push back
    # on, and one that teaches them why that setup was not worth taking.
    # Funding is not computed on spot: it belongs to the perpetual, and here it would be an
    # invented cost.
    cost_r = (2 * cfg.fee_bps_taker / 10_000) * entry / risk
    rr_t2 = abs(targets[1] - entry) / risk
    # The same `abs()` trap the headline was just guarded against, one line down. A zone midpoint
    # on the far side of its own stop measures as ordinary positive risk and yields a plausible
    # ratio for a trade that is a loss on entry. It is unreachable through `match_impulses` — 1,532
    # plans measured, zero crossings, nearest approach 0.058R — but `build_plan` is exported and
    # takes a hand-set invalidation, and "unreachable today" is how the headline case got here.
    # 0.0 reads as "no reward", which is the honest answer for a geometry that makes no sense.
    risk_zone = s * (entry_zone - stop)
    rr_in_zone = abs(targets[1] - entry_zone) / risk_zone if risk_zone > 0 else 0.0
    p_req = required_hit_rate(rr_t2, cost_r, cfg)

    size = 1.0
    if stop_atr > cfg.max_stop_atr:
        # CUT THE SIZE, never tighten the stop: tightening it severs the link to the invalidation.
        size = cfg.max_stop_atr / stop_atr
        reasons.append(f"stop at {stop_atr:.1f} ATR (>{cfg.max_stop_atr}): size cut to "
                       f"{size:.0%}, the stop is NOT tightened")
    if stop_atr < cfg.min_stop_atr:
        return PlanResult(None, False, rr_t2, rr_in_zone, cost_r, p_req, stop_atr, 1.0,
                          ((f"stop at {stop_atr:.2f} ATR: it sits inside the noise floor "
                            f"(<{cfg.min_stop_atr} ATR) and any wick would sweep it."),))

    if cost_r > cfg.max_cost_r:
        return PlanResult(None, False, rr_t2, rr_in_zone, cost_r, p_req, stop_atr, size,
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
    return PlanResult(plan, in_zone, rr_t2, rr_in_zone, cost_r, p_req, stop_atr, size,
                      tuple(reasons))
