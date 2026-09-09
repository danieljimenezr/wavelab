"""The types that define the seams of the system.

No logic here: only the contracts. If a type in this module has to change in order to add a feature
that was always foreseen (another asset, another signal source, news), the seam was drawn wrong.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from wavelab.core.timeframes import Timeframe, close_time_for

__all__ = [
    "AuxEvent",
    "Bar",
    "Decision",
    "Direction",
    "Event",
    "ExitTemplate",
    "MaturityLevel",
    "Pivot",
    "PivotKind",
    "RuleVerdict",
    "Signal",
    "SourceKind",
    "Stat",
    "TradePlan",
    "Verdict",
]


# --------------------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Bar:
    """A bar. Immutable, with the `close_time` contract asserted at construction.

    ``n_source_bars`` and ``is_gap`` travel WITH the bar, not in a parallel structure: a 1h bar
    built out of 43 minutes has to stay recognisable as such at every point in the system, and a
    separate mask is lost on the first slice.
    """

    symbol: str
    tf: Timeframe
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_base: float = 0.0
    taker_buy_quote: float = 0.0
    is_closed: bool = True
    n_source_bars: int = 0
    is_gap: bool = False

    def __post_init__(self) -> None:
        # Cheap (one integer comparison) and it catches the costliest mistake in the project:
        # an adapter that invents its own close_time convention.
        if self.open_time_ms % self.tf.ms != 0:
            raise ValueError(
                f"Bar {self.symbol} {self.tf}: open_time_ms={self.open_time_ms} does not land on "
                f"the {self.tf} UTC grid (remainder {self.open_time_ms % self.tf.ms}). "
                "The adapter must align to the grid, not round."
            )

    @property
    def close_time_ms(self) -> int:
        return close_time_for(self.open_time_ms, self.tf)

    @property
    def coverage(self) -> float:
        """Fraction of the 1m bars actually present. 1.0 = complete."""
        exp = self.tf.expected_source_bars
        return 1.0 if exp <= 1 else self.n_source_bars / exp

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class AuxEvent:
    """Anything that is not a bar: funding, liquidation, news item, macro event.

    It exists from day 1 even though the news layer is v2. The two separate timestamps are the
    reason: ``ts_event_ms`` (when it happened) and ``ts_ingest_ms`` (when we found out). That
    distinction CANNOT be reconstructed after the fact, so either it is recorded from the start or
    the news history is born useless for calibrating latency.
    """

    ts_event_ms: int
    ts_ingest_ms: int
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> int:
        return self.ts_ingest_ms - self.ts_event_ms


Event = Bar | AuxEvent


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------

class SourceKind(StrEnum):
    """Where a signal comes from. The per-kind cap is configured and applied at merge time,
    so switching news on in v2 is one line of config and not a refactor."""
    STRUCTURE = "structure"      # Elliott, break of structure, levels
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    FLOW = "flow"                # volume, CVD, VWAP
    DERIVATIVES = "derivatives"  # funding, OI, ratios (cross-instrument context on spot)
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    NEWS = "news"                # capped at 0.0 in v1; raised to 0.25 in v2 by editing config


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1

    @property
    def sign(self) -> int:
        return int(self.value)


class Verdict(StrEnum):
    """Three states, never a boolean.

    NO_TRADE is the COMMON case and always shows the arithmetic behind the rejection.
    WATCH covers two distinct situations the user has to be able to tell apart: there is structure
    but price is not in the zone, or the count is still tentative.
    """
    NO_TRADE = "no_trade"
    WATCH = "watch"
    ACTIONABLE = "actionable"


class MaturityLevel(IntEnum):
    """How much evidence backs what is being shown. Computed per cell, not globally."""
    PRIOR = 0             # hand-written table; no probability on screen
    HISTORICAL = 1        # purged walk-forward over 9 years; contaminated by selection
    HISTORICAL_RIGOR = 2  # CPCV, SPA, Deflated Sharpe, PBO
    FORWARD = 3           # uncontaminated evidence accumulated live


# --------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------

class PivotKind(IntEnum):
    LOW = -1
    HIGH = 1


@dataclass(frozen=True, slots=True)
class Pivot:
    """A ZigZag extreme, with its TWO separate timestamps.

    ``idx``/``ts_ms`` is where it is DRAWN (the bar of the extreme).
    ``confirmed_idx``/``confirmed_ts_ms`` is the first bar on which we were ALLOWED to know it.

    Conflating them is the entire class of bug. The lag between the two is a first-passage time to
    a barrier: a median of a few bars, a very heavy right tail, hundreds of bars in a strong trend.
    A fixed lag can never be assumed.

    ``thr_at_extreme`` is FROZEN at the bar of the extreme. That is what makes confirmation
    monotonic: if the threshold were recomputed with today's ATR, a pivot confirmed yesterday could
    stop being confirmed, the history of counts would stop being append-only, and a count the user
    had already seen would vanish with no invalidation event — precisely the dishonesty this design
    exists to eliminate.
    """

    idx: int
    ts_ms: int
    price: float
    kind: PivotKind
    thr_at_extreme: float
    confirmed_idx: int | None = None
    confirmed_ts_ms: int | None = None

    @property
    def is_confirmed(self) -> bool:
        return self.confirmed_idx is not None

    @property
    def confirm_price(self) -> float:
        """The price at which this pivot would become confirmed.

        Drawn as a dashed grey line: "the count confirms below 108,240". It turns Elliott's
        repainting weakness into the most actionable line on the chart, because the user stops
        seeing "this might be the top" and starts seeing the exact price at which it stops being a
        maybe.
        """
        return (self.price - self.thr_at_extreme if self.kind is PivotKind.HIGH
                else self.price + self.thr_at_extreme)

    def confirmed_at(self, idx: int, ts_ms: int) -> Pivot:
        """Return the confirmed version. WRITE-ONCE: re-confirming is a program error."""
        if self.is_confirmed:
            raise ValueError(
                f"Pivot at idx={self.idx} was already confirmed at {self.confirmed_idx}; "
                "confirmation is write-once so that the beam can only GROW."
            )
        return Pivot(self.idx, self.ts_ms, self.price, self.kind, self.thr_at_extreme, idx, ts_ms)


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    """The result of a hard rule, WITH its own invalidation price.

    Having every rule emit its own invalidation is the best property of the design: the biggest
    number on the signal card is produced by the rule engine and is not assembled downstream, so it
    cannot drift away from the rule that justifies it.
    """

    rule: str                       # "R1", "R2b", "R3", "diag_2_4"
    ok: bool
    invalidation_price: float | None
    detail: str = ""
    evaluable: bool = True          # R2 is not evaluable while the impulse is still incomplete


# --------------------------------------------------------------------------------------
# Plan and signals
# --------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ExitTemplate:
    """Exit template shared by the labeller and the interface.

    ONE single frozen object is consumed by both ``labeling/barriers.py`` and the UI cards. If the
    model trained on 2R/1R/48-bar barriers while the screen showed a 3R target with a dynamic
    stop, every probability displayed would describe a trade the user is not taking. It is asserted
    when each decision is constructed.
    """

    id: str
    tp_r: float                 # target in multiples of R
    max_bars: int               # vertical barrier
    trail: str = "none"         # "none" | "chandelier" | "structure"
    breakeven_after_r: float | None = None


@dataclass(frozen=True, slots=True)
class TradePlan:
    """What to do, where it stops making sense, and why."""

    archetype: str              # "w2_long", "w4_long", "wc_long", "diag_exit"
    direction: Direction
    entry_lo: float
    entry_hi: float
    stop: float
    invalidation_price: float
    invalidation_rule: str      # the name of the rule that produced the invalidation
    targets: tuple[float, ...]
    exit_template_id: str
    count_id: str | None = None

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_mid - self.stop)

    @property
    def entry_mid(self) -> float:
        return (self.entry_lo + self.entry_hi) / 2.0

    def rr_at(self, target: float) -> float:
        risk = self.risk_per_unit
        return abs(target - self.entry_mid) / risk if risk > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Signal:
    """The unit the engine merges. `source` and the two timestamps exist from v1 precisely so
    that the news layer does not force anything to be redrawn."""

    ts_event_ms: int
    ts_ingest_ms: int
    source: str                 # dotted name of the provider in the REGISTRY
    kind: SourceKind
    direction: Direction
    strength: float             # normalised to [-1, 1]
    detail: str = ""
    tentative: bool = False     # touched by the provisional pivot: never enters the statistics


@dataclass(frozen=True, slots=True)
class Stat:
    """A statistic is NEVER shown as a bare number.

    Always the triple (value, the n it came from, state of the precondition). A statistic that
    returns a reassuring value out of insufficient data is worse than not having it at all: it
    manufactures confidence.
    """

    name: str
    value: float | None
    n: int
    n_required: int
    detail: str = ""

    @property
    def computable(self) -> bool:
        return self.value is not None and self.n >= self.n_required

    @property
    def missing(self) -> int:
        return max(0, self.n_required - self.n)

    def render(self) -> str:
        if self.computable:
            return f"{self.name}: {self.value:.4g} (n={self.n})"
        return f"{self.name}: not computable — {self.missing} missing (n={self.n}/{self.n_required})"


@dataclass(frozen=True, slots=True)
class Decision:
    """What the engine concludes on a closed bar. This is the unit that gets journalled.

    ``stale`` and ``catching_up`` travel here and not in a global variable because they are
    properties of THIS decision: a decision emitted while catching up after an outage describes a
    price that has already gone, and the user has the right to see that on the card itself.
    """

    ts_ms: int
    symbol: str
    tf: Timeframe
    verdict: Verdict
    maturity: MaturityLevel
    plan: TradePlan | None = None
    signals: tuple[Signal, ...] = ()
    stats: tuple[Stat, ...] = ()
    reasons: tuple[str, ...] = ()
    stale: bool = False
    catching_up: bool = False

    def __post_init__(self) -> None:
        if self.verdict is Verdict.ACTIONABLE:
            if self.plan is None:
                raise ValueError("ACTIONABLE Decision with no plan: there is nothing to trade")
            if self.catching_up:
                raise ValueError(
                    "ACTIONABLE Decision during CATCH_UP: the entry zone describes a price that "
                    "has already gone. Suppress emission while the gap is being replayed."
                )
            if self.maturity is MaturityLevel.PRIOR:
                raise ValueError(
                    "ACTIONABLE Decision at PRIOR level: with no evidence the verdict is capped "
                    "at WATCH. See the maturity ladder."
                )

    @property
    def actionable(self) -> bool:
        return self.verdict is Verdict.ACTIONABLE
