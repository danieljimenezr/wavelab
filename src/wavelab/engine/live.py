"""Live engine: keeps the per-timeframe rings fed from the single 1m series.

Three modes, and the user gets to see them:

  WARMUP    loading history; nothing is emitted
  CATCH_UP  replaying a hole left by an outage; state is updated but DECISION EMISSION IS
            SUPPRESSED
  LIVE      up to date

CATCH_UP exists for one concrete reason. Because ``on_bar`` is deliberately identical live and in
replay, waking up after an outage it will happily produce an ACTIONABLE decision with an entry zone
at a price that went past forty minutes ago. That is the most likely way for the tool to lose the
user's trust in its first week, and it is not fixed with a warning: it is fixed by not emitting.

The ``Decision`` type already prevents this structurally (it raises if it is constructed ACTIONABLE
with ``catching_up=True``), but that is the last line of defence. This is the first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from wavelab.core.ring import Ring
from wavelab.core.timeframes import BY_NAME, MIN_SOURCE_COVERAGE, TF_1M, Timeframe
from wavelab.core.types import Bar, Direction, MaturityLevel, Verdict
from wavelab.waves.matcher import MatcherConfig, match_impulses
from wavelab.waves.pivots import ZigZag, ZigZagConfig
from wavelab.waves.projection import PlanConfig, build_plan

__all__ = ["EngineState", "Health", "LiveEngine", "Mode"]


class Mode(StrEnum):
    WARMUP = "warmup"
    CATCH_UP = "catch_up"
    LIVE = "live"


@dataclass(slots=True)
class Health:
    """What gets painted on the data-health badge. If anything here is off, the interface is lying
    about how fresh it is and the user has a right to know."""

    mode: Mode = Mode.WARMUP
    connected: bool = False
    reconnects: int = 0
    healed_bars: int = 0
    silent_seconds: float = 0.0
    lag_bars: float = 0.0
    gaps_in_window: int = 0
    last_closed_ms: int | None = None

    def as_dict(self) -> dict:
        return {
            "mode": self.mode.value, "connected": self.connected,
            "reconnects": self.reconnects, "healed_bars": self.healed_bars,
            "silent_seconds": round(self.silent_seconds, 1),
            "lag_bars": round(self.lag_bars, 2), "gaps_in_window": self.gaps_in_window,
            "last_closed_ms": self.last_closed_ms,
        }


@dataclass(slots=True)
class EngineState:
    """The engine's mutable state. Mutated ONLY on the event loop's thread.

    When the heavy analysis lands (M4+), it will be handed an immutable snapshot and will return a
    new state that gets applied here. Never the other way round: `asyncio.wait_for` around
    `asyncio.to_thread` does NOT cancel the thread, so an abandoned analysis would carry on
    mutating this object while the next one starts. That would be a data race inside the one
    function the whole design promises is deterministic.
    """

    symbol: str
    rings: dict[str, Ring] = field(default_factory=dict)
    #: One detector per ANALYSIS timeframe. 1m does not get one: it is the source series, not an
    #: analysis timeframe, and detecting pivots on 1m would be measuring microstructure noise.
    detectors: dict[str, ZigZag] = field(default_factory=dict)
    provisional: Bar | None = None
    health: Health = field(default_factory=Health)
    n_bars_1m: int = 0

    def ring(self, tf: Timeframe) -> Ring:
        return self.rings[tf.name]


class LiveEngine:
    """Consumes 1m bars and keeps the rings of every timeframe up to date."""

    def __init__(self, symbol: str, timeframes: list[str], ring_capacity: int = 8192,
                 trigger_tf: str = "15m", zigzag: ZigZagConfig | None = None) -> None:
        self.symbol = symbol.upper()
        self.tfs = [BY_NAME[t] for t in timeframes]
        self.trigger = BY_NAME[trigger_tf]
        self.zz_cfg = zigzag or ZigZagConfig()
        self.matcher_cfg = MatcherConfig()
        self.plan_cfg = PlanConfig()
        #: The interface only shows longs, but the engine evaluates BOTH directions: that way twice
        #: as much evidence accumulates from day 1 and turning shorts on is one config line.
        self.directions: list[Direction] = [Direction.LONG, Direction.SHORT]
        self.state = EngineState(symbol=self.symbol)
        self.state.rings[TF_1M.name] = Ring(self.symbol, TF_1M, ring_capacity)
        for tf in self.tfs:
            if tf is not TF_1M:
                self.state.rings[tf.name] = Ring(self.symbol, tf, ring_capacity)
                self.state.detectors[tf.name] = ZigZag(self.zz_cfg)
        self._last_wall = time.time()

    # ------------------------------------------------------------------ resampling

    def _close_htf(self, tf: Timeframe, htf_open_ms: int) -> Bar | None:
        """Builds the higher-timeframe bar that has just closed, out of the 1m ring.

        It is rebuilt from the source series instead of accumulated incrementally: an accumulator
        that drifts out of sync gives nothing away, whereas rebuilding always yields the same
        result the backtest would give over the same data.
        """
        r1 = self.state.rings[TF_1M.name]
        n = tf.expected_source_bars
        if len(r1) == 0:
            return None
        w = r1.window(min(n + 5, len(r1)))
        sel = (w.ts >= htf_open_ms) & (w.ts < htf_open_ms + tf.ms)
        if not sel.any():
            return None
        ts, o, h, l, c, v = w.ts[sel], w.open[sel], w.high[sel], w.low[sel], w.close[sel], w.volume[sel]
        n_src = int(ts.size)
        return Bar(
            symbol=self.symbol, tf=tf, open_time_ms=htf_open_ms,
            open=float(o[0]), high=float(h.max()), low=float(l.min()), close=float(c[-1]),
            volume=float(v.sum()), is_closed=True, n_source_bars=n_src,
            # A 1h bar built out of 43 minutes is not a 1h bar. It travels labelled as such.
            is_gap=n_src < int(MIN_SOURCE_COVERAGE * n),
        )

    # ------------------------------------------------------------------ ingestion

    def on_bar_1m(self, bar: Bar) -> list[Bar]:
        """Adds a 1m bar and returns the higher-timeframe bars that closed along with it."""
        if not bar.is_closed:
            self.state.provisional = bar
            self.state.rings[TF_1M.name].set_provisional(bar)
            return []

        r1 = self.state.rings[TF_1M.name]
        last = r1.last_closed_ts_ms
        if last is not None and bar.open_time_ms <= last:
            return []                                   # duplicate from gap healing
        r1.append(bar)
        self.state.n_bars_1m += 1
        self.state.provisional = None
        self.state.health.last_closed_ms = bar.open_time_ms

        closed: list[Bar] = []
        end = bar.open_time_ms + TF_1M.ms      # the instant this 1m bar ends
        for tf in self.tfs:
            if tf is TF_1M or end % tf.ms != 0:
                continue
            htf = self._close_htf(tf, end - tf.ms)
            if htf is None:
                continue
            ring = self.state.rings[tf.name]
            last_htf = ring.last_closed_ts_ms
            if last_htf is None or htf.open_time_ms > last_htf:
                ring.append(htf)
                # The detector is fed the SAME bar that just entered the ring, one bar at a time
                # and in order. It is never handed a whole array: calling it once over the entire
                # history and then slicing the result inflates every entry by roughly the whole
                # threshold (1.2-2.5 ATR), which is more than any real edge.
                det = self.state.detectors.get(tf.name)
                if det is not None:
                    det.update(htf.open_time_ms, htf.high, htf.low, htf.close)
                closed.append(htf)
        return closed

    def decide(self, tf_name: str, price: float) -> dict:
        """Ranked hypotheses and their plan. This is what gets painted on the decision card.

        The verdict is ALWAYS capped at WATCH while the maturity level is PRIOR: with no evidence
        of our own, nothing can be marked ACTIONABLE. `Decision`'s constructor enforces that
        structurally as well, so this is the first line of defence and that one is the last.
        """
        det = self.state.detectors.get(tf_name)
        ring = self.state.rings.get(tf_name)
        if det is None or ring is None or not len(ring) or det.atr is None:
            return {"verdict": Verdict.NO_TRADE.value, "maturity": int(MaturityLevel.PRIOR),
                    "hypotheses": [], "reasons": ["not enough structure yet"]}

        pivs = det.store.as_of(ring.last_closed_ts_ms or 0)
        hyps = match_impulses(pivs, self.matcher_cfg,
                              directions=tuple(self.directions))
        atr = det.atr
        out, best_in_zone = [], False
        for h in hyps:
            r = build_plan(h, price, atr, self.plan_cfg)
            row = {
                "id": h.id, "state": h.state.value, "direction": h.direction.name,
                "label": h.terminal_label, "score": round(h.score, 3),
                "fit": {k: round(v, 3) for k, v in h.fit.items()},
                "archetype": h.archetype, "truncated": h.truncated,
                "points": [round(x, 2) for x in h.points],
                "pivot_ts": [p.ts_ms for p in h.pivots],
                "invalidation_price": round(h.invalidation_price, 2),
                "invalidation_rule": h.invalidation_rule,
                "viable": r.viable, "in_zone": r.in_zone,
                "reasons": list(r.reasons),
            }
            if r.viable:
                row |= {
                    "entry_lo": round(r.plan.entry_lo, 2), "entry_hi": round(r.plan.entry_hi, 2),
                    "stop": round(r.plan.stop, 2),
                    "targets": [round(t, 2) for t in r.plan.targets],
                    # Both ratios travel, and the card shows which is which. `rr_t2` is quoted at
                    # `price` — the close the backtest books at, and a fill you can always get.
                    # `rr_in_zone` is the better case that needs a resting limit to actually fill,
                    # so it can never be the headline: it is the number the card used to quote
                    # alone while the results measured the other one.
                    "rr_t2": round(r.rr_t2, 2), "rr_in_zone": round(r.rr_in_zone, 2),
                    "cost_r": round(r.cost_r, 4),
                    "p_required": round(r.p_required, 4), "stop_atr": round(r.stop_atr, 2),
                    "size_factor": round(r.size_factor, 3),
                }
                best_in_zone |= r.in_zone
            out.append(row)

        # PRIOR level: the expectancy table is hand-written and not a single trade has resolved.
        # We can WATCH, never mark as actionable.
        #
        # So `best_in_zone` does NOT move the verdict, and this used to be written as a ternary
        # whose two branches were both WATCH — code shaped like a decision that decided nothing,
        # and measured dead: 311 of 446 WATCHes on the 20-day fixture had nothing in any zone and
        # were indistinguishable from the 135 that did. What being in a zone changes today is the
        # REASON below, which is the thing the reader acts on. When maturity rises past PRIOR this
        # is the line that has to learn the difference, and it should be rewritten then rather
        # than left looking as though it already had.
        verdict = Verdict.WATCH if out else Verdict.NO_TRADE
        reasons = []
        if not out:
            reasons.append("no structure satisfies the hard rules right now")
        elif not best_in_zone:
            reasons.append("there is structure, but the price is not inside any entry zone")
        reasons.append("PRIOR level (n=0): expectancies from an expert table, unvalidated. "
                       "The verdict cannot go above WATCH.")
        return {"verdict": verdict.value, "maturity": int(MaturityLevel.PRIOR),
                "hypotheses": out, "reasons": reasons, "atr": round(atr, 2),
                "price": round(price, 2)}

    def waves(self, tf_name: str, now_ms: int | None = None,
              since_ms: int | None = None) -> dict:
        """Legs and confirmation price, for drawing. Never raises if the timeframe does not exist."""
        det = self.state.detectors.get(tf_name)
        if det is None:
            return {"legs": [], "confirm_price": None, "n_confirmed": 0, "atr": None}
        ring = self.state.rings.get(tf_name)
        t = now_ms if now_ms is not None else (ring.last_closed_ts_ms if ring else 0) or 0
        return {
            "legs": det.legs_as_of(t, since_ms=since_ms),
            "confirm_price": det.confirm_price(),
            "n_confirmed": det.n_confirmed,
            "atr": det.atr,
        }

    def warmup(self, bars_1m) -> int:
        """Loads history. Replays bar by bar, exactly like the live path: if the warm-up went down
        a different route, the initial state would differ from the one a replay would produce and
        the "one single function" promise would be false from the first second."""
        self.state.health.mode = Mode.WARMUP
        n = 0
        for b in bars_1m:
            self.on_bar_1m(b)
            n += 1
        return n

    # ------------------------------------------------------------------ mode

    def check_clock(self, trigger_tf_ms: int | None = None) -> Mode:
        """Detects a clock jump (suspend, restart, long outage) and enters CATCH_UP."""
        tf_ms = trigger_tf_ms or self.trigger.ms
        now = time.time()
        jump = now - self._last_wall
        self._last_wall = now
        h = self.state.health
        if h.mode is not Mode.WARMUP and jump * 1000 > 2 * tf_ms:
            h.mode = Mode.CATCH_UP
        return h.mode

    def update_health(self, connected: bool, reconnects: int, healed: int,
                      silent_seconds: float) -> Health:
        h = self.state.health
        h.connected = connected
        h.reconnects = reconnects
        h.healed_bars = healed
        h.silent_seconds = silent_seconds
        if h.last_closed_ms:
            h.lag_bars = (time.time() * 1000 - h.last_closed_ms) / self.trigger.ms
        ring = self.state.rings.get(self.trigger.name)
        if ring is not None and len(ring):
            h.gaps_in_window = ring.window(min(500, len(ring))).n_gaps
        # CATCH_UP is left when the lag is back inside a single trigger bar.
        if h.mode is Mode.CATCH_UP and h.lag_bars <= 1.0:
            h.mode = Mode.LIVE
        elif h.mode is Mode.WARMUP and h.last_closed_ms:
            h.mode = Mode.LIVE if h.lag_bars <= 1.0 else Mode.CATCH_UP
        return h

    @property
    def emitting(self) -> bool:
        """If this is False, NO decision is emitted. This is the first line of defence;
        `Decision`'s constructor is the last."""
        return self.state.health.mode is Mode.LIVE
