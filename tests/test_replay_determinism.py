"""The M0 CI gate.

A harness that only passes correct code is worth nothing: it has to be shown that it REJECTS
incorrect code. So here engines are built broken on purpose, and the harness is required to catch
them and to name the guilty field as well.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from wavelab.core.clock import SimClock
from wavelab.core.timeframes import BY_NAME, TF_1M
from wavelab.core.types import Bar, Decision, MaturityLevel, Verdict
from wavelab.engine.live import LiveEngine
from wavelab.validation.replay import (
    ReplayDivergence,
    assert_replay_deterministic,
    diff_path,
    replay,
    streaming,
)

from .conftest import SYMBOL, make_bars


@dataclass(frozen=True, slots=True)
class StubState:
    ema: float | None = None
    n: int = 0


def _decision(bar: Bar, ema: float) -> Decision:
    return Decision(
        ts_ms=bar.close_time_ms,
        symbol=bar.symbol,
        tf=bar.tf,
        verdict=Verdict.WATCH if bar.close > ema else Verdict.NO_TRADE,
        maturity=MaturityLevel.PRIOR,
        reasons=(f"ema={ema:.4f}",),
    )


# --------------------------------------------------------------------- engines

def correct():
    """Consumes only what `on_bar` hands it. The operational definition of causal."""
    def on_bar(state: StubState, bar: Bar) -> tuple[StubState, Decision]:
        ema = bar.close if state.ema is None else 0.2 * bar.close + 0.8 * state.ema
        return StubState(ema, state.n + 1), _decision(bar, ema)
    return streaming(on_bar, StubState)


def with_lookahead(offset: int = 1):
    """The classic bug: precompute over the WHOLE series and then serve it by position.

    It is exactly what happens when you call the pivot detector once over the entire array, or when
    you use `scipy.find_peaks(prominence=...)`, whose prominence is defined against the full array.
    """
    def factory(bars: Sequence[Bar]):
        # Precomputed over EVERYTHING visible: this is where the future sneaks in.
        closes = [b.close for b in bars]
        pos = {b.open_time_ms: i for i, b in enumerate(bars)}

        def on_bar(state: StubState, bar: Bar) -> tuple[StubState, Decision]:
            i = pos[bar.open_time_ms]
            future = closes[min(i + offset, len(closes) - 1)]
            ema = future if state.ema is None else 0.2 * future + 0.8 * state.ema
            return StubState(ema, state.n + 1), _decision(bar, ema)

        return on_bar, StubState
    return factory


def nondeterministic():
    def on_bar(state: StubState, bar: Bar) -> tuple[StubState, Decision]:
        ema = bar.close * (1.0 + random.random() * 1e-9)
        return StubState(ema, state.n + 1), _decision(bar, ema)
    return streaming(on_bar, StubState)


# --------------------------------------------------------------------- tests

class TestHarness:
    def test_a_correct_engine_passes_2000_bars(self, bars_2k):
        """The literal M0 verification."""
        assert_replay_deterministic(correct(), bars_2k)

    def test_replay_is_reproducible(self, bars_2k):
        assert replay(correct(), bars_2k) == replay(correct(), bars_2k)

    def test_it_catches_the_lookahead_and_names_the_field(self, bars_2k):
        with pytest.raises(ReplayDivergence) as ei:
            assert_replay_deterministic(with_lookahead(), bars_2k)
        msg = str(ei.value)
        assert "LOOKAHEAD DETECTED" in msg
        # It must point at the exact path, not a generic "the outputs differ".
        assert "reasons" in msg or "verdict" in msg, msg
        assert "prefix(k=" in msg

    @pytest.mark.parametrize("offset", [1, 2, 5, 20])
    def test_it_catches_the_lookahead_at_any_distance(self, bars_small, offset):
        with pytest.raises(ReplayDivergence, match="LOOKAHEAD"):
            assert_replay_deterministic(with_lookahead(offset), bars_small)

    def test_it_catches_the_nondeterminism(self, bars_small):
        with pytest.raises(ReplayDivergence, match="NON-DETERMINISTIC"):
            assert_replay_deterministic(nondeterministic(), bars_small)

    def test_it_rejects_unclosed_bars(self, bars_small):
        b = bars_small[0]
        unclosed = Bar(
            symbol=b.symbol, tf=b.tf, open_time_ms=b.open_time_ms, open=b.open, high=b.high,
            low=b.low, close=b.close, volume=b.volume, is_closed=False,
        )
        with pytest.raises(ValueError, match="unclosed"):
            replay(correct(), [unclosed])


class TestTheHarnessContract:
    """The harness is only worth what it REFUSES. These are the ways it could go quiet."""

    def test_it_refuses_to_certify_an_engine_on_three_bars(self, bars_small):
        """A green tick from a harness that only saw three bars is worse than no harness at all.

        The lookahead detector works by replaying PREFIXES: below four bars there is no prefix
        left to compare, so the check runs over nothing and returns quietly. If the minimum is
        ever dropped, someone points CI at a short fixture, sees green, and ships a repainting
        engine believing it was checked.
        """
        with pytest.raises(ValueError, match="at least 4"):
            assert_replay_deterministic(correct(), bars_small[:3])

        # The refusal is not pedantry. On one bar there is no prefix at all, so an engine we KNOW
        # reads the future would come back certified instead of caught.
        with pytest.raises(ValueError, match="at least 4"):
            assert_replay_deterministic(with_lookahead(), bars_small[:1])

    def test_it_advances_the_sim_clock_to_the_close_of_every_bar(self, bars_small):
        """Everything causal in this project asks the clock what time it is.

        `@causal` refuses a `now_ms` it was not handed, and the pivot store answers `as_of(now)`.
        If replay left the clock parked at the first bar, every one of those guards would be
        evaluated against the wrong instant — and they would all evaluate as SAFE, because a clock
        stuck in the past can never see anything from the future. The guards would keep passing
        while protecting nothing, which is the failure mode this whole file exists to prevent.
        """
        clock = SimClock(bars_small[0].open_time_ms)
        seen: list[tuple[int, int]] = []

        def on_bar(state, bar):
            seen.append((clock.now_ms(), bar.close_time_ms))
            return state, None

        replay(streaming(on_bar, lambda: None), bars_small, clock=clock)

        assert len(seen) == len(bars_small), "the engine was not called once per bar"
        off = [(t, exp) for t, exp in seen if t != exp]
        assert not off, (
            f"the clock did not follow the bars: {len(off)} of {len(seen)} bars were processed at "
            f"the wrong instant, first {off[0][0]} while the bar closed at {off[0][1]}"
        )


class TestTheGateIsPointedAtTheEngineThatShips:
    """Every engine above is a stub defined in this file.

    So what the gate is proven to do, it is proven to do to twenty lines of EMA written to be
    caught. `LiveEngine` — the one function the backtest and the live route both run, the thing the
    gate was written for — is never handed to it. The stubs prove the harness works; this proves it
    is aimed at something.
    """

    TRIGGER = "15m"

    def _engine_factory(self):
        """`LiveEngine` behind the harness's contract: state in, one bar in, output out.

        `streaming` throws the bar array away before the engine ever sees it, which is the
        operational definition of causal — the engine can only answer from what `on_bar` handed it.
        The output is the decision card, because that is what reaches the user: verdict, points,
        pivot timestamps, entry zone, stop, targets and R:R, every one of them a number somebody is
        asked to risk money on.
        """
        trigger = BY_NAME[self.TRIGGER]

        def on_bar(eng: LiveEngine, bar: Bar):
            closed = eng.on_bar_1m(bar)
            if not any(b.tf is trigger for b in closed):
                return eng, None
            return eng, eng.decide(self.TRIGGER, bar.close)

        return streaming(on_bar, lambda: LiveEngine(SYMBOL, ["1m", self.TRIGGER], 4096,
                                                    self.TRIGGER))

    def test_the_real_engine_replays_identically_and_reads_no_bar_it_was_not_given(self):
        """Two full replays must agree, and every prefix must agree with the full run.

        The determinism half is the live one for an engine of this shape: `decide()` walks dicts
        and detector state, and the day anybody stamps the card with the wall clock, seeds anything
        unseeded, or lets an iteration order leak in, two runs over the same history stop matching
        and no expected-value test in the suite notices — the numbers all still look plausible.
        The prefix half is the standing guard: today `on_bar_1m` can only reach bars it was handed,
        and this is what keeps that true the day somebody speeds up the warm-up by computing over
        the whole array and slicing it.

        33 hours of 1m bars: 133 trigger closes, 108 of them carrying hypotheses, so the card being
        compared is a populated one rather than an empty shell.
        """
        bars = make_bars(2000, tf=TF_1M, seed=3)
        cards = [c for c in replay(self._engine_factory(), bars) if c]
        assert sum(1 for c in cards if c["hypotheses"]) >= 50, (
            f"only {sum(1 for c in cards if c['hypotheses'])} of {len(cards)} cards carried a "
            "hypothesis: the fixture is comparing mostly empty cards and would not notice a "
            "divergence in the numbers that matter"
        )
        assert_replay_deterministic(self._engine_factory(), bars)


class TestDiffPath:
    def test_it_names_the_nested_path(self):
        a = _decision(BAR := None, 1.0) if False else None  # noqa: F841
        assert diff_path((1, 2, 3), (1, 2, 3)) is None
        assert "[2]" in diff_path((1, 2, 3), (1, 2, 4))
        assert "lengths" in diff_path((1,), (1, 2))
        assert "different types" in diff_path(1, "1")

    def test_it_names_the_key_inside_a_dict(self):
        """The engine's real output is a dict: `decide()` returns one, hypotheses and all.

        So dicts are the shape this walker actually meets in CI. If it stops comparing key SETS,
        a divergence in a missing key surfaces as a KeyError thrown from inside the harness: the
        lookahead report the user needs turns into a stack trace with no field name in it, which
        is exactly the half hour of guessing `diff_path` was written to save.
        """
        assert diff_path({"verdict": "watch"}, {"verdict": "watch"}) is None

        p = diff_path({"plan": {"stop": 106880.0}}, {"plan": {"stop": 106884.5}})
        assert p is not None and "'plan'" in p and "'stop'" in p and "106884.5" in p, p

        missing = diff_path({"verdict": "watch"}, {"maturity": 0})
        assert missing is not None and "keys" in missing, (
            f"a dict that lost a key was not reported as a key-set difference: {missing!r}"
        )

    def test_it_walks_into_dataclasses(self, bars_small):
        d1 = _decision(bars_small[0], 1.0)
        d2 = _decision(bars_small[0], 2.0)
        p = diff_path(d1, d2)
        assert "reasons" in p and "ema=" in p

    def test_it_names_the_field_instead_of_dumping_the_object(self, bars_small):
        """`output[137].plan.stop: 106880.0 != 106884.5` is the entire point of this function.

        A `Decision` prints its every field in its repr, so a walker that stopped descending into
        dataclasses and just compared the two objects would still produce a message with the words
        `reasons` and `ema=` in it — it would look like it worked. What the reader gets instead is
        two forty-field objects side by side and the half hour of eye-work this was written to
        save, at the exact moment they are being told their engine reads the future.
        """
        p = diff_path(_decision(bars_small[0], 1.0), _decision(bars_small[0], 2.0))
        assert p is not None
        assert p.startswith("reasons"), (
            f"the first difference is in `reasons`, but the report begins with {p[:60]!r}: the "
            "walker is comparing whole objects instead of locating the field"
        )
        assert "Decision(" not in p, f"the report dumps the whole object instead of the field: {p}"
