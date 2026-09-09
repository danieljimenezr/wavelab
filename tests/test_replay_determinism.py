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

from wavelab.core.types import Bar, Decision, MaturityLevel, Verdict
from wavelab.validation.replay import (
    ReplayDivergence,
    assert_replay_deterministic,
    diff_path,
    replay,
    streaming,
)


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


class TestDiffPath:
    def test_it_names_the_nested_path(self):
        a = _decision(BAR := None, 1.0) if False else None  # noqa: F841
        assert diff_path((1, 2, 3), (1, 2, 3)) is None
        assert "[2]" in diff_path((1, 2, 3), (1, 2, 4))
        assert "lengths" in diff_path((1,), (1, 2))
        assert "different types" in diff_path(1, "1")

    def test_it_walks_into_dataclasses(self, bars_small):
        d1 = _decision(bars_small[0], 1.0)
        d2 = _decision(bars_small[0], 2.0)
        p = diff_path(d1, d2)
        assert "reasons" in p and "ema=" in p
