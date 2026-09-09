"""The determinism harness. Written BEFORE the first indicator, and it ships as a CI gate.

It checks two distinct properties, and the second one is the one that really matters:

**1. Determinism.** Two full replays of the same bars produce identical outputs. Catches global
state, dependence on the wall clock, iteration over unordered sets, and unseeded randomness.

**2. Prefix — the lookahead detector.** Replaying only ``bars[:k]`` has to produce exactly the same
first ``k`` outputs as the full replay. If the engine peeks even one bar ahead, its output at bar
``k`` changes with whatever comes AFTER it, and the two series diverge.

The second property is irreplaceable because repainting **fails upward**: an engine that looks at
the future produces a prettier backtest, not an error. No suite of expected-value tests detects it,
because the expected values were computed with the same lookahead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from wavelab.core.clock import SimClock
from wavelab.core.types import Bar

__all__ = [
    "EngineFactory",
    "ReplayDivergence",
    "assert_replay_deterministic",
    "diff_path",
    "replay",
    "streaming",
]


class ReplayDivergence(AssertionError):
    """Two replays that ought to agree do not. It always names the guilty field."""


# --------------------------------------------------------------------------- diff

def diff_path(a: Any, b: Any, path: str = "") -> str | None:
    """First difference between two structures, as a readable path.

    Without this, a harness failure says "the outputs differ" and leaves you half an hour working
    out which of forty fields it was. With it, it says ``output[137].plan.stop: 106880.0 !=
    106884.5``.
    """
    if type(a) is not type(b):
        return f"{path or '<root>'}: different types {type(a).__name__} != {type(b).__name__}"

    if is_dataclass(a) and not isinstance(a, type):
        for f in fields(a):
            sub = diff_path(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}" if path else f.name)
            if sub:
                return sub
        return None

    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return f"{path or '<root>'}: lengths {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            sub = diff_path(x, y, f"{path}[{i}]")
            if sub:
                return sub
        return None

    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path or '<root>'}: keys {sorted(a.keys())} != {sorted(b.keys())}"
        for k in a:
            sub = diff_path(a[k], b[k], f"{path}[{k!r}]")
            if sub:
                return sub
        return None

    if a != b:
        return f"{path or '<root>'}: {a!r} != {b!r}"
    return None


# --------------------------------------------------------------------------- replay

def streaming(on_bar: Callable[[Any, Bar], tuple[Any, Any]],
              initial_state: Callable[[], Any]) -> EngineFactory:
    """Factory for a correct engine: it ignores the bars it is offered.

    A causal engine consumes only what ``on_bar`` hands it, one bar at a time. That this factory
    throws its argument away is not a matter of convenience: it is the operational definition of
    "causal".
    """
    def factory(_bars: Sequence[Bar]) -> tuple[Any, Any]:
        return on_bar, initial_state
    return factory


#: Receives the bars THIS run is about to replay and returns ``(on_bar, initial_state)``.
#:
#: The argument stands for "everything the engine could possibly see in its store". A correct
#: engine ignores it. One that precomputes over the whole series and then slices it up — the
#: classic bug of calling the pivot detector once over the entire array — uses it, and that is why
#: the prefix test catches it: on the prefix run it only receives ``bars[:k]``, so its output
#: changes.
EngineFactory = Callable[[Sequence[Bar]], tuple[Callable[[Any, Bar], tuple[Any, Any]], Callable[[], Any]]]


def replay(
    factory: EngineFactory,
    bars: Sequence[Bar],
    clock: SimClock | None = None,
) -> list[Any]:
    """Runs the engine bar by bar and returns the output of each one.

    It is the SAME function the engine runs live. There is no vectorised path, and there will not
    be one: if backtest and live were two implementations, their divergence would quietly
    reintroduce lookahead, and in a project that labels waves from pivots that repaint, that is
    the likeliest way for everything to break without anyone finding out.
    """
    on_bar, initial_state = factory(bars)
    clock = clock or SimClock(bars[0].open_time_ms if bars else 0)
    state = initial_state()
    out: list[Any] = []
    for bar in bars:
        if not bar.is_closed:
            raise ValueError(
                f"replay was handed an unclosed bar at {bar.open_time_ms}. "
                "Replay consumes closed bars only, exactly like the live path."
            )
        clock.set(bar.close_time_ms)
        state, result = on_bar(state, bar)
        out.append(result)
    return out


def assert_replay_deterministic(
    factory: EngineFactory,
    bars: Sequence[Bar],
    checkpoints: Sequence[int] | None = None,
) -> None:
    """Verifies determinism and the absence of lookahead. Raises ``ReplayDivergence`` on failure."""
    if len(bars) < 4:
        raise ValueError("at least 4 bars are needed for the test to mean anything")

    full_a = replay(factory, bars)
    full_b = replay(factory, bars)

    d = diff_path(full_a, full_b, "output")
    if d:
        raise ReplayDivergence(
            "NON-DETERMINISTIC: two identical replays differ.\n"
            f"  {d}\n"
            "  Typical causes: global state shared between instances, reading the wall clock, "
            "iterating over an unordered set/dict, or randomness without a seed."
        )

    if checkpoints is None:
        n = len(bars)
        checkpoints = sorted({max(2, n // 4), max(3, n // 2), max(4, (3 * n) // 4), n - 1})

    for k in checkpoints:
        if not (2 <= k <= len(bars)):
            continue
        prefix = replay(factory, bars[:k])
        d = diff_path(prefix, full_a[:k], f"prefix(k={k})")
        if d:
            raise ReplayDivergence(
                f"LOOKAHEAD DETECTED in the prefix k={k} of {len(bars)} bars.\n"
                f"  {d}\n"
                "  A bar's output changes depending on which bars EXIST after it, so the engine "
                "is reading the future.\n"
                "  Usual suspects: calling the pivot detector over the whole array and then "
                "slicing it; using scipy.find_peaks' `prominence` (which is defined against the "
                "ENTIRE array); indexing with [-1] a series that includes the bar in progress; or "
                "consuming a pivot by its `idx` instead of its `confirmed_idx`."
            )
