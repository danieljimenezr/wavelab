"""Causality is a type, not a convention.

Repainting is the one class of bug in this project that **fails upward**: if the engine peeks at
the future, the backtest comes out *better*, not worse. So testing for it is not enough — it has to
be impossible to write.

Three mechanisms, in increasing order of strength:

1. ``AsOf[T]`` carries ``available_at_ms``. Reading it too early raises.
2. ``@causal`` inspects every argument and raises if any of them was not available at ``now_ms``.
3. ``ProvisionalWindow`` is a DIFFERENT type from ``Window``: a ``@causal`` function rejects it
   always, so the provisional channel cannot feed the signal path even by accident.

This module deliberately imports nothing from ``wavelab.core.types``: it checks by attribute (duck
typing) so as not to create a cycle, and so that any future type exposing the same contract is
protected without touching this file.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["AsOf", "CausalityError", "causal", "is_causal"]


class CausalityError(RuntimeError):
    """Something tried to read a value before the instant it became available.

    Never caught in order to carry on: this is a programming error, not a runtime condition.
    """


@dataclass(frozen=True, slots=True)
class AsOf[T]:
    """A value together with the instant it became knowable.

    ``available_at_ms`` is NOT when the fact happened, but the first millisecond the system had any
    right to know it. For a ZigZag pivot those are very different things: the extreme happens on
    bar ``t``, but it is only confirmed hundreds of bars later. Conflating them is the entire class
    of bug this module exists to prevent.
    """

    value: T
    available_at_ms: int

    def get(self, now_ms: int) -> T:
        if now_ms < self.available_at_ms:
            raise CausalityError(
                f"acausal read: the value became available at {self.available_at_ms} "
                f"and was requested at {now_ms} "
                f"({self.available_at_ms - now_ms} ms into the future)"
            )
        return self.value

    def known_at(self, now_ms: int) -> bool:
        """Same as ``get`` but without raising. For branching, never for reading."""
        return now_ms >= self.available_at_ms


def _reject(what: str, detail: str, fn_name: str) -> None:
    raise CausalityError(f"{fn_name}: {what} — {detail}")


def _check(name: str, v: Any, now_ms: int, fn_name: str) -> None:
    """Reject any argument that was not available at ``now_ms``.

    Recursive over tuples and lists because sequences of pivots are passed that way.
    """
    # 1. Provisional channel: banned in any causal function, no exceptions, without even
    #    looking at the dates.
    if getattr(type(v), "__wavelab_provisional__", False):
        _reject(
            f"argument `{name}`",
            "is a PROVISIONAL channel and cannot feed the causal path. "
            "Provisional data may only produce tentative annotations, "
            "never signals, journal entries or statistics.",
            fn_name,
        )

    # 2. AsOf: the explicit case.
    if isinstance(v, AsOf):
        if now_ms < v.available_at_ms:
            _reject(
                f"argument `{name}`",
                f"AsOf available at {v.available_at_ms}, requested at {now_ms} "
                f"({v.available_at_ms - now_ms} ms into the future)",
                fn_name,
            )
        return

    # 3. Bar: neither still open, nor closing after `now_ms`.
    close = getattr(v, "close_time_ms", None)
    if close is not None and getattr(v, "open_time_ms", None) is not None:
        if getattr(v, "is_closed", True) is False:
            _reject(
                f"argument `{name}`",
                "is an UNCLOSED bar. Causal features only consume closed bars; "
                "use the provisional channel if you really do want the in-flight price.",
                fn_name,
            )
        if close > now_ms:
            _reject(
                f"argument `{name}`",
                f"the bar closes at {close}, after now_ms={now_ms}",
                fn_name,
            )
        return

    # 4. Window: its last closed index cannot fall in the future.
    end = getattr(v, "end_closed_ts_ms", None)
    if end is not None:
        if end > now_ms:
            _reject(
                f"argument `{name}`",
                f"the window ends at {end}, after now_ms={now_ms}",
                fn_name,
            )
        return

    # 5. Sequences: pivots, signals, hypotheses.
    if isinstance(v, (tuple, list)):
        for i, item in enumerate(v):
            _check(f"{name}[{i}]", item, now_ms, fn_name)
        return

    # 6. Objects carrying a timestamp of their own (confirmed Pivot, Signal, ...).
    for attr in ("confirmed_ts_ms", "available_at_ms", "ts_event_ms"):
        ts = getattr(v, attr, None)
        if ts is not None and isinstance(ts, int) and ts > now_ms:
            _reject(
                f"argument `{name}`",
                f"its `{attr}`={ts} is later than now_ms={now_ms}",
                fn_name,
            )
            return


def causal[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Mark a function as causal and verify its arguments on every call.

    The function MUST accept a ``now_ms`` parameter: without an explicit reference instant there is
    no way to decide what was knowable, and a check that guesses the instant is not a check.

    Cost: ~2-4 µs per call (it does not use ``Signature.bind``, which would be ~20x dearer). The
    analysis runs once per closed bar, so that is irrelevant — and this check is NOT disabled in
    production, because the bug it prevents does not show up as a failure but as a pretty backtest.
    """
    params = list(inspect.signature(fn).parameters)
    if "now_ms" not in params:
        raise TypeError(
            f"@causal demands a `now_ms` parameter on {fn.__qualname__}: "
            "without a reference instant causality cannot be verified."
        )
    now_pos = params.index("now_ms")
    name = fn.__qualname__

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if "now_ms" in kwargs:
            now_ms = kwargs["now_ms"]
        elif len(args) > now_pos:
            now_ms = args[now_pos]
        else:
            raise TypeError(f"{name}: missing `now_ms` (mandatory in a @causal function)")
        if not isinstance(now_ms, int):
            raise TypeError(f"{name}: `now_ms` must be an int in ms, not {type(now_ms).__name__}")

        for i, v in enumerate(args):
            if i != now_pos:
                _check(params[i] if i < len(params) else f"arg{i}", v, now_ms, name)
        for k, v in kwargs.items():
            if k != "now_ms":
                _check(k, v, now_ms, name)
        return fn(*args, **kwargs)

    wrapper.__wavelab_causal__ = True  # type: ignore[attr-defined]
    return wrapper


def is_causal(fn: Any) -> bool:
    """Is this function marked as causal? Used by the feature registry to reject unmarked
    providers on the live path."""
    return bool(getattr(fn, "__wavelab_causal__", False))
