"""Fixtures deterministas. Ni una sola llamada a `random` sin semilla ni al reloj de pared."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.core.timeframes import TF_1H, TF_15M, Timeframe
from wavelab.core.types import Bar

SYMBOL = "BTCUSDT"
T0 = 1_600_000_000_000 - (1_600_000_000_000 % TF_1H.ms)  # alineado a rejilla UTC


def make_bars(
    n: int,
    tf: Timeframe = TF_15M,
    seed: int = 7,
    start_ms: int = T0,
    drop: set[int] | None = None,
    start_price: float = 30_000.0,
) -> list[Bar]:
    """Paseo aleatorio con semilla fija. ``drop`` elimina índices para simular huecos reales."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n) * 0.004
    close = start_price * np.exp(np.cumsum(r))
    spread = np.abs(rng.standard_normal(n)) * close * 0.0015
    drop = drop or set()
    out: list[Bar] = []
    for i in range(n):
        if i in drop:
            continue
        c = float(close[i])
        o = float(close[i - 1]) if i else start_price
        out.append(
            Bar(
                symbol=SYMBOL,
                tf=tf,
                open_time_ms=start_ms + i * tf.ms,
                open=o,
                high=max(o, c) + float(spread[i]),
                low=min(o, c) - float(spread[i]),
                close=c,
                volume=float(abs(rng.standard_normal()) * 100 + 10),
                n_source_bars=tf.expected_source_bars,
            )
        )
    return out


@pytest.fixture(scope="session")
def bars_2k() -> list[Bar]:
    """Las 2.000 velas que exige la verificación de M0."""
    return make_bars(2000)


@pytest.fixture(scope="session")
def bars_small() -> list[Bar]:
    return make_bars(60)
