"""La puerta de CI de M0.

Un arnés que solo aprueba código correcto no vale nada: hay que demostrar que RECHAZA código
incorrecto. Por eso aquí se construyen motores rotos a propósito y se exige que el arnés los cace y
además nombre el campo culpable.
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


# --------------------------------------------------------------------- motores

def correcto():
    """Solo consume lo que `on_bar` le entrega. Definición operativa de causal."""
    def on_bar(state: StubState, bar: Bar) -> tuple[StubState, Decision]:
        ema = bar.close if state.ema is None else 0.2 * bar.close + 0.8 * state.ema
        return StubState(ema, state.n + 1), _decision(bar, ema)
    return streaming(on_bar, StubState)


def con_lookahead(offset: int = 1):
    """El bug clásico: precalcular sobre la serie ENTERA y luego servirla por posición.

    Es exactamente lo que pasa al llamar al detector de pivotes una vez sobre todo el array, o al
    usar `scipy.find_peaks(prominence=...)`, cuya prominencia se define contra el array completo.
    """
    def factory(bars: Sequence[Bar]):
        # Precálculo sobre TODO lo visible: aquí se cuela el futuro.
        closes = [b.close for b in bars]
        pos = {b.open_time_ms: i for i, b in enumerate(bars)}

        def on_bar(state: StubState, bar: Bar) -> tuple[StubState, Decision]:
            i = pos[bar.open_time_ms]
            futuro = closes[min(i + offset, len(closes) - 1)]
            ema = futuro if state.ema is None else 0.2 * futuro + 0.8 * state.ema
            return StubState(ema, state.n + 1), _decision(bar, ema)

        return on_bar, StubState
    return factory


def no_determinista():
    def on_bar(state: StubState, bar: Bar) -> tuple[StubState, Decision]:
        ema = bar.close * (1.0 + random.random() * 1e-9)  # noqa: S311 — sin semilla, a propósito
        return StubState(ema, state.n + 1), _decision(bar, ema)
    return streaming(on_bar, StubState)


# --------------------------------------------------------------------- tests

class TestArnes:
    def test_motor_correcto_pasa_2000_velas(self, bars_2k):
        """La verificación literal de M0."""
        assert_replay_deterministic(correcto(), bars_2k)

    def test_replay_es_reproducible(self, bars_2k):
        assert replay(correcto(), bars_2k) == replay(correcto(), bars_2k)

    def test_caza_el_lookahead_y_nombra_el_campo(self, bars_2k):
        with pytest.raises(ReplayDivergence) as ei:
            assert_replay_deterministic(con_lookahead(), bars_2k)
        msg = str(ei.value)
        assert "LOOKAHEAD DETECTADO" in msg
        # Debe señalar la ruta exacta, no un genérico "las salidas difieren".
        assert "reasons" in msg or "verdict" in msg, msg
        assert "prefijo(k=" in msg

    @pytest.mark.parametrize("offset", [1, 2, 5, 20])
    def test_caza_el_lookahead_a_cualquier_distancia(self, bars_small, offset):
        with pytest.raises(ReplayDivergence, match="LOOKAHEAD"):
            assert_replay_deterministic(con_lookahead(offset), bars_small)

    def test_caza_el_no_determinismo(self, bars_small):
        with pytest.raises(ReplayDivergence, match="NO DETERMINISTA"):
            assert_replay_deterministic(no_determinista(), bars_small)

    def test_rechaza_velas_sin_cerrar(self, bars_small):
        b = bars_small[0]
        abierta = Bar(
            symbol=b.symbol, tf=b.tf, open_time_ms=b.open_time_ms, open=b.open, high=b.high,
            low=b.low, close=b.close, volume=b.volume, is_closed=False,
        )
        with pytest.raises(ValueError, match="sin cerrar"):
            replay(correcto(), [abierta])


class TestDiffPath:
    def test_nombra_la_ruta_anidada(self):
        a = _decision(BAR := None, 1.0) if False else None  # noqa: F841
        assert diff_path((1, 2, 3), (1, 2, 3)) is None
        assert "[2]" in diff_path((1, 2, 3), (1, 2, 4))
        assert "longitudes" in diff_path((1,), (1, 2))
        assert "tipos distintos" in diff_path(1, "1")

    def test_recorre_dataclasses(self, bars_small):
        d1 = _decision(bars_small[0], 1.0)
        d2 = _decision(bars_small[0], 2.0)
        p = diff_path(d1, d2)
        assert "reasons" in p and "ema=" in p
