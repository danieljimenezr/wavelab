"""Los tests de M3. Si estos fallan, todo lo que se construya encima miente.

El repintado **falla hacia arriba**: un detector que mira al futuro produce un backtest más bonito,
no un error. Ninguna suite de valores esperados lo detecta, porque los valores esperados se
calcularon con el mismo lookahead. Solo lo detecta comparar lo que el sistema decía EN SU MOMENTO
con lo que dice después.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wavelab.core.timeframes import TF_4H
from wavelab.core.types import PivotKind
from wavelab.waves.pivots import WilderATR, ZigZag, ZigZagConfig, detect_batch

from .conftest import make_bars


def _serie(n: int, seed: int = 3, tf=TF_4H):
    bars = make_bars(n, tf=tf, seed=seed)
    return (np.array([b.open_time_ms for b in bars], dtype=np.int64),
            np.array([b.high for b in bars]),
            np.array([b.low for b in bars]),
            np.array([b.close for b in bars]))


class TestCausalidad:
    """La propiedad de la que cuelga todo lo demás."""

    def test_as_of_es_identico_a_recalcular_solo_con_el_pasado(self):
        """★ EL test de M3.

        `pivots_as_of(t)` calculado en el momento t debe ser IDÉNTICO al que se obtiene
        reproduciendo únicamente las velas hasta t. Si difiriesen, el detector estaría usando
        información posterior a t para decidir qué había en t.
        """
        ts, h, l, c = _serie(400)
        completo = detect_batch(ts, h, l, c)

        for k in (60, 120, 200, 300, 399):
            parcial = detect_batch(ts[:k + 1], h[:k + 1], l[:k + 1], c[:k + 1])
            a = parcial.store.as_of(int(ts[k]))
            b = completo.store.as_of(int(ts[k]))
            assert a == b, (
                f"en la vela {k} el detector ve {len(a)} pivotes reproduciendo solo el pasado "
                f"y {len(b)} conociendo el futuro: está mirando hacia adelante"
            )

    def test_el_historico_solo_puede_crecer(self):
        """Un conteo que el usuario ya vio no puede desaparecer sin evento de invalidación."""
        ts, h, l, c = _serie(500, seed=11)
        z = ZigZag()
        previo: tuple = ()
        for i in range(ts.size):
            z.update(int(ts[i]), float(h[i]), float(l[i]), float(c[i]))
            actual = z.store.as_of(int(ts[i]))
            assert actual[: len(previo)] == previo, (
                f"en la vela {i} el histórico confirmado CAMBIÓ en vez de crecer"
            )
            assert len(actual) >= len(previo)
            previo = actual

    def test_ningun_pivote_es_visible_antes_de_confirmarse(self):
        ts, h, l, c = _serie(400, seed=5)
        z = detect_batch(ts, h, l, c)
        for p in z.store.as_of(10**15):
            assert p not in z.store.as_of(p.confirmed_ts_ms - 1), (
                f"el pivote localizado en {p.ts_ms} es visible antes de confirmarse "
                f"en {p.confirmed_ts_ms}: eso ES el repintado"
            )
            assert p in z.store.as_of(p.confirmed_ts_ms)

    def test_el_retardo_de_confirmacion_es_variable_y_a_veces_enorme(self):
        """Documenta por qué «desplazar N barras» no vale como mitigación."""
        ts, h, l, c = _serie(1500, seed=2)
        z = detect_batch(ts, h, l, c)
        pivs = z.store.as_of(10**15)
        lags = [(p.confirmed_ts_ms - p.ts_ms) // TF_4H.ms for p in pivs]
        assert len(set(lags)) > 3, "el retardo debería variar mucho, no ser casi constante"
        assert max(lags) > 3 * (sorted(lags)[len(lags) // 2] or 1), (
            "debería haber cola derecha pesada: algunos pivotes tardan muchísimo en confirmarse"
        )


class TestUmbralCongelado:
    """El umbral se congela en la vela del extremo. De ahí sale la monotonía."""

    def test_el_umbral_queda_grabado_en_el_pivote(self):
        ts, h, l, c = _serie(300)
        z = detect_batch(ts, h, l, c)
        for p in z.store.as_of(10**15):
            assert p.thr_at_extreme > 0
            esperado = (p.price - p.thr_at_extreme if p.kind is PivotKind.HIGH
                        else p.price + p.thr_at_extreme)
            assert p.confirm_price == pytest.approx(esperado)

    def test_una_explosion_de_volatilidad_no_desconfirma_nada(self):
        """Sin congelar el umbral, un ATR que se dispara podría invalidar pivotes ya confirmados
        y el gráfico cambiaría de opinión sobre el pasado, sin lanzar nada."""
        ts, h, l, c = _serie(300, seed=9)
        z = ZigZag()
        for i in range(200):
            z.update(int(ts[i]), float(h[i]), float(l[i]), float(c[i]))
        antes = z.store.as_of(int(ts[199]))

        # Cien velas de volatilidad brutal: el ATR se multiplica.
        base = float(c[199])
        for j in range(100):
            t = int(ts[199]) + (j + 1) * TF_4H.ms
            z.update(t, base * 1.30, base * 0.70, base)
        despues = z.store.as_of(int(ts[199]))
        assert despues == antes, (
            "una explosión de volatilidad ha reescrito el pasado: el umbral no estaba congelado"
        )


class TestPrecioDeConfirmacion:
    def test_esta_al_otro_lado_del_extremo(self):
        ts, h, l, c = _serie(300)
        z = detect_batch(ts, h, l, c)
        prov = z.store.provisional_as_of(10**15)
        cp = z.confirm_price()
        assert prov is not None and cp is not None
        if prov.kind is PivotKind.HIGH:
            assert cp < prov.price, "un máximo provisional confirma cayendo POR DEBAJO"
        else:
            assert cp > prov.price, "un mínimo provisional confirma subiendo POR ENCIMA"

    def test_hay_como_mucho_un_provisional(self):
        ts, h, l, c = _serie(200)
        z = detect_batch(ts, h, l, c)
        legs = z.legs_as_of(10**15)
        assert sum(1 for x in legs if x["tentative"]) <= 1


class TestATR:
    def test_coincide_con_talib(self):
        """Oráculo independiente: si nuestro ATR se desvía, todo el umbral se desvía con él."""
        talib = pytest.importorskip("talib")
        ts, h, l, c = _serie(300)
        ref = talib.ATR(h, l, c, timeperiod=14)
        a = WilderATR(14)
        propio = [a.update(float(h[i]), float(l[i]), float(c[i])) for i in range(ts.size)]
        for i in range(20, ts.size):
            if not np.isnan(ref[i]):
                assert propio[i] == pytest.approx(ref[i], rel=1e-9), f"divergencia en {i}"

    def test_no_esta_listo_antes_del_periodo(self):
        """Primer ATR(14) en el índice 14, no en el 13: la vela inicial no aporta rango
        verdadero porque no tiene cierre anterior. Es lo que hace TA-Lib."""
        a = WilderATR(14)
        for i in range(14):
            assert a.update(10 + i, 9 + i, 9.5 + i) is None, f"listo demasiado pronto en {i}"
        assert a.update(24, 23, 23.5) is not None


class TestEscalaInvariante:
    """El umbral es absoluto y escalado por volatilidad, así que k es adimensional: el mismo
    k=1.5 significa lo mismo en BTC, EURUSD y AAPL. Ahí está el multi-activo hecho real."""

    @given(factor=st.floats(min_value=0.01, max_value=100.0, allow_nan=False))
    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_escalar_los_precios_no_cambia_los_pivotes(self, factor):
        ts, h, l, c = _serie(250, seed=4)
        a = detect_batch(ts, h, l, c, ZigZagConfig(min_pct=0.0))  # min_pct=0 -> puro ATR
        b = detect_batch(ts, h * factor, l * factor, c * factor, ZigZagConfig(min_pct=0.0))
        pa = [(p.idx, int(p.kind)) for p in a.store.as_of(10**15)]
        pb = [(p.idx, int(p.kind)) for p in b.store.as_of(10**15)]
        assert pa == pb, f"escalar x{factor} cambió los pivotes: k no es adimensional"
