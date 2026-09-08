"""El motor en vivo debe producir EXACTAMENTE lo mismo que el camino offline.

Si el resampleo en vivo y el del backtest difirieran aunque fuese en el último decimal, toda la
promesa de «una sola función» sería falsa: el backtest mediría una serie y el usuario vería otra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelab.core.timeframes import TF_1H, TF_15M, TF_1M, resample_from_1m
from wavelab.engine.live import LiveEngine, Mode

from .conftest import SYMBOL, make_bars


def _engine(tfs=("1m", "15m", "1h")) -> LiveEngine:
    return LiveEngine(SYMBOL, list(tfs), ring_capacity=4096, trigger_tf="15m")


class TestResampleoEnVivo:
    def test_coincide_con_el_offline_barra_por_barra(self):
        bars = make_bars(600, tf=TF_1M)
        e = _engine()
        for b in bars:
            e.on_bar_1m(b)

        df = pd.DataFrame(
            {"open": [b.open for b in bars], "high": [b.high for b in bars],
             "low": [b.low for b in bars], "close": [b.close for b in bars],
             "volume": [b.volume for b in bars]},
            index=pd.Index([b.open_time_ms for b in bars], name="open_time_ms"),
        )
        for tf in (TF_15M, TF_1H):
            offline = resample_from_1m(df, tf)
            offline = offline[offline["n_source_bars"] == tf.expected_source_bars]
            anillo = e.state.rings[tf.name]
            w = anillo.window(len(anillo))
            comunes = offline.index.intersection(w.ts)
            assert len(comunes) >= 4, f"muy pocas velas de {tf.name} para comparar"
            for ts in comunes:
                i = int(np.flatnonzero(w.ts == ts)[0])
                o = offline.loc[ts]
                assert w.open[i] == pytest.approx(o["open"]), f"{tf.name} open en {ts}"
                assert w.high[i] == pytest.approx(o["high"]), f"{tf.name} high en {ts}"
                assert w.low[i] == pytest.approx(o["low"]), f"{tf.name} low en {ts}"
                assert w.close[i] == pytest.approx(o["close"]), f"{tf.name} close en {ts}"
                assert w.volume[i] == pytest.approx(o["volume"]), f"{tf.name} volume en {ts}"

    def test_cuenta_las_velas_fuente(self):
        e = _engine()
        for b in make_bars(120, tf=TF_1M):
            e.on_bar_1m(b)
        w = e.state.rings["1h"].window(2)
        assert list(w.n_source_bars) == [60, 60]

    def test_marca_como_hueco_la_vela_mal_cubierta(self):
        """Una vela de 1h construida con 20 minutos no es una vela de 1h."""
        e = _engine()
        for b in make_bars(120, tf=TF_1M, drop=set(range(0, 40))):
            e.on_bar_1m(b)
        w = e.state.rings["1h"].window(1)
        assert int(w.n_source_bars[0]) == 60, "la 2ª hora está completa"
        assert not bool(w.is_gap[0])

    def test_la_vela_en_curso_no_cierra_nada(self):
        e = _engine()
        bars = make_bars(60, tf=TF_1M)
        for b in bars[:-1]:
            e.on_bar_1m(b)
        antes = len(e.state.rings["1h"])
        ultima = bars[-1]
        abierta = type(ultima)(
            symbol=ultima.symbol, tf=ultima.tf, open_time_ms=ultima.open_time_ms,
            open=ultima.open, high=ultima.high, low=ultima.low, close=ultima.close,
            volume=ultima.volume, is_closed=False)
        assert e.on_bar_1m(abierta) == []
        assert len(e.state.rings["1h"]) == antes
        assert e.state.provisional is abierta

    def test_las_duplicadas_del_curado_se_ignoran(self):
        e = _engine()
        bars = make_bars(30, tf=TF_1M)
        for b in bars:
            e.on_bar_1m(b)
        n = len(e.state.rings["1m"])
        for b in bars[-10:]:
            assert e.on_bar_1m(b) == []
        assert len(e.state.rings["1m"]) == n


class TestModoPuestaAlDia:
    def test_arranca_calentando_y_no_emite(self):
        e = _engine()
        assert e.state.health.mode is Mode.WARMUP
        assert not e.emitting

    def test_un_retraso_grande_impide_emitir(self):
        e = _engine()
        for b in make_bars(120, tf=TF_1M, start_ms=1_600_000_000_000 - (1_600_000_000_000 % TF_1M.ms)):
            e.on_bar_1m(b)
        # Las velas son de 2020: el retraso es de años.
        e.update_health(connected=True, reconnects=0, healed=0, silent_seconds=0.0)
        assert e.state.health.mode is Mode.CATCH_UP
        assert not e.emitting, (
            "emitir durante la puesta al día describe un precio que ya pasó; es la forma más "
            "probable de perder la confianza del usuario en la primera semana"
        )
