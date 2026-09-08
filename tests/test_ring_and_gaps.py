"""Huecos: el bug que renderiza sin error.

Una serie de 1m con agujeros y un buffer indexado por POSICIÓN hacen que `i-20` para un ER(20)
atraviese un hueco de horas y devuelva un float perfectamente válido y completamente equivocado.
Nada lanza. Por eso el paso se asevera desde los timestamps en cada construcción de ventana.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wavelab.core.ring import GapError, Ring
from wavelab.core.timeframes import MIN_SOURCE_COVERAGE, TF_1H, TF_1M, TF_15M, resample_from_1m
from wavelab.core.types import Bar

from .conftest import SYMBOL, make_bars


class TestRing:
    def test_ventana_continua_es_completa(self):
        r = Ring(SYMBOL, TF_15M, capacity=64)
        for b in make_bars(20):
            r.append(b)
        w = r.window(20)
        assert w.complete and w.n_gaps == 0
        assert w.end_closed_ts_ms == w.ts[-1] + TF_15M.ms - 1

    def test_detecta_el_hueco_desde_los_timestamps(self):
        r = Ring(SYMBOL, TF_15M, capacity=64)
        for b in make_bars(20, drop={7, 8, 9}):
            r.append(b)
        w = r.window(20)
        assert not w.complete
        assert w.n_gaps == 1        # una discontinuidad, aunque falten tres velas
        with pytest.raises(GapError, match="Rehúsa calcular"):
            w.require_complete("ER(20)")

    def test_el_hueco_sobrevive_a_dar_la_vuelta_al_ring(self):
        """La máscara se RECALCULA desde los timestamps, no se hereda de la escritura."""
        r = Ring(SYMBOL, TF_15M, capacity=8)
        for b in make_bars(20, drop={17}):
            r.append(b)
        w = r.window(8)
        assert len(w) == 8 and not w.complete

    def test_rechaza_duplicados_y_desorden(self):
        r = Ring(SYMBOL, TF_15M, capacity=16)
        bars = make_bars(5)
        for b in bars:
            r.append(b)
        with pytest.raises(ValueError, match="fuera de orden o duplicada"):
            r.append(bars[-1])

    def test_rechaza_velas_sin_cerrar(self):
        r = Ring(SYMBOL, TF_15M, capacity=16)
        b = make_bars(1)[0]
        abierta = Bar(symbol=b.symbol, tf=b.tf, open_time_ms=b.open_time_ms, open=b.open,
                      high=b.high, low=b.low, close=b.close, volume=b.volume, is_closed=False)
        with pytest.raises(ValueError, match="no está cerrada"):
            r.append(abierta)

    def test_la_vela_en_curso_no_toca_la_ventana_cerrada(self):
        r = Ring(SYMBOL, TF_15M, capacity=16)
        bars = make_bars(5)
        for b in bars:
            r.append(b)
        antes = r.window(5)
        r.set_provisional(Bar(symbol=SYMBOL, tf=TF_15M,
                              open_time_ms=bars[-1].open_time_ms + TF_15M.ms,
                              open=1, high=99999, low=0.1, close=50000, volume=1, is_closed=False))
        despues = r.window(5)
        assert (antes.high == despues.high).all(), (
            "la vela en curso ha contaminado la ventana cerrada: el ATR se movería intra-vela y con "
            "él el umbral del ZigZag y la confirmación de pivotes"
        )
        assert len(r.provisional_window(5)) == 5


class TestResampleo:
    def _df(self, n: int, drop: set[int] | None = None) -> pd.DataFrame:
        bars = make_bars(n, tf=TF_1M, drop=drop)
        return pd.DataFrame(
            {"open": [b.open for b in bars], "high": [b.high for b in bars],
             "low": [b.low for b in bars], "close": [b.close for b in bars],
             "volume": [b.volume for b in bars]},
            index=pd.Index([b.open_time_ms for b in bars], name="open_time_ms"),
        )

    def test_cuenta_las_velas_fuente(self):
        out = resample_from_1m(self._df(120), TF_1H)
        assert len(out) == 2
        assert (out["n_source_bars"] == 60).all()
        assert not out["is_gap"].any()
        assert (out["close_time_ms"] == out.index + TF_1H.ms - 1).all()

    def test_marca_como_hueco_la_vela_mal_cubierta(self):
        """Una vela de 1h construida con 43 minutos no es una vela de 1h."""
        faltan = set(range(20))          # 40 de 60 minutos -> 0.67 < 0.9
        out = resample_from_1m(self._df(120, drop=faltan), TF_1H)
        assert out.iloc[0]["n_source_bars"] == 40
        assert bool(out.iloc[0]["is_gap"]) is True
        assert bool(out.iloc[1]["is_gap"]) is False

    def test_el_umbral_de_cobertura_es_el_declarado(self):
        justo = set(range(6))            # 54/60 = 0.90 -> pasa
        out = resample_from_1m(self._df(60, drop=justo), TF_1H)
        assert out.iloc[0]["n_source_bars"] == 54
        assert bool(out.iloc[0]["is_gap"]) is (54 < int(MIN_SOURCE_COVERAGE * 60))

    def test_rechaza_indice_no_creciente(self):
        df = self._df(10)
        df.index = df.index[::-1]
        with pytest.raises(ValueError, match="estrictamente creciente"):
            resample_from_1m(df, TF_1H)
