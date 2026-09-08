"""Las dos trampas de corrupción silenciosa del archivo, y la idempotencia del almacén.

Herméticos: los ZIP se fabrican aquí reproduciendo los DOS formatos reales de Binance. La prueba
contra datos de verdad existe también, marcada `net`, y está excluida por defecto.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from wavelab.core.timeframes import TF_1H, TF_1M
from wavelab.feeds.base import Market
from wavelab.feeds.binance_archive import (
    MicrosecondBoundaryError,
    monthly_url,
    parse_klines_zip,
    verify_checksum,
)
from wavelab.store.bars import BarStore

ENE_2025 = 1735689600000   # 2025-01-01T00:00:00Z en ms
DIC_2024 = 1733011200000   # 2024-12-01T00:00:00Z en ms


def _csv(start_ms: int, n: int, tf_ms: int, *, micros: bool, header: bool) -> bytes:
    mul = 1000 if micros else 1
    filas = []
    if header:
        filas.append(",".join([
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]))
    for i in range(n):
        ot = (start_ms + i * tf_ms) * mul
        ct = ot + tf_ms * mul - (1 * mul)
        filas.append(f"{ot},100.0,101.0,99.0,100.5,10.0,{ct},1005.0,42,5.0,502.5,0")
    return ("\n".join(filas) + "\n").encode()


def _zip(csv: bytes, name: str = "x.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, csv)
    return buf.getvalue()


class TestTrampaMicrosegundos:
    """Los CSV de klines de SPOT pasaron a microsegundos el 2025-01-01. Los de futuros UM y toda la
    API REST siguen en milisegundos. Parsear todo como ms manda 2025-01-01 al año 56971 sin lanzar."""

    def test_milisegundos_se_leen_tal_cual(self):
        df = parse_klines_zip(_zip(_csv(DIC_2024, 60, TF_1M.ms, micros=False, header=False)),
                              TF_1M, "BTCUSDT", Market.SPOT, date(2024, 12, 1))
        assert len(df) == 60
        assert int(df.index[0]) == DIC_2024

    def test_microsegundos_se_convierten(self):
        df = parse_klines_zip(_zip(_csv(ENE_2025, 60, TF_1M.ms, micros=True, header=False)),
                              TF_1M, "BTCUSDT", Market.SPOT, date(2025, 1, 1))
        assert int(df.index[0]) == ENE_2025, "no se convirtió de µs a ms"
        assert 2020 < pd.Timestamp(int(df.index[0]), unit="ms").year < 2040

    def test_la_frontera_del_cambio_de_formato_es_continua(self):
        """El test literal de M1: 2024-12-31 -> 2025-01-01 sin salto."""
        dic = parse_klines_zip(
            _zip(_csv(DIC_2024, 44_640, TF_1M.ms, micros=False, header=False)),
            TF_1M, "BTCUSDT", Market.SPOT, date(2024, 12, 1))
        ene = parse_klines_zip(
            _zip(_csv(ENE_2025, 60, TF_1M.ms, micros=True, header=False)),
            TF_1M, "BTCUSDT", Market.SPOT, date(2025, 1, 1))
        assert int(ene.index[0]) - int(dic.index[-1]) == TF_1M.ms

    def test_un_fichero_mixto_lanza_en_vez_de_adivinar(self):
        a = _csv(DIC_2024, 10, TF_1M.ms, micros=False, header=False)
        b = _csv(ENE_2025, 10, TF_1M.ms, micros=True, header=False)
        with pytest.raises(MicrosecondBoundaryError, match="mezcla"):
            parse_klines_zip(_zip(a + b), TF_1M, "BTCUSDT", Market.SPOT)


class TestTrampaCabecera:
    """Los CSV de futuros llevan cabecera y los de spot no."""

    def test_futuros_con_cabecera(self):
        df = parse_klines_zip(_zip(_csv(ENE_2025, 24, TF_1H.ms, micros=False, header=True)),
                              TF_1H, "BTCUSDT", Market.FUTURES_UM, date(2025, 1, 1))
        assert len(df) == 24
        assert df["open"].dtype == np.float64, "la cabecera se coló como dato"

    def test_spot_sin_cabecera(self):
        df = parse_klines_zip(_zip(_csv(DIC_2024, 24, TF_1H.ms, micros=False, header=False)),
                              TF_1H, "BTCUSDT", Market.SPOT, date(2024, 12, 1))
        assert len(df) == 24


class TestGuardas:
    def test_checksum_corrupto_lanza(self):
        raw = _zip(_csv(DIC_2024, 5, TF_1M.ms, micros=False, header=False))
        with pytest.raises(ValueError, match="checksum"):
            verify_checksum(raw, "0" * 64 + "  fichero.zip")

    def test_filas_fuera_del_mes_anunciado_lanzan(self):
        with pytest.raises(ValueError, match="fuera del mes"):
            parse_klines_zip(_zip(_csv(ENE_2025, 60, TF_1M.ms, micros=False, header=False)),
                             TF_1M, "BTCUSDT", Market.SPOT, date(2024, 12, 1))

    def test_url_correcta(self):
        assert monthly_url("btcusdt", TF_1M, date(2025, 1, 1)).endswith(
            "/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-01.zip")
        assert "/futures/um/" in monthly_url("BTCUSDT", TF_1H, date(2025, 1, 1), Market.FUTURES_UM)


class TestIngestaIdempotente:
    """Tres caminos escriben las mismas velas: archivo, relleno REST y WebSocket. Una vela de 1m
    duplicada dobla el volumen resampleado, distorsiona el ATR y fabrica pivotes, sin ningún error."""

    def _df(self, n: int, start: int = ENE_2025) -> pd.DataFrame:
        i = pd.Index(np.arange(start, start + n * TF_1M.ms, TF_1M.ms, dtype="int64"),
                     name="open_time_ms")
        return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
                            index=i)

    def test_reingestar_no_duplica(self, tmp_path):
        s = BarStore(tmp_path)
        assert s.ingest("BTCUSDT", self._df(100)).written == 100
        r = s.ingest("BTCUSDT", self._df(100))
        assert r.written == 0 and r.duplicates == 100
        assert len(s.read("BTCUSDT", ENE_2025, ENE_2025 + 200 * TF_1M.ms)) == 100

    def test_solape_escribe_solo_lo_nuevo(self, tmp_path):
        """El caso real: reconexión a las 24 h y relleno REST sobre minutos ya guardados."""
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(100))
        r = s.ingest("BTCUSDT", self._df(100, ENE_2025 + 50 * TF_1M.ms))
        assert (r.written, r.duplicates) == (50, 50)

    def test_particiona_por_mes(self, tmp_path):
        # 89.280 minutos desde el 1-ene son 62 días: enero (31) + febrero (28) + 3 de marzo.
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(2 * 44_640))
        assert s.months("BTCUSDT") == ["2025-01", "2025-02", "2025-03"]
        assert (tmp_path / "BTCUSDT" / "2025-02.parquet").exists()

    def test_la_lectura_hace_explicito_el_hueco(self, tmp_path):
        s = BarStore(tmp_path)
        df = self._df(100)
        s.ingest("BTCUSDT", df.drop(df.index[40:60]))
        out = s.read("BTCUSDT", ENE_2025, ENE_2025 + 99 * TF_1M.ms)
        assert len(out) == 100, "la rejilla debe estar completa aunque falten velas"
        assert int(out["is_gap"].sum()) == 20
        present, total, frac = s.coverage("BTCUSDT", ENE_2025, ENE_2025 + 99 * TF_1M.ms)
        assert (present, total) == (80, 100) and frac == 0.8

    def test_rechaza_fuera_de_rejilla(self, tmp_path):
        s = BarStore(tmp_path)
        df = self._df(10)
        df.index = pd.Index(df.index.to_numpy() + 1, name="open_time_ms")
        with pytest.raises(ValueError, match="rejilla"):
            s.ingest("BTCUSDT", df)


@pytest.mark.net
def test_datos_reales_a_ambos_lados_del_cambio_de_formato():
    """Contra el archivo de verdad. `pytest -m net` para ejecutarlo."""
    import httpx
    from wavelab.feeds.binance_archive import verify_checksum as vc

    with httpx.Client(timeout=60.0, follow_redirects=True) as c:
        for month, n in ((date(2024, 12, 1), 44_640), (date(2025, 1, 1), 44_640)):
            url = monthly_url("BTCUSDT", TF_1M, month)
            raw = c.get(url).raise_for_status().content
            vc(raw, c.get(url + ".CHECKSUM").raise_for_status().text)
            df = parse_klines_zip(raw, TF_1M, "BTCUSDT", Market.SPOT, month)
            assert len(df) == n
            assert 2024 <= pd.Timestamp(int(df.index[0]), unit="ms").year <= 2025


class TestElContenidoExiste:
    """Regresión de un bug que pasó por delante de cinco tests.

    Construir un DataFrame con Series y un `index=` distinto hace que pandas ALINEE en vez de
    asignar: el resultado tiene el índice correcto, la longitud correcta y el dtype correcto
    (NaN es float64) y TODAS las columnas en NaN. Los tests que comprobaban forma pasaban.
    Estos comprueban contenido.
    """

    def _df(self):
        return parse_klines_zip(_zip(_csv(ENE_2025, 120, TF_1M.ms, micros=False, header=False)),
                                TF_1M, "BTCUSDT", Market.SPOT, date(2025, 1, 1))

    def test_ninguna_columna_es_nan(self):
        df = self._df()
        nan = df.columns[df.isna().all()].tolist()
        assert not nan, f"columnas enteras en NaN: {nan}"
        assert not df.isna().any().any(), "hay NaN sueltos"

    def test_los_valores_son_los_del_csv(self):
        df = self._df()
        assert df["open"].iloc[0] == 100.0
        assert df["high"].iloc[0] == 101.0
        assert df["low"].iloc[0] == 99.0
        assert df["close"].iloc[0] == 100.5
        assert df["volume"].iloc[0] == 10.0
        assert df["trades"].iloc[0] == 42

    def test_las_sumas_no_son_cero(self):
        df = self._df()
        for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades"):
            assert df[c].sum() > 0, f"la columna {c} suma cero: probablemente NaN o vacía"


class TestRejillaDesplazada:
    """2017-12-04 06:00 → 2017-12-18 10:00: la rejilla de Binance estuvo desplazada 20,799 s.
    Son 20.401 velas de BTCUSDT con 144.678 BTC de volumen real, no relleno."""

    def test_se_alinean_y_se_cuentan(self):
        base = 1512367200000  # 2017-12-04 06:00:00 UTC
        filas = [f"{base + i*60000 + 20799},100.0,101.0,99.0,100.5,10.0,"
                 f"{base + (i+1)*60000 + 20798},1005.0,7,5.0,502.5,0" for i in range(30)]
        raw = _zip(("\n".join(filas) + "\n").encode())
        df = parse_klines_zip(raw, TF_1M, "BTCUSDT", Market.SPOT)
        assert (df.index.to_numpy() % TF_1M.ms == 0).all(), "quedan velas fuera de rejilla"
        assert df.attrs["realigned"] == 30, "el realineado debe QUEDAR REGISTRADO, no ser silencioso"
        assert df.attrs["max_offset_ms"] == 20799
        assert df["volume"].sum() == 300.0, "se ha perdido volumen real al realinear"

    def test_un_desfase_de_medio_timeframe_lanza(self):
        """Eso ya no es rejilla desplazada, es conversión mal hecha."""
        base = 1735689600000
        filas = [f"{base + i*60000 + 45000},1,1,1,1,1,{base+(i+1)*60000},1,1,1,1,0" for i in range(5)]
        with pytest.raises(ValueError, match="conversión mal hecha"):
            parse_klines_zip(_zip(("\n".join(filas) + "\n").encode()), TF_1M, "BTCUSDT")
