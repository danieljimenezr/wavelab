"""Archivo histórico de Binance (data.binance.vision).

Es la ÚNICA fuente gratuita de 9 años de velas de 1m de BTC, y es ~40x más rápida que REST: dos años
de 1m son 24 ficheros / ~50 MB / ~50 s a 1 MB/s medido desde España, frente a 1051 llamadas REST /
~33 min. Hidratar es la primera acción irreversible del proyecto, no una tarea de mantenimiento:
Binance retiró su solicitud MiCA el 2026-06-24 y opera sin licencia en la UE, así que un bloqueo
geográfico de borde puede caer cualquier mañana sin aviso.

DOS TRAMPAS DE CORRUPCIÓN SILENCIOSA, ambas confirmadas y ambas tratadas aquí:

1. **Microsegundos.** Los CSV de klines de SPOT pasaron a microsegundos el 2025-01-01, mientras los
   de futuros UM y toda la API REST siguen en milisegundos. Parsear todo como ms convierte
   2025-01-01 en el año 55000 y el índice queda inservible — sin lanzar ningún error.

2. **Cabecera.** Los CSV de futuros llevan fila de cabecera y los de spot no. Leer con
   ``header=None`` sobre un fichero de futuros mete la palabra "open_time" como primer dato.

Ambas se detectan por CONTENIDO, no por fecha ni por ruta: una regla basada en la fecha del cambio
falla el día que Binance rellene un mes antiguo con el formato nuevo.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from xml.etree import ElementTree

import httpx
import numpy as np
import pandas as pd

from wavelab.core.timeframes import Timeframe
from wavelab.feeds.base import Market

__all__ = [
    "BASE_URL",
    "KLINE_COLUMNS",
    "MicrosecondBoundaryError",
    "list_available_months",
    "monthly_url",
    "parse_klines_zip",
    "verify_checksum",
]

BASE_URL = "https://data.binance.vision"
_S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

#: Las 12 columnas del CSV de klines de Binance, en orden. La última ("ignore") existe de verdad.
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

#: Frontera para distinguir milisegundos de microsegundos.
#: Un instante en ms del periodo 2017-2030 vive en ~1,5e12..1,9e12; en µs, en ~1,5e15..1,9e15.
#: 1e14 los separa con tres órdenes de magnitud de margen por ambos lados, así que no hay ninguna
#: fecha plausible que pueda caer del lado equivocado.
_US_THRESHOLD = 1e14


class MicrosecondBoundaryError(ValueError):
    """Un mismo fichero mezcla milisegundos y microsegundos: el formato cambió a mitad."""


def _market_path(market: str) -> str:
    return "spot" if market == Market.SPOT else f"futures/{market}"


def monthly_url(symbol: str, tf: Timeframe, month: date, market: str = Market.SPOT) -> str:
    p = _market_path(market)
    stem = f"{symbol.upper()}-{tf.name}-{month:%Y-%m}"
    return f"{BASE_URL}/data/{p}/monthly/klines/{symbol.upper()}/{tf.name}/{stem}.zip"


def list_available_months(
    symbol: str, tf: Timeframe, market: str = Market.SPOT, client: httpx.Client | None = None
) -> list[date]:
    """Lista los meses realmente publicados, paginando el listado S3.

    Se consulta en vez de generar el rango a ciegas: un símbolo listado a mitad de mes, un mes que
    Binance no publicó, o un par retirado producirían 404 en bucle, y distinguir "no existe" de
    "no responde" a base de códigos de error es exactamente la clase de fragilidad que hace fallar
    una hidratación a las 3 de la mañana.
    """
    own = client is None
    client = client or httpx.Client(timeout=30.0, follow_redirects=True)
    prefix = f"data/{_market_path(market)}/monthly/klines/{symbol.upper()}/{tf.name}/"
    months: list[date] = []
    marker: str | None = None
    try:
        while True:
            params = {"delimiter": "/", "prefix": prefix}
            if marker:
                params["marker"] = marker
            r = client.get(_S3_LIST, params=params)
            r.raise_for_status()
            root = ElementTree.fromstring(r.text)
            ns = {"s3": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
            # `ns` se reasigna en cada vuelta del bucle, así que se enlaza por VALOR con un
            # argumento por defecto. Hoy funcionaría igual porque el lambda se usa en la misma
            # iteración, pero capturar por referencia una variable de bucle es la clase de código
            # que se rompe en silencio en cuanto alguien lo refactoriza.
            find = ((lambda el, t, _ns=ns: el.findall(f"s3:{t}", _ns)) if ns
                    else (lambda el, t: el.findall(t)))
            keys = [k.text or "" for c in find(root, "Contents") for k in find(c, "Key")]
            for k in keys:
                if not k.endswith(".zip"):
                    continue
                stem = k.rsplit("/", 1)[-1].removesuffix(".zip")
                ym = stem.rsplit("-", 2)[-2:]
                try:
                    months.append(date(int(ym[0]), int(ym[1]), 1))
                except (ValueError, IndexError):
                    continue
            truncated = (find(root, "IsTruncated") or [None])[0]
            if truncated is None or (truncated.text or "false").lower() != "true":
                break
            nxt = (find(root, "NextMarker") or [None])[0]
            marker = nxt.text if nxt is not None else (keys[-1] if keys else None)
            if not marker:
                break
    finally:
        if own:
            client.close()
    return sorted(set(months))


def verify_checksum(raw: bytes, expected_line: str) -> None:
    """Compara con el .CHECKSUM publicado. Un ZIP truncado descomprime «bien» a veces."""
    expected = expected_line.split()[0].strip().lower()
    got = hashlib.sha256(raw).hexdigest()
    if got != expected:
        raise ValueError(
            f"checksum SHA-256 no coincide: esperado {expected[:16]}…, obtenido {got[:16]}…. "
            "Descarga corrupta o truncada; NO ingestar."
        )


def _has_header(first_line: bytes) -> bool:
    """Detecta la cabecera por CONTENIDO: el primer campo de una fila de datos es un entero."""
    first_field = first_line.split(b",", 1)[0].strip().strip(b'"')
    try:
        int(first_field)
    except ValueError:
        return True
    return False


def _normalize_epoch(values: np.ndarray, what: str) -> np.ndarray:
    """Convierte a milisegundos, decidiendo POR FILA si venían en microsegundos.

    Por fila y no por fichero: si algún día Binance republica un mes antiguo con el formato nuevo,
    una regla por fecha o por ruta fallaría en silencio. El coste es una comparación vectorizada.
    """
    v = values.astype(np.int64, copy=False)
    is_us = v > _US_THRESHOLD
    n_us = int(is_us.sum())
    if 0 < n_us < v.size:
        raise MicrosecondBoundaryError(
            f"{what}: el fichero mezcla milisegundos y microsegundos "
            f"({n_us} de {v.size} filas en µs). Binance cambió el formato de klines de spot el "
            "2025-01-01; un fichero mixto significa que el supuesto de un solo formato por fichero "
            "ya no vale y hay que revisar el parser antes de ingestar nada."
        )
    return v // 1000 if n_us else v


def parse_klines_zip(
    raw: bytes,
    tf: Timeframe,
    symbol: str,
    market: str = Market.SPOT,
    month: date | None = None,
) -> pd.DataFrame:
    """ZIP mensual de Binance -> DataFrame indexado por ``open_time_ms`` en MILISEGUNDOS.

    Aplica las dos ramas (cabecera y microsegundos) por contenido, y asevera la frontera del mes:
    un fichero cuyas filas se salen del mes que anuncia su nombre significa que la convención de
    nombres cambió, y descubrirlo tras ingestar nueve años es muchísimo peor que fallar aquí.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"se esperaba 1 CSV en el ZIP, hay {len(names)}: {names}")
        data = zf.read(names[0])

    if not data.strip():
        raise ValueError("CSV vacío")

    first_line = data.split(b"\n", 1)[0]
    df = pd.read_csv(
        io.BytesIO(data),
        header=0 if _has_header(first_line) else None,
        names=KLINE_COLUMNS,
        usecols=range(len(KLINE_COLUMNS)),
    )

    open_ms = _normalize_epoch(df["open_time"].to_numpy(), f"{symbol} {tf.name} open_time")
    # .to_numpy() NO es opcional. Construir un DataFrame con Series (que traen su propio índice
    # 0..n-1) y pasar index= con otros valores hace que pandas ALINEE en vez de asignar: sin
    # solape, el resultado son NaN en todas las columnas, con el índice correcto, la longitud
    # correcta y el dtype correcto (NaN es float64). Renderiza sin error y pasa cualquier test
    # que compruebe forma en vez de contenido.
    out = pd.DataFrame(
        {
            "open": df["open"].to_numpy(dtype="float64"),
            "high": df["high"].to_numpy(dtype="float64"),
            "low": df["low"].to_numpy(dtype="float64"),
            "close": df["close"].to_numpy(dtype="float64"),
            "volume": df["volume"].to_numpy(dtype="float64"),
            "quote_volume": df["quote_volume"].to_numpy(dtype="float64"),
            "trades": df["trades"].to_numpy(dtype="int64"),
            "taker_buy_base": df["taker_buy_base"].to_numpy(dtype="float64"),
            "taker_buy_quote": df["taker_buy_quote"].to_numpy(dtype="float64"),
        },
        index=pd.Index(open_ms, name="open_time_ms"),
    ).sort_index()

    if out.index.has_duplicates:
        n = int(out.index.duplicated().sum())
        out = out[~out.index.duplicated(keep="first")]
        # No es fatal: la deduplicación es responsabilidad del almacén. Pero se cuenta.
        out.attrs["duplicates_dropped"] = n

    # --- rejilla ---------------------------------------------------------------------
    # Un open_time fuera de la rejilla suele delatar una conversión µs/ms equivocada, pero NO
    # siempre. Caso real y verificado: entre el 2017-12-04 06:00:20.799 y el 2017-12-18
    # 10:00:20.799, la rejilla de klines de Binance estuvo desplazada 20,799 s. Son 20.401 velas
    # de BTCUSDT, de las que 20.320 tienen operaciones y suman 144.678 BTC de volumen: datos
    # reales, no relleno. Rechazarlas costaría dos semanas de histórico; aceptarlas en silencio
    # dejaría un índice mentiroso.
    #
    # Se alinean al minuto que las contiene y se CUENTA cuántas, para que el desfase quede en el
    # registro y no en la memoria de nadie. Si el desfase fuese >= la mitad del timeframe, o si
    # afectase a la mayoría del fichero de un año moderno, casi seguro es un error de conversión
    # y entonces sí hay que mirar antes de tragárselo.
    resto = out.index.to_numpy() % tf.ms
    n_desalineadas = int((resto != 0).sum())
    if n_desalineadas:
        desfase_max = int(resto.max())
        if desfase_max >= tf.ms // 2:
            raise ValueError(
                f"{symbol} {tf.name}: desfase de {desfase_max} ms, la mitad o más de un "
                f"{tf.name}. Eso no es una rejilla desplazada, es una conversión mal hecha."
            )
        out.index = pd.Index(out.index.to_numpy() - resto, name="open_time_ms")
        # Al alinear pueden chocar dos velas en el mismo minuto. Se conserva la que tiene
        # operaciones: en el único choque real observado (2017-12-04 06:00) la desplazada
        # tenía trades=0 y volumen 0, y la alineada trades=4.
        if out.index.has_duplicates:
            out = (out.sort_values("trades", ascending=False)
                      .loc[~out.sort_values("trades", ascending=False).index.duplicated(keep="first")]
                      .sort_index())
        out.attrs["realigned"] = n_desalineadas
        out.attrs["max_offset_ms"] = desfase_max

    if month is not None:
        lo = pd.Timestamp(month, tz="UTC").value // 1_000_000
        nxt = date(month.year + (month.month == 12), (month.month % 12) + 1, 1)
        hi = pd.Timestamp(nxt, tz="UTC").value // 1_000_000
        first, last = int(out.index[0]), int(out.index[-1])
        if first < lo or last >= hi:
            raise ValueError(
                f"{symbol} {tf.name} {month:%Y-%m}: filas fuera del mes anunciado "
                f"({first}..{last} vs {lo}..{hi}). La convención de nombres del archivo ha cambiado."
            )

    out.attrs["market"] = market
    out.attrs["symbol"] = symbol.upper()
    out.attrs["timeframe"] = tf.name
    return out
