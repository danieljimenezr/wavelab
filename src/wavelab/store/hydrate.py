"""Hidratación del histórico. Cuatro etapas: listar, descargar, verificar, ingestar.

    python -m wavelab.store.hydrate            # todo lo publicado (2017-08 -> hoy)
    python -m wavelab.store.hydrate --quick    # últimos 2 años, para ver un gráfico ya

Por qué el archivo y no REST: dos años de velas de 1m son 24 ficheros / ~50 MB / ~50 s a la
velocidad medida desde España (1,1 MB/s), frente a 1.051 llamadas REST y ~33 minutos. Y el archivo
llega hasta 2017-08; REST también, pero paginando durante horas.

Tres niveles de granularidad, porque el archivo no publica el mes en curso hasta que termina:
  1. mensual  — meses completos (lo masivo)
  2. diario   — días del mes en curso
  3. REST     — las últimas horas, que aún no tienen fichero diario
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from wavelab.core.timeframes import TF_1M
from wavelab.feeds.base import Market
from wavelab.feeds.binance_archive import (
    BASE_URL,
    list_available_months,
    parse_klines_zip,
    verify_checksum,
)
from wavelab.store.bars import BarStore

__all__ = ["hydrate"]


def _daily_url(symbol: str, day: date) -> str:
    stem = f"{symbol.upper()}-1m-{day:%Y-%m-%d}"
    return f"{BASE_URL}/data/spot/daily/klines/{symbol.upper()}/1m/{stem}.zip"


async def _fetch(client: httpx.AsyncClient, url: str) -> bytes | None:
    """Descarga y verifica el checksum. Devuelve None si el fichero no existe (404 legítimo)."""
    r = await client.get(url)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    raw = r.content
    chk = await client.get(url + ".CHECKSUM")
    if chk.status_code == 200:
        # Un ZIP truncado a veces descomprime "bien" y produce un mes con la mitad de las velas.
        verify_checksum(raw, chk.text)
    return raw


async def hydrate(
    symbol: str = "BTCUSDT",
    data_dir: Path | str = "data",
    quick: bool = False,
    concurrency: int = 6,
) -> int:
    store = BarStore(Path(data_dir) / "bars")
    t0 = time.perf_counter()

    print(f"[hydrate] listando meses publicados de {symbol} 1m…", flush=True)
    meses = list_available_months(symbol, TF_1M, Market.SPOT)
    if not meses:
        print("[hydrate] el archivo no devolvió ningún mes", file=sys.stderr)
        return 1
    if quick:
        corte = date.today().replace(day=1) - timedelta(days=730)
        meses = [m for m in meses if m >= corte]
    print(f"[hydrate] {len(meses)} meses: {meses[0]:%Y-%m} → {meses[-1]:%Y-%m}", flush=True)

    ya = set(store.months(symbol))
    # El mes más reciente ya guardado puede estar incompleto (se ingestó a mitad de mes),
    # así que se vuelve a bajar. Ingestar es idempotente, no cuesta nada.
    pendientes = [m for m in meses if f"{m:%Y-%m}" not in ya or f"{m:%Y-%m}" == max(ya, default="")]
    print(f"[hydrate] {len(ya)} ya presentes, {len(pendientes)} por descargar", flush=True)

    total_bytes = escritas = duplicadas = 0
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 2)

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, limits=limits) as client:

        async def uno(mes: date) -> tuple[date, bytes | None]:
            async with sem:
                url = f"{BASE_URL}/data/spot/monthly/klines/{symbol.upper()}/1m/{symbol.upper()}-1m-{mes:%Y-%m}.zip"
                return mes, await _fetch(client, url)

        for i in range(0, len(pendientes), concurrency):
            lote = pendientes[i:i + concurrency]
            for mes, raw in await asyncio.gather(*(uno(m) for m in lote)):
                if raw is None:
                    print(f"  {mes:%Y-%m}: no publicado", flush=True)
                    continue
                total_bytes += len(raw)
                df = parse_klines_zip(raw, TF_1M, symbol, Market.SPOT, mes)
                r = store.ingest(symbol, df)
                escritas += r.written
                duplicadas += r.duplicates
                print(f"  {mes:%Y-%m}: {len(df):>6,} velas → {r.written:>6,} nuevas "
                      f"({len(raw)/1e6:.1f} MB)", flush=True)

        # --- días del mes en curso, que aún no tienen fichero mensual ---
        hoy = datetime.now(UTC).date()
        dias = [hoy - timedelta(days=k) for k in range(1, 35)]
        dias = [d for d in dias if d >= hoy.replace(day=1)] or [hoy - timedelta(days=1)]
        print(f"[hydrate] {len(dias)} día(s) del mes en curso…", flush=True)

        async def un_dia(d: date) -> tuple[date, bytes | None]:
            async with sem:
                return d, await _fetch(client, _daily_url(symbol, d))

        for d, raw in await asyncio.gather(*(un_dia(d) for d in sorted(dias))):
            if raw is None:
                continue
            total_bytes += len(raw)
            df = parse_klines_zip(raw, TF_1M, symbol, Market.SPOT)
            r = store.ingest(symbol, df)
            escritas += r.written
            duplicadas += r.duplicates

    dt = time.perf_counter() - t0
    mb = total_bytes / 1e6
    print(f"\n[hydrate] {escritas:,} velas nuevas, {duplicadas:,} duplicadas ignoradas", flush=True)
    print(f"[hydrate] {mb:.0f} MB en {dt:.0f} s ({mb/dt if dt else 0:.1f} MB/s)", flush=True)

    ms = store.months(symbol)
    if ms:
        lo = int(datetime.strptime(ms[0], "%Y-%m").replace(tzinfo=UTC).timestamp() * 1000)
        hi = int(time.time() * 1000)
        pres, tot, frac = store.coverage(symbol, lo, hi)
        print(f"[hydrate] cobertura {ms[0]} → {ms[-1]}: {pres:,}/{tot:,} velas ({frac:.2%})",
              flush=True)
        faltan = tot - pres
        if faltan:
            print(f"[hydrate] faltan {faltan:,} velas de 1m. Algunas son paradas reales de "
                  "Binance; el relleno REST cubrirá el resto en el arranque del motor.", flush=True)
    return 0


def _main() -> int:
    ap = argparse.ArgumentParser(description="Hidrata el histórico de velas desde data.binance.vision")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--quick", action="store_true", help="solo los últimos 2 años")
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    import os
    data = a.data_dir or os.environ.get("WAVELAB_DATA", "data")
    return asyncio.run(hydrate(a.symbol, data, a.quick, a.concurrency))


if __name__ == "__main__":
    sys.exit(_main())
