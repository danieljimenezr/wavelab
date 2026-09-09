"""History hydration. Four stages: list, download, verify, ingest.

    python -m wavelab.store.hydrate            # everything published (2017-08 -> today)
    python -m wavelab.store.hydrate --quick    # last 2 years, to get a chart up right away

Why the archive and not REST: two years of 1m bars is 24 files / ~50 MB / ~50 s at the speed
measured from Spain (1.1 MB/s), against 1,051 REST calls and ~33 minutes. And the archive reaches
back to 2017-08; REST does too, but only by paginating for hours.

Three levels of granularity, because the archive does not publish the current month until it ends:
  1. monthly — complete months (the bulk of it)
  2. daily   — days of the current month
  3. REST    — the last few hours, which do not have a daily file yet
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
    """Downloads and verifies the checksum. Returns None if the file does not exist (a legitimate
    404)."""
    r = await client.get(url)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    raw = r.content
    chk = await client.get(url + ".CHECKSUM")
    if chk.status_code == 200:
        # A truncated ZIP sometimes decompresses "fine" and yields a month with half the bars.
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

    print(f"[hydrate] listing published months for {symbol} 1m…", flush=True)
    months = list_available_months(symbol, TF_1M, Market.SPOT)
    if not months:
        print("[hydrate] the archive returned no months at all", file=sys.stderr)
        return 1
    if quick:
        # UTC, not local. `date.today()` is the machine's civil date, and the archive's month keys
        # are UTC. The two disagree for a 2h window in Spain (CEST = UTC+2) — and it only CHANGES
        # the answer when that window also crosses a month boundary, because of the .replace(day=1)
        # snap: run at 00:30 local on the 1st and the local date says "1st" while UTC still says
        # "last day of the previous month", so the cutoff moves a whole month and --quick fetches
        # 25 months instead of 24. Harmless for the bar grid (ingest is idempotent and these are
        # whole-month keys), but it made the same run answer differently depending on the box's
        # timezone, and the daily-file loop below already computes its `today` as UTC — the two
        # notions of "today" inside one hydrate must not come from different clocks.
        cutoff = datetime.now(UTC).date().replace(day=1) - timedelta(days=730)
        months = [m for m in months if m >= cutoff]
    print(f"[hydrate] {len(months)} months: {months[0]:%Y-%m} → {months[-1]:%Y-%m}", flush=True)

    already = set(store.months(symbol))
    # The most recent stored month may be incomplete (it was ingested mid-month), so it gets
    # downloaded again. Ingesting is idempotent, it costs nothing.
    pending = [m for m in months
               if f"{m:%Y-%m}" not in already or f"{m:%Y-%m}" == max(already, default="")]
    print(f"[hydrate] {len(already)} already present, {len(pending)} to download", flush=True)

    total_bytes = written = duplicated = 0
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 2)

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, limits=limits) as client:

        async def one(month: date) -> tuple[date, bytes | None]:
            async with sem:
                url = f"{BASE_URL}/data/spot/monthly/klines/{symbol.upper()}/1m/{symbol.upper()}-1m-{month:%Y-%m}.zip"
                return month, await _fetch(client, url)

        for i in range(0, len(pending), concurrency):
            batch = pending[i:i + concurrency]
            for month, raw in await asyncio.gather(*(one(m) for m in batch)):
                if raw is None:
                    print(f"  {month:%Y-%m}: not published", flush=True)
                    continue
                total_bytes += len(raw)
                df = parse_klines_zip(raw, TF_1M, symbol, Market.SPOT, month)
                r = store.ingest(symbol, df)
                written += r.written
                duplicated += r.duplicates
                print(f"  {month:%Y-%m}: {len(df):>6,} bars → {r.written:>6,} new "
                      f"({len(raw)/1e6:.1f} MB)", flush=True)

        # --- days of the current month, which have no monthly file yet ---
        today = datetime.now(UTC).date()
        days = [today - timedelta(days=k) for k in range(1, 35)]
        days = [d for d in days if d >= today.replace(day=1)] or [today - timedelta(days=1)]
        print(f"[hydrate] {len(days)} day(s) of the current month…", flush=True)

        async def one_day(d: date) -> tuple[date, bytes | None]:
            async with sem:
                return d, await _fetch(client, _daily_url(symbol, d))

        for d, raw in await asyncio.gather(*(one_day(d) for d in sorted(days))):
            if raw is None:
                continue
            total_bytes += len(raw)
            df = parse_klines_zip(raw, TF_1M, symbol, Market.SPOT)
            r = store.ingest(symbol, df)
            written += r.written
            duplicated += r.duplicates

    dt = time.perf_counter() - t0
    mb = total_bytes / 1e6
    print(f"\n[hydrate] {written:,} new bars, {duplicated:,} duplicates ignored", flush=True)
    print(f"[hydrate] {mb:.0f} MB in {dt:.0f} s ({mb/dt if dt else 0:.1f} MB/s)", flush=True)

    ms = store.months(symbol)
    if ms:
        lo = int(datetime.strptime(ms[0], "%Y-%m").replace(tzinfo=UTC).timestamp() * 1000)
        hi = int(time.time() * 1000)
        pres, tot, frac = store.coverage(symbol, lo, hi)
        print(f"[hydrate] coverage {ms[0]} → {ms[-1]}: {pres:,}/{tot:,} bars ({frac:.2%})",
              flush=True)
        missing = tot - pres
        if missing:
            print(f"[hydrate] {missing:,} 1m bars missing. Some of them are real Binance outages; "
                  "the REST backfill will cover the rest when the engine starts.", flush=True)
    return 0


def _main() -> int:
    ap = argparse.ArgumentParser(
        description="Hydrates the bar history from data.binance.vision")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--quick", action="store_true", help="only the last 2 years")
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    import os
    data = a.data_dir or os.environ.get("WAVELAB_DATA", "data")
    return asyncio.run(hydrate(a.symbol, data, a.quick, a.concurrency))


if __name__ == "__main__":
    sys.exit(_main())
