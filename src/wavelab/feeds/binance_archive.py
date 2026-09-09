"""Binance historical archive (data.binance.vision).

It is the ONLY free source of 9 years of 1m BTC bars, and it is ~40x faster than REST: two years of
1m data is 24 files / ~50 MB / ~50 s at 1 MB/s measured from Spain, against 1051 REST calls /
~33 min. Hydrating is the project's first irreversible action, not a maintenance chore: Binance
withdrew its MiCA application on 2026-06-24 and operates in the EU without a licence, so an edge
geo-block could land any morning without warning.

TWO SILENT-CORRUPTION TRAPS, both confirmed and both handled here:

1. **Microseconds.** SPOT kline CSVs switched to microseconds on 2025-01-01, while the UM futures
   ones and the whole REST API stayed on milliseconds. Parsing everything as ms turns 2025-01-01
   into the year 55000 and leaves the index useless -- without raising any error at all.

2. **Header.** Futures CSVs carry a header row and spot ones do not. Reading a futures file with
   ``header=None`` slips the word "open_time" in as the first data point.

Both are detected by CONTENT, not by date and not by path: a rule keyed on the changeover date
breaks the day Binance backfills an old month in the new format.
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

#: The 12 columns of the Binance kline CSV, in order. The last one ("ignore") really does exist.
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

#: Boundary that separates milliseconds from microseconds.
#: An instant from 2017-2030 lives at ~1.5e12..1.9e12 in ms, and at ~1.5e15..1.9e15 in µs.
#: 1e14 splits them with three orders of magnitude of headroom on either side, so there is no
#: plausible date that could fall on the wrong side of it.
_US_THRESHOLD = 1e14


class MicrosecondBoundaryError(ValueError):
    """One and the same file mixes milliseconds and microseconds: the format changed midway."""


def _market_path(market: str) -> str:
    return "spot" if market == Market.SPOT else f"futures/{market}"


def monthly_url(symbol: str, tf: Timeframe, month: date, market: str = Market.SPOT) -> str:
    p = _market_path(market)
    stem = f"{symbol.upper()}-{tf.name}-{month:%Y-%m}"
    return f"{BASE_URL}/data/{p}/monthly/klines/{symbol.upper()}/{tf.name}/{stem}.zip"


def list_available_months(
    symbol: str, tf: Timeframe, market: str = Market.SPOT, client: httpx.Client | None = None
) -> list[date]:
    """List the months actually published, paginating the S3 listing.

    We ask instead of generating the range blindly: a symbol listed mid-month, a month Binance never
    published, or a delisted pair would each produce 404s in a loop, and telling "does not exist"
    apart from "is not answering" on the strength of status codes is exactly the kind of fragility
    that makes a hydration fail at 3 in the morning.
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
            # `ns` is rebound on every turn of the loop, so it is captured by VALUE through a
            # default argument. It would work either way today because the lambda is used within
            # the same iteration, but capturing a loop variable by reference is the kind of code
            # that breaks silently the moment somebody refactors it.
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
    """Check against the published .CHECKSUM. A truncated ZIP sometimes unzips just "fine"."""
    expected = expected_line.split()[0].strip().lower()
    got = hashlib.sha256(raw).hexdigest()
    if got != expected:
        raise ValueError(
            f"SHA-256 checksum mismatch: expected {expected[:16]}…, got {got[:16]}…. "
            "Corrupt or truncated download; do NOT ingest."
        )


def _has_header(first_line: bytes) -> bool:
    """Detect the header by CONTENT: the first field of a data row is an integer."""
    first_field = first_line.split(b",", 1)[0].strip().strip(b'"')
    try:
        int(first_field)
    except ValueError:
        return True
    return False


def _normalize_epoch(values: np.ndarray, what: str) -> np.ndarray:
    """Convert to milliseconds, deciding PER ROW whether the values came in microseconds.

    Per row, not per file: if Binance ever republishes an old month in the new format, a date-based
    or path-based rule would fail silently. The cost is one vectorised compare.
    """
    v = values.astype(np.int64, copy=False)
    is_us = v > _US_THRESHOLD
    n_us = int(is_us.sum())
    if 0 < n_us < v.size:
        raise MicrosecondBoundaryError(
            f"{what}: the file mixes milliseconds and microseconds "
            f"({n_us} of {v.size} rows in µs). Binance changed the spot kline format on "
            "2025-01-01; a mixed file means the one-format-per-file assumption no longer holds "
            "and the parser has to be revisited before anything is ingested."
        )
    return v // 1000 if n_us else v


def parse_klines_zip(
    raw: bytes,
    tf: Timeframe,
    symbol: str,
    market: str = Market.SPOT,
    month: date | None = None,
) -> pd.DataFrame:
    """Binance monthly ZIP -> DataFrame indexed by ``open_time_ms`` in MILLISECONDS.

    Applies both branches (header and microseconds) by content, and asserts the month boundary: a
    file whose rows fall outside the month its own name advertises means the naming convention has
    changed, and finding that out after ingesting nine years is vastly worse than failing here.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected 1 CSV in the ZIP, found {len(names)}: {names}")
        data = zf.read(names[0])

    if not data.strip():
        raise ValueError("empty CSV")

    first_line = data.split(b"\n", 1)[0]
    df = pd.read_csv(
        io.BytesIO(data),
        header=0 if _has_header(first_line) else None,
        names=KLINE_COLUMNS,
        usecols=range(len(KLINE_COLUMNS)),
    )

    open_ms = _normalize_epoch(df["open_time"].to_numpy(), f"{symbol} {tf.name} open_time")
    # .to_numpy() is NOT optional. Building a DataFrame out of Series (which bring their own 0..n-1
    # index) while passing an index= with different values makes pandas ALIGN instead of assign:
    # with no overlap, the result is NaN in every column, with the right index, the right length
    # and the right dtype (NaN is float64). It renders without any error and passes any test that
    # checks shape instead of content.
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
        # Not fatal: deduplication is the store's job. But it gets counted.
        out.attrs["duplicates_dropped"] = n

    # --- grid ------------------------------------------------------------------------
    # An open_time off the grid usually gives away a botched µs/ms conversion, but NOT always.
    # Real, verified case: between 2017-12-04 06:00:20.799 and 2017-12-18 10:00:20.799 the Binance
    # kline grid was shifted by 20.799 s. That is 20,401 BTCUSDT bars, of which 20,320 have trades
    # and together add up to 144,678 BTC of volume: real data, not filler. Rejecting them would
    # cost two weeks of history; accepting them silently would leave a lying index.
    #
    # They are aligned to the minute that contains them and COUNTED, so the shift ends up in the
    # record and not in somebody's memory. If the shift were >= half the timeframe, or if it hit
    # most of a file from a modern year, it is almost certainly a conversion error and then it does
    # need a look before being swallowed.
    remainder = out.index.to_numpy() % tf.ms
    n_misaligned = int((remainder != 0).sum())
    if n_misaligned:
        max_offset = int(remainder.max())
        if max_offset >= tf.ms // 2:
            raise ValueError(
                f"{symbol} {tf.name}: offset of {max_offset} ms, half a "
                f"{tf.name} or more. That is not a shifted grid, it is a botched conversion."
            )
        out.index = pd.Index(out.index.to_numpy() - remainder, name="open_time_ms")
        # Aligning can push two bars onto the same minute. The one with trades is kept: in the
        # only real collision observed (2017-12-04 06:00) the shifted bar had trades=0 and volume
        # 0, and the aligned one had trades=4.
        if out.index.has_duplicates:
            out = (out.sort_values("trades", ascending=False)
                      .loc[~out.sort_values("trades", ascending=False).index.duplicated(keep="first")]
                      .sort_index())
        out.attrs["realigned"] = n_misaligned
        out.attrs["max_offset_ms"] = max_offset

    if month is not None:
        lo = pd.Timestamp(month, tz="UTC").value // 1_000_000
        nxt = date(month.year + (month.month == 12), (month.month % 12) + 1, 1)
        hi = pd.Timestamp(nxt, tz="UTC").value // 1_000_000
        first, last = int(out.index[0]), int(out.index[-1])
        if first < lo or last >= hi:
            raise ValueError(
                f"{symbol} {tf.name} {month:%Y-%m}: rows outside the advertised month "
                f"({first}..{last} vs {lo}..{hi}). The archive's naming convention has changed."
            )

    out.attrs["market"] = market
    out.attrs["symbol"] = symbol.upper()
    out.attrs["timeframe"] = tf.name
    return out
