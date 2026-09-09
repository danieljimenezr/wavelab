"""Binance REST client with a weight governor.

The budget is 6000 weight per minute per IP, and the response reports it in the
``x-mbx-used-weight-1m`` header. Going over escalates like this: **429** (with ``Retry-After``) and
then **418**, which is an IP ban lasting from two minutes to **three days**. For an application that
lives off a public feed, a 418 is total downtime, so the governor brakes at 70% instead of running
all the way to the edge.

The 418 is NOT retried: it is treated as a switch to the fallback feed. Retrying a ban extends the
ban.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from wavelab.core.timeframes import Timeframe
from wavelab.core.types import Bar
from wavelab.feeds.base import FeedCaps

__all__ = ["BinanceREST", "RateLimitCircuitOpen", "WeightGovernor"]

#: Official data-only mirror. Binance's own docs say in as many words that it needs no
#: authentication. It is the default so that we never so much as brush against the account surface.
SPOT_BASE = "https://data-api.binance.vision"
SPOT_FALLBACK = "https://api.binance.com"
FAPI_BASE = "https://fapi.binance.com"

WEIGHT_BUDGET = 6000
KLINE_WEIGHT = 2
MAX_LIMIT = 1000


class RateLimitCircuitOpen(RuntimeError):
    """418: IP banned. Not retried — we switch feeds instead."""


class WeightGovernor:
    """Brakes before the edge, not at it."""

    __slots__ = ("banned_until", "brake_at", "budget", "used")

    def __init__(self, budget: int = WEIGHT_BUDGET, brake_ratio: float = 0.70) -> None:
        self.budget = budget
        self.brake_at = int(budget * brake_ratio)
        self.used = 0
        self.banned_until = 0.0

    def observe(self, headers) -> None:
        v = headers.get("x-mbx-used-weight-1m") or headers.get("X-MBX-USED-WEIGHT-1M")
        if v:
            try:
                self.used = int(v)
            except ValueError:
                pass

    async def wait_if_needed(self) -> None:
        if self.banned_until > time.monotonic():
            raise RateLimitCircuitOpen(
                f"IP banned by Binance for another {self.banned_until - time.monotonic():.0f} s. "
                "Switch to the fallback feed; retrying extends the ban."
            )
        if self.used >= self.brake_at:
            # The counter resets on every wall-clock minute. Waiting for the next one is cheaper
            # than risking a 418 that can last three days.
            wait_s = 60 - (time.time() % 60) + 0.5
            await asyncio.sleep(wait_s)
            self.used = 0

    @property
    def headroom(self) -> float:
        return 1.0 - (self.used / self.budget)


@dataclass(slots=True)
class BinanceREST:
    """Klines and derivatives data. No key, no account, no orders."""

    name: str = "binance_rest"
    base: str = SPOT_BASE
    _client: httpx.AsyncClient | None = None
    gov: WeightGovernor = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.gov is None:
            self.gov = WeightGovernor()

    def caps(self) -> FeedCaps:
        return FeedCaps(
            timeframes=frozenset({"1m", "5m", "15m", "1h", "4h", "1d"}),
            has_websocket=True,
            max_klines_per_request=MAX_LIMIT,
            weight_budget_per_min=WEIGHT_BUDGET,
            deep_history_from_ms=1_502_942_400_000,  # 2017-08-17, BTCUSDT listing
            supports_aux=frozenset({"funding", "open_interest", "long_short", "liquidation"}),
        )

    async def __aenter__(self) -> BinanceREST:
        self._client = httpx.AsyncClient(timeout=30.0, http2=True, follow_redirects=True)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()

    async def _get(self, url: str, params: dict, weight: int = 1) -> object:
        assert self._client is not None, "use BinanceREST as a context manager"
        for attempt in range(4):
            await self.gov.wait_if_needed()
            r = await self._client.get(url, params=params)
            self.gov.observe(r.headers)

            if r.status_code == 418:
                self.gov.banned_until = time.monotonic() + 3600
                raise RateLimitCircuitOpen(f"418 from {url}: IP banned. Do NOT retry.")
            if r.status_code == 429:
                wait_s = float(r.headers.get("Retry-After", 2 ** attempt))
                await asyncio.sleep(wait_s)
                continue
            if r.status_code >= 500:
                await asyncio.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"{url}: retries exhausted")

    # ------------------------------------------------------------------ klines

    async def fetch_klines(
        self, symbol: str, tf: Timeframe, start_ms: int, end_ms: int, limit: int = MAX_LIMIT
    ) -> list[Bar]:
        raw = await self._get(
            f"{self.base}/api/v3/klines",
            {"symbol": symbol.upper(), "interval": tf.name,
             "startTime": start_ms, "endTime": end_ms, "limit": min(limit, MAX_LIMIT)},
            weight=KLINE_WEIGHT,
        )
        out: list[Bar] = []
        for k in raw:  # type: ignore[union-attr]
            out.append(Bar(
                symbol=symbol.upper(), tf=tf, open_time_ms=int(k[0]),
                open=float(k[1]), high=float(k[2]), low=float(k[3]), close=float(k[4]),
                volume=float(k[5]), quote_volume=float(k[7]), trades=int(k[8]),
                taker_buy_base=float(k[9]), taker_buy_quote=float(k[10]),
                is_closed=True, n_source_bars=tf.expected_source_bars,
            ))
        return out

    async def heal_gap(self, symbol: str, tf: Timeframe, start_ms: int, end_ms: int) -> list[Bar]:
        """Fill a gap by paginating. Runs on EVERY reconnect, not only after the 24 h one: the cut
        can come from the network, from a service restart or from a deploy."""
        out: list[Bar] = []
        cur = start_ms
        while cur <= end_ms:
            batch = await self.fetch_klines(symbol, tf, cur, end_ms)
            if not batch:
                break
            out.extend(batch)
            nxt = batch[-1].open_time_ms + tf.ms
            if nxt <= cur:
                break
            cur = nxt
            if len(batch) < MAX_LIMIT:
                break
        return out

    # ------------------------------------------------------------------ derivatives

    async def funding_rate(self, symbol: str, limit: int = 1000) -> list[dict]:
        return await self._get(f"{FAPI_BASE}/fapi/v1/fundingRate",
                               {"symbol": symbol.upper(), "limit": limit})  # type: ignore[return-value]

    async def open_interest_hist(self, symbol: str, period: str = "5m", limit: int = 500) -> list[dict]:
        """WATCH OUT: /futures/data/* keeps only ~30 DAYS and for earlier dates returns error -1130
        ('parameter startTime is invalid'), NOT an empty list. A naive backfill loop either breaks
        or swallows the error in silence. Deep history exists only in the archive."""
        return await self._get(f"{FAPI_BASE}/futures/data/openInterestHist",
                               {"symbol": symbol.upper(), "period": period, "limit": limit})  # type: ignore[return-value]

    async def long_short_ratio(self, symbol: str, period: str = "5m", limit: int = 500) -> list[dict]:
        return await self._get(f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio",
                               {"symbol": symbol.upper(), "period": period, "limit": limit})  # type: ignore[return-value]

    async def taker_ratio(self, symbol: str, period: str = "5m", limit: int = 500) -> list[dict]:
        return await self._get(f"{FAPI_BASE}/futures/data/takerlongshortRatio",
                               {"symbol": symbol.upper(), "period": period, "limit": limit})  # type: ignore[return-value]
