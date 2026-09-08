"""Cliente REST de Binance con gobernador de peso.

El presupuesto es de 6000 de peso por minuto y por IP, y la respuesta lo dice en la cabecera
``x-mbx-used-weight-1m``. Pasarse escala así: **429** (con ``Retry-After``) y luego **418**, que es
un baneo de IP de dos minutos a **tres días**. Para una aplicación que vive de un feed público, un
418 es una caída total, así que el gobernador frena al 70% en vez de correr hasta el borde.

El 418 NO se reintenta: se trata como conmutador a feed de reserva. Reintentar un baneo alarga el
baneo.
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

#: Mirror oficial de solo-datos. Los documentos de Binance dicen literalmente que no requiere
#: autenticación. Se usa por defecto para no rozar siquiera la superficie con cuentas.
SPOT_BASE = "https://data-api.binance.vision"
SPOT_FALLBACK = "https://api.binance.com"
FAPI_BASE = "https://fapi.binance.com"

WEIGHT_BUDGET = 6000
KLINE_WEIGHT = 2
MAX_LIMIT = 1000


class RateLimitCircuitOpen(RuntimeError):
    """418: IP baneada. No se reintenta — se conmuta de feed."""


class WeightGovernor:
    """Frena antes del borde, no en él."""

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
                f"IP baneada por Binance durante {self.banned_until - time.monotonic():.0f} s más. "
                "Conmuta a feed de reserva; reintentar alarga el baneo."
            )
        if self.used >= self.brake_at:
            # El contador se reinicia cada minuto natural. Esperar al siguiente es más barato
            # que arriesgar un 418 de hasta tres días.
            espera = 60 - (time.time() % 60) + 0.5
            await asyncio.sleep(espera)
            self.used = 0

    @property
    def headroom(self) -> float:
        return 1.0 - (self.used / self.budget)


@dataclass(slots=True)
class BinanceREST:
    """Klines y datos de derivados. Sin clave, sin cuenta, sin órdenes."""

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
            deep_history_from_ms=1_502_942_400_000,  # 2017-08-17, listado de BTCUSDT
            supports_aux=frozenset({"funding", "open_interest", "long_short", "liquidation"}),
        )

    async def __aenter__(self) -> BinanceREST:
        self._client = httpx.AsyncClient(timeout=30.0, http2=True, follow_redirects=True)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()

    async def _get(self, url: str, params: dict, weight: int = 1) -> object:
        assert self._client is not None, "usa BinanceREST como context manager"
        for intento in range(4):
            await self.gov.wait_if_needed()
            r = await self._client.get(url, params=params)
            self.gov.observe(r.headers)

            if r.status_code == 418:
                self.gov.banned_until = time.monotonic() + 3600
                raise RateLimitCircuitOpen(f"418 de {url}: IP baneada. NO reintentar.")
            if r.status_code == 429:
                espera = float(r.headers.get("Retry-After", 2 ** intento))
                await asyncio.sleep(espera)
                continue
            if r.status_code >= 500:
                await asyncio.sleep(2 ** intento)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"{url}: agotados los reintentos")

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
        """Rellena un hueco paginando. Se ejecuta en CADA reconexión, no solo tras la de 24 h:
        el corte puede venir de la red, de un reinicio del servicio o de un despliegue."""
        out: list[Bar] = []
        cur = start_ms
        while cur <= end_ms:
            lote = await self.fetch_klines(symbol, tf, cur, end_ms)
            if not lote:
                break
            out.extend(lote)
            nxt = lote[-1].open_time_ms + tf.ms
            if nxt <= cur:
                break
            cur = nxt
            if len(lote) < MAX_LIMIT:
                break
        return out

    # ------------------------------------------------------------------ derivados

    async def funding_rate(self, symbol: str, limit: int = 1000) -> list[dict]:
        return await self._get(f"{FAPI_BASE}/fapi/v1/fundingRate",
                               {"symbol": symbol.upper(), "limit": limit})  # type: ignore[return-value]

    async def open_interest_hist(self, symbol: str, period: str = "5m", limit: int = 500) -> list[dict]:
        """OJO: /futures/data/* retiene solo ~30 DÍAS y para fechas anteriores devuelve el error
        -1130 ('parameter startTime is invalid'), NO una lista vacía. Un bucle de backfill ingenuo
        se rompe o se traga el error en silencio. El histórico profundo solo está en el archivo."""
        return await self._get(f"{FAPI_BASE}/futures/data/openInterestHist",
                               {"symbol": symbol.upper(), "period": period, "limit": limit})  # type: ignore[return-value]

    async def long_short_ratio(self, symbol: str, period: str = "5m", limit: int = 500) -> list[dict]:
        return await self._get(f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio",
                               {"symbol": symbol.upper(), "period": period, "limit": limit})  # type: ignore[return-value]

    async def taker_ratio(self, symbol: str, period: str = "5m", limit: int = 500) -> list[dict]:
        return await self._get(f"{FAPI_BASE}/futures/data/takerlongshortRatio",
                               {"symbol": symbol.upper(), "period": period, "limit": limit})  # type: ignore[return-value]
