"""Resilient WebSocket client for Binance.

Real service limits, not assumptions:
  - A connection is valid for **exactly 24 hours**. Binance closes it by design: that is not a
    failure, it is normal operation, and it has to be planned for.
  - The server sends a ping every 20 s and disconnects if there is no pong within 1 minute. That is
    why the receive loop can NEVER block: the library answers the pong from that same loop.
  - At most 5 INCOMING messages per second per connection (subscriptions, manual pongs...). Going
    over disconnects you, and doing it again bans the IP.
  - At most 300 connections per 5 minutes per IP: reconnection carries jitter so as not to create a
    storm when the outage is on Binance's side.

Symbols go LOWERCASE in the stream path. In uppercase the connection opens and not a single message
arrives: a perfect silent failure.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Callable

import websockets

__all__ = ["FAPI_WS", "MAX_CONNECTION_SECONDS", "SPOT_WS", "stream_json"]

#: Data-only mirror: it exposes no user streams, which is exactly what we want.
SPOT_WS = "wss://data-stream.binance.vision"
FAPI_WS = "wss://fstream.binance.com"

#: Binance closes at 24 h. We get there first so the cut is ours and controlled, instead of an
#: exception in the middle of a message.
MAX_CONNECTION_SECONDS = 23 * 3600 + 30 * 60


async def stream_json(
    base: str,
    streams: list[str],
    *,
    on_connect: Callable[[], None] | None = None,
    on_disconnect: Callable[[str], None] | None = None,
    max_seconds: int = MAX_CONNECTION_SECONDS,
) -> AsyncIterator[dict]:
    """Iterate decoded messages, reconnecting indefinitely.

    Every message comes out enriched with ``_ts_ingest_ms``: when we found out, as against when it
    happened. That distinction cannot be reconstructed afterwards, so it is recorded from the start
    even though nothing consumes it today.
    """
    if any(s != s.lower() for s in streams):
        raise ValueError(
            f"stream names must be LOWERCASE: {streams}. "
            "In uppercase the connection opens and no message ever arrives."
        )
    path = "/ws/" + streams[0] if len(streams) == 1 else "/stream?streams=" + "/".join(streams)
    url = base + path
    attempt = 0

    while True:
        try:
            async with websockets.connect(
                url, ping_interval=None,      # Binance does the pinging; we do not add traffic
                ping_timeout=None, close_timeout=5, max_queue=2048,
            ) as ws:
                attempt = 0
                if on_connect:
                    on_connect()
                deadline = time.monotonic() + max_seconds
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if on_disconnect:
                            on_disconnect("pre-emptive rotation ahead of the 24 h cut")
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 90))
                    except TimeoutError:
                        # 90 s of nothing when there is a ping every 20 s: the connection is dead
                        # even if the socket does not know it yet.
                        if on_disconnect:
                            on_disconnect("no messages in 90 s")
                        break
                    msg = json.loads(raw)
                    if "stream" in msg and "data" in msg:   # combined format
                        msg = {**msg["data"], "_stream": msg["stream"]}
                    msg["_ts_ingest_ms"] = int(time.time() * 1000)
                    yield msg
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if on_disconnect:
                on_disconnect(f"{type(e).__name__}: {e}")

        attempt += 1
        # Exponential backoff with jitter: if the outage is Binance's, a thousand clients
        # reconnecting at once create the storm that trips the 300 connections / 5 min limit.
        wait_s = min(60.0, 1.5 ** min(attempt, 10)) * (0.5 + random.random())
        await asyncio.sleep(wait_s)
