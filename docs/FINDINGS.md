# Findings that contradict the plan

Things the plan took for granted that turned out to be false once checked. They are documented here
because half of them are of the "fails silently" kind: nothing raises, nothing warns, and the system
looks like it is working.

## 1. The Binance futures WebSocket delivers no data from Spain

**Verified on 2026-09-08 from two independent machines** (a Telefónica residential line in Sitges
and a Clouding datacenter in Barcelona), with the `websockets` library and with raw TLS sockets:

| Endpoint | Handshake | SUBSCRIBE | Data frames |
|---|---|---|---|
| `wss://fstream.binance.com/ws/btcusdt@markPrice@1s` | 101 OK | — | **0 in 90 s** |
| `wss://fstream.binance.com/ws/!forceOrder@arr` | 101 OK | — | **0 in 90 s** |
| `wss://fstream.binance.com/ws` + SUBSCRIBE | 101 OK | `{"result":null,"id":1}` | **0 in 30 s** |
| `wss://data-stream.binance.vision/ws/btcusdt@aggTrade` | 101 OK | `{"result":null,"id":1}` | 99 in 12 s |
| `https://fapi.binance.com/fapi/v1/*` (REST) | — | — | **works** |

Binance accepts the connection, accepts the subscription and answers with success, and then sends
nothing. `markPrice@1s` pushes one message per second by definition, so zero in 90 seconds admits no
other reading.

**This is the perfect silent failure**: a naive recorder would write an empty file for months
convinced it was working, because the socket is open and there is no error anywhere.

**Consequence:** the plan treated the `@forceOrder` recorder as the most urgent and most
unrecoverable piece of the project. It is not reachable from here. Replaced by **OKX
`liquidation-orders`** (verified delivering data) with **Bybit `allLiquidation`** as the secondary.
The Binance futures REST API still serves funding, open interest and long/short ratios.

**Permanent mitigation:** `LiquidationRecorder.silent_seconds` watches the SILENCE, not the socket
state, and warns if a source has been connected for more than 6 h without delivering anything.

**Caveat the project inherits:** no single-exchange feed sees the whole market, and neither OKX nor
Bybit see Binance's liquidations, which is the largest perpetuals market. It works as a stress and
cascade signal, not as a census or an absolute magnitude.

## 2. `ccxt` pins `orjson==3.11.9` exactly

The plan declared `orjson>=3.12`. The lockfile was unresolvable and `uv sync` failed on day one.

## 3. TA-Lib does not need `brew install ta-lib`

The wheel bundles the C library as of 0.6.5. Putting it in the install docs ruins the first run for
no reason.

## 4. The performance figures were off by 25x

See `BENCHMARKS.md`. The pure-Python ZigZag over 500k bars is 122 ms, not the ~5 ms of the
synthesis. Over the real working window the budget is met with 18x of headroom: numba ruled out by
measurement.

## 5. `fallocate` leaves the image sparse

8 GB apparent, 69 MB real. The disk "reservation" reserved nothing, and the host's free space went
down as wavelab wrote. It is now verified with `du` and materialised if it needs to be.

## 6. `MemoryMax` does not bound swap

A runaway process throttled at 1.2 GB of RAM and pushed 1,957 MB of the host swapfile's 2,048 MB,
with `oom_kill=0`. Nobody was killing it and `OOMScoreAdjust` never got a chance to fire. Fixed with
`MemorySwapMax`.

## 7. `MemoryHigh` below `MemoryMax` hangs instead of killing

The process does not die: it crawls along at 1 MB/s indefinitely. For a validation batch that is
acceptable; for the live service it is worse than a crash, because `Restart=always` never fires and
you are left with a frozen chart and no error.

## 8. Resampling by `close_time` worked by accident

Binance returns `open + duration − 1 ms`; OKX, Bybit and Coinbase return only `open_time`.
Resampling indexed on `close_time` got the right answer *because of* that 1 ms, and would have
broken with any provider that rounded up. Resampling is now done on `open_time`, which is
unambiguous.
