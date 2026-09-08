# Hallazgos que contradicen el plan

Cosas que el plan daba por buenas y que resultaron falsas al verificarlas. Se documentan aquí porque
la mitad son de la clase «falla en silencio»: nada lanza, nada avisa, y el sistema parece funcionar.

## 1. El WebSocket de futuros de Binance no entrega datos desde España

**Verificado el 2026-09-08 desde dos máquinas independientes** (residencial de Telefónica en Sitges y
datacenter de Clouding en Barcelona), con la librería `websockets` y con sockets TLS crudos:

| Endpoint | Handshake | SUBSCRIBE | Frames de datos |
|---|---|---|---|
| `wss://fstream.binance.com/ws/btcusdt@markPrice@1s` | 101 OK | — | **0 en 90 s** |
| `wss://fstream.binance.com/ws/!forceOrder@arr` | 101 OK | — | **0 en 90 s** |
| `wss://fstream.binance.com/ws` + SUBSCRIBE | 101 OK | `{"result":null,"id":1}` | **0 en 30 s** |
| `wss://data-stream.binance.vision/ws/btcusdt@aggTrade` | 101 OK | `{"result":null,"id":1}` | 99 en 12 s |
| `https://fapi.binance.com/fapi/v1/*` (REST) | — | — | **funciona** |

Binance acepta la conexión, acepta la suscripción respondiendo éxito, y no envía nada. `markPrice@1s`
empuja un mensaje por segundo por definición, así que cero en 90 segundos no admite otra lectura.

**Es el fallo silencioso perfecto**: un grabador ingenuo escribiría un fichero vacío durante meses
convencido de estar funcionando, porque el socket está abierto y no hay ningún error.

**Consecuencia:** el plan trataba el grabador `@forceOrder` como lo más urgente e irrecuperable del
proyecto. No es alcanzable desde aquí. Sustituido por **OKX `liquidation-orders`** (verificado
entregando datos) con **Bybit `allLiquidation`** como secundaria. La REST de futuros de Binance sigue
sirviendo para funding, open interest y ratios long/short.

**Mitigación permanente:** `LiquidationRecorder.silent_seconds` vigila el SILENCIO, no el estado del
socket, y avisa si una fuente lleva más de 6 h conectada sin entregar nada.

**Caveat que hereda el proyecto:** ningún feed de un solo exchange ve el mercado entero, y ni OKX ni
Bybit ven las liquidaciones de Binance, que es el mayor mercado de perpetuos. Sirve como señal de
estrés y de cascada, no como censo ni como magnitud absoluta.

## 2. `ccxt` fija `orjson==3.11.9` exactamente

El plan declaraba `orjson>=3.12`. El lockfile era irresoluble y `uv sync` fallaba el día uno.

## 3. TA-Lib no necesita `brew install ta-lib`

La wheel empaqueta la librería C desde la 0.6.5. Ponerlo en la documentación de instalación arruina la
primera ejecución sin motivo.

## 4. Las cifras de rendimiento estaban mal por 25x

Ver `BENCHMARKS.md`. El ZigZag en Python puro sobre 500k velas son 122 ms, no los ~5 ms de la síntesis.
Sobre la ventana real de trabajo el presupuesto se cumple con 18x de margen: numba descartado por
medición.

## 5. `fallocate` deja la imagen dispersa

8 GB aparentes, 69 MB reales. La «reserva» de disco no reservaba nada y el host bajaba según wavelab
escribía. Ahora se verifica con `du` y se materializa si hace falta.

## 6. `MemoryMax` no acota el swap

Un proceso desbocado se estrangulaba en 1,2 GB de RAM y empujaba 1.957 MB de los 2.048 MB del swapfile
del host, con `oom_kill=0`. Nadie lo mataba y `OOMScoreAdjust` no llegaba a dispararse nunca.
Corregido con `MemorySwapMax`.

## 7. `MemoryHigh` por debajo de `MemoryMax` cuelga en vez de matar

El proceso no muere: se arrastra a 1 MB/s indefinidamente. Para un lote de validación es aceptable;
para el servicio en vivo es peor que una caída, porque `Restart=always` no se dispara nunca y te
quedas con un gráfico congelado y ningún error.

## 8. El resampleo por `close_time` funcionaba por accidente

Binance devuelve `open + duración − 1 ms`; OKX, Bybit y Coinbase solo `open_time`. El resampleo
indexado por `close_time` acertaba por ese 1 ms y se habría roto con cualquier proveedor que redondease
al alza. Ahora se resamplea sobre `open_time`, que es inequívoco.
