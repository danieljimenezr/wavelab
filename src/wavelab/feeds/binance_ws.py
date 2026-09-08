"""Cliente WebSocket resiliente para Binance.

Límites reales del servicio, no supuestos:
  - Una conexión es válida **24 horas exactas**. Binance la cierra por diseño: no es un fallo,
    es el funcionamiento normal, y hay que planificarlo.
  - El servidor manda un ping cada 20 s y desconecta si no hay pong en 1 minuto. Por eso el bucle
    de recepción no puede bloquearse NUNCA: la librería responde al pong desde ese mismo bucle.
  - Máximo 5 mensajes ENTRANTES por segundo y conexión (suscripciones, pongs manuales...). Pasarse
    desconecta, y reincidir banea la IP.
  - Máximo 300 conexiones por cada 5 minutos y por IP: la reconexión lleva jitter para no crear
    una tormenta si el corte es del lado de Binance.

Los símbolos van en MINÚSCULAS en la ruta del stream. En mayúsculas la conexión se abre y no
llega ni un solo mensaje: un fallo silencioso perfecto.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from collections.abc import AsyncIterator, Callable

import websockets

__all__ = ["stream_json", "SPOT_WS", "FAPI_WS", "MAX_CONNECTION_SECONDS"]

#: Mirror de solo-datos: no expone streams de usuario, que es justo lo que queremos.
SPOT_WS = "wss://data-stream.binance.vision"
FAPI_WS = "wss://fstream.binance.com"

#: Binance cierra a las 24 h. Nos adelantamos para que el corte sea nuestro y controlado,
#: en vez de una excepción a mitad de un mensaje.
MAX_CONNECTION_SECONDS = 23 * 3600 + 30 * 60


async def stream_json(
    base: str,
    streams: list[str],
    *,
    on_connect: Callable[[], None] | None = None,
    on_disconnect: Callable[[str], None] | None = None,
    max_seconds: int = MAX_CONNECTION_SECONDS,
) -> AsyncIterator[dict]:
    """Itera mensajes decodificados, reconectando indefinidamente.

    Cada mensaje sale enriquecido con ``_ts_ingest_ms``: cuándo nos enteramos, frente a cuándo
    ocurrió. Esa distinción no se puede reconstruir después, así que se registra desde el principio
    aunque hoy no la consuma nadie.
    """
    if any(s != s.lower() for s in streams):
        raise ValueError(
            f"los nombres de stream deben ir en MINÚSCULAS: {streams}. "
            "En mayúsculas la conexión se abre y no llega ningún mensaje."
        )
    path = "/ws/" + streams[0] if len(streams) == 1 else "/stream?streams=" + "/".join(streams)
    url = base + path
    intento = 0

    while True:
        try:
            async with websockets.connect(
                url, ping_interval=None,      # es Binance quien hace ping; no añadimos tráfico
                ping_timeout=None, close_timeout=5, max_queue=2048,
            ) as ws:
                intento = 0
                if on_connect:
                    on_connect()
                deadline = time.monotonic() + max_seconds
                while True:
                    restante = deadline - time.monotonic()
                    if restante <= 0:
                        if on_disconnect:
                            on_disconnect("rotación preventiva antes del corte de 24 h")
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=min(restante, 90))
                    except TimeoutError:
                        # 90 s sin nada teniendo ping cada 20 s: la conexión está muerta
                        # aunque el socket no lo sepa todavía.
                        if on_disconnect:
                            on_disconnect("sin mensajes en 90 s")
                        break
                    msg = json.loads(raw)
                    if "stream" in msg and "data" in msg:   # formato combinado
                        msg = {**msg["data"], "_stream": msg["stream"]}
                    msg["_ts_ingest_ms"] = int(time.time() * 1000)
                    yield msg
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if on_disconnect:
                on_disconnect(f"{type(e).__name__}: {e}")

        intento += 1
        # Retroceso exponencial con jitter: si el corte es de Binance, mil clientes reconectando
        # a la vez crean la tormenta que dispara el límite de 300 conexiones / 5 min.
        espera = min(60.0, 1.5 ** min(intento, 10)) * (0.5 + random.random())
        await asyncio.sleep(espera)
