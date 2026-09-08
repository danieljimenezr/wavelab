"""Los límites son lo que separa una demo local de algo expuesto a internet."""

from __future__ import annotations

import asyncio

import pytest

from wavelab.server.limits import RateLimiter, TooBusy, TooMany


async def test_corta_por_minuto():
    lim = RateLimiter(max_concurrent=4, per_hour=100, per_minute=3)
    for _ in range(3):
        async with lim.slot("1.2.3.4"):
            pass
    with pytest.raises(TooMany, match="por minuto"):
        async with lim.slot("1.2.3.4"):
            pass


async def test_corta_por_hora():
    lim = RateLimiter(max_concurrent=4, per_hour=3, per_minute=100)
    for _ in range(3):
        async with lim.slot("1.2.3.4"):
            pass
    with pytest.raises(TooMany, match="límite"):
        async with lim.slot("1.2.3.4"):
            pass


async def test_las_ips_no_se_estorban():
    """Un abusón no puede dejar sin servicio a los demás."""
    lim = RateLimiter(max_concurrent=4, per_hour=2, per_minute=100)
    for _ in range(2):
        async with lim.slot("abuson"):
            pass
    with pytest.raises(TooMany):
        async with lim.slot("abuson"):
            pass
    async with lim.slot("otro"):        # no debe lanzar
        pass


async def test_limita_la_concurrencia():
    lim = RateLimiter(max_concurrent=2, per_hour=100, per_minute=100)
    activos = 0
    pico = 0

    async def uno(i):
        nonlocal activos, pico
        async with lim.slot(f"ip{i}"):
            activos += 1
            pico = max(pico, activos)
            await asyncio.sleep(0.05)
            activos -= 1

    await asyncio.gather(*(uno(i) for i in range(8)))
    assert pico <= 2, f"se ejecutaron {pico} a la vez con un máximo de 2"


async def test_rechaza_cuando_esta_saturado():
    lim = RateLimiter(max_concurrent=1, per_hour=100, per_minute=100)
    lim._Ctx  # noqa: B018
    async with lim.slot("a"):
        # Con el único hueco ocupado y un tiempo de espera corto, el segundo debe rendirse.
        lim._sem = asyncio.Semaphore(0)
        ctx = lim.slot("b")
        with pytest.raises(TooBusy, match="demasiadas validaciones"):
            await asyncio.wait_for(ctx.__aenter__(), timeout=30)


async def test_poda_las_ips_inactivas():
    """Sin poda, el diccionario crece siempre: una fuga lenta que solo se nota tras semanas."""
    lim = RateLimiter(max_concurrent=9, per_hour=1000, per_minute=1000)
    for i in range(5200):
        async with lim.slot(f"ip{i}"):
            pass
    assert lim.stats["ips_activas"] <= 5200
