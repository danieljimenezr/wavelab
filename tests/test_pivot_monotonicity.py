"""La propiedad de la que cuelgan cuatro afirmaciones arquitectónicas.

`as_of(t)` debe ser un PREFIJO ESTRICTO de `as_of(t+1)`, siempre. Si no lo fuese:
  - el histórico de conteos dejaría de ser append-only,
  - la instantánea «como estaba en la vela t» dejaría de ser gratis,
  - el arnés de replay pasaría de O(n) a O(n²),
  - y un conteo que el usuario ya vio desaparecería sin evento de invalidación.

Esa última consecuencia es la grave: es exactamente la deshonestidad que el diseño existe para
eliminar.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from wavelab.core.types import Pivot, PivotKind
from wavelab.waves.store import PivotStore


@st.composite
def historial_de_pivotes(draw, max_n: int = 40):
    """Pivotes válidos: alternos, con idx y confirmación monótonos, umbral congelado."""
    n = draw(st.integers(min_value=1, max_value=max_n))
    idx, conf_ts, out = 0, 0, []
    for i in range(n):
        idx += draw(st.integers(min_value=1, max_value=50))
        lag = draw(st.integers(min_value=1, max_value=200))
        conf_ts += draw(st.integers(min_value=1, max_value=10_000))
        precio = draw(st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False))
        thr = draw(st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False))
        kind = PivotKind.HIGH if i % 2 == 0 else PivotKind.LOW
        out.append(Pivot(idx, conf_ts, precio, kind, thr).confirmed_at(idx + lag, conf_ts))
    return out


@given(pivotes=historial_de_pivotes(), ts=st.lists(st.integers(0, 500_000), min_size=1, max_size=30))
@settings(max_examples=200, deadline=None)
def test_as_of_siempre_devuelve_un_prefijo(pivotes, ts):
    s = PivotStore()
    for p in pivotes:
        s.append_confirmed(p)
    completo = s.as_of(10**12)
    for t in sorted(ts):
        parcial = s.as_of(t)
        assert parcial == completo[: len(parcial)]


@given(pivotes=historial_de_pivotes())
@settings(max_examples=200, deadline=None)
def test_as_of_es_monotono_no_decreciente(pivotes):
    s = PivotStore()
    for p in pivotes:
        s.append_confirmed(p)
    marcas = sorted({p.confirmed_ts_ms for p in pivotes} | {0})
    previo = ()
    for t in marcas:
        actual = s.as_of(t)
        assert len(actual) >= len(previo)
        assert actual[: len(previo)] == previo, "el histórico confirmado solo puede CRECER"
        previo = actual


@given(pivotes=historial_de_pivotes())
@settings(max_examples=100, deadline=None)
def test_ningun_pivote_visible_antes_de_confirmarse(pivotes):
    s = PivotStore()
    for p in pivotes:
        s.append_confirmed(p)
    for p in pivotes:
        visibles = s.as_of(p.confirmed_ts_ms - 1)
        assert p not in visibles, (
            f"pivote localizado en ts={p.ts_ms} visible antes de su confirmación "
            f"en {p.confirmed_ts_ms}: eso es exactamente el repintado"
        )
        assert p in s.as_of(p.confirmed_ts_ms)
