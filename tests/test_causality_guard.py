"""La causalidad tiene que ser imposible de violar, no meramente improbable."""

from __future__ import annotations

import pytest

from wavelab.core.causality import AsOf, CausalityError, causal, is_causal
from wavelab.core.ring import Ring
from wavelab.core.timeframes import TF_15M
from wavelab.core.types import Bar, Pivot, PivotKind
from wavelab.waves.store import PivotStore

from .conftest import SYMBOL, make_bars


class TestAsOf:
    def test_lectura_antes_de_disponible_lanza(self):
        v = AsOf(42.0, available_at_ms=1000)
        assert v.get(1000) == 42.0
        assert v.get(5000) == 42.0
        with pytest.raises(CausalityError, match="acausal"):
            v.get(999)

    def test_known_at_no_lanza(self):
        v = AsOf("x", 1000)
        assert not v.known_at(999)
        assert v.known_at(1000)


class TestPivotStore:
    """`as_of` es el ÚNICO camino alcanzable. Todo lo demás lanza."""

    @pytest.fixture
    def store(self) -> PivotStore:
        s = PivotStore()
        s.append_confirmed(Pivot(10, 1000, 100.0, PivotKind.HIGH, 5.0).confirmed_at(20, 2000))
        s.append_confirmed(Pivot(30, 3000, 90.0, PivotKind.LOW, 5.0).confirmed_at(40, 4000))
        return s

    @pytest.mark.parametrize("op", [lambda s: s[0], lambda s: list(s), lambda s: len(s)])
    def test_accesores_no_causales_lanzan(self, store, op):
        with pytest.raises(CausalityError):
            op(store)

    def test_as_of_devuelve_solo_lo_confirmado(self, store):
        assert store.n_as_of(1999) == 0   # el extremo ya ocurrió, pero aún no estaba confirmado
        assert store.n_as_of(2000) == 1
        assert store.n_as_of(9999) == 2

    def test_as_of_es_prefijo_estricto(self, store):
        completo = store.as_of(10_000)
        for t in (0, 1500, 2000, 3500, 4000, 9999):
            parcial = store.as_of(t)
            assert parcial == completo[: len(parcial)], (
                "as_of debe devolver un PREFIJO: si no, el histórico de conteos no es append-only "
                "y un conteo ya mostrado podría desaparecer sin invalidación."
            )

    def test_reconfirmar_lanza(self):
        p = Pivot(1, 100, 10.0, PivotKind.LOW, 1.0).confirmed_at(2, 200)
        with pytest.raises(ValueError, match="escritura única"):
            p.confirmed_at(3, 300)

    def test_confirmacion_fuera_de_orden_lanza(self, store):
        tarde = Pivot(50, 5000, 80.0, PivotKind.HIGH, 5.0).confirmed_at(60, 1000)
        with pytest.raises(ValueError, match="monótono"):
            store.append_confirmed(tarde)


class TestDecoradorCausal:
    def test_exige_now_ms(self):
        with pytest.raises(TypeError, match="now_ms"):
            @causal
            def sin_reloj(x: int) -> int:
                return x

    def test_marca_la_funcion(self):
        @causal
        def f(now_ms: int) -> int:
            return now_ms
        assert is_causal(f)
        assert not is_causal(lambda: None)

    def test_rechaza_asof_del_futuro(self):
        @causal
        def f(now_ms: int, dato: AsOf[float]) -> float:
            return dato.get(now_ms)
        assert f(2000, AsOf(1.0, 1000)) == 1.0
        with pytest.raises(CausalityError, match="en el futuro"):
            f(500, AsOf(1.0, 1000))

    def test_rechaza_vela_sin_cerrar(self):
        @causal
        def f(now_ms: int, bar: Bar) -> float:
            return bar.close
        b = make_bars(1)[0]
        abierta = Bar(**{**{k: getattr(b, k) for k in
                           ("symbol", "tf", "open_time_ms", "open", "high", "low", "close", "volume")},
                        "is_closed": False})
        with pytest.raises(CausalityError, match="SIN CERRAR"):
            f(b.close_time_ms, abierta)

    def test_rechaza_vela_que_cierra_despues(self):
        @causal
        def f(now_ms: int, bar: Bar) -> float:
            return bar.close
        b = make_bars(1)[0]
        with pytest.raises(CausalityError, match="después de now_ms"):
            f(b.open_time_ms, b)

    def test_rechaza_ventana_provisional_siempre(self):
        """El canal provisional es estructuralmente incapaz de alimentar la ruta causal."""
        @causal
        def f(now_ms: int, w) -> int:
            return len(w)

        ring = Ring(SYMBOL, TF_15M, capacity=16)
        bars = make_bars(5)
        for b in bars:
            ring.append(b)
        en_curso = Bar(
            symbol=SYMBOL, tf=TF_15M, open_time_ms=bars[-1].open_time_ms + TF_15M.ms,
            open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0, is_closed=False,
        )
        ring.set_provisional(en_curso)

        # La ventana cerrada pasa...
        assert f(bars[-1].close_time_ms, ring.window(3)) == 3
        # ...y la provisional NO, ni siquiera con un now_ms muy posterior.
        with pytest.raises(CausalityError, match="PROVISIONAL"):
            f(en_curso.open_time_ms + 10**9, ring.provisional_window(3))

    def test_recorre_secuencias(self):
        @causal
        def f(now_ms: int, pivotes: tuple) -> int:
            return len(pivotes)
        futuro = Pivot(1, 100, 10.0, PivotKind.LOW, 1.0).confirmed_at(2, 9_000)
        with pytest.raises(CausalityError, match=r"pivotes\[0\]"):
            f(1000, (futuro,))
