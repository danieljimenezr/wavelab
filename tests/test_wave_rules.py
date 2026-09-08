"""Las reglas duras contra fixtures etiquetadas a mano sobre BTC real.

Un motor que no RECHAZA es tan inútil como uno que no acepta. Seis de los diez casos son negativos,
y cuatro de ellos son secuencias reales de mercado que a ojo parecen impulsos de cinco ondas
perfectos y no lo son.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wavelab.core.types import Direction
from wavelab.waves.rules import (
    ImpulseState,
    check_impulse,
    fib_projection,
    fib_retracement,
    invalidation_for,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "impulsos_btc.json").read_text())
CASOS = FIXTURES["casos"]
DIR = {"long": Direction.LONG, "short": Direction.SHORT}


@pytest.mark.parametrize("caso", CASOS, ids=[c["id"] for c in CASOS])
def test_las_fixtures_dan_el_veredicto_etiquetado(caso):
    r = check_impulse(caso["puntos"], DIR[caso["direccion"]])
    assert r.valid is caso["espera_valido"], (
        f"{caso['id']}: se esperaba valido={caso['espera_valido']} y salió {r.valid} "
        f"(rotas: {r.broken}). Nota de la etiqueta: {caso.get('notas','')[:120]}"
    )
    if not caso["espera_valido"]:
        assert set(caso["rompe"]) == set(r.broken), (
            f"{caso['id']}: se esperaba que rompiese {caso['rompe']} y rompió {list(r.broken)}"
        )
    if "estado" in caso:
        assert r.state.value == caso["estado"]
    if caso.get("espera_truncado"):
        assert r.truncated, "la quinta truncada debe MARCARSE, nunca rechazarse"


class TestNoTerminalesParciales:
    """★ El 100% de las entradas viven en impulsos INCOMPLETOS. Un motor que solo evalúa
    estructuras terminadas es un juguete de anotación."""

    P = [100.0, 130.0, 115.0, 160.0, 140.0, 155.0]

    @pytest.mark.parametrize("n,estado,arquetipo", [
        (3, ImpulseState.AT_2, "w2"),
        (4, ImpulseState.AT_3, None),
        (5, ImpulseState.AT_4, "w4"),
        (6, ImpulseState.COMPLETE, "abc"),
    ])
    def test_cada_estado_ofrece_su_arquetipo(self, n, estado, arquetipo):
        r = check_impulse(self.P[:n])
        assert r.state is estado
        assert r.archetype == arquetipo, (
            "dentro de la onda 3 no se entra, se gestiona: AT_3 no debe ofrecer arquetipo"
        )

    @pytest.mark.parametrize("n,evaluables", [
        (3, {"R1"}),
        (4, {"R1", "R2b"}),
        (5, {"R1", "R2b", "R3"}),
        (6, {"R1", "R2b", "R3", "R2"}),
    ])
    def test_solo_son_evaluables_las_reglas_que_pueden_serlo(self, n, evaluables):
        r = check_impulse(self.P[:n])
        assert {v.rule for v in r.verdicts if v.evaluable} == evaluables

    def test_r2_es_guia_mientras_falte_w5(self):
        """"La 3 nunca es la más corta" no se puede afirmar sin w5. Fingir un veredicto sobre
        estructura que aún no existe es inventarse certeza."""
        # w1=40, w3=25 (corta pero R2b y R3 se cumplen: w3 supera w1 y w4 no la invade).
        parcial = check_impulse([100.0, 140.0, 120.0, 145.0, 142.0])
        r2 = next(v for v in parcial.verdicts if v.rule == "R2")
        assert not r2.evaluable
        assert parcial.valid, (
            "en impulso@4 todas las reglas evaluables se cumplen; R2 no evaluable no invalida"
        )

        # Con w5=58, la onda 3 (25) pasa a ser la MÁS corta y R2 sí invalida.
        completo = check_impulse([100.0, 140.0, 120.0, 145.0, 142.0, 200.0])
        r2 = next(v for v in completo.verdicts if v.rule == "R2")
        assert r2.evaluable and not r2.ok
        assert not completo.valid, "con w5 presente, R2 sí invalida"

    def test_la_invalidacion_se_mueve_de_P0_a_P1(self):
        """Dentro de w3 el stop es P0 por R1; a partir de w4 pasa a P1, que es más ceñido.
        Por eso la entrada en onda 4 tiene peor R:R aunque intuitivamente parezca más segura."""
        p0, _ = invalidation_for(ImpulseState.AT_2, self.P[:3])
        p1, _ = invalidation_for(ImpulseState.AT_4, self.P[:5])
        assert p0 == self.P[0] and p1 == self.P[1]
        assert abs(self.P[4] - p1) < abs(self.P[2] - p0), "el stop de w4 debe ser más ceñido"

    def test_la_invalidacion_lleva_el_nombre_de_su_regla(self):
        for n in (3, 4, 5, 6):
            r = check_impulse(self.P[:n])
            assert r.invalidation_rule.startswith("R"), (
                "el número más grande de la tarjeta debe llevar escrita la regla que lo produce"
            )


class TestSolape:
    """La excepción de Prechter: en futuros y materias primas la onda 4 puede solapar."""

    P = [100.0, 130.0, 115.0, 160.0, 125.0, 175.0]   # w4=125 < w1=130 -> solapa

    def test_por_defecto_se_rechaza(self):
        assert not check_impulse(self.P).valid

    def test_con_allow_overlap_se_acepta(self):
        r = check_impulse(self.P, allow_overlap=True)
        assert r.valid
        assert "Prechter" in next(v for v in r.verdicts if v.rule == "R3").detail


class TestSimetria:
    """Las reglas se escriben una vez con s=+1/-1. Un bajista es un alcista reflejado."""

    @given(pts=st.lists(st.floats(50, 500, allow_nan=False), min_size=6, max_size=6, unique=True))
    @settings(max_examples=150, deadline=None)
    def test_reflejar_los_precios_da_el_mismo_veredicto(self, pts):
        a = check_impulse(pts, Direction.LONG)
        b = check_impulse([1000.0 - p for p in pts], Direction.SHORT)
        assert a.valid == b.valid
        assert set(a.broken) == set(b.broken)

    @given(f=st.floats(0.1, 50.0, allow_nan=False))
    @settings(max_examples=50, deadline=None)
    def test_escalar_no_cambia_el_veredicto(self, f):
        base = [100.0, 130.0, 115.0, 160.0, 140.0, 155.0]
        assert check_impulse(base).valid == check_impulse([p * f for p in base]).valid


class TestFibonacci:
    def test_el_retroceso_cae_entre_los_extremos(self):
        niveles = fib_retracement(100.0, 200.0, (0.382, 0.5, 0.618, 0.786))
        assert all(100.0 < n < 200.0 for n in niveles)
        assert niveles == sorted(niveles, reverse=True)

    def test_el_bolsillo_dorado_esta_donde_debe(self):
        """El bolsillo dorado 0,618-0,65 de un tramo 100->200 cae en 135-138,2, no en 161,8.
        Un retroceso se mide DESDE el final del tramo hacia atrás."""
        lo, hi = fib_retracement(100.0, 200.0, (0.65, 0.618))
        assert lo == pytest.approx(135.0)
        assert hi == pytest.approx(138.2)
        assert 130.0 < lo < hi < 140.0

    def test_las_extensiones_proyectan_hacia_delante(self):
        objetivos = fib_projection(150.0, 100.0, 200.0)
        assert objetivos[0] == pytest.approx(250.0)     # 1.000
        assert objetivos[2] == pytest.approx(311.8)     # 1.618


class TestGuardas:
    @pytest.mark.parametrize("n", [0, 1, 2, 7])
    def test_numero_de_vertices_invalido(self, n):
        with pytest.raises(ValueError):
            check_impulse([100.0 + i for i in range(n)])
