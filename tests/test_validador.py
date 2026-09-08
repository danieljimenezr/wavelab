"""El validador es el producto, así que sus guardas son lo que más importa que no se rompa."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.validation.csv_import import ImportError_, align_to_bars, parse_signals_csv
from wavelab.validation.expr import ExprError, build_series, evaluate_rule

DIA = 86_400_000
BASE = 1_600_000_000_000 - (1_600_000_000_000 % DIA)


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(4)
    n = 800
    c = 30_000 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
    sp = np.abs(rng.standard_normal(n)) * c * 0.01
    o = np.concatenate([[c[0]], c[:-1]])
    return build_series(o, c + sp, c - sp, c, np.abs(rng.standard_normal(n)) * 100 + 10)


class TestSeguridadDelEditor:
    """Si esto se ofrece como servicio, la regla la escribe un desconocido."""

    @pytest.mark.parametrize("ataque", [
        "__import__('os').system('ls')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd')",
        "exec('x=1')",
        "eval('1+1')",
        "[x for x in range(10)]",
        "lambda x: x",
        "cierre[0]",
        "globals()",
        "cierre.__class__",
    ])
    def test_rechaza_ejecucion_de_codigo(self, series, ataque):
        with pytest.raises(ExprError):
            evaluate_rule(series, ataque)

    def test_no_existe_eval_en_el_modulo(self):
        """Regresión: el intérprete es recursivo justamente para no tener eval()."""
        from pathlib import Path

        import wavelab.validation.expr as m
        src = Path(m.__file__).read_text()
        assert "eval(compile" not in src and "exec(" not in src


class TestNoSePuedeMirarAlFuturo:
    """La restricción propia del dominio: el error más caro del oficio no debe ser escribible."""

    @pytest.mark.parametrize("n", [-1, -5, -100])
    def test_desplazamiento_negativo_prohibido(self, series, n):
        with pytest.raises(ExprError, match="FUTURO"):
            evaluate_rule(series, f"desplazar(cierre, {n}) > cierre")

    def test_desplazamiento_positivo_permitido(self, series):
        r = evaluate_rule(series, "cierre > desplazar(cierre, 5)")
        assert r.n_largo > 0

    def test_desplazar_devuelve_el_pasado(self):
        from wavelab.validation.expr import desplazar
        x = np.arange(10.0)
        d = desplazar(x, 3)
        assert np.isnan(d[:3]).all()
        assert (d[3:] == x[:-3]).all()


class TestOperadores:
    """`and`/`or` tienen que funcionar elemento a elemento: es lo que escribe cualquiera."""

    def test_and_elemento_a_elemento(self, series):
        a = evaluate_rule(series, "cierre > ema(cierre,50)").n_largo
        b = evaluate_rule(series, "rsi(cierre,14) < 70").n_largo
        amb = evaluate_rule(series, "cierre > ema(cierre,50) and rsi(cierre,14) < 70").n_largo
        assert amb <= min(a, b), "la conjunción no puede dar más barras que sus partes"

    def test_or_elemento_a_elemento(self, series):
        r = evaluate_rule(series, "rsi(cierre,14) < 30 or rsi(cierre,14) > 70")
        assert r.n_largo > 0

    def test_largo_y_corto_a_la_vez_deja_fuera(self, series):
        """Una regla que dice comprar y vender a la vez es una contradicción, no una señal.
        Resolverla en silencio a favor de una ocultaría el error al usuario."""
        r = evaluate_rule(series, "cierre > 0", "cierre > 0")
        assert r.n_largo == 0 and r.n_corto == 0

    def test_comparacion_encadenada_da_mensaje_util(self, series):
        with pytest.raises(ExprError, match="de una en una"):
            evaluate_rule(series, "1 < cierre < 2")


class TestImportacionCSV:
    @pytest.mark.parametrize("csv,n_largo", [
        ("fecha,señal\n2024-01-01,largo\n2024-02-01,fuera\n", 1),
        ("timestamp,position\n1704067200000,1\n1706745600000,0\n", 1),
        ("time,signal\n1704067200,1\n1706745600,-1\n", 1),
        ("date,side\n2024-01-01,buy\n2024-02-01,sell\n", 1),
        ("date;pos\n2024-01-01;1\n2024-02-01;-1\n", 1),
    ])
    def test_acepta_formatos_humanos(self, csv, n_largo):
        assert parse_signals_csv(csv).n_largo == n_largo

    def test_normaliza_el_tamano_de_posicion_al_signo(self):
        r = parse_signals_csv("date,position\n2024-01-01,0.5\n2024-02-01,-2.5\n2024-03-01,0\n")
        assert (r.n_largo, r.n_corto, r.n_fuera) == (1, 1, 1)
        assert any("SIGNO" in i for i in r.informe), "el cambio debe AVISARSE, no hacerse callando"

    @pytest.mark.parametrize("csv,frag", [
        ("fecha,precio\n2024-01-01,100\n", "columna de señal"),
        ("fecha,señal\n2024-01-01,quizá\n", "no entiendo"),
        ("fecha,señal\n5,1\n7,-1\n", "no parecen"),
        ("", "no se pudo leer"),
    ])
    def test_explica_bien_los_errores(self, csv, frag):
        with pytest.raises(ImportError_, match=frag):
            parse_signals_csv(csv)

    def test_la_posicion_persiste_hasta_que_cambia(self):
        """Es lo que la gente quiere decir con «el día 3 estaba largo». Interpretarlo como
        «solo ese día» convertiría una estrategia de tenencia en una de un día."""
        r = parse_signals_csv("fecha,señal\n2020-09-13,largo\n2020-09-16,fuera\n")
        bars = np.array([BASE + i * DIA for i in range(6)])
        assert align_to_bars(r, bars, DIA).tolist() == [1, 1, 1, 0, 0, 0]

    def test_ordena_y_deduplica(self):
        r = parse_signals_csv("fecha,señal\n2024-03-01,corto\n2024-01-01,largo\n2024-01-01,fuera\n")
        assert r.ts_ms[0] < r.ts_ms[-1]
        assert len(r.ts_ms) == 2, "la marca duplicada debe colapsar"


class TestBateria:
    def test_caza_una_senal_que_mira_al_futuro(self):
        """La prueba de retraso sobre una señal con lookahead deliberado."""
        from wavelab.validation.battery import run_battery
        rng = np.random.default_rng(1)
        n = 1200
        c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
        ts = np.array([BASE + i * DIA for i in range(n)])
        tramposa = np.zeros(n)
        tramposa[:-1] = (np.diff(c) > 0).astype(float)   # sabe si mañana sube
        r = run_battery(c, ts, tramposa, bar_ms=DIA, n_random=60)
        assert r.veredicto == "fuga", f"no cazó el lookahead: {r.veredicto}"
        assert not next(t for t in r.tests if t.id == "retraso").passed

    def test_una_senal_aleatoria_no_sobrevive(self):
        from wavelab.validation.battery import run_battery
        rng = np.random.default_rng(2)
        n = 1200
        c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
        ts = np.array([BASE + i * DIA for i in range(n)])
        sig = (rng.random(n) < 0.3).astype(float)
        r = run_battery(c, ts, sig, bar_ms=DIA, n_random=60)
        assert r.veredicto != "sobrevive"
