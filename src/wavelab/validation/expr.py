"""Intérprete seguro de reglas de trading escritas por el usuario.

NO usa eval(). Recorre el AST de Python con lista blanca: si un nodo no está permitido, se rechaza.
Esto no es paranoia: si el validador se ofrece como servicio, un eval() de una cadena que envía un
desconocido es ejecución remota de código.

Prohibido: importaciones, acceso a atributos (`().__class__`), suscripción (`x[i]`), lambdas,
comprensiones, asignaciones, f-strings, walrus y cualquier función que no esté en la lista.

★ Y una restricción propia del dominio: `desplazar(x, n)` SOLO acepta n positivo. Desplazar hacia
el futuro es mirar al futuro, y el error más caro de este oficio no debería ser ni escribible.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import talib

__all__ = ["ExprError", "SAFE_FUNCS", "build_series", "evaluate_rule", "FUNC_DOCS", "SERIE_DOCS"]


class ExprError(ValueError):
    """La regla no es válida. El mensaje se le enseña al usuario tal cual."""


# ------------------------------------------------------------------ funciones permitidas

def _n(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sma(x, n): return talib.SMA(_n(x), int(n))
def ema(x, n): return talib.EMA(_n(x), int(n))
def rsi(x, n=14): return talib.RSI(_n(x), int(n))
def std(x, n): return talib.STDDEV(_n(x), int(n))
def maximo(x, n): return talib.MAX(_n(x), int(n))
def minimo(x, n): return talib.MIN(_n(x), int(n))
def cambio(x, n=1): return talib.ROC(_n(x), int(n))


def desplazar(x, n=1):
    """Desplaza la serie n barras hacia ATRÁS. n negativo está PROHIBIDO: sería mirar al futuro."""
    n = int(n)
    if n < 0:
        raise ExprError(
            "desplazar() no admite valores negativos. Un desplazamiento negativo trae datos del "
            "FUTURO, y es el error que hace que un backtest precioso pierda dinero en real. "
            "Si de verdad quieres mirar hacia adelante, esta herramienta no es para ti."
        )
    a = _n(x)
    if n == 0:
        return a
    out = np.full_like(a, np.nan)
    out[n:] = a[:-n]
    return out


def cruza_arriba(a, b):
    a, b = _n(a), _n(b)
    prev_a, prev_b = desplazar(a), desplazar(b)
    return ((a > b) & (prev_a <= prev_b)).astype(float)


def cruza_abajo(a, b):
    a, b = _n(a), _n(b)
    prev_a, prev_b = desplazar(a), desplazar(b)
    return ((a < b) & (prev_a >= prev_b)).astype(float)


SAFE_FUNCS = {
    "sma": sma, "ema": ema, "rsi": rsi, "std": std,
    "maximo": maximo, "minimo": minimo, "cambio": cambio,
    "desplazar": desplazar, "cruza_arriba": cruza_arriba, "cruza_abajo": cruza_abajo,
    "abs": lambda x: np.abs(_n(x)),
}

FUNC_DOCS = {
    "sma(x, n)": "media simple de n barras",
    "ema(x, n)": "media exponencial de n barras",
    "rsi(x, n)": "RSI de n barras (por defecto 14)",
    "std(x, n)": "desviación típica de n barras",
    "maximo(x, n)": "máximo de las últimas n barras",
    "minimo(x, n)": "mínimo de las últimas n barras",
    "cambio(x, n)": "variación porcentual respecto a n barras atrás",
    "desplazar(x, n)": "el valor de hace n barras (n negativo PROHIBIDO)",
    "cruza_arriba(a, b)": "1 en la barra en que a cruza por encima de b",
    "cruza_abajo(a, b)": "1 en la barra en que a cruza por debajo de b",
    "abs(x)": "valor absoluto",
}

SERIE_DOCS = {
    "cierre": "precio de cierre", "apertura": "precio de apertura",
    "maximo_": "máximo de la barra", "minimo_": "mínimo de la barra",
    "volumen": "volumen", "atr": "ATR de 14 barras",
    "rango": "máximo − mínimo de la barra", "cuerpo": "cierre − apertura",
}

_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
        ast.Div: np.divide, ast.Mod: np.mod, ast.Pow: np.power,
        ast.BitAnd: np.logical_and, ast.BitOr: np.logical_or}
_CMP = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
        ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}


class _Interprete:
    """Intérprete recursivo del AST. NO hay eval() en ninguna parte.

    Evaluar a mano en vez de compilar tiene dos ventajas que justifican el código extra:
    control total sobre qué se puede ejecutar, y poder tratar `and`/`or` ELEMENTO A ELEMENTO.
    Python evalúa `a and b` sobre la verdad global del array y lanza "the truth value of an array
    is ambiguous" — que es un error incomprensible para quien solo quería escribir una regla.
    """

    def __init__(self, entorno: dict) -> None:
        self.env = entorno

    def visit(self, n: ast.AST):
        m = getattr(self, "v_" + type(n).__name__, None)
        if m is None:
            raise ExprError(
                f"expresión no permitida: {type(n).__name__}. Solo se admiten comparaciones, "
                "operaciones aritméticas, `and`/`or`/`not` y las funciones de la lista. "
                "Nada de importaciones, atributos, índices ni lambdas.")
        return m(n)

    def v_Expression(self, n): return self.visit(n.body)

    def v_Constant(self, n):
        if not isinstance(n.value, (int, float, bool)):
            raise ExprError(f"solo se admiten números, no {type(n.value).__name__}")
        return float(n.value)

    def v_Name(self, n):
        if n.id not in self.env:
            raise ExprError(
                f"nombre desconocido: {n.id}. "
                f"Series: {', '.join(sorted(k for k in self.env if not callable(self.env[k])))}. "
                f"Funciones: {', '.join(sorted(SAFE_FUNCS))}")
        return self.env[n.id]

    def v_BinOp(self, n):
        op = _BIN.get(type(n.op))
        if op is None:
            raise ExprError(f"operador no permitido: {type(n.op).__name__}")
        with np.errstate(all="ignore"):
            return op(self.visit(n.left), self.visit(n.right))

    def v_UnaryOp(self, n):
        v = self.visit(n.operand)
        if isinstance(n.op, ast.USub): return np.negative(v)
        if isinstance(n.op, ast.UAdd): return v
        if isinstance(n.op, (ast.Not, ast.Invert)): return np.logical_not(_bool(v))
        raise ExprError(f"operador unario no permitido: {type(n.op).__name__}")

    def v_BoolOp(self, n):
        # ELEMENTO A ELEMENTO. Es la razón principal de escribir este intérprete.
        vals = [_bool(self.visit(v)) for v in n.values]
        f = np.logical_and if isinstance(n.op, ast.And) else np.logical_or
        out = vals[0]
        for v in vals[1:]:
            out = f(out, v)
        return out

    def v_Compare(self, n):
        if len(n.ops) != 1:
            raise ExprError("escribe las comparaciones de una en una: `a > b and b > c`, "
                            "no `a > b > c`")
        op = _CMP.get(type(n.ops[0]))
        if op is None:
            raise ExprError(f"comparación no permitida: {type(n.ops[0]).__name__}")
        with np.errstate(all="ignore"):
            return op(self.visit(n.left), self.visit(n.comparators[0]))

    def v_Call(self, n):
        if not isinstance(n.func, ast.Name):
            raise ExprError("solo se pueden llamar funciones por su nombre")
        f = SAFE_FUNCS.get(n.func.id)
        if f is None:
            raise ExprError(f"función desconocida: {n.func.id}(). "
                            f"Disponibles: {', '.join(sorted(SAFE_FUNCS))}")
        if n.keywords:
            raise ExprError("las funciones no admiten argumentos con nombre")
        return f(*[self.visit(a) for a in n.args])


def _bool(v) -> np.ndarray:
    a = np.asarray(v)
    return a if a.dtype == bool else np.nan_to_num(a, nan=0.0) != 0


def build_series(o, h, l, c, v) -> dict[str, np.ndarray]:
    """Las series que la regla puede nombrar. Todas causales por construcción."""
    o, h, l, c, v = (_n(x) for x in (o, h, l, c, v))
    return {
        "cierre": c, "apertura": o, "maximo_": h, "minimo_": l, "volumen": v,
        "atr": talib.ATR(h, l, c, 14),
        "rango": h - l, "cuerpo": c - o,
    }


def _check(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(n, _NODOS):
            raise ExprError(
                f"expresión no permitida: {type(n).__name__}. Solo se admiten comparaciones, "
                "operaciones aritméticas, y las funciones de la lista. Nada de importaciones, "
                "atributos, índices ni lambdas."
            )
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise ExprError("solo se pueden llamar funciones por su nombre")
            if n.func.id not in SAFE_FUNCS:
                raise ExprError(f"función desconocida: {n.func.id}(). "
                                f"Disponibles: {', '.join(sorted(SAFE_FUNCS))}")
            if n.keywords:
                raise ExprError("las funciones no admiten argumentos con nombre")


@dataclass(frozen=True, slots=True)
class RuleResult:
    signal: np.ndarray
    n_largo: int
    n_corto: int


def evaluate_rule(series: dict[str, np.ndarray], largo: str, corto: str = "") -> RuleResult:
    """Evalúa las reglas de entrada y devuelve la señal en {-1, 0, +1}.

    Si ambas se cumplen en la misma barra, gana 0 (fuera): una regla que dice a la vez compra y
    vende no es una señal, es una contradicción, y resolverla en silencio a favor de una de las
    dos ocultaría el error al usuario.
    """
    entorno = {**series, **SAFE_FUNCS}

    def _eval(src: str) -> np.ndarray:
        src = src.strip()
        if not src:
            return np.zeros_like(series["cierre"], dtype=bool)
        try:
            arbol = ast.parse(src, mode="eval")
        except SyntaxError as e:
            raise ExprError(f"error de sintaxis: {e.msg}") from None
        try:
            v = _Interprete(entorno).visit(arbol)
        except ExprError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ExprError(f"error al evaluar: {type(e).__name__}: {e}") from None
        a = np.asarray(v)
        if a.shape != series["cierre"].shape:
            raise ExprError("la regla debe producir una serie del mismo tamaño que los precios "
                            "(¿has escrito una constante en vez de una comparación?)")
        return np.nan_to_num(a, nan=0.0).astype(bool)

    l_ = _eval(largo)
    s_ = _eval(corto)
    sig = np.zeros(series["cierre"].size, dtype=np.int8)
    sig[l_ & ~s_] = 1
    sig[s_ & ~l_] = -1
    return RuleResult(sig, int((sig == 1).sum()), int((sig == -1).sum()))
