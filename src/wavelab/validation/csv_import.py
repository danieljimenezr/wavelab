"""Importa la estrategia del usuario desde un CSV y la alinea con nuestras velas.

Filosofía: ser MUY tolerante con el formato de entrada y MUY explícito sobre lo que se ha
entendido. Nadie tiene sus señales en el formato que a nosotros nos convenga, y rechazar un fichero
por el nombre de una columna es la forma más tonta de perder un usuario. Pero adivinar en silencio
es peor: si interpretamos "1" como largo cuando el usuario quería decir "operación número 1",
produciríamos un informe precioso sobre una estrategia que no existe.

Por eso siempre se devuelve un `informe` con lo que se detectó, y la interfaz lo enseña antes de
validar nada.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["ImportError_", "ImportedStrategy", "parse_signals_csv", "align_to_bars"]


class ImportError_(ValueError):
    """El CSV no se pudo interpretar. El mensaje va directo al usuario."""


_COL_TIEMPO = ("time", "timestamp", "date", "datetime", "fecha", "ts", "open_time", "día", "dia")
_COL_SENAL = ("signal", "señal", "senal", "position", "posicion", "posición", "side", "lado",
              "direction", "direccion", "dirección", "pos")

_TEXTO_LARGO = {"long", "largo", "buy", "compra", "comprar", "l", "b", "1", "alcista", "up"}
_TEXTO_CORTO = {"short", "corto", "sell", "venta", "vender", "s", "-1", "bajista", "down"}
_TEXTO_FUERA = {"flat", "fuera", "none", "cash", "0", "neutral", "", "nan", "hold"}


@dataclass(slots=True)
class ImportedStrategy:
    ts_ms: np.ndarray
    signal: np.ndarray
    informe: list[str] = field(default_factory=list)
    n_largo: int = 0
    n_corto: int = 0
    n_fuera: int = 0


def _detectar(cols: list[str], candidatos: tuple[str, ...]) -> str | None:
    bajas = {c.strip().lower(): c for c in cols}
    for cand in candidatos:
        if cand in bajas:
            return bajas[cand]
    for baja, orig in bajas.items():
        if any(cand in baja for cand in candidatos):
            return orig
    return None


def _a_ms(serie: pd.Series, informe: list[str]) -> np.ndarray:
    """Interpreta la columna de tiempo. Acepta epoch en s/ms y cualquier fecha legible."""
    if pd.api.types.is_numeric_dtype(serie):
        v = serie.astype("int64").to_numpy()
        med = float(np.median(np.abs(v)))
        if med > 1e17:
            informe.append("tiempo interpretado como epoch en NANOsegundos")
            return v // 1_000_000
        if med > 1e14:
            informe.append("tiempo interpretado como epoch en MICROsegundos")
            return v // 1000
        if med > 1e11:
            informe.append("tiempo interpretado como epoch en milisegundos")
            return v
        if med > 1e8:
            informe.append("tiempo interpretado como epoch en SEGUNDOS")
            return v * 1000
        raise ImportError_(
            f"la columna de tiempo tiene valores en torno a {med:.0f}, que no parecen ni una "
            "marca de tiempo ni una fecha. ¿Es la columna correcta?")
    try:
        dt = pd.to_datetime(serie, utc=True, format="mixed")
    except Exception as e:  # noqa: BLE001
        raise ImportError_(f"no se pudieron interpretar las fechas: {e}") from None
    informe.append(f"fechas interpretadas como texto (ej. «{serie.iloc[0]}»)")
    return (dt.astype("int64") // 1_000_000).to_numpy()


def _a_senal(serie: pd.Series, informe: list[str]) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(serie):
        v = serie.fillna(0).to_numpy(dtype=float)
        distintos = np.unique(v[~np.isnan(v)])
        if len(distintos) > 12 or np.abs(v).max() > 1.0001:
            # No son -1/0/1: se interpreta como tamaño de posición y se normaliza al signo,
            # avisando, porque cambiar la magnitud del usuario sin decirlo sería falsear su idea.
            informe.append(f"la columna de señal tiene {len(distintos)} valores distintos "
                           f"(máximo {np.abs(v).max():.4g}): se usa solo el SIGNO de la posición")
            return np.sign(v).astype(np.int8)
        informe.append(f"señal numérica con valores {sorted(distintos.tolist())[:6]}")
        return np.sign(v).astype(np.int8)

    txt = serie.fillna("").astype(str).str.strip().str.lower()
    out = np.zeros(len(txt), dtype=np.int8)
    desconocidos: set[str] = set()
    for i, t in enumerate(txt):
        if t in _TEXTO_LARGO:
            out[i] = 1
        elif t in _TEXTO_CORTO:
            out[i] = -1
        elif t not in _TEXTO_FUERA:
            desconocidos.add(t)
    if desconocidos:
        raise ImportError_(
            f"no entiendo estos valores de señal: {sorted(desconocidos)[:8]}. "
            f"Usa números (-1/0/1) o palabras: {sorted(_TEXTO_LARGO)[:5]} / "
            f"{sorted(_TEXTO_CORTO)[:5]} / {sorted(_TEXTO_FUERA)[:4]}")
    informe.append("señal interpretada desde texto (largo/corto/fuera)")
    return out


def parse_signals_csv(contenido: bytes | str, col_tiempo: str | None = None,
                      col_senal: str | None = None) -> ImportedStrategy:
    import io
    datos = contenido.decode("utf-8-sig", errors="replace") if isinstance(contenido, bytes) else contenido
    try:
        df = pd.read_csv(io.StringIO(datos), sep=None, engine="python")
    except Exception as e:  # noqa: BLE001
        raise ImportError_(f"no se pudo leer el CSV: {e}") from None
    if df.empty:
        raise ImportError_("el fichero no tiene filas")

    informe = [f"{len(df):,} filas y {len(df.columns)} columnas: {', '.join(map(str, df.columns[:8]))}"]
    ct = col_tiempo or _detectar(list(df.columns), _COL_TIEMPO)
    cs = col_senal or _detectar(list(df.columns), _COL_SENAL)
    if ct is None:
        raise ImportError_(
            f"no encuentro la columna de tiempo. Columnas: {list(df.columns)}. "
            f"Debería llamarse algo como: {', '.join(_COL_TIEMPO[:6])}")
    if cs is None:
        raise ImportError_(
            f"no encuentro la columna de señal. Columnas: {list(df.columns)}. "
            f"Debería llamarse algo como: {', '.join(_COL_SENAL[:6])}")
    informe.append(f"columna de tiempo: «{ct}» · columna de señal: «{cs}»")

    ts = _a_ms(df[ct], informe)
    sig = _a_senal(df[cs], informe)
    orden = np.argsort(ts, kind="stable")
    ts, sig = ts[orden], sig[orden]
    if len(np.unique(ts)) != len(ts):
        _, keep = np.unique(ts[::-1], return_index=True)
        keep = len(ts) - 1 - keep
        informe.append(f"{len(ts)-len(keep)} marcas de tiempo duplicadas: se conserva la última")
        ts, sig = ts[np.sort(keep)], sig[np.sort(keep)]

    return ImportedStrategy(ts, sig, informe,
                            int((sig == 1).sum()), int((sig == -1).sum()), int((sig == 0).sum()))


def align_to_bars(imp: ImportedStrategy, bar_ts: np.ndarray, tf_ms: int) -> np.ndarray:
    """Alinea la señal del usuario a nuestra rejilla de velas.

    Semántica: una posición PERSISTE hasta que el usuario la cambia. Es lo que casi todo el mundo
    quiere decir con "el día 3 estaba largo", y es lo que hace un `ffill`. La alternativa —posición
    solo en las barras listadas— convertiría una estrategia de tenencia en una de un solo día y
    daría un resultado absurdo sin que nadie lo notara.
    """
    grid = (imp.ts_ms // tf_ms) * tf_ms
    s = pd.Series(imp.signal, index=grid).groupby(level=0).last()
    alineada = s.reindex(pd.Index(bar_ts)).ffill().fillna(0).to_numpy()
    return alineada.astype(np.int8)
