"""Almacén de velas: Parquet particionado por mes, con ingesta IDEMPOTENTE.

Por qué la idempotencia no es opcional. Tres caminos escriben las mismas velas:
el archivo mensual al hidratar, el relleno REST tras cada reconexión, y el WebSocket en vivo. El
WebSocket de Binance se desconecta a las 24 h **por diseño**, y en cada reconexión hay que re-ejecutar
el relleno REST sobre minutos que ya están guardados. Sin una clave declarada, un simple append
duplica filas — y una vela de 1m duplicada **dobla el volumen resampleado, distorsiona el ATR y
fabrica pivotes**. Todo eso renderiza sin ningún error.

Por qué Parquet y no DuckDB como almacén vivo: DuckDB es de un solo escritor y sus llamadas son C
bloqueante. Una escritura por lotes o un CHECKPOINT ejecutados en el hilo del bucle de eventos paran
el consumidor del WebSocket; pasado el plazo de 20 s del pong, Binance desconecta, y las reconexiones
en cascada escalan hacia un 418 de hasta 3 días. DuckDB se abre POR TRABAJO para análisis offline.

Por qué se guarda solo 1m: los conteos multi-timeframe tienen que derivar de una única serie fuente
idéntica, o la lógica de acuerdo entre timeframes mide ruido de alineación en vez de estructura.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wavelab.core.timeframes import TF_1M, Timeframe

__all__ = ["BarStore", "IngestResult"]

_COLUMNS = ["open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Qué pasó al ingestar. `duplicates` se cuenta y se registra: si crece sin parar, algún
    camino de escritura está mal y hay que verlo antes de que envenene el ATR."""

    written: int
    duplicates: int
    months_touched: tuple[str, ...]

    def __str__(self) -> str:
        return (f"{self.written:,} velas nuevas, {self.duplicates:,} duplicadas ignoradas, "
                f"{len(self.months_touched)} mes(es)")


class BarStore:
    """Velas de 1m en ``<root>/<symbol>/<YYYY-MM>.parquet``.

    Un fichero por mes: una hidratación interrumpida deja meses completos y meses ausentes, nunca un
    fichero a medio escribir; y reprocesar un mes concreto no reescribe nueve años.
    """

    def __init__(self, root: Path | str, tf: Timeframe = TF_1M) -> None:
        if tf is not TF_1M:
            raise ValueError(
                "BarStore solo almacena 1m. Los demás timeframes se resamplean en local desde esta "
                "única serie fuente; pedirlos a la API por separado desalinea las fronteras."
            )
        self.root = Path(root)
        self.tf = tf

    # ------------------------------------------------------------------ rutas

    def _dir(self, symbol: str) -> Path:
        return self.root / symbol.upper()

    def _path(self, symbol: str, month_key: str) -> Path:
        return self._dir(symbol) / f"{month_key}.parquet"

    def months(self, symbol: str) -> list[str]:
        d = self._dir(symbol)
        return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []

    #: pandas indexa en nanosegundos, así que su rango acaba en 2262. Un centinela como 10**14 ms
    #: (año 5138) revienta con OutOfBoundsDatetime en vez de significar "todo". Se acota aquí, en
    #: la frontera, para que ningún llamante tenga que conocer el detalle.
    _MAX_MS = 4_102_444_800_000   # 2100-01-01
    _MIN_MS = 1_262_304_000_000   # 2010-01-01

    @classmethod
    def _clamp(cls, ms: int) -> int:
        return int(min(max(int(ms), cls._MIN_MS), cls._MAX_MS))

    @staticmethod
    def _month_keys(index: np.ndarray) -> np.ndarray:
        return pd.to_datetime(index, unit="ms", utc=True).strftime("%Y-%m").to_numpy()

    # ------------------------------------------------------------------ escritura

    def ingest(self, symbol: str, df: pd.DataFrame) -> IngestResult:
        """Escribe velas nuevas y **descarta silenciosamente las ya presentes**.

        Equivalente a ``INSERT ... ON CONFLICT DO NOTHING`` con clave
        ``(symbol, timeframe, open_time_ms)``. Parquet no tiene claves primarias, así que la
        restricción se aplica aquí con un anti-join — y por eso este es el ÚNICO camino de escritura
        permitido al almacén.
        """
        if df.empty:
            return IngestResult(0, 0, ())
        df = df[[c for c in _COLUMNS if c in df.columns]].sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="first")]

        bad = df.index.to_numpy() % self.tf.ms
        if (bad != 0).any():
            raise ValueError("ingest: hay open_time_ms fuera de la rejilla de 1m")

        self._dir(symbol).mkdir(parents=True, exist_ok=True)
        written = dupes = 0
        touched: list[str] = []

        for key, chunk in df.groupby(self._month_keys(df.index.to_numpy())):
            path = self._path(symbol, str(key))
            if path.exists():
                existing = pd.read_parquet(path)
                nuevas = chunk[~chunk.index.isin(existing.index)]
                dupes += len(chunk) - len(nuevas)
                if nuevas.empty:
                    continue
                merged = pd.concat([existing, nuevas]).sort_index()
            else:
                nuevas, merged = chunk, chunk
            # Escritura atómica: un fichero .tmp renombrado. Un Ctrl-C a mitad no deja un Parquet
            # corrupto que luego lea como si fuese válido.
            tmp = path.with_suffix(".parquet.tmp")
            merged.to_parquet(tmp, engine="pyarrow", compression="zstd", index=True)
            tmp.replace(path)
            written += len(nuevas)
            touched.append(str(key))

        return IngestResult(written, dupes, tuple(sorted(set(touched))))

    # ------------------------------------------------------------------ lectura

    def read(
        self, symbol: str, start_ms: int, end_ms: int, *, fill_grid: bool = True
    ) -> pd.DataFrame:
        """Lee un rango. Con ``fill_grid`` reindexa a la rejilla UTC de minutos COMPLETA.

        Reindexar es lo que convierte un hueco invisible en un hueco explícito: sin esto, veinte
        filas consecutivas del DataFrame pueden abarcar tres horas de reloj y cualquier ventana de
        N velas mentiría sobre qué periodo cubre.
        """
        start_ms, end_ms = self._clamp(start_ms), self._clamp(end_ms)
        if end_ms < start_ms:
            return pd.DataFrame(columns=_COLUMNS,
                                index=pd.Index([], name="open_time_ms", dtype="int64"))
        keys = set(self._month_keys(np.array([start_ms, end_ms], dtype=np.int64)))
        keys |= set(pd.date_range(pd.Timestamp(start_ms, unit="ms", tz="UTC"),
                                  pd.Timestamp(end_ms, unit="ms", tz="UTC"),
                                  freq="MS").strftime("%Y-%m"))
        frames = [pd.read_parquet(p) for k in sorted(keys)
                  if (p := self._path(symbol, k)).exists()]
        if not frames:
            return pd.DataFrame(columns=_COLUMNS, index=pd.Index([], name="open_time_ms", dtype="int64"))

        df = pd.concat(frames).sort_index()
        df = df[(df.index >= start_ms) & (df.index <= end_ms)]
        if not fill_grid or df.empty:
            return df

        grid = np.arange(df.index[0], df.index[-1] + 1, self.tf.ms, dtype=np.int64)
        out = df.reindex(pd.Index(grid, name="open_time_ms"))
        out["is_gap"] = out["close"].isna()
        return out

    def coverage(self, symbol: str, start_ms: int, end_ms: int) -> tuple[int, int, float]:
        """(presentes, esperadas, fracción). La insignia de salud de datos de la interfaz."""
        df = self.read(symbol, start_ms, end_ms, fill_grid=True)
        if df.empty:
            return 0, max(0, (end_ms - start_ms) // self.tf.ms + 1), 0.0
        total = len(df)
        present = int(total - df["is_gap"].sum())
        return present, total, present / total if total else 0.0
