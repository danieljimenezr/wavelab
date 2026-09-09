"""Bar store: Parquet partitioned by month, with IDEMPOTENT ingestion.

Why idempotency is not optional. Three paths write the same bars: the monthly archive during
hydration, the REST backfill after every reconnect, and the live WebSocket. Binance's WebSocket
disconnects after 24 h **by design**, and on each reconnect the REST backfill has to run again over
minutes that are already stored. Without a declared key, a plain append duplicates rows — and a
duplicated 1m bar **doubles the resampled volume, distorts the ATR and manufactures pivots**. All
of that renders without a single error.

Why Parquet and not DuckDB as the live store: DuckDB is single-writer and its calls are blocking C.
A batch write or a CHECKPOINT running on the event loop's thread stalls the WebSocket consumer;
past the 20 s pong deadline Binance disconnects, and the cascading reconnects escalate into a 418
lasting up to 3 days. DuckDB is opened PER JOB for offline analysis.

Why only 1m is stored: multi-timeframe counts have to derive from one identical source series, or
the cross-timeframe agreement logic measures alignment noise instead of structure.
"""

from __future__ import annotations

import os
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
    """What happened during an ingest. `duplicates` is counted and logged: if it grows without
    stopping, some write path is wrong and it needs looking at before it poisons the ATR."""

    written: int
    duplicates: int
    months_touched: tuple[str, ...]

    def __str__(self) -> str:
        return (f"{self.written:,} new bars, {self.duplicates:,} duplicates ignored, "
                f"{len(self.months_touched)} month(s)")


class BarStore:
    """1m bars under ``<root>/<symbol>/<YYYY-MM>.parquet``.

    One file per month: an interrupted hydration leaves whole months and missing months, never a
    half-written file; and reprocessing one particular month does not rewrite nine years.
    """

    def __init__(self, root: Path | str, tf: Timeframe = TF_1M) -> None:
        if tf is not TF_1M:
            raise ValueError(
                "BarStore only stores 1m. Every other timeframe is resampled locally from this one "
                "source series; asking the API for them separately misaligns the boundaries."
            )
        self.root = Path(root)
        self.tf = tf

    # ------------------------------------------------------------------ paths

    def _dir(self, symbol: str) -> Path:
        return self.root / symbol.upper()

    def _path(self, symbol: str, month_key: str) -> Path:
        return self._dir(symbol) / f"{month_key}.parquet"

    def months(self, symbol: str) -> list[str]:
        d = self._dir(symbol)
        return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []

    #: pandas indexes in nanoseconds, so its range ends in 2262. A sentinel like 10**14 ms
    #: (the year 5138) blows up with OutOfBoundsDatetime instead of meaning "everything". It is
    #: clamped here, at the boundary, so that no caller has to know the detail.
    _MAX_MS = 4_102_444_800_000   # 2100-01-01
    _MIN_MS = 1_262_304_000_000   # 2010-01-01

    @classmethod
    def _clamp(cls, ms: int) -> int:
        return int(min(max(int(ms), cls._MIN_MS), cls._MAX_MS))

    @staticmethod
    def _month_keys(index: np.ndarray) -> np.ndarray:
        return pd.to_datetime(index, unit="ms", utc=True).strftime("%Y-%m").to_numpy()

    # ------------------------------------------------------------------ writing

    def ingest(self, symbol: str, df: pd.DataFrame) -> IngestResult:
        """Writes new bars and **silently drops the ones already present**.

        Equivalent to ``INSERT ... ON CONFLICT DO NOTHING`` keyed on
        ``(symbol, timeframe, open_time_ms)``. Parquet has no primary keys, so the constraint is
        enforced here with an anti-join — and that is why this is the ONLY write path into the
        store that is allowed.
        """
        if df.empty:
            return IngestResult(0, 0, ())
        df = df[[c for c in _COLUMNS if c in df.columns]].sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="first")]

        bad = df.index.to_numpy() % self.tf.ms
        if (bad != 0).any():
            raise ValueError("ingest: there are open_time_ms values off the 1m grid")

        self._dir(symbol).mkdir(parents=True, exist_ok=True)
        written = dupes = 0
        touched: list[str] = []

        for key, chunk in df.groupby(self._month_keys(df.index.to_numpy())):
            path = self._path(symbol, str(key))
            if path.exists():
                existing = pd.read_parquet(path)
                fresh = chunk[~chunk.index.isin(existing.index)]
                dupes += len(chunk) - len(fresh)
                if fresh.empty:
                    continue
                merged = pd.concat([existing, fresh]).sort_index()
            else:
                fresh, merged = chunk, chunk
            # Atomic write: a .tmp file, then a rename. A Ctrl-C halfway through does not leave a
            # corrupt Parquet behind that would later read as if it were valid.
            # The temporary name carries the PID. A FIXED name is not enough: two processes
            # pointed at the same data directory — a second server started by hand next to the
            # one already running is all it takes — write into the SAME .tmp and then rename it,
            # and what lands in place is half of one file and half of the other. Observed, not
            # theoretical. This does not make concurrent writers correct (the read-merge-write
            # cycle still loses the other's rows); it only stops them corrupting the store.
            tmp = path.with_suffix(f".parquet.{os.getpid()}.tmp")
            try:
                merged.to_parquet(tmp, engine="pyarrow", compression="zstd", index=True)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
            written += len(fresh)
            touched.append(str(key))

        return IngestResult(written, dupes, tuple(sorted(set(touched))))

    # ------------------------------------------------------------------ reading

    def read(
        self, symbol: str, start_ms: int, end_ms: int, *, fill_grid: bool = True
    ) -> pd.DataFrame:
        """Reads a range. With ``fill_grid`` it reindexes onto the COMPLETE UTC minute grid.

        Reindexing is what turns an invisible gap into an explicit one: without it, twenty
        consecutive DataFrame rows can span three hours of wall clock and any window of N bars
        would be lying about which period it covers.
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

    def iter_months(self, symbol: str, start_ms: int = 0, end_ms: int | None = None):
        """Streams the history MONTH BY MONTH, without loading all of it.

        Nine years of 1m data is 4.76M rows: read in one go that is ~340 MB of DataFrame plus the
        cost of concatenating 110 Parquet files, and that kills the service against its 768 MB cap
        before it serves a single request. Month by month, the peak stays around 44,000 rows.

        It is the FILES that are paginated, which is the store's natural unit: each one is a whole
        month and is already sorted, so in the general case there is nothing to sort or trim.
        """
        end_ms = self._clamp(end_ms if end_ms is not None else self._MAX_MS)
        start_ms = self._clamp(start_ms)
        for key in self.months(symbol):
            path = self._path(symbol, key)
            df = pd.read_parquet(path)
            if df.empty:
                continue
            if int(df.index[0]) > end_ms or int(df.index[-1]) < start_ms:
                continue
            if int(df.index[0]) < start_ms or int(df.index[-1]) > end_ms:
                df = df[(df.index >= start_ms) & (df.index <= end_ms)]
            if not df.empty:
                yield key, df.sort_index()

    def coverage(self, symbol: str, start_ms: int, end_ms: int) -> tuple[int, int, float]:
        """(present, expected, fraction). The interface's data-health badge."""
        df = self.read(symbol, start_ms, end_ms, fill_grid=True)
        if df.empty:
            return 0, max(0, (end_ms - start_ms) // self.tf.ms + 1), 0.0
        total = len(df)
        present = int(total - df["is_gap"].sum())
        return present, total, present / total if total else 0.0
