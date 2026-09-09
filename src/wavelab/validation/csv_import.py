"""Imports the user's strategy from a CSV and aligns it with our bars.

Philosophy: be VERY tolerant about the input format and VERY explicit about what was understood.
Nobody keeps their signals in whatever format happens to suit us, and turning a file away over the
name of a column is the dumbest possible way to lose a user. But guessing in silence is worse: if
we read "1" as long when the user meant "trade number 1", we would produce a beautiful report
about a strategy that does not exist.

That is why a `report` of everything that was detected is always returned, and the interface shows
it before validating anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["ImportError_", "ImportedStrategy", "align_to_bars", "parse_signals_csv"]


class ImportError_(ValueError):
    """The CSV could not be interpreted. The message goes straight to the user."""


# The vocabulary below is USER DATA, not code, and that is why it is the one thing here that is
# not in English. A Spanish spreadsheet says «fecha» and a Catalan one says «data»; the moment we
# accept English headers only, we stop working for exactly the people this was built for. Note
# that «data» is deliberately last among the time candidates: matching is exact-first and then by
# substring, so the unambiguous names get their turn before it does.
_TIME_COLS = ("time", "timestamp", "date", "datetime", "fecha", "ts", "open_time", "día", "dia",
              "data")
_SIGNAL_COLS = ("signal", "señal", "senal", "senyal", "position", "posicion", "posición",
                "posicio", "posició", "side", "lado", "direction", "direccion", "dirección",
                "direccio", "direcció", "pos")

_LONG_WORDS = {"long", "largo", "llarg", "buy", "compra", "comprar", "l", "b", "1", "alcista",
               "up"}
_SHORT_WORDS = {"short", "corto", "curt", "sell", "venta", "venda", "vender", "vendre", "s", "-1",
                "bajista", "baixista", "down"}
_FLAT_WORDS = {"flat", "fuera", "fora", "none", "cash", "0", "neutral", "", "nan", "hold"}
_PRICE_COLS = ("close", "cierre", "tancament", "price", "precio", "preu", "adj close", "adj_close",
               "last", "último", "ultimo", "c")


@dataclass(slots=True)
class ImportedStrategy:
    ts_ms: np.ndarray
    signal: np.ndarray
    report: list[str] = field(default_factory=list)
    n_long: int = 0
    n_short: int = 0
    n_flat: int = 0
    #: The user's own prices, when they supply them. This is what turns the tool into something
    #: anyone can use: without it, it only serves people trading exactly BTC on Binance.
    price: np.ndarray | None = None

    @property
    def has_prices(self) -> bool:
        return self.price is not None


def _detect(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {c.strip().lower(): c for c in cols}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for low, original in lowered.items():
        if any(cand in low for cand in candidates):
            return original
    return None


def _to_ms(col: pd.Series, report: list[str]) -> np.ndarray:
    """Interprets the time column. Accepts epochs in s/ms and any readable date."""
    if pd.api.types.is_numeric_dtype(col):
        v = col.astype("int64").to_numpy()
        med = float(np.median(np.abs(v)))
        if med > 1e17:
            report.append("time read as an epoch in NANOseconds")
            return v // 1_000_000
        if med > 1e14:
            report.append("time read as an epoch in MICROseconds")
            return v // 1000
        if med > 1e11:
            report.append("time read as an epoch in milliseconds")
            return v
        if med > 1e8:
            report.append("time read as an epoch in SECONDS")
            return v * 1000
        raise ImportError_(
            f"the time column holds values around {med:.0f}, which look like neither a timestamp "
            "nor a date. Is this the right column?")
    try:
        dt = pd.to_datetime(col, utc=True, format="mixed")
    except Exception as e:  # noqa: BLE001
        raise ImportError_(f"the dates could not be interpreted: {e}") from None
    report.append(f"dates read as text (e.g. «{col.iloc[0]}»)")
    return (dt.astype("int64") // 1_000_000).to_numpy()


def _to_signal(col: pd.Series, report: list[str]) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(col):
        v = col.fillna(0).to_numpy(dtype=float)
        distinct = np.unique(v[~np.isnan(v)])
        if len(distinct) > 12 or np.abs(v).max() > 1.0001:
            # These are not -1/0/1: read them as position size and normalise to the sign, saying
            # so, because changing the user's magnitude without a word would misstate their idea.
            report.append(f"the signal column has {len(distinct)} distinct values "
                          f"(max {np.abs(v).max():.4g}): only the SIGN of the position is used")
            return np.sign(v).astype(np.int8)
        report.append(f"numeric signal with values {sorted(distinct.tolist())[:6]}")
        return np.sign(v).astype(np.int8)

    txt = col.fillna("").astype(str).str.strip().str.lower()
    out = np.zeros(len(txt), dtype=np.int8)
    unknown: set[str] = set()
    for i, t in enumerate(txt):
        if t in _LONG_WORDS:
            out[i] = 1
        elif t in _SHORT_WORDS:
            out[i] = -1
        elif t not in _FLAT_WORDS:
            unknown.add(t)
    if unknown:
        raise ImportError_(
            f"I don't understand these signal values: {sorted(unknown)[:8]}. "
            f"Use numbers (-1/0/1) or words: {sorted(_LONG_WORDS)[:5]} / "
            f"{sorted(_SHORT_WORDS)[:5]} / {sorted(_FLAT_WORDS)[:4]}")
    report.append("signal read from text (long/short/flat)")
    return out


def parse_signals_csv(content: bytes | str, time_col: str | None = None,
                      signal_col: str | None = None,
                      price_col: str | None = None) -> ImportedStrategy:
    import io
    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as e:  # noqa: BLE001
        raise ImportError_(f"could not read the CSV: {e}") from None
    if df.empty:
        raise ImportError_("the file has no rows")

    report = [(f"{len(df):,} rows and {len(df.columns)} columns: "
               f"{', '.join(map(str, df.columns[:8]))}")]
    ct = time_col or _detect(list(df.columns), _TIME_COLS)
    cs = signal_col or _detect(list(df.columns), _SIGNAL_COLS)
    if ct is None:
        raise ImportError_(
            f"I can't find the time column. Columns: {list(df.columns)}. "
            f"It should be called something like: {', '.join(_TIME_COLS[:6])}")
    if cs is None:
        raise ImportError_(
            f"I can't find the signal column. Columns: {list(df.columns)}. "
            f"It should be called something like: {', '.join(_SIGNAL_COLS[:6])}")
    report.append(f"time column: «{ct}» · signal column: «{cs}»")

    ts = _to_ms(df[ct], report)
    sig = _to_signal(df[cs], report)
    order = np.argsort(ts, kind="stable")
    ts, sig = ts[order], sig[order]
    if len(np.unique(ts)) != len(ts):
        _, keep = np.unique(ts[::-1], return_index=True)
        keep = len(ts) - 1 - keep
        report.append(f"{len(ts)-len(keep)} duplicate timestamps: the last one is kept")
        ts, sig = ts[np.sort(keep)], sig[np.sort(keep)]

    # --- the user's own prices, if the file carries them ---------------------------------------
    price = None
    cp = price_col or _detect(list(df.columns), _PRICE_COLS)
    if cp is not None and cp not in (ct, cs):
        try:
            pr = pd.to_numeric(df[cp], errors="coerce").to_numpy(dtype=float)[order]
        except Exception:  # noqa: BLE001
            pr = None
        if pr is not None and np.isfinite(pr).sum() >= len(pr) * 0.9 and np.nanmin(pr) > 0:
            price = pr[np.sort(keep)] if "keep" in dir() else pr
            if len(price) != len(ts):
                price = pr[: len(ts)] if len(pr) >= len(ts) else None
            if price is not None:
                report.append(f"price column: «{cp}» — YOUR price series will be used, "
                              f"not ours ({np.nanmin(price):,.4g} to {np.nanmax(price):,.4g})")
        elif pr is not None:
            report.append(f"column «{cp}» discarded as a price: it has invalid or non-positive "
                          "values")

    return ImportedStrategy(ts, sig, report,
                            int((sig == 1).sum()), int((sig == -1).sum()), int((sig == 0).sum()),
                            price)


def align_to_bars(imp: ImportedStrategy, bar_ts: np.ndarray, tf_ms: int) -> np.ndarray:
    """Aligns the user's signal onto our grid of bars.

    Semantics: a position PERSISTS until the user changes it. That is what nearly everyone means
    by "on the 3rd I was long", and it is what a `ffill` does. The alternative — a position only
    on the bars actually listed — would turn a hold-for-weeks strategy into a one-day one and give
    an absurd result without anybody noticing.
    """
    grid = (imp.ts_ms // tf_ms) * tf_ms
    s = pd.Series(imp.signal, index=grid).groupby(level=0).last()
    aligned = s.reindex(pd.Index(bar_ts)).ffill().fillna(0).to_numpy()
    return aligned.astype(np.int8)
