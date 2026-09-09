"""The archive's two silent-corruption traps, and the store's idempotence.

Hermetic: the ZIPs are manufactured here, reproducing Binance's TWO real formats. The test against
real data exists too, marked `net`, and is excluded by default.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from wavelab.core.timeframes import TF_1H, TF_1M
from wavelab.feeds.base import Market
from wavelab.feeds.binance_archive import (
    MicrosecondBoundaryError,
    monthly_url,
    parse_klines_zip,
    verify_checksum,
)
from wavelab.store.bars import BarStore

JAN_2025 = 1735689600000   # 2025-01-01T00:00:00Z in ms
DEC_2024 = 1733011200000   # 2024-12-01T00:00:00Z in ms


def _csv(start_ms: int, n: int, tf_ms: int, *, micros: bool, header: bool) -> bytes:
    mul = 1000 if micros else 1
    rows = []
    if header:
        # Binance's own header, verbatim and in order: the parser keys off these names, so this
        # is one constant line, not a list to be joined at runtime.
        rows.append("open_time,open,high,low,close,volume,close_time,"
                    "quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore")
    for i in range(n):
        ot = (start_ms + i * tf_ms) * mul
        ct = ot + tf_ms * mul - (1 * mul)
        rows.append(f"{ot},100.0,101.0,99.0,100.5,10.0,{ct},1005.0,42,5.0,502.5,0")
    return ("\n".join(rows) + "\n").encode()


def _zip(csv: bytes, name: str = "x.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, csv)
    return buf.getvalue()


class TestMicrosecondTrap:
    """SPOT kline CSVs switched to microseconds on 2025-01-01. The UM futures ones and the whole
    REST API are still in milliseconds. Parsing everything as ms sends 2025-01-01 to the year
    56971 without raising."""

    def test_milliseconds_are_read_as_they_come(self):
        df = parse_klines_zip(_zip(_csv(DEC_2024, 60, TF_1M.ms, micros=False, header=False)),
                              TF_1M, "BTCUSDT", Market.SPOT, date(2024, 12, 1))
        assert len(df) == 60
        assert int(df.index[0]) == DEC_2024

    def test_microseconds_are_converted(self):
        df = parse_klines_zip(_zip(_csv(JAN_2025, 60, TF_1M.ms, micros=True, header=False)),
                              TF_1M, "BTCUSDT", Market.SPOT, date(2025, 1, 1))
        assert int(df.index[0]) == JAN_2025, "µs were not converted to ms"
        assert 2020 < pd.Timestamp(int(df.index[0]), unit="ms").year < 2040

    def test_the_format_change_boundary_is_continuous(self):
        """The literal M1 test: 2024-12-31 -> 2025-01-01 with no jump."""
        dec = parse_klines_zip(
            _zip(_csv(DEC_2024, 44_640, TF_1M.ms, micros=False, header=False)),
            TF_1M, "BTCUSDT", Market.SPOT, date(2024, 12, 1))
        jan = parse_klines_zip(
            _zip(_csv(JAN_2025, 60, TF_1M.ms, micros=True, header=False)),
            TF_1M, "BTCUSDT", Market.SPOT, date(2025, 1, 1))
        assert int(jan.index[0]) - int(dec.index[-1]) == TF_1M.ms

    def test_a_mixed_file_raises_instead_of_guessing(self):
        a = _csv(DEC_2024, 10, TF_1M.ms, micros=False, header=False)
        b = _csv(JAN_2025, 10, TF_1M.ms, micros=True, header=False)
        with pytest.raises(MicrosecondBoundaryError, match="mixes"):
            parse_klines_zip(_zip(a + b), TF_1M, "BTCUSDT", Market.SPOT)


class TestHeaderTrap:
    """Futures CSVs carry a header and spot ones do not."""

    def test_futures_with_header(self):
        df = parse_klines_zip(_zip(_csv(JAN_2025, 24, TF_1H.ms, micros=False, header=True)),
                              TF_1H, "BTCUSDT", Market.FUTURES_UM, date(2025, 1, 1))
        assert len(df) == 24
        assert df["open"].dtype == np.float64, "the header slipped through as data"

    def test_spot_without_header(self):
        df = parse_klines_zip(_zip(_csv(DEC_2024, 24, TF_1H.ms, micros=False, header=False)),
                              TF_1H, "BTCUSDT", Market.SPOT, date(2024, 12, 1))
        assert len(df) == 24


class TestGuards:
    def test_a_corrupt_checksum_raises(self):
        raw = _zip(_csv(DEC_2024, 5, TF_1M.ms, micros=False, header=False))
        with pytest.raises(ValueError, match="checksum"):
            verify_checksum(raw, "0" * 64 + "  file.zip")

    def test_rows_outside_the_advertised_month_raise(self):
        with pytest.raises(ValueError, match="outside the advertised month"):
            parse_klines_zip(_zip(_csv(JAN_2025, 60, TF_1M.ms, micros=False, header=False)),
                             TF_1M, "BTCUSDT", Market.SPOT, date(2024, 12, 1))

    def test_correct_url(self):
        assert monthly_url("btcusdt", TF_1M, date(2025, 1, 1)).endswith(
            "/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-01.zip")
        assert "/futures/um/" in monthly_url("BTCUSDT", TF_1H, date(2025, 1, 1), Market.FUTURES_UM)


class TestIdempotentIngest:
    """Three paths write the same bars: archive, REST backfill and WebSocket. One duplicated 1m
    bar doubles the resampled volume, distorts the ATR and manufactures pivots, with no error."""

    def _df(self, n: int, start: int = JAN_2025) -> pd.DataFrame:
        i = pd.Index(np.arange(start, start + n * TF_1M.ms, TF_1M.ms, dtype="int64"),
                     name="open_time_ms")
        return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
                            index=i)

    def test_reingesting_does_not_duplicate(self, tmp_path):
        s = BarStore(tmp_path)
        assert s.ingest("BTCUSDT", self._df(100)).written == 100
        r = s.ingest("BTCUSDT", self._df(100))
        assert r.written == 0 and r.duplicates == 100
        assert len(s.read("BTCUSDT", JAN_2025, JAN_2025 + 200 * TF_1M.ms)) == 100

    def test_an_overlap_writes_only_what_is_new(self, tmp_path):
        """The real case: reconnecting after 24 h and backfilling over minutes already stored."""
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(100))
        r = s.ingest("BTCUSDT", self._df(100, JAN_2025 + 50 * TF_1M.ms))
        assert (r.written, r.duplicates) == (50, 50)

    def test_partitions_by_month(self, tmp_path):
        # 89,280 minutes from 1 Jan are 62 days: January (31) + February (28) + 3 of March.
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(2 * 44_640))
        assert s.months("BTCUSDT") == ["2025-01", "2025-02", "2025-03"]
        assert (tmp_path / "BTCUSDT" / "2025-02.parquet").exists()

    def test_reading_makes_the_gap_explicit(self, tmp_path):
        s = BarStore(tmp_path)
        df = self._df(100)
        s.ingest("BTCUSDT", df.drop(df.index[40:60]))
        out = s.read("BTCUSDT", JAN_2025, JAN_2025 + 99 * TF_1M.ms)
        assert len(out) == 100, "the grid must be complete even when bars are missing"
        assert int(out["is_gap"].sum()) == 20
        present, total, frac = s.coverage("BTCUSDT", JAN_2025, JAN_2025 + 99 * TF_1M.ms)
        assert (present, total) == (80, 100) and frac == 0.8

    def test_rejects_off_grid_bars(self, tmp_path):
        s = BarStore(tmp_path)
        df = self._df(10)
        df.index = pd.Index(df.index.to_numpy() + 1, name="open_time_ms")
        with pytest.raises(ValueError, match="grid"):
            s.ingest("BTCUSDT", df)


@pytest.mark.net
def test_real_data_on_both_sides_of_the_format_change():
    """Against the real archive. `pytest -m net` to run it."""
    import httpx

    from wavelab.feeds.binance_archive import verify_checksum as vc

    with httpx.Client(timeout=60.0, follow_redirects=True) as c:
        for month, n in ((date(2024, 12, 1), 44_640), (date(2025, 1, 1), 44_640)):
            url = monthly_url("BTCUSDT", TF_1M, month)
            raw = c.get(url).raise_for_status().content
            vc(raw, c.get(url + ".CHECKSUM").raise_for_status().text)
            df = parse_klines_zip(raw, TF_1M, "BTCUSDT", Market.SPOT, month)
            assert len(df) == n
            assert 2024 <= pd.Timestamp(int(df.index[0]), unit="ms").year <= 2025


class TestTheContentIsActuallyThere:
    """Regression for a bug that walked straight past five tests.

    Building a DataFrame from Series with a different `index=` makes pandas ALIGN instead of
    assign: the result has the right index, the right length and the right dtype (NaN is float64)
    and EVERY column full of NaN. The tests that checked shape passed. These check content.
    """

    def _df(self):
        return parse_klines_zip(_zip(_csv(JAN_2025, 120, TF_1M.ms, micros=False, header=False)),
                                TF_1M, "BTCUSDT", Market.SPOT, date(2025, 1, 1))

    def test_no_column_is_all_nan(self):
        df = self._df()
        nan = df.columns[df.isna().all()].tolist()
        assert not nan, f"whole columns are NaN: {nan}"
        assert not df.isna().any().any(), "there are stray NaNs"

    def test_the_values_are_the_ones_from_the_csv(self):
        df = self._df()
        assert df["open"].iloc[0] == 100.0
        assert df["high"].iloc[0] == 101.0
        assert df["low"].iloc[0] == 99.0
        assert df["close"].iloc[0] == 100.5
        assert df["volume"].iloc[0] == 10.0
        assert df["trades"].iloc[0] == 42

    def test_the_sums_are_not_zero(self):
        df = self._df()
        for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades"):
            assert df[c].sum() > 0, f"column {c} sums to zero: probably NaN or empty"


class TestShiftedGrid:
    """2017-12-04 06:00 → 2017-12-18 10:00: Binance's grid was shifted by 20.799 s. That is
    20,401 BTCUSDT bars carrying 144,678 BTC of real volume, not padding."""

    def test_they_are_realigned_and_counted(self):
        base = 1512367200000  # 2017-12-04 06:00:00 UTC
        rows = [f"{base + i*60000 + 20799},100.0,101.0,99.0,100.5,10.0,"
                f"{base + (i+1)*60000 + 20798},1005.0,7,5.0,502.5,0" for i in range(30)]
        raw = _zip(("\n".join(rows) + "\n").encode())
        df = parse_klines_zip(raw, TF_1M, "BTCUSDT", Market.SPOT)
        assert (df.index.to_numpy() % TF_1M.ms == 0).all(), "some bars are still off the grid"
        assert df.attrs["realigned"] == 30, "the realignment must be RECORDED, not silent"
        assert df.attrs["max_offset_ms"] == 20799
        assert df["volume"].sum() == 300.0, "real volume was lost in the realignment"

    def test_an_offset_of_half_a_timeframe_raises(self):
        """That is no longer a shifted grid, it is a botched conversion."""
        base = 1735689600000
        rows = [f"{base + i*60000 + 45000},1,1,1,1,1,{base+(i+1)*60000},1,1,1,1,0" for i in range(5)]
        with pytest.raises(ValueError, match="botched conversion"):
            parse_klines_zip(_zip(("\n".join(rows) + "\n").encode()), TF_1M, "BTCUSDT")
