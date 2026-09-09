"""The archive's two silent-corruption traps, and the store's idempotence.

Hermetic: the ZIPs are manufactured here, reproducing Binance's TWO real formats. The test against
real data exists too, marked `net`, and is excluded by default.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

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


def _row(open_ms: int, *, trades: int = 7, volume: float = 10.0, close: float = 100.5) -> str:
    """One kline row with the open_time placed EXACTLY where the caller wants it.

    `_csv` above generates a clean grid; these tests need individual rows off the grid, out of
    order or duplicated, which is what the archive actually ships.
    """
    return (f"{open_ms},100.0,101.0,99.0,{close},{volume},{open_ms + 59_999},"
            f"1005.0,{trades},5.0,502.5,0")


def _zip_rows(rows: list[str]) -> bytes:
    return _zip(("\n".join(rows) + "\n").encode())


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

    def test_a_good_checksum_passes_in_the_format_binance_publishes(self):
        """A verifier that rejects everything is as broken as one that accepts everything, and
        only the rejecting half was covered.

        The published .CHECKSUM is one line, ``<sha256>  <filename>`` — two spaces, sha256sum's
        own format. Reading the wrong field off that line (the filename instead of the digest)
        makes every single download fail verification, and hydration — the project's one
        irreversible action, against an archive that could be geo-blocked any morning — never
        ingests a byte. The failure would look like universal corruption, which is exactly the
        wrong thing to go debugging.
        """
        raw = _zip(_csv(DEC_2024, 5, TF_1M.ms, micros=False, header=False))
        digest = hashlib.sha256(raw).hexdigest()
        verify_checksum(raw, f"{digest}  BTCUSDT-1m-2024-12.zip")

    def test_checksum_comparison_ignores_hex_case(self):
        """sha256sum writes lowercase; plenty of other tooling writes uppercase, and the digest
        is the same number either way. Comparing the raw strings would reject a perfectly good
        download the day the archive's publishing script changes."""
        raw = _zip(_csv(DEC_2024, 5, TF_1M.ms, micros=False, header=False))
        digest = hashlib.sha256(raw).hexdigest()
        verify_checksum(raw, f"{digest.upper()}  BTCUSDT-1m-2024-12.zip")

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

    def test_the_guard_is_driven_by_the_WORST_offset_not_a_typical_one(self):
        """A real file is not uniformly shifted: most rows sit on the grid and a few do not.

        The botched-conversion guard has to fire on the worst row in the file. If it looks at a
        typical offset instead — the smallest, or one averaged down by all the aligned rows — a
        file with one badly converted timestamp reports an offset of 0, sails past the guard and
        gets silently realigned. That is precisely the corruption the guard exists to stop, and
        it would arrive wearing the shape of a file that is 99% fine.
        """
        base = JAN_2025
        with pytest.raises(ValueError, match="botched conversion") as e:
            parse_klines_zip(_zip_rows([_row(base), _row(base + 60_000),
                                        _row(base + 120_000 + 45_000)]), TF_1M, "BTCUSDT")
        assert "45000" in str(e.value), (
            f"the error must name the worst offset (45000 ms), not a milder one: {e.value}")

    def test_the_recorded_offset_and_count_describe_the_whole_file(self):
        """`realigned` and `max_offset_ms` ARE the audit record of a silent index rewrite.

        They are what a human reads later to decide whether a month is trustworthy. If the count
        silently means "rows in the file" instead of "rows we moved", or the offset reports the
        mildest shift instead of the worst, the record understates the damage and the next person
        to look concludes the data was fine.
        """
        base = JAN_2025
        df = parse_klines_zip(
            _zip_rows([_row(base), _row(base + 60_000 + 5_000), _row(base + 120_000 + 20_799)]),
            TF_1M, "BTCUSDT")
        assert df.attrs["realigned"] == 2, (
            f"2 of the 3 rows were off the grid; realigned says {df.attrs['realigned']}")
        assert df.attrs["max_offset_ms"] == 20_799, (
            f"the worst offset in the file is 20799 ms, recorded as {df.attrs['max_offset_ms']}")
        assert (df.index.to_numpy() % TF_1M.ms == 0).all(), "some bars are still off the grid"

    def test_a_realignment_collision_keeps_the_bar_that_has_the_trades(self):
        """The observed case at 2017-12-04 06:00: realigning pushed a shifted EMPTY bar
        (trades=0, volume=0) onto the same minute as a real one (trades=4).

        Whichever bar survives is the bar the whole system then treats as that minute. Keeping
        the empty one throws away real trades and real volume and writes a flat, zero-volume
        candle into the store — which then depresses the ATR and fakes a volume anomaly, with no
        error anywhere. The rule is: the bar carrying the trades wins.
        """
        base = JAN_2025
        df = parse_klines_zip(_zip_rows([
            _row(base, trades=4, volume=5.0, close=111.0),           # on the grid, has trades
            _row(base + 20_799, trades=0, volume=0.0, close=222.0),  # shifted, empty
            _row(base + 60_000, trades=9, volume=1.0),
        ]), TF_1M, "BTCUSDT")
        assert len(df) == 2, f"the collision should leave one bar per minute, got {len(df)}"
        kept = df.loc[base]
        assert kept["trades"] == 4, (
            f"the empty shifted bar won the collision: trades={kept['trades']}, expected 4")
        assert kept["volume"] == 5.0, (
            f"real volume was replaced by the empty bar's: {kept['volume']}, expected 5.0")
        assert kept["close"] == 111.0, (
            f"the surviving bar is the empty one: close={kept['close']}, expected 111.0")


class TestTheAdvertisedMonthIsEnforcedFromBothEnds:
    """The month check is what catches a change in the archive's naming convention.

    Only the "rows after the month" half was covered. Both halves matter: a file whose name says
    one month and whose contents are another gets ingested under the wrong partition key, and the
    error is found nine years of history later.
    """

    FEB_2025 = 1738368000000  # 2025-02-01T00:00:00Z in ms

    def test_rows_BEFORE_the_advertised_month_raise(self):
        """The mirror image of the covered case: a file named 2025-02 carrying January bars.

        Binance republishing an old month under a new name, or an off-by-one in a URL builder,
        both land here. Unchecked, January bars get written into the February partition and the
        store's month files stop meaning what their names say — which is the one thing
        `iter_months` and `coverage` both rely on.
        """
        with pytest.raises(ValueError, match="outside the advertised month"):
            parse_klines_zip(_zip_rows([_row(JAN_2025)]),
                             TF_1M, "BTCUSDT", Market.SPOT, date(2025, 2, 1))

    def test_the_first_millisecond_of_the_month_is_INSIDE_it(self):
        """The lower bound is inclusive. If it were exclusive, every single monthly file would be
        rejected on its own first bar, because that bar is always exactly midnight on the 1st."""
        df = parse_klines_zip(_zip_rows([_row(JAN_2025), _row(JAN_2025 + TF_1M.ms)]),
                              TF_1M, "BTCUSDT", Market.SPOT, date(2025, 1, 1))
        assert len(df) == 2, "the month's own first bar was rejected as being outside the month"

    def test_the_first_millisecond_of_the_NEXT_month_is_outside_it(self):
        """The upper bound is exclusive, and the boundary is where an off-by-one actually shows
        up: exactly one bar of the next month bleeding into this month's file. One duplicated 1m
        bar is all it takes to double a resampled volume, so the boundary has to be exact, not
        approximately right."""
        with pytest.raises(ValueError, match="outside the advertised month"):
            parse_klines_zip(_zip_rows([_row(JAN_2025), _row(self.FEB_2025)]),
                             TF_1M, "BTCUSDT", Market.SPOT, date(2025, 1, 1))

    def test_out_of_month_rows_are_caught_even_when_the_file_is_not_in_time_order(self):
        """The month check compares the FIRST and LAST rows, which only means min and max if the
        output is sorted.

        Given an unsorted file, an unsorted output makes the check read two arbitrary rows: here
        a February bar sits in the middle, both end rows are in January, and the guard would wave
        the file through. So sortedness is not cosmetic — it is the precondition that makes the
        boundary check true at all, and every downstream consumer (the store's month partitioning,
        the reindexed grid, the path-dependent warmup) assumes it too.
        """
        rows = [_row(JAN_2025 + 5 * 86_400_000),
                _row(self.FEB_2025 + 2 * 86_400_000),   # out of month, hidden in the middle
                _row(JAN_2025 + 6 * 86_400_000)]
        with pytest.raises(ValueError, match="outside the advertised month"):
            parse_klines_zip(_zip_rows(rows), TF_1M, "BTCUSDT", Market.SPOT, date(2025, 1, 1))

    def test_the_output_is_sorted_whatever_order_the_file_came_in(self):
        rows = [_row(JAN_2025 + 2 * TF_1M.ms), _row(JAN_2025), _row(JAN_2025 + TF_1M.ms)]
        df = parse_klines_zip(_zip_rows(rows), TF_1M, "BTCUSDT")
        assert df.index.is_monotonic_increasing, f"parse returned an unsorted index: {df.index}"


class TestDuplicateRowsInOneFile:
    """A repeated open_time inside a single archive file."""

    def test_duplicates_are_dropped_and_COUNTED(self):
        """The store would also drop these, so the value here is the count, not the drop.

        `duplicates_dropped` is the only signal that a published file was malformed. Dropping the
        rows silently means the archive could start shipping duplicates tomorrow and nothing
        would ever say so — while the first row of each pair, which is what gets kept, quietly
        decides the price of that minute.
        """
        df = parse_klines_zip(_zip_rows([_row(JAN_2025, close=1.0),
                                         _row(JAN_2025, close=2.0),
                                         _row(JAN_2025 + TF_1M.ms, close=3.0)]),
                              TF_1M, "BTCUSDT")
        assert len(df) == 2, f"the duplicate open_time was not collapsed: {len(df)} rows"
        assert df.attrs["duplicates_dropped"] == 1, (
            "the duplicate was dropped without being recorded in attrs['duplicates_dropped']")
        assert df["close"].iloc[0] == 1.0, (
            f"the FIRST row of a duplicate pair must win, got close={df['close'].iloc[0]}")

    def test_a_clean_file_records_no_duplicates(self):
        """The flip side: the marker must not appear on healthy files, or it means nothing."""
        df = parse_klines_zip(_zip_rows([_row(JAN_2025), _row(JAN_2025 + TF_1M.ms)]),
                              TF_1M, "BTCUSDT")
        assert "duplicates_dropped" not in df.attrs, (
            "a clean file was flagged as containing duplicates")


class TestStoreBoundaries:
    """Small store contracts with real callers behind them, none of them previously covered."""

    def _df(self, n: int, start: int = JAN_2025) -> pd.DataFrame:
        i = pd.Index(np.arange(start, start + n * TF_1M.ms, TF_1M.ms, dtype="int64"),
                     name="open_time_ms")
        return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
                            index=i)

    def test_the_store_refuses_any_timeframe_but_1m(self, tmp_path):
        """Every other timeframe is resampled from this one series on purpose: if 5m bars could
        be stored alongside, the cross-timeframe agreement logic would end up comparing bars from
        two different sources and measuring alignment noise instead of structure. The constructor
        is the only place that invariant can be enforced."""
        with pytest.raises(ValueError, match="only stores 1m"):
            BarStore(tmp_path, TF_1H)

    def test_an_open_ended_read_does_not_explode_on_the_sentinel(self, tmp_path):
        """Callers say "everything up to now" with a sentinel far past today.

        pandas indexes in nanoseconds, so its timestamps stop in 2262: a sentinel like 1e14 ms is
        the year 5138 and `to_datetime` raises OutOfBoundsDatetime. That is a crash on the
        server's warmup path — `iter_months` defaults to exactly this sentinel — so the clamp is
        load-bearing, not defensive dressing.
        """
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(100))
        out = s.read("BTCUSDT", 0, 10**14)
        assert len(out) == 100, f"an open-ended read returned {len(out)} bars instead of 100"
        assert [k for k, _ in s.iter_months("BTCUSDT")] == ["2025-01"], (
            "iter_months with its default end_ms did not return the stored month")

    def test_a_single_minute_read_returns_that_minute(self, tmp_path):
        """start == end is one bar, not an empty range. The chart asks for exactly one minute
        whenever a user hovers a candle; returning nothing renders a blank tooltip over data that
        is sitting right there in the store."""
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(100))
        t = JAN_2025 + 5 * TF_1M.ms
        out = s.read("BTCUSDT", t, t)
        assert len(out) == 1, f"a start==end read returned {len(out)} bars, expected exactly 1"
        assert int(out.index[0]) == t, f"got the wrong minute back: {int(out.index[0])} != {t}"

    def test_coverage_of_an_unknown_symbol_still_reports_what_was_expected(self, tmp_path):
        """The data-health badge has to read "0 / 100", not "0 / 0".

        With an expected count of zero the badge shows a symbol with no history at all as
        indistinguishable from one that is fully covered — the empty store is the exact case
        where the badge matters most, right after a hydration that silently did nothing.
        """
        s = BarStore(tmp_path)
        present, total, frac = s.coverage("ETHUSDT", JAN_2025, JAN_2025 + 99 * TF_1M.ms)
        assert present == 0, f"an empty store reported {present} bars present"
        assert total == 100, f"expected 100 bars over that range, the badge would show {total}"
        assert frac == 0.0, f"coverage fraction for an empty store is {frac}, expected 0.0"


class TestWritesAreAtomic:
    """One file per month is only safe because the write is a temp-file-then-rename."""

    def _df(self, n: int, start: int = JAN_2025) -> pd.DataFrame:
        i = pd.Index(np.arange(start, start + n * TF_1M.ms, TF_1M.ms, dtype="int64"),
                     name="open_time_ms")
        return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
                            index=i)

    def test_a_write_that_dies_halfway_leaves_the_stored_month_intact(self, tmp_path,
                                                                     monkeypatch):
        """A hydration is hours long and gets interrupted: Ctrl-C, OOM kill, a full disk.

        Writing in place means the process dies with the month file half rewritten, and a
        truncated Parquet does not announce itself — it either fails to read on the next startup
        or reads back short, silently costing bars nobody notices are gone. Writing to a temp file
        and renaming makes the update all-or-nothing: the previous month survives untouched, and
        re-running the hydration fixes it, because ingest is idempotent.

        Simulated the way it actually happens: the writer puts bytes on disk and only then fails.
        """
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(100))
        path = tmp_path / "BTCUSDT" / "2025-01.parquet"
        before = path.read_bytes()

        def dies_after_writing(self, target, *a, **kw):
            Path(target).write_bytes(b"PAR1 half a file")
            raise OSError("No space left on device")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", dies_after_writing)
        with pytest.raises(OSError, match="No space left"):
            s.ingest("BTCUSDT", self._df(200))
        monkeypatch.undo()

        assert path.read_bytes() == before, (
            "the failed write reached the real month file: 2025-01.parquet was modified")
        assert len(pd.read_parquet(path)) == 100, (
            "the stored month no longer reads back as 100 bars after a failed write")
        # The other half of the contract: it must not leave its half-written temp behind either.
        # A failing hydration is a RETRIED hydration, and each attempt would strand another
        # month-sized fragment on a box that is, in this exact scenario, already out of disk.
        leftovers = [p.name for p in (tmp_path / "BTCUSDT").iterdir()
                     if not p.name.endswith(".parquet")]
        assert not leftovers, f"the failed write stranded its temp file: {leftovers}"

    def test_a_successful_ingest_leaves_no_temp_files_behind(self, tmp_path):
        """The temp name carries the PID so two processes cannot rename each other's half-file
        into place. That only holds if the temps are also cleaned up: left behind, they are
        unreadable junk accumulating a full month of bars each, on a box with a disk quota."""
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(2 * 44_640))
        leftovers = [p.name for p in (tmp_path / "BTCUSDT").iterdir()
                     if not p.name.endswith(".parquet")]
        assert not leftovers, f"temp files survived a successful ingest: {leftovers}"


class TestIterMonthsStaysInsideTheRange:
    """`iter_months` feeds the server's warmup, and the pivot detector is PATH DEPENDENT."""

    def _df(self, n: int, start: int = JAN_2025) -> pd.DataFrame:
        i = pd.Index(np.arange(start, start + n * TF_1M.ms, TF_1M.ms, dtype="int64"),
                     name="open_time_ms")
        return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5,
                             "close": np.arange(n, dtype="float64"), "volume": 10.0}, index=i)

    def test_it_yields_whole_months_in_order_and_loses_nothing(self, tmp_path):
        """Warmup replays these bars in the order they arrive to rebuild the live structure. A
        month yielded out of order, or dropped, changes the pivots — and the promise of the design
        is that the live state matches what a full replay produces. A wrong warmup breaks that
        silently: the chart still renders, it just shows structure that never happened."""
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(3 * 44_640))
        keys = [k for k, _ in s.iter_months("BTCUSDT")]
        assert keys == sorted(keys), f"months came out of chronological order: {keys}"
        seen = pd.concat([d for _, d in s.iter_months("BTCUSDT")])
        assert len(seen) == 3 * 44_640, (
            f"streaming the history yielded {len(seen)} bars, the store holds {3 * 44_640}")
        assert seen.index.is_monotonic_increasing, "the streamed bars are not in time order"

    def test_a_partly_overlapping_month_is_TRIMMED_to_the_requested_range(self, tmp_path):
        """The month file is the pagination unit, so a range that covers part of a month still
        reads the whole file — and every row outside the range has to be cut before it is yielded.

        Warmup passes `until_ms` precisely to stop at a point in time. Bars past it leaking
        through means the warmup consumes bars from the future relative to the state it is
        rebuilding, which is the causality violation the whole engine is built to prevent.
        """
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(3 * 44_640))
        lo, hi = JAN_2025 + 10 * TF_1M.ms, JAN_2025 + 19 * TF_1M.ms
        out = [(k, d) for k, d in s.iter_months("BTCUSDT", lo, hi)]
        assert [k for k, _ in out] == ["2025-01"], (
            f"months outside the requested range were yielded: {[k for k, _ in out]}")
        df = out[0][1]
        assert len(df) == 10, f"the month was not trimmed to the range: {len(df)} bars, expected 10"
        assert int(df.index[0]) == lo and int(df.index[-1]) == hi, (
            f"bars outside [{lo}, {hi}] leaked through: "
            f"{int(df.index[0])}..{int(df.index[-1])}")
