"""The archive's two silent-corruption traps, the store's idempotence, and the feeds that write it.

Hermetic: the ZIPs are manufactured here, reproducing Binance's TWO real formats; the HTTP and
WebSocket layers are driven through stub transports and scripted messages, so nothing here touches
the network. The test against real data exists too, marked `net`, and is excluded by default.

This file is the ONLY place that exercises `wavelab.feeds` and `wavelab.store`: hydration, the REST
weight governor, the reconnect gap-heal seam and the liquidation log have no other coverage, so a
fault in any of them is invisible everywhere else.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import os
import time
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from wavelab.core.timeframes import TF_1H, TF_1M
from wavelab.feeds import binance_klines, binance_ws
from wavelab.feeds.base import Market
from wavelab.feeds.binance_archive import (
    MicrosecondBoundaryError,
    monthly_url,
    parse_klines_zip,
    verify_checksum,
)
from wavelab.feeds.binance_klines import KlineFeed
from wavelab.feeds.binance_rest import (
    WEIGHT_BUDGET,
    BinanceREST,
    RateLimitCircuitOpen,
    WeightGovernor,
)
from wavelab.feeds.liquidations import LiquidationRecorder
from wavelab.store import hydrate as hydrate_mod
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

    @pytest.mark.parametrize("pos", [0, 5, 9])
    def test_ONE_bar_off_the_grid_is_enough_to_refuse_the_frame(self, tmp_path, pos):
        """The realistic corruption is one stray timestamp in 44,640, not a whole file shifted.

        That is what a partly-applied µs/ms conversion or a single realigned archive row looks
        like, and it is the case the guard has to catch: an off-grid row that gets written is
        then dropped by `read`'s reindex onto the minute grid, which deletes a real bar AND
        reports the minute it belonged to as a gap. No exception, no log line, one candle gone
        and one hole invented. A guard that only fires when EVERY row is off the grid passes a
        uniformly shifted test file and waves the real thing through.
        """
        s = BarStore(tmp_path)
        df = self._df(10)
        idx = df.index.to_numpy().copy()
        idx[pos] += 1
        df.index = pd.Index(idx, name="open_time_ms")
        with pytest.raises(ValueError, match="grid"):
            s.ingest("BTCUSDT", df)
        assert s.months("BTCUSDT") == [], (
            "a frame refused for being off the grid still wrote bars into the store")

    def test_bars_that_arrive_out_of_order_are_STORED_in_time_order(self, tmp_path):
        """A REST page arrives newest-first and a spreadsheet export is often reversed too.

        Sortedness of the month file is not cosmetic: `iter_months` decides whether a month
        overlaps the requested range by reading `index[0]` and `index[-1]`, which only mean
        min and max on a sorted file. Stored in arrival order, a whole month reads as being
        outside every range — the warmup then replays ZERO bars and the live engine starts with
        no structure at all, while the store cheerfully reports the month as present.
        """
        s = BarStore(tmp_path)
        df = self._df(100)
        s.ingest("BTCUSDT", df.iloc[::-1])
        stored = pd.read_parquet(tmp_path / "BTCUSDT" / "2025-01.parquet")
        assert stored.index.is_monotonic_increasing, (
            "the month file was written in arrival order; its index is not sorted")
        lo, hi = JAN_2025, JAN_2025 + 9 * TF_1M.ms
        assert [k for k, _ in s.iter_months("BTCUSDT", lo, hi)] == ["2025-01"], (
            "iter_months skipped a month it holds, because the stored index is unsorted")


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

    def test_a_duplicate_inside_one_frame_resolves_the_same_way_the_parser_resolves_it(
            self, tmp_path):
        """Two write paths answer the same question — which row wins when a minute arrives twice —
        and only one of them was pinned.

        `parse_klines_zip` keeps the FIRST row and has a test saying so. `ingest` keeps the first
        too, and nothing said so, which means the two could silently drift apart. That matters
        because they see the same minute: the archive publishes a month, the REST backfill re-reads
        its tail, and a minute present in both arrives once through each path. If the paths
        disagree about which copy wins, the price of that minute depends on which path happened
        to run — and the anti-join against what is already stored makes that choice permanent,
        because the loser is never revisited.

        Whichever rule is chosen is defensible; the two being written down in one place and tested
        in the other is not.
        """
        i = pd.Index([JAN_2025, JAN_2025, JAN_2025 + TF_1M.ms], name="open_time_ms")
        df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5,
                           "close": [1.0, 2.0, 3.0], "volume": 10.0}, index=i)
        s = BarStore(tmp_path)
        r = s.ingest("BTCUSDT", df)

        assert r.written == 2, f"the duplicated minute was stored twice: {r.written} rows written"
        stored = s.read("BTCUSDT", JAN_2025, JAN_2025 + TF_1M.ms)
        assert stored["close"].iloc[0] == 1.0, (
            f"the store kept close={stored['close'].iloc[0]} for a minute that arrived as 1.0 then "
            "2.0, while `parse_klines_zip` keeps the first. The two write paths disagree about "
            "the price of the same minute, and the anti-join makes the winner permanent")


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

    @pytest.mark.parametrize("offset", [TF_1M.ms // 2 - 1, TF_1M.ms // 2])
    def test_half_a_timeframe_is_the_exact_line_between_shifted_and_botched(self, offset):
        """Where the guard sits decides whether a file is repaired or refused, and the two
        outcomes are opposites: below the line the bars are real and get realigned, on or above it
        they are a unit-conversion artefact and must never be ingested.

        Tested at the boundary itself, because that is the only place the two can be told apart.
        A guard written one millisecond loose accepts an exactly-half-a-minute shift — the shape a
        half-applied µs/ms conversion actually produces — and silently rewrites every timestamp in
        the month instead of stopping the hydration.
        """
        base = JAN_2025
        rows = [_row(base + offset), _row(base + 60_000 + offset)]
        if offset >= TF_1M.ms // 2:
            with pytest.raises(ValueError, match="botched conversion"):
                parse_klines_zip(_zip_rows(rows), TF_1M, "BTCUSDT")
            return
        df = parse_klines_zip(_zip_rows(rows), TF_1M, "BTCUSDT")
        assert df.attrs["max_offset_ms"] == offset, (
            f"a {offset} ms shift is under half a minute and must be realigned and recorded, "
            f"attrs say {df.attrs.get('max_offset_ms')}")
        assert (df.index.to_numpy() % TF_1M.ms == 0).all(), "some bars are still off the grid"

    def test_a_shifted_bar_belongs_to_the_minute_that_CONTAINS_it(self):
        """Realignment may only move a timestamp BACKWARDS, onto the minute already in progress.

        Landing on the grid is not enough: rounding a 06:00:20.799 bar up to 06:01 is just as
        aligned and files a whole fortnight of real trades one bar into the future. Every ATR
        window, every resample and every pivot index in that stretch is then off by one, forever,
        with no error and no attr to give it away — the audit record would say `realigned: 20401`
        either way.
        """
        base = JAN_2025
        originals = [base + 20_799, base + 60_000 + 20_799, base + 120_000 + 20_799]
        df = parse_klines_zip(_zip_rows([_row(t) for t in originals]), TF_1M, "BTCUSDT")
        stored = [int(t) for t in df.index]
        assert stored == [t - (t % TF_1M.ms) for t in originals], (
            f"bars were not filed under the minute containing them: {stored} from {originals}")
        assert all(s <= o for s, o in zip(stored, originals, strict=True)), (
            f"realignment moved a bar FORWARD in time: {stored} from {originals}")

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

    def test_two_processes_never_share_a_temp_file(self, tmp_path, monkeypatch):
        """The rename is only atomic if the thing being renamed belongs to one writer.

        Two processes over the same data directory — a second server started by hand beside the
        one already running is all it takes — that write into the SAME temp name interleave their
        bytes and then rename the mixture into place: half of one month and half of another,
        landing as a file that looks valid. Observed, not theoretical. Cleanup on failure is
        already covered; what is not is that the name distinguishes the writers at all.
        """
        seen: list[str] = []
        real = pd.DataFrame.to_parquet

        def spy(self, target, *a, **kw):
            seen.append(Path(target).name)
            return real(self, target, *a, **kw)

        monkeypatch.setattr(pd.DataFrame, "to_parquet", spy)
        for fake_pid, start in ((4242, JAN_2025), (9797, JAN_2025 + 10 * TF_1M.ms)):
            monkeypatch.setattr(os, "getpid", lambda pid=fake_pid: pid)
            BarStore(tmp_path).ingest("BTCUSDT", self._df(10, start))

        assert len(seen) == 2, f"expected one write per ingest, got {seen}"
        # Inequality alone is NOT the property. A per-process counter, a random suffix seeded at
        # import, a monotonic serial — all of them make these two names differ inside one process
        # and collide across two, which is the only case that matters. I verified it: replacing
        # the PID with a module-level counter leaves the whole suite green, and two servers over
        # the same directory then pick the same temp name on their first write apiece. So the
        # assertion has to be that the name is a function of the PID that was patched.
        assert seen[0] != seen[1], (
            f"two processes wrote the same month through the same temp file ({seen[0]}): "
            "whichever renames last publishes a mixture of both")
        for pid, name in zip((4242, 9797), seen, strict=True):
            assert str(pid) in name, (
                f"process {pid} wrote to {name!r}, which does not carry its pid. Whatever "
                "distinguishes these two names is something both processes would agree on if "
                "they started separately, so the collision this test is named for is still open")


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

    FEB_2025 = 1738368000000  # 2025-02-01T00:00:00Z in ms

    def test_a_month_overlapping_only_ONE_end_of_the_range_is_trimmed_too(self, tmp_path):
        """The production call is `iter_months(symbol, 0, until_ms)`: every month after the first
        overlaps the range on one side only.

        A trim that needs BOTH ends to stick out therefore never fires on the case the server
        actually makes, and the month holding `until_ms` is yielded whole. The warmup then replays
        weeks of bars that had not happened yet at `until_ms` and rebuilds the pivot state — which
        is path dependent — out of the future. Nothing raises; the chart renders structure that
        never existed, which is the exact look-ahead the engine is built to prevent.
        """
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(3 * 44_640))

        end = self.FEB_2025 + 10 * TF_1M.ms       # range starts before every month, ends mid-Feb
        seen = list(s.iter_months("BTCUSDT", 0, end))
        for key, df in seen:
            assert int(df.index[-1]) <= end, (
                f"{key} yielded bars past until_ms: {int(df.index[-1])} > {end}")
        assert sum(len(d) for _, d in seen) == 44_640 + 11, (
            "the warmup was fed a different number of bars than the range contains")

        start = self.FEB_2025 + 10 * TF_1M.ms     # the mirror: range starts mid-Feb, never ends
        mirror = list(s.iter_months("BTCUSDT", start))
        # A generator that silently yields nothing satisfies the loop below without executing it
        # once, so the count comes first: two months and change, minus the ten trimmed minutes.
        assert sum(len(d) for _, d in mirror) == 2 * 44_640 - 10, (
            f"the mirror range yielded {sum(len(d) for _, d in mirror)} bars across "
            f"{len(mirror)} months; an empty or short generator would satisfy the loop below "
            "without ever running it")
        for key, df in mirror:
            assert int(df.index[0]) >= start, (
                f"{key} yielded bars from before the requested start: "
                f"{int(df.index[0])} < {start}")

    def test_the_month_that_begins_exactly_at_the_range_END_is_not_dropped(self, tmp_path):
        """`until_ms` is inclusive everywhere else in the store, and it has to be here too.

        A cutoff landing exactly on midnight of the 1st is not an edge case a caller avoids: the
        chart and the warmup both ask for whole periods. Treating that month as "past the range"
        drops its first bar — and one missing 1m bar at a month boundary is a hole the resampler
        turns into a short hour, silently.
        """
        s = BarStore(tmp_path)
        s.ingest("BTCUSDT", self._df(3 * 44_640))
        out = dict(s.iter_months("BTCUSDT", 0, self.FEB_2025))
        assert "2025-02" in out, (
            "the month starting exactly at end_ms was skipped whole; its first bar is inside "
            "an inclusive range")
        assert [int(t) for t in out["2025-02"].index] == [self.FEB_2025], (
            f"expected exactly the boundary bar from 2025-02, got {len(out['2025-02'])} bars")


# ----------------------------------------------------------------------------------------
# Hydration. The project's one irreversible action, against an archive that could be
# geo-blocked any morning — and until now not reachable from a single test.
# ----------------------------------------------------------------------------------------

def _month_zip(month: date, n: int = 3, *, content_month: date | None = None) -> bytes:
    """A well-formed monthly ZIP. `content_month` lets the bytes disagree with the URL."""
    src = content_month or month
    start = int(pd.Timestamp(src, tz="UTC").value // 1_000_000)
    return _zip(_csv(start, n, TF_1M.ms, micros=False, header=False))


def _archive_transport(files: dict[str, bytes], *, checksums: dict[str, str] | None = None):
    """Serves the manufactured archive: the ZIP, its .CHECKSUM, and 404 for everything else."""
    checksums = checksums or {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith(".CHECKSUM"):
            raw = files.get(url.removesuffix(".CHECKSUM"))
            if raw is None:
                return httpx.Response(404)
            digest = checksums.get(url.removesuffix(".CHECKSUM"), hashlib.sha256(raw).hexdigest())
            return httpx.Response(200, text=f"{digest}  x.zip")
        raw = files.get(url)
        return httpx.Response(200, content=raw) if raw is not None else httpx.Response(404)

    return httpx.MockTransport(handler)


class TestTheDownloadIsVerifiedBeforeItIsBelieved:
    """`_fetch` is the only thing standing between a truncated download and nine years of history."""

    @pytest.mark.parametrize("kind", ["all_zeros", "last_character", "first_character"])
    async def test_a_body_that_does_not_match_its_published_checksum_never_reaches_the_caller(
            self, kind):
        """A truncated ZIP frequently decompresses "fine" and yields a month with half its bars.

        Nothing downstream can tell that month from a real one — the parser sees valid rows, the
        store ingests them, and `coverage` reports the missing minutes as an ordinary Binance
        outage. Skipping the verification is therefore not a smaller safety margin, it is the
        difference between a loud failure now and a permanently short history nobody re-derives.

        The near-misses are the point of the parametrisation. Every other checksum case in this
        file serves a digest that matches exactly, and this one used to serve `"0" * 64` — maximally
        wrong, so the suite pinned that A comparison happens and not that the WHOLE digest is
        compared. I verified a prefix-only comparison (`got[:16] != expected[:16]`) survives the
        all-zeros case; a digest differing in its last character alone does not let it.
        """
        url = monthly_url("BTCUSDT", TF_1M, date(2025, 1, 1))
        raw = _month_zip(date(2025, 1, 1))
        good = hashlib.sha256(raw).hexdigest()
        bad = {
            "all_zeros": "0" * 64,
            "last_character": good[:-1] + ("f" if good[-1] != "f" else "0"),
            "first_character": ("f" if good[0] != "f" else "0") + good[1:],
        }[kind]
        assert bad != good and len(bad) == len(good), "the fixture must be a well-formed near-miss"

        transport = _archive_transport({url: raw}, checksums={url: bad})
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError, match="checksum"):
                await hydrate_mod._fetch(client, url)

    async def test_a_body_that_matches_is_returned_untouched(self):
        """The other half: a verifier that rejects everything is as broken as one that accepts
        everything, and only the rejecting half would be caught by a hydration that ingests
        nothing."""
        url = monthly_url("BTCUSDT", TF_1M, date(2025, 1, 1))
        raw = _month_zip(date(2025, 1, 1))
        async with httpx.AsyncClient(transport=_archive_transport({url: raw})) as client:
            assert await hydrate_mod._fetch(client, url) == raw

    async def test_a_month_the_archive_never_published_is_not_an_error(self):
        """A symbol listed mid-month leaves legitimate 404s in the range; a hydration that treats
        them as failures stops on the first one and never reaches the months that do exist."""
        url = monthly_url("BTCUSDT", TF_1M, date(2017, 1, 1))
        async with httpx.AsyncClient(transport=_archive_transport({})) as client:
            assert await hydrate_mod._fetch(client, url) is None


class TestHydrationWritesWhatTheURLPromised:
    """End-to-end over a stub transport: no network, real parse, real store."""

    def _run(self, tmp_path, monkeypatch, months: list[date], files: dict[str, bytes],
             *, concurrency: int = 6):
        monkeypatch.setattr(hydrate_mod, "list_available_months", lambda *a, **kw: list(months))
        transport = _archive_transport(files)
        real_client = httpx.AsyncClient

        def fake_client(*a, **kw):
            return real_client(transport=transport, follow_redirects=True)

        monkeypatch.setattr(hydrate_mod.httpx, "AsyncClient", fake_client)
        return hydrate_mod.hydrate("BTCUSDT", tmp_path, concurrency=concurrency)

    async def test_a_month_whose_contents_disagree_with_its_name_stops_the_hydration(
            self, tmp_path, monkeypatch):
        """The archive's naming convention is checked at the download that relies on it.

        `parse_klines_zip` can only enforce the advertised month if hydration tells it which month
        it asked for. Left out, a file republished under the wrong name — or an off-by-one in a
        URL builder — writes January bars into whatever partition their timestamps fall in, and
        the month files stop meaning what their names say. `iter_months` and `coverage` both read
        those names, and the mistake is found nine years of history later.
        """
        feb = date(2025, 2, 1)
        url = monthly_url("BTCUSDT", TF_1M, feb)
        files = {url: _month_zip(feb, content_month=date(2025, 1, 1))}
        with pytest.raises(ValueError, match="outside the advertised month"):
            await self._run(tmp_path, monkeypatch, [feb], files)
        assert BarStore(tmp_path / "bars").months("BTCUSDT") == [], (
            "a file whose contents disagree with its name was ingested anyway")

    async def test_every_published_month_is_downloaded_and_ingested(self, tmp_path, monkeypatch):
        """Hydration downloads in batches, and a batch is the one place a slice can quietly lose
        an element: no error, no retry, just months absent from the store. The hole then sits in
        the middle of the history, where `read` reports it as a gap indistinguishable from a real
        Binance outage — and hydration is not re-run, because it said it succeeded.
        """
        months = [date(2025, m, 1) for m in range(1, 6)]
        files = {monthly_url("BTCUSDT", TF_1M, m): _month_zip(m) for m in months}
        assert await self._run(tmp_path, monkeypatch, months, files, concurrency=2) == 0
        store = BarStore(tmp_path / "bars")
        assert store.months("BTCUSDT") == [f"{m:%Y-%m}" for m in months], (
            "months published by the archive are missing from the store after a clean hydration")

    async def test_the_newest_stored_month_is_downloaded_again(self, tmp_path, monkeypatch):
        """The most recent stored month was ingested mid-month, so it is INCOMPLETE by
        construction.

        Skipping it because its file exists leaves a permanent hole at the join between the
        archive and the live feed — the newest end of the history, which is the part every chart
        and every warmup reads. Ingest is idempotent, so re-downloading it costs one file.
        """
        jan = date(2025, 1, 1)
        store = BarStore(tmp_path / "bars")
        partial = pd.DataFrame(
            {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
            index=pd.Index([JAN_2025], name="open_time_ms"))
        store.ingest("BTCUSDT", partial)

        files = {monthly_url("BTCUSDT", TF_1M, jan): _month_zip(jan, n=3)}
        await self._run(tmp_path, monkeypatch, [jan], files)
        out = store.read("BTCUSDT", JAN_2025, JAN_2025 + 2 * TF_1M.ms)
        assert len(out) == 3 and not out["is_gap"].any(), (
            f"the incomplete newest month was not refreshed: {len(out)} bars, "
            f"{int(out['is_gap'].sum())} still missing")


def _clock_stopped_at(instant: datetime) -> type[datetime]:
    """A `datetime` stand-in stopped at one instant that still answers `now` exactly as the real
    one does: `now(UTC)` is that instant in UTC, `now()` is the same instant rendered as the
    machine's naive local civil time.

    The instant is the CONTROLLED variable — it is the only way to put two timezones at the same
    moment — never the thing asserted. No test below cares which instant it is.
    """

    class _Stopped(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return instant.astimezone().replace(tzinfo=None)
            return instant.astimezone(tz)

    return _Stopped


@contextlib.contextmanager
def _a_machine_in(tz_name: str, instant: datetime, monkeypatch):
    """Runs the body as if on a box configured for `tz_name`, with the clock stopped at `instant`.

    `time.tzset()` is the part that bites: setting `os.environ["TZ"]` changes nothing until the C
    library is told to re-read it, on macOS and on Linux both — without it every zone here would
    quietly report the developer's own. Restored on the way out, because a leaked TZ re-times every
    test that runs after this one.
    """
    before = os.environ.get("TZ")
    os.environ["TZ"] = tz_name
    time.tzset()
    try:
        monkeypatch.setattr(hydrate_mod, "datetime", _clock_stopped_at(instant))
        yield
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time.tzset()


# 23:30 UTC on the last day of a month. In Spain that is 01:30 on the 1st (CEST = UTC+2) — the run
# that produced the fault — and in Kiritimati (UTC+14) it is the afternoon of the 1st. The hour is
# inside the window where the local civil date and the UTC date name different MONTHS, which is the
# only window in which the two answers can differ at all.
A_MONTH_EDGE = datetime(2024, 3, 31, 23, 30, tzinfo=UTC)


class TestQuickAsksForTheSameTwoYearsOnEveryMachine:
    """`--quick` picks its months off a clock, and a clock has a timezone. The archive's month keys
    do not: they are UTC, always. These tests are the only thing standing between that line and the
    next person who "simplifies" `datetime.now(UTC)` to `datetime.now()`."""

    # UTC-12 through UTC+14: the full span of inhabited civil dates at any one instant.
    ZONES = ("Etc/GMT+12", "UTC", "Europe/Madrid", "Pacific/Kiritimati")

    def test_machines_in_different_timezones_agree_at_the_same_instant(self, monkeypatch):
        """Two boxes running `--quick` at the same moment must ask the archive for the same months.

        Computed from the local civil date they do not. At 23:30 UTC on the last day of a month the
        box in Spain is already on the 1st, and the `.replace(day=1)` snap turns that one hour into
        a whole month: the same command fetches 25 months there and 24 here. Nothing downstream
        corrupts — whole-month keys, idempotent ingest — which is exactly why it needs a test of its
        own. The only symptom is that two machines disagree about what the history IS, on a run that
        reported success and that nobody repeats.

        The instant is frozen so that "the same moment" means something. The assertion is across
        timezones, not across the calendar, so it says nothing about what today is.
        """
        seen = {}
        for tz in self.ZONES:
            with _a_machine_in(tz, A_MONTH_EDGE, monkeypatch):
                seen[tz] = hydrate_mod._quick_cutoff()

        assert len(set(seen.values())) == 1, (
            "the --quick cutoff depends on the machine's timezone: at one single instant "
            f"({A_MONTH_EDGE:%Y-%m-%d %H:%M} UTC) these boxes disagree about how far back to go — "
            + ", ".join(f"{tz}→{c}" for tz, c in seen.items()))

    def test_the_cutoff_still_follows_the_utc_month(self, monkeypatch):
        """The agreement above is also satisfied by a cutoff that ignores the clock altogether — a
        constant, or a snap that never moves — so on its own it would be worth nothing.

        This is the other half: one hour that carries the UTC clock into a new month must move the
        cutoff by exactly the length of the month it left, because the cutoff is snapped to the 1st.
        Together the two say the cutoff is a function of the UTC month and of nothing else, without
        pinning any particular date.
        """
        one_hour_later = A_MONTH_EDGE + timedelta(hours=1)   # 2024-04-01T00:30Z, a new UTC month
        with _a_machine_in("UTC", A_MONTH_EDGE, monkeypatch):
            before = hydrate_mod._quick_cutoff()
        with _a_machine_in("UTC", one_hour_later, monkeypatch):
            after = hydrate_mod._quick_cutoff()

        march = date(2024, 4, 1) - date(2024, 3, 1)
        assert after - before == march, (
            f"crossing into a new UTC month moved the --quick cutoff by {after - before} "
            f"({before} → {after}), not by the {march.days} days of the month it left: the cutoff "
            "is not tracking the UTC month it is supposed to be snapped to")

    async def test_the_quick_branch_uses_that_cutoff_and_not_a_clock_of_its_own(
            self, tmp_path, monkeypatch):
        """The rule is worth nothing unless the `--quick` branch is what applies it, and nothing
        else in this suite runs `hydrate(quick=True)` — a branch left computing its own local date
        would pass both tests above and still download a different history on every box.

        So: two complete hydrations over the same stub archive, at the same instant, one in UTC and
        one in Spain. 2022-04 is the month at stake — inside the cutoff the UTC clock produces,
        outside the one the Spanish civil date produces — and it has to reach the store either way.
        """
        months = [date(2022, 3, 1), date(2022, 4, 1), date(2022, 5, 1)]
        at_stake = date(2022, 4, 1)
        files = {monthly_url("BTCUSDT", TF_1M, at_stake): _month_zip(at_stake)}
        # Captured before the loop: `hydrate_mod.httpx` IS the httpx module, so the first patch
        # would otherwise make the second iteration wrap the stub in another stub.
        real_client = httpx.AsyncClient
        transport = _archive_transport(files)

        stored = {}
        for tz in ("UTC", "Europe/Madrid"):
            root = tmp_path / tz.replace("/", "_")
            with _a_machine_in(tz, A_MONTH_EDGE, monkeypatch):
                monkeypatch.setattr(
                    hydrate_mod, "list_available_months", lambda *a, **kw: list(months))
                monkeypatch.setattr(
                    hydrate_mod.httpx, "AsyncClient",
                    lambda *a, **kw: real_client(transport=transport, follow_redirects=True))
                assert await hydrate_mod.hydrate("BTCUSDT", root, quick=True) == 0
            stored[tz] = BarStore(root / "bars").months("BTCUSDT")

        assert stored["UTC"] == stored["Europe/Madrid"], (
            "the same --quick hydration, at the same instant, stored different months depending on "
            f"the box's timezone: UTC got {stored['UTC']}, Europe/Madrid got "
            f"{stored['Europe/Madrid']}")
        assert f"{at_stake:%Y-%m}" in stored["UTC"], (
            f"{at_stake:%Y-%m} is inside the two years --quick asks for and never reached the "
            f"store ({stored['UTC']}); the two runs agree only because both are short")


class TestTheWeightGovernor:
    """6000 weight per minute per IP. Going over gives a 429 and then a 418 — an IP ban lasting
    from two minutes to three days, which for an application living off a public feed is total
    downtime. The brake is the single number this class exists to enforce, and it had no test."""

    @pytest.fixture
    def slept(self, monkeypatch):
        calls: list[float] = []

        async def fake_sleep(s):
            calls.append(s)

        monkeypatch.setattr("wavelab.feeds.binance_rest.asyncio.sleep", fake_sleep)
        return calls

    @pytest.mark.parametrize("ratio", [0.5, 0.7])
    async def test_it_brakes_at_the_configured_ratio_and_not_at_the_edge(self, slept, ratio):
        """Braking at the budget instead of before it is not a smaller margin: the used-weight
        header arrives with the response, so by the time the counter reads 6000 the request that
        took it there is already gone. The whole point of the ratio is to stop while the answer
        can still change the outcome."""
        brake = int(WEIGHT_BUDGET * ratio)
        g = WeightGovernor(WEIGHT_BUDGET, ratio)
        g.used = brake - 1
        await g.wait_if_needed()
        assert not slept, f"the governor waited at {g.used} of {WEIGHT_BUDGET}, below its brake"

        g.used = brake
        await g.wait_if_needed()
        assert slept, (
            f"the governor did not brake at {brake} of {WEIGHT_BUDGET} ({ratio:.0%}); it is "
            "running to the edge of the budget, where the next request earns a 418")
        assert g.used == 0, "the used-weight counter was not reset after waiting out the minute"

    async def test_a_ban_is_raised_and_never_slept_off(self, slept):
        """A 418 is a switch to the fallback feed, not a retry. Sleeping and trying again extends
        the ban, which is how two minutes becomes three days."""
        g = WeightGovernor()
        g.banned_until = time.monotonic() + 60
        with pytest.raises(RateLimitCircuitOpen, match="banned"):
            await g.wait_if_needed()
        assert not slept, "the client waited out an IP ban instead of failing over"


class TestTheRestKlineMapping:
    """`fetch_klines` and `heal_gap` are how every bar missed during an outage gets back in."""

    def _kline(self, open_ms: int) -> list:
        # Binance's array, in order. Every field a different number, so an index slip shows up.
        return [open_ms, "100.0", "102.0", "98.0", "101.0", "5.0",
                open_ms + 59_999, "7.0", 8, "9.0", "10.0", "0"]

    def _rest(self, handler) -> BinanceREST:
        c = BinanceREST()
        c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return c

    async def test_every_field_of_the_kline_array_lands_in_its_own_column(self):
        """The REST array is twelve positional fields with no names on them, and the two that are
        easiest to confuse — `close_time` at 6 and `quote_volume` at 7 — sit next to each other.

        Reading 6 as the quote volume writes a 13-digit timestamp into a price-times-size field on
        every bar the backfill produces. It is not a crash: the bars merge into the store beside
        archive bars that carry a real number, and any volume statistic computed across the join
        is nonsense from then on.
        """
        def handler(request):
            return httpx.Response(200, json=[self._kline(JAN_2025)])

        rest = self._rest(handler)
        bars = await rest.fetch_klines("btcusdt", TF_1M, JAN_2025, JAN_2025 + TF_1M.ms)
        assert len(bars) == 1
        b = bars[0]
        got = (b.open_time_ms, b.open, b.high, b.low, b.close, b.volume,
               b.quote_volume, b.trades, b.taker_buy_base, b.taker_buy_quote)
        assert got == (JAN_2025, 100.0, 102.0, 98.0, 101.0, 5.0, 7.0, 8, 9.0, 10.0), (
            f"a kline field was read from the wrong position in the array: {got}")
        assert b.is_closed and b.n_source_bars == TF_1M.expected_source_bars

    async def test_paging_a_gap_resumes_AFTER_the_last_bar_it_received(self):
        """Healing an outage is a paginated loop, and where the next page starts is the whole of
        it: resume from the first bar of the page just read and the loop asks for the same window
        again, one minute further on each time. A day-long gap becomes ~1,400 identical REST calls
        instead of two — 2 weight each, against a 6,000/min budget — which is the request storm
        that earns the 418 the governor exists to avoid, while the gap itself never closes.
        """
        starts: list[int] = []
        page1 = [self._kline(JAN_2025 + i * TF_1M.ms) for i in range(1000)]
        page2 = [self._kline(JAN_2025 + i * TF_1M.ms) for i in range(1000, 1500)]

        def handler(request):
            st = int(request.url.params["startTime"])
            starts.append(st)
            if st == JAN_2025:
                return httpx.Response(200, json=page1)
            if st == JAN_2025 + 1000 * TF_1M.ms:
                return httpx.Response(200, json=page2)
            return httpx.Response(200, json=[])

        rest = self._rest(handler)
        end = JAN_2025 + 1499 * TF_1M.ms
        bars = await rest.heal_gap("BTCUSDT", TF_1M, JAN_2025, end)
        ts = [b.open_time_ms for b in bars]
        assert starts == [JAN_2025, JAN_2025 + 1000 * TF_1M.ms], (
            f"the second page did not resume after the first one's last bar: {starts}")
        assert ts == list(range(JAN_2025, end + TF_1M.ms, TF_1M.ms)), (
            f"the healed gap is not the contiguous range asked for: {len(ts)} bars, "
            f"{ts[:1]}..{ts[-1:]}")


class TestTheGapHealSeam:
    """Binance cuts the WebSocket every 24 h BY DESIGN, so this seam is crossed daily. A hole
    there raises nothing at all: it just makes a window of N bars span more time than it claims."""

    def _msg(self, open_ms: int, *, closed: bool = True) -> dict:
        return {
            "_ts_ingest_ms": open_ms + TF_1M.ms,
            "k": {"t": open_ms, "o": "100.0", "h": "102.0", "l": "98.0", "c": "101.0",
                  "v": "5.0", "q": "7.0", "n": 8, "V": "9.0", "Q": "10.0", "x": closed},
        }

    async def test_the_seam_repeats_no_bar_and_loses_none(self, monkeypatch):
        """Both directions are silent and both are damaging. A minute lost at the seam is a hole
        the resampler turns into a short hour; a minute delivered twice doubles that minute's
        volume, distorts the ATR and manufactures a pivot — and the engine consumes the bar before
        the store ever gets the chance to deduplicate it.
        """
        t0 = JAN_2025
        feed = KlineFeed("BTCUSDT")
        asked: list[tuple[int, int]] = []

        async def fake_heal(start_ms: int, end_ms: int):
            asked.append((start_ms, end_ms))
            # What REST returns: every bar from start_ms onwards, including ones we already have.
            return [binance_klines._to_bar(self._msg(t)["k"], "BTCUSDT", TF_1M)
                    for t in range(t0 - TF_1M.ms, t0 + 3 * TF_1M.ms, TF_1M.ms)
                    if t >= start_ms]

        async def fake_stream_json(base, streams, **kw):
            if kw.get("on_connect"):
                kw["on_connect"]()
            for t in (t0 + 2 * TF_1M.ms, t0 + 3 * TF_1M.ms):
                yield self._msg(t)

        feed.heal = fake_heal
        monkeypatch.setattr(binance_klines, "stream_json", fake_stream_json)

        out = [b.open_time_ms async for b in feed.stream(start_ms=t0)]

        assert asked, "no heal was requested at all across the seam"
        assert asked[0][0] == t0 - TF_1M.ms, (
            f"the heal window starts at {asked[0][0]}, not one bar back at {t0 - TF_1M.ms}. "
            "The overlap is exactly one minute and it is written down in a source comment and "
            "nowhere else: start later and the boundary bar falls between the two feeds and is "
            "lost on every reconnect; start earlier and every reconnect re-fetches history for "
            "nothing, spending REST weight on the one code path that runs when the feed is "
            "already in trouble")
        assert out == sorted(set(out)), f"the seam delivered a bar twice or out of order: {out}"
        assert all(t > t0 for t in out), (
            f"a bar the caller already had was delivered again across the seam: {out}")
        assert out == list(range(t0 + TF_1M.ms, out[-1] + TF_1M.ms, TF_1M.ms)), (
            f"a minute is missing between the healed bars and the live ones: {out}")

    def test_a_bar_that_is_still_forming_does_not_claim_a_full_period(self):
        """The in-flight bar is emitted between closes so the chart can move, and it is the one
        bar whose coverage is a lie waiting to be believed: stamped with a full period's source
        count it is indistinguishable from a closed bar, and a partial minute's high and low then
        get treated as a settled extreme by the pivot detector."""
        forming = binance_klines._to_bar(self._msg(JAN_2025, closed=False)["k"], "BTCUSDT", TF_1M)
        assert not forming.is_closed
        assert forming.n_source_bars == 0, (
            f"a forming bar claims {forming.n_source_bars} source bars, so it reads as complete")
        closed = binance_klines._to_bar(self._msg(JAN_2025)["k"], "BTCUSDT", TF_1M)
        assert closed.n_source_bars == TF_1M.expected_source_bars


class TestTheSocketStampsAndAddressesItsOwnMessages:
    """`stream_json` itself, which no test reached: every WebSocket test in this file scripts
    messages straight past it, so its three lines of real logic ran nowhere.

    Nothing here opens a socket. `websockets.connect` is replaced with a fake that records the URL
    it was handed and yields one scripted frame, which is enough to exercise the guard, the path
    builder and the stamping.
    """

    @staticmethod
    def _fake_connect(seen: list[str], frames: list[str]):
        class _WS:
            async def recv(self):
                if not frames:
                    raise asyncio.CancelledError
                return frames.pop(0)

        class _Conn:
            def __init__(self, url, **kw):
                seen.append(url)

            async def __aenter__(self):
                return _WS()

            async def __aexit__(self, *a):
                return False

        return _Conn

    async def _first(self, monkeypatch, streams, frames):
        seen: list[str] = []
        monkeypatch.setattr(binance_ws.websockets, "connect",
                            self._fake_connect(seen, list(frames)))
        out = []
        try:
            async for msg in binance_ws.stream_json(binance_ws.SPOT_WS, streams):
                out.append(msg)
        except asyncio.CancelledError:
            pass
        return seen, out

    @pytest.mark.parametrize("streams", [
        ["BTCUSDT@kline_1m"],                          # the whole name shouted
        ["btcusdt@kline_1m", "ETHUSDT@kline_1m"],      # one of several, which is the likelier slip
    ])
    async def test_an_uppercase_stream_name_is_refused_before_anything_is_opened(
            self, monkeypatch, streams):
        """The module's own docstring calls this «a perfect silent failure»: in uppercase the
        connection opens, Binance accepts it, and not one message ever arrives. There is no error
        and no disconnect to notice — the feed simply goes quiet forever, which is the single
        hardest failure in the system to attribute, because every other symptom points at the
        market being closed or the machine being asleep. And `stream_json` reconnects INDEFINITELY,
        so nothing ever gives up and reports.

        The transport is stubbed even though the guard is supposed to raise before reaching it.
        That is deliberate and it is not belt-and-braces: without the stub, a regression that
        removed the guard would send this test to the real Binance endpoint and then loop on the
        backoff forever, so the suite would hang rather than fail. A test whose failure mode is a
        hung deploy gate is worse than the bug it was meant to catch.
        """
        seen: list[str] = []
        monkeypatch.setattr(binance_ws.websockets, "connect", self._fake_connect(seen, []))
        with pytest.raises(ValueError, match="LOWERCASE"):
            async for _ in binance_ws.stream_json(binance_ws.SPOT_WS, streams):
                pass
        assert not seen, (
            f"the guard let an uppercase name through and a connection to {seen} was attempted: "
            "that socket opens, stays open, and delivers nothing for the life of the process")

    async def test_the_ingest_stamp_is_in_MILLISECONDS_like_everything_else(self, monkeypatch):
        """★ `_ts_ingest_ms` is not a diagnostic. `KlineFeed.stream` passes it straight through as
        the END of the gap-heal window, and `heal` opens with `if end_ms <= start_ms: return []`.

        Stamped in seconds it is ~1.7e9 against a `last_closed_ms` of ~1.7e12, so the end is
        always before the start and gap healing is silently and permanently DISABLED on every
        reconnect — not mis-stamped, disabled — while the seam it exists to close is crossed daily
        by design. The same value drives `silent_seconds`, so the watchdog would also read about
        1.7 million seconds of silence, forever.

        The seam test above cannot see this: it scripts `_ts_ingest_ms` itself, so the line that
        actually stamps it never runs there.
        """
        frame = json.dumps({"e": "kline", "k": {"t": JAN_2025, "x": True}})
        before = int(time.time() * 1000)
        _, out = await self._first(monkeypatch, ["btcusdt@kline_1m"], [frame])
        after = int(time.time() * 1000)

        assert len(out) == 1, "the fake transport should have delivered exactly one message"
        stamp = out[0]["_ts_ingest_ms"]
        assert isinstance(stamp, int) and before <= stamp <= after, (
            f"_ts_ingest_ms came back {stamp}, outside the [{before}, {after}] window this call "
            "ran in. Everything in the system compares against integer milliseconds; in seconds "
            "the heal window's end lands three orders of magnitude before its start and the gap "
            "heal is dead on every reconnect, with nothing raising"
        )
        assert 1_600_000_000_000 < stamp < 4_000_000_000_000, (
            f"{stamp} is not a plausible epoch in MILLISECONDS")

    @pytest.mark.parametrize("streams,expect", [
        (["btcusdt@kline_1m"], "/ws/btcusdt@kline_1m"),
        (["btcusdt@kline_1m", "ethusdt@kline_1m"],
         "/stream?streams=btcusdt@kline_1m/ethusdt@kline_1m"),
    ])
    async def test_one_stream_and_several_use_the_addresses_binance_defines(
            self, monkeypatch, streams, expect):
        """Binance has two endpoints and they are not interchangeable. `/ws/<name>` delivers bare
        payloads; `/stream?streams=` wraps every message in `{"stream":…, "data":…}`, which the
        unwrapping branch below depends on. Address a single stream through the combined endpoint
        and the payload arrives wrapped while the caller reads it bare — every field comes back
        missing, on a connection that is up and delivering.
        """
        seen, _ = await self._first(monkeypatch, streams, [])
        assert seen and seen[0] == binance_ws.SPOT_WS + expect, (
            f"{len(streams)} stream(s) were addressed to {seen[0] if seen else None!r}, not "
            f"{binance_ws.SPOT_WS + expect!r}")

    async def test_a_combined_frame_is_unwrapped_and_labelled_with_its_stream(self, monkeypatch):
        """The other half of the same contract: a wrapped frame has to be flattened, and the
        stream it came from kept, or a multi-symbol feed cannot tell whose bar it is holding."""
        frame = json.dumps({"stream": "ethusdt@kline_1m", "data": {"e": "kline", "k": {"t": 1}}})
        _, out = await self._first(
            monkeypatch, ["btcusdt@kline_1m", "ethusdt@kline_1m"], [frame])
        assert out and out[0]["e"] == "kline" and out[0]["k"] == {"t": 1}, (
            f"the combined frame was not unwrapped: {out}")
        assert out[0]["_stream"] == "ethusdt@kline_1m", (
            "the stream name is the only thing saying which symbol this bar belongs to")


class TestTheLiquidationLogIsAppendOnly:
    """Every hour not recorded is lost forever: there is no archive to backfill liquidations from."""

    def test_reopening_a_day_appends_to_it_instead_of_truncating_it(self, tmp_path):
        """The recorder reconnects on any exception, and a day's file is reopened whenever the
        day key changes — including back to a day already written, which a clock skew, a late
        message or a restart all produce.

        Opened for writing rather than appending, that reopen empties the file: hours of recorded
        cascades disappear at the moment of the reconnect, and the recorder carries on happily
        counting messages. Nothing reads the file until much later, when the gap is unexplainable.
        """
        rec = LiquidationRecorder(tmp_path, "okx")
        d1 = int(datetime(2026, 9, 9, 12, 0, tzinfo=UTC).timestamp() * 1000)
        d2 = d1 + int(timedelta(days=1).total_seconds() * 1000)

        for ts, line in ((d1, "first"), (d2, "next day"), (d1, "back again")):
            fh = rec._file(ts)
            fh.write(line + "\n")
            fh.flush()

        day1 = tmp_path / "okx-2026-09-09.jsonl"
        assert day1.exists(), (
            f"the day's file is not named after the day: {sorted(p.name for p in tmp_path.iterdir())}")
        assert day1.read_text().splitlines() == ["first", "back again"], (
            "reopening the day truncated it: the liquidations recorded before the reconnect "
            f"are gone ({day1.read_text()!r})")
        assert (tmp_path / "okx-2026-09-10.jsonl").read_text().splitlines() == ["next day"], (
            "two different days were written into one file, so the daily rollover is not a "
            "rollover at all")
