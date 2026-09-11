"""Adding an asset must be ONE TOML file and zero code. This is where that gets proved."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wavelab.config import AppConfig, AssetConfig, EngineConfig, config_hash, load_config


def test_loads_the_real_config():
    c = load_config(Path(__file__).parent.parent / "config")
    assert "BTCUSDT" in c.assets
    btc = c.assets["BTCUSDT"]
    assert btc.instrument == "spot" and btc.has_funding is False
    assert btc.price_space == "log"


def test_separation_of_roles():
    c = load_config(Path(__file__).parent.parent / "config")
    assert c.trigger_tf == "15m", "only the trigger timeframe draws entries"
    assert c.regime_tf not in c.gate_tfs


def test_adding_an_asset_requires_no_code(tmp_path: Path):
    root = tmp_path / "config"
    (root / "assets").mkdir(parents=True)
    (root / "wavelab.toml").write_text('trigger_tf = "15m"\n')
    (root / "assets" / "solusdt.toml").write_text(
        'symbol="SOLUSDT"\ninstrument="spot"\nasset_class="crypto"\n'
        "tick_size=0.001\nqty_step=0.01\nfee_bps_taker=7.5\n"
    )
    c = load_config(root)
    assert set(c.assets) == {"SOLUSDT"}
    assert c.assets["SOLUSDT"].tick_size == 0.001


def test_the_assets_load_in_a_deterministic_order(tmp_path: Path):
    """`Path.glob` yields directory order, and the FIRST asset is the one the server trades.

    `server/app.py` takes `next(iter(cfg.assets), "BTCUSDT")` as the symbol the whole service runs
    on. Leave the listing unsorted and which instrument that is becomes a fact about the order the
    files happened to land in the directory — different on the machine that wrote them and the
    machine that restored them from a backup, with nothing in the config to explain it. The same
    unsorted order reaches `config_hash` through `assets`, so the trial identity would travel with
    the filesystem too.

    (On a filesystem that happens to return names already ordered this test still passes and
    proves less; APFS and ext4 both return hash order, so it has teeth where it runs.)
    """
    root = tmp_path / "config"
    (root / "assets").mkdir(parents=True)
    for sym in ("ZZZUSDT", "AAAUSDT", "MMMUSDT", "KKKUSDT", "BBBUSDT"):   # not alphabetical
        (root / "assets" / f"{sym.lower()}.toml").write_text(
            f'symbol="{sym}"\ntick_size=0.01\nqty_step=0.01\n'
        )
    loaded = list(load_config(root).assets)
    assert loaded == sorted(loaded), (
        f"the assets loaded as {loaded}: the directory listing was not sorted, so the first "
        "asset — the single symbol the server trades — is whatever the filesystem hands back "
        "first, and it can change without a line of config changing"
    )


def test_the_global_toml_is_read_and_not_merely_looked_for(tmp_path: Path):
    """Every value in the shipped `config/wavelab.toml` restates a code default, so nothing else
    in this suite can tell the file apart from the defaults sitting behind it.

    `load_config` could stop reading it entirely — an inverted `.exists()`, a renamed file, a
    reordered merge — and `test_loads_the_real_config` and `test_separation_of_roles` would both
    stay green while the running engine quietly ignored every line an operator had edited. The
    only way to tell is a file whose values are NOT the defaults, which is exactly the file a
    second deployment would have.
    """
    root = tmp_path / "config"
    (root / "assets").mkdir(parents=True)
    (root / "wavelab.toml").write_text(
        'trigger_tf = "5m"\nregime_tf = "2h"\n\n[engine]\natr_period = 21\n'
    )
    c = load_config(root)
    assert (c.trigger_tf, c.regime_tf) == ("5m", "2h"), (
        f"the config declared trigger_tf=5m and regime_tf=2h and loaded as "
        f"({c.trigger_tf!r}, {c.regime_tf!r}): wavelab.toml is not being read, so the deployment "
        "is running on the defaults regardless of what the operator wrote"
    )
    assert c.engine.atr_period == 21, (
        f"[engine] atr_period=21 loaded as {c.engine.atr_period}: the engine table of the global "
        "TOML is dropped, so every engine parameter an operator sets is silently ignored — and "
        "the config_hash recorded in trials.sqlite describes a configuration nobody ran"
    )


def test_the_defaults_are_the_ones_the_project_ships():
    """The shipped TOML restates all of these, which is precisely why they need pinning here.

    Anything that does not read `config/` — a script, a notebook, a test, a deployment whose file
    is missing — runs on these values, and they are also what `config_hash` is taken over. Because
    the file overrides them, a changed default is invisible to every test that goes through
    `load_config`: `test_separation_of_roles` passes with `regime_tf` defaulting to "4h", a GATE
    timeframe that would put the veto in charge of classifying the regime, because the file sets
    it back to "1h" before the assertion is reached.
    """
    c = AppConfig()
    assert (c.trigger_tf, c.regime_tf, c.gate_tfs) == ("15m", "1h", ["4h", "1d"]), (
        f"the default role assignment is ({c.trigger_tf!r}, {c.regime_tf!r}, {c.gate_tfs}) and "
        "not (15m trigger, 1h regime, 4h+1d gates): the separation of roles the whole engine is "
        "built on is different for anyone who does not load a TOML"
    )
    assert c.regime_tf not in c.gate_tfs, (
        "the regime timeframe is also a gate: one timeframe would both classify the regime and "
        "veto on it, so the veto can never disagree with the classification"
    )
    e = EngineConfig()
    assert (e.atr_period, e.er_period) == (14, 20), (
        f"the default ATR and ER periods are ({e.atr_period}, {e.er_period}), not (14, 20): the "
        "ZigZag threshold and the regime label are computed over a different amount of history "
        "than the one every recorded trial was measured with"
    )


def test_the_ring_can_hold_the_window_the_engine_asks_for():
    """`ring_capacity` is one of the handful of config values the running server actually reads —
    it is passed straight to `LiveEngine` and sizes every `Ring` — and it appears in no TOML, so
    this default is the live value.

    Nothing ties it to `window_bars`, which is the number of bars the engine asks each ring for.
    Undersize it and no exception is raised anywhere: `Ring._take` clamps to what is retained and
    every feature is computed over a silently shorter history than it was validated on. At 819
    slots the 1m ring holds under fourteen hours, and the ER, ATR and vol percentile carry on
    returning perfectly plausible floats.
    """
    e = EngineConfig()
    assert e.ring_capacity >= e.window_bars, (
        f"the ring holds {e.ring_capacity} bars and the engine asks its windows for "
        f"{e.window_bars}: every window is silently clamped to what survived the wrap, and no "
        "call raises to say so"
    )
    assert e.ring_capacity == 8192, "the shipped capacity, two powers of two above window_bars"


def test_instrument_metadata_is_mandatory():
    """With no tick_size the stop is undefined; with no fees you inherit BTC's economics."""
    # `pytest.raises(Exception)` here asserted almost nothing: pydantic's ValidationError IS a
    # ValueError, but so is every accident — a typo in the field name would raise too and the test
    # would still have gone green while proving the opposite of its name. Pin the exception AND
    # which fields were missing, so that giving tick_size or qty_step a default breaks this test.
    with pytest.raises(ValidationError) as exc:
        AssetConfig(symbol="X")
    assert {e["loc"][0] for e in exc.value.errors() if e["type"] == "missing"} == {
        "tick_size", "qty_step"}


class TestInvariants:
    def test_spot_pays_no_funding(self):
        with pytest.raises(ValueError, match="funding"):
            AssetConfig(symbol="X", instrument="spot", has_funding=True, tick_size=1, qty_step=1)

    def test_crypto_is_24x7(self):
        with pytest.raises(ValueError, match="24/7"):
            AssetConfig(symbol="X", asset_class="crypto", session="us_equity_rth",
                        tick_size=1, qty_step=1)

    def test_a_zero_tick_or_qty_step_is_refused(self):
        """Zero is the value that actually arrives, and it is the one `> 0` is written for.

        Both are hydrated from the exchange's own filters: a field that failed to parse, or an
        asset TOML written from memory, arrives as 0.0 rather than as something obviously mad. A
        zero `tick_size` silently deletes the `2 ticks` term from the stop
        `invalidation ∓ max(0.25·ATR, 2 ticks, 0.1%)`, so the stop quietly stops respecting the
        instrument's own resolution; a zero `qty_step` divides by zero the moment a size is
        rounded. Zero IS the boundary, which is why the guard reads `<= 0` — and why every case
        below it has to be exercised, not just the absurd ones.
        """
        for kw in ({"tick_size": 0.0, "qty_step": 1.0}, {"tick_size": 1.0, "qty_step": 0.0}):
            with pytest.raises(ValueError, match="must be > 0"):
                AssetConfig(symbol="X", **kw)

    @pytest.mark.parametrize("kw", [
        {"er_trend_exit": 0.9}, {"er_chop_exit": 0.01}, {"min_stop_atr": 9.0},
        # The three boundaries themselves. Every case above sits far on the wrong side of its
        # guard, so all three comparisons could be relaxed from `>=`/`<=` to `>`/`<` — the classic
        # "tidy up the operator" edit — and stay green. Equal thresholds are hysteresis of zero
        # width, which is exactly the regime-label flicker the enter/exit pair exists to stop, and
        # `min_stop_atr == max_stop_atr` leaves the stop distance no interval to live in at all.
        {"er_trend_enter": 0.35, "er_trend_exit": 0.35},
        {"er_chop_enter": 0.20, "er_chop_exit": 0.20},
        {"min_stop_atr": 3.0, "max_stop_atr": 3.0},
    ])
    def test_incoherent_hysteresis_raises(self, kw):
        """Every one of these is a config that cannot mean anything, refused at construction.

        None of them raises later, which is the point: a zero-width hysteresis just makes the
        regime label flip on every bar that grazes the threshold, and an empty stop interval makes
        the planner clamp every stop to a single value. Both produce output that looks ordinary.
        """
        with pytest.raises(ValueError):
            EngineConfig(**kw)


class TestConfigHash:
    """The hash feeds the effective N of the Deflated Sharpe. If it does not change when a
    parameter is touched, the trial accounting lies and the DSR comes out anti-conservative."""

    def test_it_is_stable(self):
        a, b = AppConfig(), AppConfig()
        assert config_hash(a) == config_hash(b)

    def test_it_changes_with_any_engine_parameter(self):
        base = AppConfig()
        for kw in ({"zigzag_k_atr": 1.6}, {"er_trend_enter": 0.36}, {"ev_min_r": 0.16}):
            other = AppConfig(engine=EngineConfig(**kw))
            assert config_hash(base) != config_hash(other), (
                f"touching {kw} did not change the hash: that trial would go uncounted"
            )

    def test_it_does_not_change_with_the_data_path(self):
        assert config_hash(AppConfig()) == config_hash(AppConfig(data_dir=Path("/elsewhere")))

    def test_it_is_a_fact_about_the_content_and_not_about_the_order(self):
        """`assets` is filled from a directory listing, so its insertion order is a fact about the
        filesystem rather than about the configuration.

        Let that order reach the digest and the same deployment hashes differently on two machines
        — or on the same machine after one asset file is rewritten — so one engine is counted as
        two trials. `effective_n` climbs with nothing having changed, and the Deflated Sharpe
        corrects for searching nobody did.
        """
        a = AssetConfig(symbol="AAAUSDT", tick_size=0.1, qty_step=0.1)
        b = AssetConfig(symbol="BBBUSDT", tick_size=0.2, qty_step=0.2)
        assert config_hash(AppConfig(assets={"AAAUSDT": a, "BBBUSDT": b})) == config_hash(
            AppConfig(assets={"BBBUSDT": b, "AAAUSDT": a})
        ), (
            "the same two assets in the other order hashed differently: the trial identity "
            "depends on the order the asset files were read, not on what is in them"
        )

    def test_it_is_the_full_declared_width(self):
        """The digest is written into `trials.sqlite` and matched against rows written months ago.

        Widen or narrow it by one character and every historical hash stops matching the
        configuration that produced it: from that day on the running engine looks like a brand new
        trial, for ever. That is the anti-conservative direction — `effective_n` inflates against a
        record that can no longer be reconciled — and the row it corrupts is the one thing the
        project keeps specifically so the DSR cannot be gamed after the fact.
        """
        h = config_hash(AppConfig())
        assert len(h) == 16, (
            f"config_hash returned {len(h)} characters ({h!r}) instead of 16: every hash already "
            "recorded in trials.sqlite now belongs to a configuration nothing can find"
        )
