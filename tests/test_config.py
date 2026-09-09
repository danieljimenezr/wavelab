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

    @pytest.mark.parametrize("kw", [
        {"er_trend_exit": 0.9}, {"er_chop_exit": 0.01}, {"min_stop_atr": 9.0},
    ])
    def test_incoherent_hysteresis_raises(self, kw):
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
