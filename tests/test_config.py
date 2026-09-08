"""Añadir un activo debe ser UN fichero TOML y cero código. Aquí se demuestra."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from wavelab.config import AppConfig, AssetConfig, EngineConfig, config_hash, load_config


def test_carga_la_configuracion_real():
    c = load_config(Path(__file__).parent.parent / "config")
    assert "BTCUSDT" in c.assets
    btc = c.assets["BTCUSDT"]
    assert btc.instrument == "spot" and btc.has_funding is False
    assert btc.price_space == "log"


def test_separacion_de_roles():
    c = load_config(Path(__file__).parent.parent / "config")
    assert c.trigger_tf == "15m", "solo el timeframe de disparo dibuja entradas"
    assert c.regime_tf not in c.gate_tfs


def test_anadir_un_activo_no_requiere_codigo(tmp_path: Path):
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


def test_metadata_de_instrumento_es_obligatoria():
    """Sin tick_size el stop no está definido; sin comisiones se hereda la economía de BTC."""
    with pytest.raises(Exception):
        AssetConfig(symbol="X")


class TestInvariantes:
    def test_spot_no_paga_funding(self):
        with pytest.raises(ValueError, match="funding"):
            AssetConfig(symbol="X", instrument="spot", has_funding=True, tick_size=1, qty_step=1)

    def test_cripto_es_24x7(self):
        with pytest.raises(ValueError, match="24/7"):
            AssetConfig(symbol="X", asset_class="crypto", session="us_equity_rth",
                        tick_size=1, qty_step=1)

    @pytest.mark.parametrize("kw", [
        {"er_trend_exit": 0.9}, {"er_chop_exit": 0.01}, {"min_stop_atr": 9.0},
    ])
    def test_histeresis_incoherente_lanza(self, kw):
        with pytest.raises(ValueError):
            EngineConfig(**kw)


class TestHashDeConfiguracion:
    """El hash alimenta el N efectivo del Deflated Sharpe. Si no cambia al tocar un parámetro,
    la contabilidad de ensayos miente y el DSR sale anti-conservador."""

    def test_es_estable(self):
        a, b = AppConfig(), AppConfig()
        assert config_hash(a) == config_hash(b)

    def test_cambia_con_cualquier_parametro_del_motor(self):
        base = AppConfig()
        for kw in ({"zigzag_k_atr": 1.6}, {"er_trend_enter": 0.36}, {"ev_min_r": 0.16}):
            otro = AppConfig(engine=EngineConfig(**kw))
            assert config_hash(base) != config_hash(otro), (
                f"tocar {kw} no cambió el hash: ese ensayo no se contaría"
            )

    def test_no_cambia_con_la_ruta_de_datos(self):
        assert config_hash(AppConfig()) == config_hash(AppConfig(data_dir=Path("/otro")))
