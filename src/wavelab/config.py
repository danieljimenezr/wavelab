"""TOML configuration, and the hash that feeds the trials registry.

TOML over YAML: zero extra dependencies (``tomllib`` is stdlib as of 3.11) and none of YAML's type
coercion traps, where ``no`` turns into False and ``1.10`` into 1.1.

Adding an asset is **one file** in ``config/assets/`` and zero code changes. For that to be true
rather than aspirational, ``AssetConfig`` carries the full instrument metadata: without
``tick_size`` the stop "invalidation ∓ max(0.25·ATR, 2 ticks, 0.1%)" is undefined, and without
per-asset fees a newly added SOLUSDT would silently reuse BTC's economics.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

__all__ = ["AppConfig", "AssetConfig", "EngineConfig", "config_hash", "load_config"]


class AssetConfig(BaseModel, frozen=True):
    """Everything that tells one asset apart from another. One TOML per asset."""

    symbol: str
    exchange: str = "binance"
    instrument: Literal["spot", "perp"] = "spot"
    asset_class: Literal["crypto", "equity", "fx", "futures", "commodity"] = "crypto"

    # --- microstructure: without this, the stop and the costs are undefined ---
    tick_size: float
    qty_step: float
    min_notional: float = 0.0
    fee_bps_taker: float = 7.5
    fee_bps_maker: float = 2.0
    #: There is no funding on spot. Kept explicit so the cost model does not drag the economics
    #: of the perpetual onto a spot price series.
    has_funding: bool = False

    #: BTC spans ~100x over the history: in linear space, a 0-2-4 channel and the wave equality
    #: test mean nothing at a high degree, and the "wave 3 is not the shortest" comparison changes
    #: its answer. Log for crypto and equities; linear for FX and rates.
    price_space: Literal["log", "linear"] = "log"

    #: 24/7 for crypto. Equities need a session calendar or the resampling fabricates ghost bars
    #: across the 17.5 overnight hours and poisons the ATR and the ZigZag threshold.
    session: Literal["24x7", "us_equity_rth"] = "24x7"

    #: Prechter's exception: wave 4 may overlap wave 1's territory in futures and commodities. In
    #: diagonals it is always allowed (decided per pattern, not here).
    allow_overlap: bool = False

    #: Invalidating rule R3 on a wick throws away a huge fraction of good counts on BTC, where
    #: liquidation hunting is endemic. Identical on the live path and in replay.
    r3_on: Literal["close", "wick"] = "close"

    timeframes: list[str] = Field(default_factory=lambda: ["15m", "1h", "4h", "1d"])
    enabled: bool = True

    @model_validator(mode="after")
    def _coherent(self) -> AssetConfig:
        if self.instrument == "spot" and self.has_funding:
            raise ValueError(
                f"{self.symbol}: instrument='spot' with has_funding=true. Funding belongs to the "
                "perpetual; on spot it is cross-instrument context, not a cost."
            )
        if self.asset_class == "crypto" and self.session != "24x7":
            raise ValueError(f"{self.symbol}: crypto trades 24/7")
        if self.tick_size <= 0 or self.qty_step <= 0:
            raise ValueError(f"{self.symbol}: tick_size and qty_step must be > 0")
        return self


class EngineConfig(BaseModel, frozen=True):
    """Engine parameters. Fixed, and not tuned: parameter search IS the generator of backtest
    overfitting. Changing any one of these writes a row in `trials`."""

    # Causal ZigZag
    zigzag_k_atr: float = 1.5
    zigzag_min_pct: float = 0.005
    atr_period: int = 14

    # Regime, with hysteresis: entering and leaving on different thresholds stops the label flicker
    er_period: int = 20
    er_trend_enter: float = 0.35
    er_trend_exit: float = 0.28
    er_chop_enter: float = 0.20
    er_chop_exit: float = 0.26
    vol_percentile_window_days: int = 365

    # Risk
    stop_buffer_atr: float = 0.25
    stop_buffer_pct: float = 0.001
    max_stop_atr: float = 3.0
    min_stop_atr: float = 0.75
    ev_min_r: float = 0.15
    expected_loss_r: float = 1.10

    # Windows
    window_bars: int = 5000
    ring_capacity: int = 8192

    @model_validator(mode="after")
    def _hysteresis_coherent(self) -> EngineConfig:
        if self.er_trend_exit >= self.er_trend_enter:
            raise ValueError("er_trend_exit must be < er_trend_enter (hysteresis)")
        if self.er_chop_exit <= self.er_chop_enter:
            raise ValueError("er_chop_exit must be > er_chop_enter (hysteresis)")
        if self.min_stop_atr >= self.max_stop_atr:
            raise ValueError("min_stop_atr must be < max_stop_atr")
        return self


class AppConfig(BaseModel, frozen=True):
    data_dir: Path = Path("data")
    trigger_tf: str = "15m"
    regime_tf: str = "1h"
    gate_tfs: list[str] = Field(default_factory=lambda: ["4h", "1d"])
    directions: list[Literal["long", "short"]] = Field(default_factory=lambda: ["long"])
    #: The engine evaluates and RECORDS both directions even though the interface only shows one:
    #: that way twice as much evidence piles up and turning shorts on is one line, not a refactor.
    shadow_opposite_direction: bool = True
    engine: EngineConfig = Field(default_factory=EngineConfig)
    assets: dict[str, AssetConfig] = Field(default_factory=dict)


def load_config(root: Path | str = "config") -> AppConfig:
    root = Path(root)
    base = tomllib.loads((root / "wavelab.toml").read_text()) if (root / "wavelab.toml").exists() else {}
    assets: dict[str, AssetConfig] = {}
    for f in sorted((root / "assets").glob("*.toml")):
        a = AssetConfig(**tomllib.loads(f.read_text()))
        assets[a.symbol] = a
    base["assets"] = assets
    return AppConfig(**base)


def config_hash(cfg: AppConfig) -> str:
    """Stable hash of the resolved configuration.

    Written to `trials` at startup. `effective_n` counts DISTINCT hashes ever evaluated — not just
    the ones from a programmatic search — because nudging a constant by eye while staring at a
    chart is a trial too, and if it goes uncounted the Deflated Sharpe comes out anti-conservative.
    """
    payload = cfg.model_dump(mode="json")
    payload.pop("data_dir", None)  # the path does not change the model
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
