"""Configuración por TOML, y el hash que alimenta el registro de ensayos.

TOML sobre YAML: cero dependencias extra (``tomllib`` es stdlib desde 3.11) y sin las trampas de
coerción de tipos de YAML, donde ``no`` se convierte en False y ``1.10`` en 1.1.

Añadir un activo es **un fichero** en ``config/assets/`` y cero cambios de código. Para que eso sea
cierto y no aspiracional, ``AssetConfig`` lleva la metadata de instrumento completa: sin ``tick_size``
el stop «invalidación ∓ max(0.25·ATR, 2 ticks, 0.1%)» no está definido, y sin comisiones por activo
un SOLUSDT nuevo reutilizaría en silencio la economía de BTC.
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
    """Todo lo que distingue un activo de otro. Un TOML por activo."""

    symbol: str
    exchange: str = "binance"
    instrument: Literal["spot", "perp"] = "spot"
    asset_class: Literal["crypto", "equity", "fx", "futures", "commodity"] = "crypto"

    # --- microestructura: sin esto, el stop y los costes no están definidos ---
    tick_size: float
    qty_step: float
    min_notional: float = 0.0
    fee_bps_taker: float = 7.5
    fee_bps_maker: float = 2.0
    #: En spot no se paga funding. Se deja explícito para que el modelo de costes no arrastre
    #: la economía del perpetuo sobre una serie de precios spot.
    has_funding: bool = False

    #: BTC abarca ~100x en el histórico: en espacio lineal, un canal 0-2-4 y el test de igualdad
    #: de ondas no significan nada a grado alto, y la comparación "la 3 no es la más corta" cambia
    #: de respuesta. Log para cripto y acciones; lineal para divisas y tipos.
    price_space: Literal["log", "linear"] = "log"

    #: 24/7 para cripto. Para acciones hace falta calendario de sesión o el resampleo fabrica velas
    #: fantasma en las 17,5 horas nocturnas y envenena el ATR y el umbral del ZigZag.
    session: Literal["24x7", "us_equity_rth"] = "24x7"

    #: Excepción de Prechter: la onda 4 puede solaparse con el territorio de la 1 en futuros y
    #: materias primas. En diagonales se permite siempre (se decide por patrón, no aquí).
    allow_overlap: bool = False

    #: Invalidar la regla R3 por mecha descarta una fracción enorme de conteos buenos en BTC,
    #: donde la caza de liquidaciones es endémica. Idéntico en la ruta viva y en el replay.
    r3_on: Literal["close", "wick"] = "close"

    timeframes: list[str] = Field(default_factory=lambda: ["15m", "1h", "4h", "1d"])
    enabled: bool = True

    @model_validator(mode="after")
    def _coherencia(self) -> AssetConfig:
        if self.instrument == "spot" and self.has_funding:
            raise ValueError(
                f"{self.symbol}: instrument='spot' con has_funding=true. El funding es del "
                "perpetuo; en spot es contexto cross-instrumento, no un coste."
            )
        if self.asset_class == "crypto" and self.session != "24x7":
            raise ValueError(f"{self.symbol}: cripto opera 24/7")
        if self.tick_size <= 0 or self.qty_step <= 0:
            raise ValueError(f"{self.symbol}: tick_size y qty_step deben ser > 0")
        return self


class EngineConfig(BaseModel, frozen=True):
    """Parámetros del motor. Fijos y no se ajustan: la búsqueda de parámetros ES el generador
    de sobreajuste de backtest. Cambiar cualquiera de estos escribe una fila en `trials`."""

    # ZigZag causal
    zigzag_k_atr: float = 1.5
    zigzag_min_pct: float = 0.005
    atr_period: int = 14

    # Régimen, con histéresis: entrar y salir por umbrales distintos evita el parpadeo de etiqueta
    er_period: int = 20
    er_trend_enter: float = 0.35
    er_trend_exit: float = 0.28
    er_chop_enter: float = 0.20
    er_chop_exit: float = 0.26
    vol_percentile_window_days: int = 365

    # Riesgo
    stop_buffer_atr: float = 0.25
    stop_buffer_pct: float = 0.001
    max_stop_atr: float = 3.0
    min_stop_atr: float = 0.75
    ev_min_r: float = 0.15
    expected_loss_r: float = 1.10

    # Ventanas
    window_bars: int = 5000
    ring_capacity: int = 8192

    @model_validator(mode="after")
    def _histeresis_coherente(self) -> EngineConfig:
        if self.er_trend_exit >= self.er_trend_enter:
            raise ValueError("er_trend_exit debe ser < er_trend_enter (histéresis)")
        if self.er_chop_exit <= self.er_chop_enter:
            raise ValueError("er_chop_exit debe ser > er_chop_enter (histéresis)")
        if self.min_stop_atr >= self.max_stop_atr:
            raise ValueError("min_stop_atr debe ser < max_stop_atr")
        return self


class AppConfig(BaseModel, frozen=True):
    data_dir: Path = Path("data")
    trigger_tf: str = "15m"
    regime_tf: str = "1h"
    gate_tfs: list[str] = Field(default_factory=lambda: ["4h", "1d"])
    directions: list[Literal["long", "short"]] = Field(default_factory=lambda: ["long"])
    #: El motor evalúa y REGISTRA ambas direcciones aunque la interfaz solo muestre una:
    #: así se acumula el doble de evidencia y activar cortos es una línea, no un refactor.
    shadow_opposite_direction: bool = True
    engine: EngineConfig = Field(default_factory=EngineConfig)
    assets: dict[str, AssetConfig] = Field(default_factory=dict)
    plugin_modules: list[str] = Field(default_factory=list)


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
    """Hash estable de la configuración resuelta.

    Se escribe en `trials` al arrancar. `effective_n` cuenta hashes DISTINTOS jamás evaluados —
    no solo los de una búsqueda programática — porque ajustar una constante a ojo mirando un
    gráfico también es un ensayo, y si no se cuenta, el Deflated Sharpe sale anti-conservador.
    """
    payload = cfg.model_dump(mode="json")
    payload.pop("data_dir", None)  # la ruta no cambia el modelo
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
