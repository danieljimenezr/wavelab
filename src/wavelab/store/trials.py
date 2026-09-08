"""Registro de ensayos. Existe para que el N efectivo no mienta.

La política de "no buscamos parámetros" NO elimina la búsqueda: la mueve a ajuste a ojo sin
registrar. Cada configuración evaluada, cada subgrupo mirado, cada umbral probado es un ensayo, y si
no se cuentan, el Deflated Sharpe sale ANTI-CONSERVADOR: la maquinaria de honestidad acabaría
mintiendo justo en la dirección de la que dice proteger.

Se escribe desde el día 1 aunque la prueba que lo consume llegue mucho después, porque **no se puede
reconstruir hacia atrás**.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from pathlib import Path

__all__ = ["TrialLog"]

SCHEMA = (Path(__file__).parent / "schema" / "trials.sql")


def _git_sha() -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5).stdout.strip() or "?"
        sucio = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=5).stdout.strip())
        return sha, sucio
    except Exception:  # noqa: BLE001
        return "?", False


class TrialLog:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA.read_text())

    def record(self, config_hash: str, kind: str, *, fixture_set: str | None = None,
               metric: str | None = None, value: float | None = None,
               note: str | None = None) -> None:
        sha, sucio = _git_sha()
        self.db.execute(
            "INSERT OR REPLACE INTO trials"
            " (ts_ms, config_hash, git_sha, dirty, kind, fixture_set, metric, value, note)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (int(time.time() * 1000), config_hash, sha, int(sucio), kind,
             fixture_set, metric, value, note))
        self.db.commit()

    @property
    def effective_n(self) -> int:
        """Hashes de configuración DISTINTOS jamás evaluados. Es el N que deflacta el Sharpe."""
        return int(self.db.execute("SELECT n FROM effective_n").fetchone()[0])

    def summary(self) -> list[tuple]:
        return self.db.execute(
            "SELECT kind, COUNT(*), MIN(ts_ms), MAX(ts_ms) FROM trials GROUP BY kind").fetchall()

    def close(self) -> None:
        self.db.close()
