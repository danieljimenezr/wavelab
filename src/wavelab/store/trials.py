"""Trial log. It exists so that the effective N cannot lie.

The "we do not search for parameters" policy does NOT get rid of the search: it moves it to
eyeballed tweaking that nobody records. Every configuration evaluated, every subgroup looked at,
every threshold tried is a trial, and if they are not counted the Deflated Sharpe comes out
ANTI-CONSERVATIVE: the honesty machinery would end up lying in exactly the direction it claims to
protect against.

It is written from day 1 even though the test that consumes it arrives much later, because **it
cannot be reconstructed after the fact**.
"""

from __future__ import annotations

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
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=5).stdout.strip())
        return sha, dirty
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
        sha, dirty = _git_sha()
        self.db.execute(
            "INSERT OR REPLACE INTO trials"
            " (ts_ms, config_hash, git_sha, dirty, kind, fixture_set, metric, value, note)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (int(time.time() * 1000), config_hash, sha, int(dirty), kind,
             fixture_set, metric, value, note))
        self.db.commit()

    @property
    def effective_n(self) -> int:
        """DISTINCT configuration hashes ever evaluated. This is the N that deflates the Sharpe."""
        return int(self.db.execute("SELECT n FROM effective_n").fetchone()[0])

    def summary(self) -> list[tuple]:
        return self.db.execute(
            "SELECT kind, COUNT(*), MIN(ts_ms), MAX(ts_ms) FROM trials GROUP BY kind").fetchall()

    def close(self) -> None:
        self.db.close()
