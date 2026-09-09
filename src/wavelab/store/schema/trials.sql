-- The trial log. It is built on DAY 1 even though the Deflated Sharpe that consumes it arrives
-- much later, because it CANNOT be reconstructed after the fact.
--
-- The "we do not search for parameters" policy does not get rid of the search: it moves it to
-- eyeballed tweaking that nobody records. There are ~60 hand-picked constants in this project
-- (k, min_pct, the ER thresholds, the MDL penalty, the guideline lambdas, the Fibonacci mixture
-- weights, eta, alpha, the caps, the whole prior table). If they get tuned by staring at charts
-- and leaving no trace, effective_n will count ~1 while the real multiple-comparison burden is in
-- the hundreds, and the DSR will come out ANTI-CONSERVATIVE: the honesty machinery would be lying
-- in exactly the direction it claims to protect against.
--
-- That is why `effective_n` counts DISTINCT config_hash values ever evaluated against the fixture
-- set, not just programmatic trials. And CI rejects a configuration change that has no row here.

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS trials (
    id                INTEGER PRIMARY KEY,
    ts_ms             INTEGER NOT NULL,
    config_hash       TEXT    NOT NULL,
    git_sha           TEXT    NOT NULL,
    dirty             INTEGER NOT NULL DEFAULT 0,   -- working tree dirty at evaluation time
    kind              TEXT    NOT NULL CHECK (kind IN ('startup','fixture_eval','walkforward','manual')),
    fixture_set       TEXT,
    metric            TEXT,
    value             REAL,
    note              TEXT
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_trials_hash_kind ON trials (config_hash, kind, fixture_set);
CREATE INDEX IF NOT EXISTS ix_trials_ts ON trials (ts_ms);

-- The view the Deflated Sharpe consumes. It deliberately does NOT filter on `kind`: a config hash
-- evaluated "just to have a look" is still a trial.
CREATE VIEW IF NOT EXISTS effective_n AS
    SELECT COUNT(DISTINCT config_hash) AS n FROM trials;
