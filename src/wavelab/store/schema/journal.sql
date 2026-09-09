-- The journal: the only uncontaminated evidence this project is ever going to have.
--
-- It lives in its own SQLite file (WAL) and does NOT share a destination with the bar store:
-- bars can be downloaded again, this cannot. It accumulates ~60 signals a year, so two years of
-- journal are irreplaceable and impossible to reconstruct — the weights, the configuration and the
-- code will all have drifted, so there is no way to re-derive what the system would have said.
--
-- Every row is stamped with engine_sha, config_hash and weights_version so that the evidence is
-- SELF-DESCRIBING: a year from now, knowing which version produced a signal must not depend on
-- anybody's memory or on the git history.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY,
    ts_ms             INTEGER NOT NULL,
    symbol            TEXT    NOT NULL,
    timeframe         TEXT    NOT NULL,
    verdict           TEXT    NOT NULL CHECK (verdict IN ('no_trade','watch','actionable')),
    maturity          INTEGER NOT NULL,
    archetype         TEXT,
    direction         INTEGER,
    entry_lo          REAL,
    entry_hi          REAL,
    stop              REAL,
    -- The product. Stored next to the NAME of the rule that produced it, so that it cannot drift
    -- away from its own justification.
    invalidation_price REAL,
    invalidation_rule  TEXT,
    targets_json      TEXT,
    exit_template_id  TEXT,
    count_id          TEXT,
    rr_at_t2          REAL,
    cost_r            REAL,
    p_required        REAL,
    ev_r_lo           REAL,      -- lower bound of the bootstrap CI; NULL when not computable
    n_cell            INTEGER,   -- the n behind ev_r_lo: never a number without its n
    stale             INTEGER NOT NULL DEFAULT 0,
    catching_up       INTEGER NOT NULL DEFAULT 0,
    manual            INTEGER NOT NULL DEFAULT 0,  -- hand-constrained count: out of the statistics
    reasons_json      TEXT,
    engine_sha        TEXT    NOT NULL,
    config_hash       TEXT    NOT NULL,
    weights_version   TEXT    NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_decisions_ts     ON decisions (symbol, timeframe, ts_ms);
CREATE INDEX IF NOT EXISTS ix_decisions_arch   ON decisions (archetype, verdict);

-- Every signal emitted ALSO writes a null-arm row with the same ExitTemplate.
-- The paired delta against a random arm under identical exit rules becomes interpretable long
-- before the absolute hit rate does.
CREATE TABLE IF NOT EXISTS null_arm (
    id                INTEGER PRIMARY KEY,
    decision_id       INTEGER NOT NULL REFERENCES decisions(id),
    entry_price       REAL    NOT NULL,
    stop              REAL    NOT NULL,
    exit_template_id  TEXT    NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS outcomes (
    decision_id       INTEGER PRIMARY KEY REFERENCES decisions(id),
    is_null_arm       INTEGER NOT NULL DEFAULT 0,
    resolved_ts_ms    INTEGER NOT NULL,
    barrier           TEXT    NOT NULL CHECK (barrier IN ('tp','sl','vertical')),
    r_realized        REAL    NOT NULL,
    mae_r             REAL    NOT NULL,
    mfe_r             REAL    NOT NULL,
    bars_to_resolve   INTEGER NOT NULL,
    -- When TP and SL both fit inside the same bar, it is resolved on the 1m series. The RATE of
    -- ambiguity is recorded: above 5% it means the barriers are too tight and the labels are not
    -- describing real trades.
    intrabar_ambiguous INTEGER NOT NULL DEFAULT 0
) STRICT;

-- Positions the user declares having taken. This closes the second half of "entry AND EXIT points":
-- without it the tool emits a plan and never looks at it again.
-- The requirement forbids EXECUTION, not follow-up.
CREATE TABLE IF NOT EXISTS manual_positions (
    id                INTEGER PRIMARY KEY,
    decision_id       INTEGER REFERENCES decisions(id),
    opened_ts_ms      INTEGER NOT NULL,
    symbol            TEXT    NOT NULL,
    direction         INTEGER NOT NULL,
    fill_price        REAL    NOT NULL,
    size              REAL    NOT NULL,
    stop_current      REAL    NOT NULL,
    closed_ts_ms      INTEGER,
    close_price       REAL,
    note              TEXT
) STRICT;

-- Count snapshots for the time scrutineer. `provenance` keeps the live path apart from the offline
-- re-parse: the offline one is NOT causal and must never reach the live reading.
CREATE TABLE IF NOT EXISTS wave_snapshots (
    id                INTEGER PRIMARY KEY,
    ts_ms             INTEGER NOT NULL,
    symbol            TEXT    NOT NULL,
    timeframe         TEXT    NOT NULL,
    count_id          TEXT    NOT NULL,
    rank              INTEGER NOT NULL,
    fit_score         REAL    NOT NULL,
    terminal_label    TEXT    NOT NULL,
    invalidation_price REAL   NOT NULL,
    invalidation_rule TEXT    NOT NULL,
    tentative         INTEGER NOT NULL DEFAULT 0,
    revision_n        INTEGER NOT NULL DEFAULT 0,
    provenance        TEXT    NOT NULL CHECK (provenance IN ('live','offline_reparse')),
    engine_sha        TEXT    NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_snap_asof ON wave_snapshots (symbol, timeframe, ts_ms, provenance);

-- Pivots pinned or forbidden by the user. Hand-constrained counts are marked manual=1 in decisions
-- and stay OUT of the statistics, so they cannot contaminate the honesty apparatus.
CREATE TABLE IF NOT EXISTS user_anchors (
    id                INTEGER PRIMARY KEY,
    symbol            TEXT    NOT NULL,
    timeframe         TEXT    NOT NULL,
    ts_ms             INTEGER NOT NULL,
    price             REAL    NOT NULL,
    kind              TEXT    NOT NULL CHECK (kind IN ('pinned','forbidden')),
    created_ts_ms     INTEGER NOT NULL
) STRICT;
