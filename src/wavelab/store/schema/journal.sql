-- El journal: la única evidencia no contaminada que este proyecto va a tener nunca.
--
-- Vive en su propio fichero SQLite (WAL) y NO comparte destino con el almacén de velas:
-- las velas se pueden volver a descargar, esto no. Acumula ~60 señales al año, así que dos años de
-- registro son irreemplazables e irreconstruibles — los pesos, la configuración y el código habrán
-- derivado, de modo que no se puede re-derivar qué habría dicho el sistema.
--
-- Cada fila lleva estampados engine_sha, config_hash y weights_version para que la evidencia sea
-- AUTODESCRIPTIVA: dentro de un año, saber qué versión produjo una señal no puede depender de la
-- memoria de nadie ni del historial de git.

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
    -- El producto. Se guarda junto al NOMBRE de la regla que lo produjo para que no pueda
    -- desviarse de su justificación.
    invalidation_price REAL,
    invalidation_rule  TEXT,
    targets_json      TEXT,
    exit_template_id  TEXT,
    count_id          TEXT,
    rr_at_t2          REAL,
    cost_r            REAL,
    p_required        REAL,
    ev_r_lo           REAL,      -- límite inferior del IC bootstrap; NULL si no es computable
    n_cell            INTEGER,   -- n del que sale ev_r_lo: nunca un número sin su n
    stale             INTEGER NOT NULL DEFAULT 0,
    catching_up       INTEGER NOT NULL DEFAULT 0,
    manual            INTEGER NOT NULL DEFAULT 0,  -- conteo restringido a mano: fuera de estadística
    reasons_json      TEXT,
    engine_sha        TEXT    NOT NULL,
    config_hash       TEXT    NOT NULL,
    weights_version   TEXT    NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_decisions_ts     ON decisions (symbol, timeframe, ts_ms);
CREATE INDEX IF NOT EXISTS ix_decisions_arch   ON decisions (archetype, verdict);

-- Cada señal emitida escribe TAMBIÉN una fila de brazo nulo con la misma ExitTemplate.
-- El delta emparejado contra un brazo aleatorio bajo reglas de salida idénticas es interpretable
-- mucho antes que la tasa de acierto absoluta.
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
    -- Si en la vela caben TP y SL a la vez, se resuelve sobre la serie de 1m. La TASA de ambigüedad
    -- se registra: por encima del 5% significa que las barreras son demasiado estrechas y las
    -- etiquetas no describen operaciones reales.
    intrabar_ambiguous INTEGER NOT NULL DEFAULT 0
) STRICT;

-- Posiciones que el usuario declara haber tomado. Cierra la segunda mitad de "puntos de entrada
-- y de SALIDA": sin esto la herramienta emite un plan y no vuelve a mirarlo.
-- El requisito prohíbe EJECUCIÓN, no seguimiento.
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

-- Instantáneas de conteo para el escrutador de tiempo. `provenance` separa la ruta viva del
-- re-parseo offline: el offline NO es causal y jamás puede llegar a la lectura en vivo.
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

-- Pivotes fijados o prohibidos por el usuario. Los conteos restringidos a mano se marcan
-- manual=1 en decisions y quedan FUERA de la estadística, para no contaminar el aparato de honestidad.
CREATE TABLE IF NOT EXISTS user_anchors (
    id                INTEGER PRIMARY KEY,
    symbol            TEXT    NOT NULL,
    timeframe         TEXT    NOT NULL,
    ts_ms             INTEGER NOT NULL,
    price             REAL    NOT NULL,
    kind              TEXT    NOT NULL CHECK (kind IN ('pinned','forbidden')),
    created_ts_ms     INTEGER NOT NULL
) STRICT;
