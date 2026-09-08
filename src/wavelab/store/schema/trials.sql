-- El registro de ensayos. Se construye el DÍA 1 aunque el Deflated Sharpe que lo consume llegue
-- mucho después, porque NO se puede reconstruir hacia atrás.
--
-- La política de "no buscamos parámetros" no elimina la búsqueda: la mueve a ajuste a ojo sin
-- registrar. Hay ~60 constantes elegidas a mano en este proyecto (k, min_pct, umbrales de ER,
-- penalización MDL, las lambdas de guías, los pesos de la mezcla de Fibonacci, eta, alfa, los topes,
-- toda la tabla de prior). Si se ajustan mirando gráficos sin dejar rastro, effective_n contará ~1
-- mientras la carga real de contraste múltiple está en los cientos, y el DSR saldrá
-- ANTI-CONSERVADOR: la maquinaria de honestidad mentiría justo en la dirección de la que dice
-- proteger.
--
-- Por eso `effective_n` cuenta config_hash DISTINTOS jamás evaluados contra el set de fixtures,
-- no solo ensayos programáticos. Y CI rechaza un cambio de configuración sin su fila aquí.

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS trials (
    id                INTEGER PRIMARY KEY,
    ts_ms             INTEGER NOT NULL,
    config_hash       TEXT    NOT NULL,
    git_sha           TEXT    NOT NULL,
    dirty             INTEGER NOT NULL DEFAULT 0,   -- árbol de trabajo sucio al evaluar
    kind              TEXT    NOT NULL CHECK (kind IN ('startup','fixture_eval','walkforward','manual')),
    fixture_set       TEXT,
    metric            TEXT,
    value             REAL,
    note              TEXT
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_trials_hash_kind ON trials (config_hash, kind, fixture_set);
CREATE INDEX IF NOT EXISTS ix_trials_ts ON trials (ts_ms);

-- Vista que consume el Deflated Sharpe. Deliberadamente NO filtra por `kind`: un hash de
-- configuración evaluado "solo para mirar" sigue siendo un ensayo.
CREATE VIEW IF NOT EXISTS effective_n AS
    SELECT COUNT(DISTINCT config_hash) AS n FROM trials;
