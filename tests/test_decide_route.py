"""`/api/decide` is the route the product is served through, and it had no test.

Three things are checked here, and they fail for three different reasons.

**The wire shape.** `web/app.js` reads the card key by key. A key renamed on either side produces a
blank panel and NO console error — `undefined` interpolates into a template literal as the string
"undefined" and `undefined.map` is the only thing that throws, so half the renames are silent even
in the browser. The required keys are therefore not retyped here: they are PARSED OUT of the real
`web/app.js`, so a rename on the JavaScript side fails this file too. A hand-written list would
only ever prove the list agrees with itself.

**The `Decision` invariants, through the route.** `Decision.__post_init__` refuses an ACTIONABLE
decision with no plan, one emitted while catching up, and one at PRIOR maturity. Each of those is a
specific way of lying to the user, and each is well tested as a unit property. What nothing checked
is that the SERVED path honours them — and it cannot inherit them, because (as
`docs/TEST_COVERAGE.md` records) `/api/decide` returns `engine.decide()`'s plain dict and never
constructs a `Decision` at all. So the card the route just served is lifted into a `Decision` here,
exactly as a caller would have to read it, and the constructor is asked to accept it. The route
passes only because the PRIOR ceiling holds; `test_catching_up_changes_nothing_on_the_card` pins
that this is the ONLY thing holding, so the day maturity rises somebody sees a red test.

**The PUBLIC gate.** With `WAVELAB_PUBLIC` set, `/api/decide`, `/api/history` and `/ws` must stop
answering. That is what keeps the owner's chart off the internet and it was protected by nothing
but a code review. `PUBLIC` is computed at IMPORT time from the environment, so monkeypatching the
module attribute would test the `if` and not the flag: the accepted-spellings list is half the
guarantee (`WAVELAB_PUBLIC=yes` silently reading as false would put the chart on the open internet).
A second, isolated copy of the module is therefore executed with the variable set — 1.4 ms — which
puts the real expression under test.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

# Must be set BEFORE the module is imported: `APP = App()` runs at import time and builds a
# BarStore rooted at WAVELAB_DATA. Pointing it away from the real store keeps a test run from ever
# touching data/. Same value as tests/test_server_status.py, which may get there first.
os.environ.setdefault("WAVELAB_DATA", "/nonexistent-wavelab-test-store")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.conftest import SYMBOL, make_bars
from wavelab.core.timeframes import BY_NAME, TF_1M
from wavelab.core.types import Decision, Direction, MaturityLevel, TradePlan, Verdict
from wavelab.engine.live import LiveEngine, Mode
from wavelab.server import app as srv

TF = "15m"

#: 6,000 synthetic 1m bars — 400 15m bars, 81 pivots — warm up in ~20 ms. Nine years of the real
#: store take two minutes and would put this file's runtime an order of magnitude over the whole
#: suite's budget. Seed 11 is not arbitrary: it is the one that lands the live price INSIDE an
#: entry zone, which is the only state in which the PRIOR ceiling has anything to hold back. On a
#: seed where nothing is in the zone, `best_in_zone` is False and lifting the WATCH cap to
#: ACTIONABLE would not change the answer, so this file's central test would pass against the very
#: mutation it exists to catch.
SEED_WITH_AN_IN_ZONE_ENTRY = 11
N_BARS_1M = 6000


# --------------------------------------------------------------------------------- fixtures


def _engine(n_bars: int, seed: int) -> LiveEngine:
    eng = LiveEngine(SYMBOL, ["1m", TF], 8192, TF)
    eng.warmup(make_bars(n_bars, tf=TF_1M, seed=seed))
    eng.state.health.mode = Mode.LIVE
    eng.state.health.last_closed_ms = eng.state.rings[TF].last_closed_ts_ms
    return eng


@pytest.fixture(scope="module")
def warm_engine() -> LiveEngine:
    """Enough structure for a full card: viable rows, non-viable rows, one of them in the zone."""
    return _engine(N_BARS_1M, SEED_WITH_AN_IN_ZONE_ENTRY)


@pytest.fixture(scope="module")
def bare_engine() -> LiveEngine:
    """Bars, an ATR, and no wave count that survives the hard rules. The empty card."""
    return _engine(400, SEED_WITH_AN_IN_ZONE_ENTRY)


@pytest.fixture
def client(request):
    """`APP` is a module-level singleton; the engine it serves is swapped and put back.

    `TestClient` is deliberately NOT used as a context manager. Entering it runs the lifespan,
    which runs `_start`, which — against an empty store — warms up in no time, sets `ready` and
    then spawns `supervise_feed`, and that opens a WebSocket to Binance. A unit test must not go
    to the internet to answer a question about a JSON key.
    """
    engine = request.getfixturevalue(getattr(request, "param", "warm_engine"))
    before = (srv.APP.engine, srv.PUBLIC)
    srv.APP.engine = engine
    # Pinned rather than assumed: this file's private-mode expectations are the mirror of its
    # public-mode ones, and a WAVELAB_PUBLIC left set in the developer's shell would otherwise turn
    # every one of them red for a reason that has nothing to do with the code.
    srv.PUBLIC = False
    engine.state.health.mode = Mode.LIVE
    yield TestClient(srv.app)
    (srv.APP.engine, srv.PUBLIC) = before


@pytest.fixture
def card(client) -> dict:
    r = client.get(f"/api/decide?tf={TF}")
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------- what web/app.js actually reads

JS = (Path(__file__).resolve().parents[1] / "web" / "app.js").read_text(encoding="utf-8")


def _fn(name: str) -> str:
    """The body of a top-level function in app.js, by brace matching.

    Reading the real file is the point: a list of keys typed into this test would go stale the
    moment somebody renames one in the JavaScript, and would then keep passing while the panel
    renders blank.
    """
    i = JS.index(f"function {name}(")
    j = JS.index("{", i)
    depth = 0
    for k in range(j, len(JS)):
        if JS[k] == "{":
            depth += 1
        elif JS[k] == "}":
            depth -= 1
            if depth == 0:
                return JS[j:k + 1]
    raise AssertionError(f"unbalanced braces reading {name}() out of web/app.js")


def _reads(src: str, var: str) -> set[str]:
    return set(re.findall(rf"\b{var}\.([A-Za-z_][A-Za-z_0-9]*)", src))


def _branch(src: str, marker: str) -> tuple[str, str]:
    """Splits a function body at an `if (…) { … }`, returning (up to and including it, the rest)."""
    i = src.index(marker)
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[:k + 1], src[k + 1:]
    raise AssertionError(f"unbalanced braces splitting on {marker!r}")


_ROW = _fn("hypothesisRow")
_ALWAYS_SRC, _VIABLE_SRC = _branch(_ROW, "if (!h.viable) {")

#: Top-level keys `paintDecision(d)` reads off the card.
CARD_KEYS = _reads(_fn("paintDecision"), "d") - {"error"}
#: Per-hypothesis keys read for EVERY row: the header, and the branch taken when it is not viable.
ROW_KEYS = _reads(_ALWAYS_SRC, "h") | _reads(_fn("paintDecision"), "h")
#: Read only for a viable row — the arithmetic block, and the price lines `drawPlan` draws.
VIABLE_KEYS = (_reads(_VIABLE_SRC, "h") | _reads(_fn("drawPlan"), "h")) - ROW_KEYS
#: The three verdict strings the panel knows how to paint, straight out of app.js's VERDICT map.
VERDICTS = set(re.findall(r"^\s*(\w+): \['v-", JS[JS.index("const VERDICT = {"):], re.MULTILINE))


def test_the_extractor_read_something_real_out_of_app_js():
    """A key list parsed out of a file that moved would be empty, and every shape test below would
    then pass by asserting nothing. This is the only thing standing between those tests and a
    vacuous green, so it is stated first and separately."""
    assert "verdict" in CARD_KEYS and "hypotheses" in CARD_KEYS and "maturity" in CARD_KEYS
    assert "invalidation_price" in ROW_KEYS and "viable" in ROW_KEYS
    assert "entry_lo" in VIABLE_KEYS and "targets" in VIABLE_KEYS
    assert "rr_t2" in VIABLE_KEYS, "the arithmetic block is what makes a rejection arguable"
    assert VERDICTS == {"no_trade", "watch", "actionable"}
    assert "async function decide" not in _ROW, "brace matching ran past the end of the function"


def test_the_card_carries_every_top_level_key_the_panel_reads(card):
    missing = CARD_KEYS - set(card)
    assert not missing, (
        f"web/app.js reads d.{{{', '.join(sorted(missing))}}} off the decision card and the route "
        "does not send it. The panel renders blank with nothing in the console.")


def test_the_verdict_is_one_the_panel_knows_how_to_paint(card):
    """`VERDICT[d.verdict] ?? ['v-no', null]` falls back SILENTLY: an unknown verdict paints the
    raw string in the no-trade colour, so a renamed verdict looks like a rejection."""
    assert card["verdict"] in VERDICTS
    assert isinstance(card["maturity"], int), "t('panel.maturity', {n}) formats it as a number"
    assert isinstance(card["reasons"], list) and card["reasons"]


def test_every_row_carries_the_keys_the_row_renderer_reads(card):
    assert card["hypotheses"], "the fixture must produce rows or this test asserts nothing"
    for h in card["hypotheses"]:
        missing = ROW_KEYS - set(h)
        assert not missing, f"row {h.get('id')} is missing h.{{{', '.join(sorted(missing))}}}"
        assert isinstance(h["reasons"], list), "h.reasons.map() is the one access that throws"
        assert h["direction"] in ("LONG", "SHORT"), "app.js tests h.direction === 'LONG'"


def test_a_viable_row_carries_the_plan_the_chart_draws_its_lines_from(card):
    viable = [h for h in card["hypotheses"] if h["viable"]]
    assert viable, "the fixture must produce a viable row or this test asserts nothing"
    for h in viable:
        missing = VIABLE_KEYS - set(h)
        assert not missing, f"viable row {h['id']} is missing h.{{{', '.join(sorted(missing))}}}"
        assert isinstance(h["targets"], list) and h["targets"], "h.targets.map() draws T1..Tn"
        assert isinstance(h["invalidation_rule"], str) and h["invalidation_rule"], \
            "drawPlan calls .split(' ') on it to title the red line"
    assert any(h["in_zone"] for h in viable), (
        "the seed must put the price inside an entry zone: without that, best_in_zone is False and "
        "the WATCH ceiling has nothing to hold back")


def test_a_timeframe_with_no_data_answers_not_ok_and_says_why(client):
    """app.js branches on `r.ok`, not on the body, so the STATUS is the load-bearing half."""
    r = client.get("/api/decide?tf=nope")
    assert r.status_code == 400
    assert r.json()["error"] == "no data for nope"


# ------------------------------------------------------ the Decision invariants, via the route


def _plan_of(h: dict) -> TradePlan:
    """The row the panel puts at the top, read back as the plan a caller would trade."""
    return TradePlan(
        archetype=h["archetype"], direction=Direction[h["direction"]],
        entry_lo=h["entry_lo"], entry_hi=h["entry_hi"], stop=h["stop"],
        invalidation_price=h["invalidation_price"], invalidation_rule=h["invalidation_rule"],
        targets=tuple(h["targets"]), exit_template_id="from-card", count_id=h["id"],
    )


def _as_decision(card: dict, engine: LiveEngine, **override) -> Decision:
    """Lifts the card the route just served into the type that guards it.

    This is the reading the route does not do for itself: `/api/decide` returns
    `engine.decide()`'s plain dict, so `Decision.__post_init__` never runs on the served path.
    Anything this constructor refuses is something the route must never have put on the wire.
    """
    top = next((h for h in card["hypotheses"] if h["viable"] and h["in_zone"]),
               next((h for h in card["hypotheses"] if h["viable"]), None))
    health = engine.state.health
    kwargs = {
        "ts_ms": health.last_closed_ms or 0, "symbol": SYMBOL, "tf": BY_NAME[TF],
        "verdict": Verdict(card["verdict"]), "maturity": MaturityLevel(card["maturity"]),
        "plan": _plan_of(top) if top else None,
        "reasons": tuple(card["reasons"]),
        "stale": health.lag_bars > 1.0,
        "catching_up": health.mode is Mode.CATCH_UP,
    }
    return Decision(**(kwargs | override))


def test_the_card_the_route_just_served_survives_the_decision_constructor(card, warm_engine):
    """End to end: nothing the route emits is something the last line of defence would refuse."""
    d = _as_decision(card, warm_engine)
    assert d.actionable is False
    assert d.plan is not None, "this card has a viable in-zone row; the lift must carry its plan"


def test_the_route_cannot_emit_an_actionable_card_at_prior_maturity(card):
    """The ceiling, at the route. Price is inside an entry zone and every hard rule is satisfied —
    the one state where an unguarded implementation says ACTIONABLE — and the answer is still
    WATCH, because not one trade has resolved."""
    assert any(h["viable"] and h["in_zone"] for h in card["hypotheses"])
    assert card["maturity"] == int(MaturityLevel.PRIOR)
    assert card["verdict"] == Verdict.WATCH.value, (
        "an in-zone plan at PRIOR maturity is a hand-written expectancy with n=0 behind it. "
        "Marking it ACTIONABLE is the tool telling the user to trade on nothing.")
    assert any("PRIOR" in r for r in card["reasons"]), "and the card has to SAY that is why"


def test_lifting_that_same_card_to_actionable_is_refused_at_prior(card, warm_engine):
    """The invariant itself, exercised on the route's own bytes: this card carries a complete,
    in-zone plan, and the ONLY thing that stops it being tradeable is the maturity level."""
    with pytest.raises(ValueError, match="PRIOR"):
        _as_decision(card, warm_engine, verdict=Verdict.ACTIONABLE)


@pytest.mark.parametrize("client", ["bare_engine"], indirect=True)
def test_a_card_with_no_structure_has_nothing_to_trade(client, bare_engine):
    """The empty card, and the no-plan invariant on it.

    There is an ATR and there are bars; what there is not is a wave count that survives the hard
    rules. An ACTIONABLE verdict here would point at no entry, no stop and no target.
    """
    card = client.get(f"/api/decide?tf={TF}").json()
    assert card["hypotheses"] == []
    assert card["verdict"] == Verdict.NO_TRADE.value
    assert any("no structure" in r for r in card["reasons"]), "a 'no' has to show its arithmetic"
    # Maturity is lifted clear of the PRIOR ceiling on purpose, so the refusal below can only be
    # the missing plan and not the ceiling standing in for it.
    with pytest.raises(ValueError, match="no plan"):
        _as_decision(card, bare_engine,
                     verdict=Verdict.ACTIONABLE, maturity=MaturityLevel.FORWARD)


def test_a_card_served_while_catching_up_cannot_be_read_as_actionable(client, warm_engine):
    """Waking up after an outage, the entry zone on the card describes a price that went past
    forty minutes ago. `catching_up` travels ON the decision for exactly this reason."""
    warm_engine.state.health.mode = Mode.CATCH_UP
    card = client.get(f"/api/decide?tf={TF}").json()

    assert card["verdict"] != Verdict.ACTIONABLE.value
    _as_decision(card, warm_engine)          # as served: still refuses nothing
    with pytest.raises(ValueError, match="CATCH_UP"):
        _as_decision(card, warm_engine,
                     verdict=Verdict.ACTIONABLE, maturity=MaturityLevel.FORWARD)


def test_catching_up_changes_nothing_on_the_card(client, warm_engine):
    """PINNED, NOT ENDORSED — read this before making it green again.

    `LiveEngine.emitting` is documented as the first line of defence and has no caller in
    production: `/api/decide` calls `engine.decide()` in every mode. Measured here: the card served
    while CATCH_UP is byte-identical to the one served while LIVE, entry zones and all. Today that
    is harmless only because the PRIOR ceiling caps every verdict at WATCH, so the stale zone is
    shown as something to watch and never as something to take.

    It stops being harmless the moment maturity rises above PRIOR. Whoever does that will land here
    first, and the fix is to consult `emitting` on the served path — not to relax this assertion.
    """
    warm_engine.state.health.mode = Mode.LIVE
    live = client.get(f"/api/decide?tf={TF}").json()
    warm_engine.state.health.mode = Mode.CATCH_UP
    catching_up = client.get(f"/api/decide?tf={TF}").json()

    assert catching_up == live
    assert live["maturity"] == int(MaturityLevel.PRIOR), (
        "the ceiling is the only thing keeping a stale entry zone off the card; if maturity has "
        "risen, the route has to start suppressing instead")


# ---------------------------------------------------------------------------- the PUBLIC gate


def _isolated(**env) -> object:
    """Executes a SECOND copy of server/app.py with `env` applied, leaving the imported one alone.

    `PUBLIC` is a module constant read from the environment at import time. Setting
    `srv.PUBLIC = True` would exercise the three `if` statements and skip the expression that
    decides what the flag MEANS — and that expression is half the guarantee, because a spelling
    that falls off its list reads as False and silently publishes the owner's chart.
    """
    spec = importlib.util.spec_from_file_location("wavelab.server._app_under_test", srv.__file__)
    mod = importlib.util.module_from_spec(spec)
    old = {k: os.environ.get(k) for k in env}

    def _apply(values: dict[str, str | None]) -> None:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    _apply(env)
    try:
        spec.loader.exec_module(mod)          # ~1.4 ms, and it never touches data/
    finally:
        _apply(old)
    return mod


@pytest.fixture(scope="module")
def public_app():
    return _isolated(WAVELAB_PUBLIC="1")


def test_public_mode_closes_the_owners_chart(public_app):
    """The whole point of the flag. Assay is the product; the chart with the Elliott signals is a
    private tool, and it must not be reachable from the internet by guessing the path."""
    assert public_app.PUBLIC is True
    c = TestClient(public_app.app)
    for path in (f"/api/decide?tf={TF}", "/api/history?tf=15m"):
        r = c.get(path)
        assert r.status_code == 404, f"{path} answered {r.status_code} on a public host"
        assert r.json() == {"error": "not available"}


def test_public_mode_refuses_the_websocket(public_app):
    """/ws cannot answer 404 — there is no HTTP response to put it in — so it is refused before
    `accept()` with a policy-violation close. The browser needs that to be distinguishable: on
    seeing `public` it stops reconnecting, and without the refusal it retries every 2 s forever."""
    c = TestClient(public_app.app)
    with pytest.raises(WebSocketDisconnect) as e, c.websocket_connect("/ws"):
        pass
    assert e.value.code == 1008
    assert public_app.APP.hub.clients == set(), "a refused socket must not be left in the hub"


def test_public_mode_still_serves_the_product_and_says_it_is_public(public_app):
    """The counterweight: a gate that 404s everything would pass the test above and ship nothing.
    `/api/status` reporting `public` is load-bearing — app.js reads it to decide which failure to
    paint and whether to give up reconnecting."""
    c = TestClient(public_app.app)
    assert c.get("/api/status").json()["public"] is True
    assert c.get("/api/rule_help").status_code == 200
    assert c.get("/").status_code == 200, "in public mode Assay is the front page"


def test_the_private_host_still_serves_all_three(client):
    """The other half of the gate, and the half a flipped condition breaks.

    Deliberately not written on top of the `card` fixture: a flipped `if PUBLIC` would trip that
    fixture's own status assertion and this test would ERROR in setup instead of failing on the
    claim it makes. A kill that lands in a fixture is a kill that moves the day the fixture does.
    """
    assert srv.PUBLIC is False
    decide = client.get(f"/api/decide?tf={TF}")
    history = client.get(f"/api/history?tf={TF}&limit=5")
    assert decide.status_code == 200, f"the owner's own host answered {decide.status_code}"
    assert history.status_code == 200, f"the owner's own host answered {history.status_code}"
    assert decide.json()["verdict"] in VERDICTS
    assert history.json()["bars"]

    with client.websocket_connect("/ws") as ws:
        first = json.loads(ws.receive_text())
    assert first["type"] == "health", "the first frame is the health badge, before any bar"
    assert srv.APP.hub.clients == set(), "the socket has to be discarded when it goes away"


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "si", "sí", "YES", "Sí"])
def test_the_flag_reads_both_vocabularies(value):
    """The list is bilingual on purpose and the value arrives from a systemd unit file. A
    `WAVELAB_PUBLIC=yes` that quietly meant "no" would cost the entire point of the flag, and it
    would cost it silently — the service starts fine and serves the private chart to the world."""
    assert _isolated(WAVELAB_PUBLIC=value).PUBLIC is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off"])
def test_anything_else_leaves_the_private_host_private(value):
    assert _isolated(WAVELAB_PUBLIC=value).PUBLIC is False
