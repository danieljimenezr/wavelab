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

**What the two served payloads say about the same bar.** `/api/decide` and `/api/history` are
read side by side — the card quotes a price, the chart draws a candle, and a user compares them
without being told to. Nothing checked that they are talking about the same bar, that the chart is
given the window it asked for, or that either of them dates a bar in the unit Lightweight Charts
reads. Each of those is a silent failure: a card priced off the previous bar still renders, a
`limit` that is ignored still draws, and a timestamp in milliseconds paints an empty chart with a
green health badge above it.

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
        "the seed must put the price inside an entry zone, or the fixture never reaches the state "
        "`test_the_route_cannot_emit_an_actionable_card_at_prior_maturity` is about — the one case "
        "where an unguarded implementation would say ACTIONABLE. (`best_in_zone` does not move "
        "the verdict; it moves the reason line. See `LiveEngine.decide`.)")


def test_a_timeframe_with_no_data_answers_not_ok_and_says_why(client):
    """app.js branches on `r.ok`, not on the body, so the STATUS is the load-bearing half."""
    r = client.get("/api/decide?tf=nope")
    assert r.status_code == 400
    assert r.json()["error"] == "no data for nope"


# ------------------------------------------------- the card, the chart, and the bar they share


def test_the_card_is_priced_off_the_bar_the_chart_ends_on(client):
    """The card's price and the last candle drawn are the same bar, or the reader is misled.

    `/api/decide` prices the whole card — the entry zone it compares against, `rr_t2`, `cost_r` —
    off one close it picks out of the ring. Reading one bar further back is a one-character edit
    (`window(1)` -> `window(2)`, whose `close[0]` is then the PREVIOUS bar) and it is invisible:
    every figure stays plausible, every line still draws, and on this fixture the price moves 0.76%
    while the verdict does not move at all. What the user sees is a card quoting a price that is
    not the one on the chart beside it, and an "inside the entry zone" that was decided against a
    bar that closed an hour ago.

    Checked across the two payloads rather than against the ring, because agreeing with each other
    is the property: these are the two numbers a reader puts side by side.
    """
    card = client.get(f"/api/decide?tf={TF}").json()
    bars = client.get(f"/api/history?tf={TF}&limit=1500").json()["bars"]
    assert "price" in card, "the fixture must produce a full card or this test asserts nothing"

    assert card["price"] == round(bars[-1]["close"], 2), (
        f"the card is quoted at {card['price']} and the last candle on the chart closed at "
        f"{bars[-1]['close']} (the one before it closed at {bars[-2]['close']}). The card and the "
        "chart are describing different bars"
    )


def test_history_sends_exactly_the_window_that_was_asked_for(client, warm_engine):
    """`limit` is the browser saying how many candles it wants, and it is a cap in both senses.

    Ignored — `max(limit, len(ring))` instead of `min(...)` — every request ships the entire ring
    instead. At the shipped `ring_capacity` of 8,192 that is a multi-megabyte JSON body answering a
    request for five candles, on a route the page calls on every timeframe switch. Nothing errors:
    the chart draws, slowly.

    The unit is asserted in the same place because it is the other half of "the chart can draw
    this": Lightweight Charts reads unix SECONDS, and a payload in milliseconds is not rejected,
    it is simply drawn 55,000 years in the future — an empty chart under a green health badge.
    """
    ring = warm_engine.state.rings[TF]
    body = client.get(f"/api/history?tf={TF}&limit=5").json()
    assert len(body["bars"]) == 5, (
        f"a request for 5 candles came back with {len(body['bars'])} of the ring's {len(ring)}"
    )

    whole = client.get(f"/api/history?tf={TF}&limit=1500").json()["bars"]
    assert len(whole) == len(ring), (
        "a limit above the ring's own length must not invent bars: it is a ceiling, not a demand"
    )

    assert body["bars"] == whole[-5:], "the window asked for is the most recent one, not the first"
    assert body["bars"][-1]["time"] * 1000 == ring.last_closed_ts_ms, (
        f"the last candle is dated {body['bars'][-1]['time']} for a bar the engine closed at "
        f"{ring.last_closed_ts_ms} ms. Lightweight Charts reads unix seconds"
    )


def test_the_socket_and_the_chart_date_a_bar_the_same_way(client):
    """The websocket path builds its bars through `bar_json`, which no test drove at all.

    `/api/history` writes its own `int(t) // 1000` inline, so the two encoders can drift apart:
    the page then loads a chart that is correct and starts receiving live bars 55,000 years away,
    freezing the candles at the warm-up state while the badge stays green and the socket stays up.
    Read back through the stdlib rather than recomputed, so this cannot agree with the expression
    it is checking.
    """
    from datetime import UTC, datetime

    b = make_bars(1, tf=TF_1M, seed=1)[0]
    t = srv.bar_json(b)["time"]
    assert datetime.fromtimestamp(t, UTC) == datetime.fromtimestamp(b.open_time_ms / 1000, UTC), (
        f"bar_json dated a bar opening at {b.open_time_ms} ms as {t}, which reads as "
        f"{datetime.fromtimestamp(t, UTC) if t < 3e10 else 'a date far outside any chart'}"
    )

    bars = client.get(f"/api/history?tf={TF}&limit=3").json()["bars"]
    assert all(datetime.fromtimestamp(x["time"], UTC).year == 2020 for x in bars), (
        "the two encoders have to agree: the fixture's bars are from 2020 on both paths"
    )


def test_the_chart_is_given_the_legs_for_the_window_it_is_drawing(client, warm_engine):
    """The zigzag is clipped to the same window as the candles, and clipped on the RIGHT key.

    `since_ms` is the first bar being drawn, so it is a vertex time. Slicing on any other instant
    of the pivot — its confirmation, or the far end of the window — silently changes how much
    structure reaches the page. Taken from the wrong end, the legs collapse to the anchor and the
    chart renders as plain candles with no zigzag at all: the one thing the private chart exists
    to show, gone, with a 200 and no console error.
    """
    whole = client.get(f"/api/history?tf={TF}&limit=1500").json()
    assert len(whole["bars"]) == len(warm_engine.state.rings[TF]), (
        "precondition: this request has to cover the WHOLE ring, or 'every pivot is drawn' is not "
        "the right claim to make about it"
    )
    legs = whole["waves"]["legs"]
    drawn = [x for x in legs if not x["tentative"]]
    assert whole["waves"]["n_confirmed"] > 10, "the fixture must have structure to draw"
    assert len(drawn) == whole["waves"]["n_confirmed"], (
        f"{len(drawn)} of the {whole['waves']['n_confirmed']} confirmed pivots reached a chart "
        "whose window is the entire ring: structure is being dropped from the picture"
    )

    narrow = client.get(f"/api/history?tf={TF}&limit=5").json()
    first_bar_ms = narrow["bars"][0]["time"] * 1000
    assert len(narrow["waves"]["legs"]) < len(legs), (
        "a five-candle window must not carry every leg of the ring: that is what stretches the "
        "time axis until the candles are unreadable"
    )
    assert narrow["waves"]["legs"][0]["ts"] <= first_bar_ms, (
        "the first leg has to start at or before the first candle drawn, or the zigzag begins in "
        "mid-air"
    )


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


def test_the_public_front_page_is_assay_and_not_the_private_chart(public_app):
    """The flag's FOURTH surface, and the only one a visitor reaches without knowing a path.

    `/api/decide`, `/api/history` and `/ws` are the three that are gated by a 404; `/` is gated by
    which file it answers with, and a 200 is what both answers look like. Swapping them serves the
    owner's chart shell on the public front page AND takes Assay — the product — off it. The page
    is not even visibly broken: `scripts/caddy_assay.conf` does not publish `/app.js`, so the
    script 404s and the visitor gets a blank dark page with nothing in the console.

    Asserted on which script the page pulls in, because that is the one line that differs between
    the two files and it is the line that decides what the browser then runs.
    """
    body = TestClient(public_app.app).get("/").text
    assert "./assay.js" in body, "the public front page must serve Assay"
    assert "./app.js" not in body, (
        "the public front page is serving the owner's private chart: its script is not published "
        "by the reverse proxy, so the visitor gets a blank page instead of the product"
    )


def test_the_private_front_page_is_still_the_chart(client):
    """The mirror. A gate that served Assay on both hosts would pass the test above and quietly
    take the chart away from the one person it is for."""
    body = client.get("/").text
    assert "./app.js" in body and "./assay.js" not in body


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


class TestTheValidationRoutesSeeTheBarsTheRingHolds:
    """The two routes that turn a ring into a `Series` each do it with five positional arrays of
    the same dtype, and nothing has ever checked the order.

    `Series(tf, w.ts, w.open, w.high, w.low, w.close, w.volume)` and
    `build_series(w.open, w.high, w.low, w.close, w.volume)` are the canonical argument-swap site:
    transpose `high` and `low` and every hypothesis and every user-written rule that mentions a
    wick is evaluated against inverted candles. The battery still runs, all five tests still
    conclude, and the verdict published is a verdict on a strategy nobody wrote. There is no
    exception and no warning: `Bar.__post_init__` validates the UTC grid and nothing asserts
    `high >= low` anywhere between the feed and here.

    Both tests below are answered against the ring itself rather than against a recorded number,
    so neither can agree with the expression it is checking.
    """

    @pytest.fixture
    def ready_client(self, client):
        """The validation routes refuse with 503 until warm-up has finished; the engine handed to
        them here is already warm."""
        before = srv.APP.ready
        srv.APP.ready = True
        yield client
        srv.APP.ready = before

    def test_the_rule_editor_is_given_candles_the_right_way_up(self, ready_client):
        """`high > low` is true of every well-formed candle and of no inverted one, so the route's
        own "not enough bars" refusal is what fires under the swap — a rule the user can see is
        universally true comes back as holding on zero bars out of four hundred."""
        r = ready_client.post("/api/validate_rule", json={"tf": TF, "long": "high > low"})
        assert r.status_code == 200, (
            f"a rule that is true of every candle was refused: {r.status_code} {r.text}. "
            "`high > low` holds on nothing only if the two are being fed in swapped"
        )
        n = int(re.search(r"(\d+) long bars", r.json()["report"][0]).group(1))
        ring = srv.APP.engine.state.rings[TF]
        assert n == len(ring), (
            f"`high > low` held on {n} of the ring's {len(ring)} candles: it holds on all of them "
            "unless the wicks arrived transposed"
        )

        inverted = ready_client.post("/api/validate_rule",
                                     json={"tf": TF, "long": "close > high"})
        assert inverted.status_code == 400 and "0 bars" in inverted.json()["error"], (
            f"`close > high` is true of no candle and came back {inverted.status_code}: "
            f"{inverted.text}"
        )

    def test_the_catalogue_route_evaluates_the_hypothesis_on_the_ring_it_was_given(
            self, ready_client):
        """The oracle is the hypothesis itself, run over a `Series` this test builds from the same
        ring in the documented order. A route that assembles those five arrays differently — or
        over a different slice of the ring — disagrees with it."""
        from wavelab.hypotheses import load_all
        from wavelab.hypotheses.base import Series

        name = "candles.outsized_wick"
        h = load_all()[name]
        ring = srv.APP.engine.state.rings[TF]
        w = ring.window(len(ring))
        expected = int((h.signals(
            Series(TF, w.ts, w.open, w.high, w.low, w.close, w.volume)) != 0).sum())
        assert expected > 0, (
            f"{name} fires on none of these {len(ring)} candles, so this test would pass against "
            "any argument order at all"
        )

        r = ready_client.get(f"/api/validate?hyp={name}&tf={TF}")
        assert r.status_code == 200, r.text
        assert r.json()["n_signals"] == expected, (
            f"the route evaluated {name} on {r.json()['n_signals']} candles where the same "
            f"hypothesis over the same ring fires on {expected}: the series the route builds is "
            "not the one the ring holds"
        )
