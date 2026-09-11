"""The es/ca prose of the hypotheses must not drift away from the Python.

`title`, `rationale` and `prior` are the pre-registration: the text a reader weighs BEFORE deciding
whether to believe a verdict — and, in the case of `title`, before deciding what to run at all. The
record itself is English and lives in the .py files; the Spanish and Catalan versions are data in
`web/hyp.{es,ca}.json`, fetched by the browser only when the reader is not reading in English.

That split has one known weakness, and it is silent: edit a rationale in Python —or register a
hypothesis number 101— and the translated files stay as they were. Nothing breaks. The es/ca reader
simply gets yesterday's argument, or the English one, with no warning anywhere. This module is the
alarm, and it catches BOTH halves: a name with no translation at all, and a translation made from
an English text that has since been reworded.

The second half is what the `en` field is for. Each entry carries a fingerprint of the exact
English it was translated from, so rewording a rationale in Python turns the stale translation red
instead of leaving it to be discovered by a reader who cannot tell. That is the whole point: a
pre-registration whose Spanish version quietly describes a different claim than its English one is
worse than no translation, because it still reads as authoritative.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from wavelab.hypotheses import load_all

WEB = Path(__file__).resolve().parent.parent / "web"
LANGS = ("es", "ca")


@pytest.fixture(scope="module")
def names() -> set[str]:
    reg = load_all()
    assert len(reg) == 100, f"the catalogue is {len(reg)} hypotheses, not the 100 registered"
    return set(reg)


@pytest.fixture(scope="module")
def catalogues() -> dict[str, dict]:
    out = {}
    for lang in LANGS:
        path = WEB / f"hyp.{lang}.json"
        assert path.exists(), f"{path} is missing: the {lang} reader would get English prose"
        out[lang] = json.loads(path.read_text(encoding="utf-8"))
    return out


@pytest.mark.parametrize("lang", LANGS)
def test_every_hypothesis_is_translated(names, catalogues, lang):
    """Names the missing ones. A count tells you there is work; a list tells you what work."""
    missing = sorted(names - set(catalogues[lang]))
    assert not missing, (
        f"{len(missing)} hypotheses have no {lang} pre-registration and will be read in English:\n"
        + "\n".join(f"  - {n}" for n in missing)
    )


@pytest.mark.parametrize("lang", LANGS)
def test_no_translation_without_a_hypothesis(names, catalogues, lang):
    """The other direction: prose for a hypothesis that no longer exists is dead weight, and it
    is also the fingerprint of a rename that only got done on one side."""
    orphans = sorted(set(catalogues[lang]) - names)
    assert not orphans, (
        f"{len(orphans)} entries in web/hyp.{lang}.json match no registered hypothesis "
        f"(renamed? deleted?):\n" + "\n".join(f"  - {n}" for n in orphans)
    )


@pytest.mark.parametrize("lang", LANGS)
def test_every_field_is_present_and_not_empty(names, catalogues, lang):
    """An entry with only `rationale` fails no fetch and shows an empty `prior`: the falsification
    condition —the part that makes this a pre-registration and not an advertisement— vanishes.

    `title` is here for a quieter version of the same thing. It is the only one of the three that
    is read BEFORE anything is chosen, and a missing one does not blank the picker: `hypLabel` in
    web/assay.js falls back to the English title, so that one hypothesis sits in a Spanish list in
    English and nothing anywhere says so. Which is the bug titles were added to fix, restored for
    one row at a time."""
    broken = []
    for name in sorted(names & set(catalogues[lang])):
        entry = catalogues[lang][name]
        for field in ("title", "rationale", "prior"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                broken.append(f"{name}.{field}")
    assert not broken, (
        f"empty or missing text in web/hyp.{lang}.json:\n" + "\n".join(f"  - {b}" for b in broken)
    )


@pytest.mark.parametrize("lang", LANGS)
def test_the_translation_is_not_the_english_text(names, catalogues, lang):
    """A key copied over untranslated to silence this suite would pass every check above. It does
    not pass this one: the file has to differ from what the Python serves."""
    reg = load_all()
    copied = []
    for name in sorted(names & set(catalogues[lang])):
        entry = catalogues[lang][name]
        h = reg[name]
        for field in ("title", "rationale", "prior"):
            if entry.get(field, "").strip() == getattr(h, field).strip():
                copied.append(f"{name}.{field}")
    assert not copied, (
        f"identical to the English source in web/hyp.{lang}.json (untranslated placeholder?):\n"
        + "\n".join(f"  - {c}" for c in copied)
    )


def fingerprint(h) -> str:
    """The English a translation was made from, as 16 hex chars. Rationale and prior are hashed
    together and separated by a NUL: reword either one and the fingerprint moves.

    `title` is deliberately NOT in the hash, and that is a known hole: reword a title in Python and
    the es/ca one goes stale with nothing to say so. Covering it would move all 200 stamps at once,
    and a re-stamp of 200 entries nobody retranslated is exactly the silencing this alarm exists to
    prevent — the staleness it would hide is a whole rationale, against one line of a title. If a
    title fingerprint is ever wanted, it belongs in a SECOND field, stamped on its own.
    """
    return hashlib.sha256(f"{h.rationale}\0{h.prior}".encode()).hexdigest()[:16]


@pytest.mark.parametrize("lang", LANGS)
def test_the_translation_matches_the_english_it_was_made_from(names, catalogues, lang):
    """Staleness, which is the failure a reader cannot detect. Every other check in this module
    passes happily when a rationale is reworded in Python and the translation is left behind \u2014 the
    text is present, non-empty and different from the English, so nothing complains. But the
    Spanish reader is now weighing a claim the project no longer makes.

    Fix, when this fails: retranslate the named fields, then re-stamp `en` with fingerprint(). Do
    not stamp without retranslating; that silences the alarm and keeps the bug."""
    reg = load_all()
    stale, unstamped = [], []
    for name in sorted(names & set(catalogues[lang])):
        stamped = catalogues[lang][name].get("en")
        if not stamped:
            unstamped.append(name)
        elif stamped != fingerprint(reg[name]):
            stale.append(name)
    assert not unstamped, (
        f"no `en` fingerprint in web/hyp.{lang}.json, so staleness cannot be detected for:\n"
        + "\n".join(f"  - {n}" for n in unstamped)
    )
    assert not stale, (
        f"the English moved and web/hyp.{lang}.json did not \u2014 these translations now describe a "
        f"different claim than the pre-registration they belong to:\n"
        + "\n".join(f"  - {n}" for n in stale)
    )


def test_the_files_are_on_the_caddy_allowlist():
    """Public mode serves an explicit path list, and a file that is not on it 404s. The panel then
    falls back to English instead of blanking, which is the worst kind of bug: nothing to see.
    This has already happened once, with /i18n.js."""
    conf = (WEB.parent / "scripts" / "caddy_assay.conf").read_text(encoding="utf-8")
    for lang in LANGS:
        assert f"/hyp.{lang}.json" in conf, (
            f"/hyp.{lang}.json is not in scripts/caddy_assay.conf: in production the {lang} "
            f"reader silently gets the English pre-registration"
        )


# --------------------------------------------------------------- the rule language's own words
#
# The catalogue prose above is the big translation surface, but it is not the only one, and the
# other one drifts more quietly. The API answers in English BY CONTRACT — test titles, import
# reports, refusals — and `tx()` in web/i18n.js translates it on arrival by exact match. A string
# that is not in that file falls through in English: no error, no blank, nothing to notice.
#
# That is how eleven of them shipped. The rule editor's whole error surface was English inside a
# Spanish page, and the only way anyone would have found out is by writing a broken rule.
#
# The vocabulary of the rule language is the part of that surface a Python change can silently
# extend: add a function to SAFE_FUNCS and you add a line of English to the help panel. So it is
# the part worth pinning here — the check maintains itself, because the source of truth is the
# dict the server actually serves.


def _js_text() -> str:
    """web/i18n.js with its wrapped string literals joined back up.

    The dictionaries wrap at 100 columns using `'...' + '...'`, so a literal search for a long
    key finds nothing even when the key is there. Splicing out the concatenation operator between
    two literals — and unescaping `\\'` — puts the strings back the way JavaScript sees them.
    """
    src = (WEB / "i18n.js").read_text(encoding="utf-8")
    src = re.sub(r"'\s*\+\s*'", "", src)
    src = re.sub(r'"\s*\+\s*"', "", src)
    return src.replace("\\'", "'")


def _server_block(lang: str) -> str:
    """Just the `SERVER_<LANG>` object literal. Checking the whole file would let a string
    translated into Spanish only satisfy the Catalan case as well, which is the exact mistake
    this is here to catch."""
    js = _js_text()
    start = js.index(f"const SERVER_{lang.upper()} = {{")
    depth = 0
    for i in range(js.index("{", start), len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return js[start:i + 1]
    raise AssertionError(f"SERVER_{lang.upper()} in web/i18n.js is not brace-balanced")


@pytest.mark.parametrize("lang", LANGS)
def test_the_rule_language_help_is_translated(lang):
    """Every series and function the editor offers is described to the user in English by the
    server. Each of those descriptions has to have an entry in THIS language's table, or the help
    panel opens half in the reader's language and half not."""
    from wavelab.validation.expr import FUNC_DOCS, SERIES_DOCS

    block = _server_block(lang)

    def present(doc: str) -> bool:
        # Most keys are quoted, but a doc that happens to be a bare JS identifier —`volume`— is
        # written without quotes, and JavaScript reads the two forms identically.
        if f"'{doc}'" in block:
            return True
        return bool(re.fullmatch(r"[A-Za-z_$][\w$]*", doc)
                    and re.search(rf"[{{,]\s*{re.escape(doc)}\s*:", block))

    missing = [d for d in [*SERIES_DOCS.values(), *FUNC_DOCS.values()] if not present(d)]
    assert not missing, (
        f"{len(missing)} rule-language descriptions have no entry in SERVER_{lang.upper()} in "
        f"web/i18n.js and will be read in English by every {lang} user:\n"
        + "\n".join(f"  - {d}" for d in missing)
    )


# ------------------------------------------------------------------- the titles, as titles
#
# The two checks above ask whether a title EXISTS and whether it DIFFERS from the English. Neither
# asks whether it says anything. `title` was added because the picker used to show
# `flow.absorption_narrow_range` and a reader had to decode it; a title that is the same identifier
# with the underscores taken out restores that bug while passing every check in this file, in
# `register()` and in the browser. Measured when this was written: one of the hundred was exactly
# that (`seasonality.turn_of_month_long` → "Long the turn of the month").

#: Words that carry no information about the claim, so a title made of these plus the identifier's
#: own words is the identifier read aloud. Deliberately short: the test's job is to catch a title
#: that adds NOTHING, not to grade prose.
_FILLER = frozenset((
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "into", "than", "that", "this", "its", "it", "is", "are", "be",
    "no", "not", "every", "all", "only", "when", "while", "after", "before",
    "over", "under", "out", "up", "down",
))


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^0-9a-z]+", text.lower()) if w]


@pytest.mark.parametrize("lang", ("en", *LANGS))
def test_a_title_is_not_the_identifier_read_aloud(names, catalogues, lang):
    """A title has to add at least one word the identifier does not already contain.

    The identifier is `family.some_words_20`, so "some words 20" is already on the screen next to
    the title — `hypLabel` in web/assay.js prints `title · name`, deliberately, because the
    identifier is the key in trials.sqlite and in the published study and a reader needs it to
    quote. A title built only from those same words plus filler therefore occupies the row and
    tells the reader nothing they could not already read two centimetres to the right.

    Domain words shared with the identifier are fine and unavoidable — "Hammer after an oversold
    fall" for `candles.hammer_oversold` says WHEN, "Flip sides on every Donchian 20 break" for
    `structure.donchian_break_20` says WHAT HAPPENS. What is refused is the case where the
    remainder is empty.

    The es/ca arm is weaker on purpose: the identifier is English, so a Catalan title shares almost
    no words with it and the rule above would be vacuous there. What is checked instead is the
    blunt version — a translator who pasted the key into the field.
    """
    reg = load_all()
    titles = {n: reg[n].title for n in names} if lang == "en" else {
        n: catalogues[lang][n].get("title", "") for n in sorted(names & set(catalogues[lang]))}
    assert titles, f"no {lang} titles were collected; this test would assert nothing"

    empty = []
    for name, title in sorted(titles.items()):
        ident = name.split(".", 1)[1]
        if _words(title) == _words(ident) or _words(title) == _words(name):
            empty.append(f"{name}: {title!r} is the identifier with the punctuation changed")
        elif lang == "en":
            own = set(_words(ident))
            if not [w for w in _words(title) if w not in own and w not in _FILLER]:
                empty.append(f"{name}: {title!r} adds nothing to the identifier beside it")
    assert not empty, (
        f"{len(empty)} {lang} titles restate their own identifier. The picker shows "
        f"`title · name`, so such a row says the same thing twice and the reader is back to "
        f"decoding a key:\n" + "\n".join(f"  - {e}" for e in empty)
    )


# ------------------------------------------------------------ the catalogue, on the wire
#
# `title` in Python and `title` in hyp.{es,ca}.json are two of the three places it has to be
# spelled the same. The third is the JSON `/api/hypotheses` puts on the wire, and it is the one
# with no test: dropping the key from `list_hypotheses` leaves every check above green, and the
# browser degrades quietly — `hypLabel` falls back to the bare identifier, which is the picker
# exactly as it was before this work, with nothing anywhere saying so. That is not hypothetical;
# the comment in `hypLabel` records it happening against a server process older than the field.
#
# The key names are PARSED OUT of web/assay.js rather than retyped, the same way
# test_decide_route.py reads web/app.js: a list typed in here would only prove it agrees with
# itself, and would keep passing after a rename on the JavaScript side.

ASSAY_JS = (WEB / "assay.js").read_text(encoding="utf-8")


def _js_function_body(src: str, name: str) -> str:
    """The body of a top-level JS function, by brace matching."""
    i = src.index(f"function {name}(")
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[j:k + 1]
    raise AssertionError(f"unbalanced braces reading {name}() out of web/assay.js")


#: Every `h.<key>` the picker reads off one entry of `/api/hypotheses`.
PICKER_KEYS = {m for fn in ("hypLabel", "paintCatalogue")
               for m in re.findall(r"\bh\.([A-Za-z_][A-Za-z_0-9]*)", _js_function_body(ASSAY_JS, fn))}


def test_the_extractor_read_something_real_out_of_assay_js():
    """The anti-vacuity guard, and the only place in this section that hard-codes a name. An
    extractor pointed at a function that has been renamed returns an empty set, and the test below
    would then pass while requiring nothing of the payload."""
    assert PICKER_KEYS >= {"name", "title", "family"}, PICKER_KEYS
    assert "hypProse(h.name)?.title" in ASSAY_JS, (
        "the picker no longer prefers the translated title, so the es/ca checks in this file are "
        "about a field nothing displays")


def test_the_catalogue_route_serves_a_title_for_every_hypothesis(names):
    """`/api/hypotheses` is what the picker actually reads; the registry is not.

    This is the one route of the four the Assay side is served through that carries prose, and
    `docs/TEST_COVERAGE.md` lists all four as unexercised. Half of that is now false.
    """
    import os

    # Before the server is imported: `APP = App()` builds a BarStore at import time and a test
    # must never be pointed at the real store. Same value as tests/test_decide_route.py.
    os.environ.setdefault("WAVELAB_DATA", "/nonexistent-wavelab-test-store")
    from fastapi.testclient import TestClient

    from wavelab.server import app as srv

    payload = TestClient(srv.app).get("/api/hypotheses").json()["hypotheses"]
    assert len(payload) == len(names) == 100, f"the route served {len(payload)} of {len(names)}"

    reg = load_all()
    broken = []
    for entry in payload:
        missing = PICKER_KEYS - set(entry)
        if missing:
            broken.append(f"{entry.get('name')}: no {', '.join(sorted(missing))} on the wire")
            continue
        if not str(entry["title"]).strip():
            broken.append(f"{entry['name']}: empty title on the wire")
        elif entry["title"] != reg[entry["name"]].title.strip():
            broken.append(f"{entry['name']}: the wire says {entry['title']!r}, the registry says "
                          f"{reg[entry['name']].title.strip()!r}")
    assert not broken, (
        "web/assay.js reads these off each entry of /api/hypotheses and the route does not send "
        "them. The picker does not break: `hypLabel` degrades to the bare identifier, which is "
        "the screen this whole change existed to replace.\n" + "\n".join(f"  - {b}" for b in broken)
    )
