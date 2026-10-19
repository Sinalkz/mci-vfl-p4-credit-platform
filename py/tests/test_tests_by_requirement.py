"""docs/tests-by-requirement.md is generated from the saved suite output (results/tests/) and the hand-written
mapping; these checks read the saved output only and never rerun the suite."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests_by_requirement as tbr  # noqa: E402

D = tbr.load_all()
DOC = tbr.DOC.read_text(encoding="utf-8")
LOG = tbr.LOG.read_text(encoding="utf-8")
SUMMARY = LOG[LOG.rindex(" passing ("):] if " passing (" in LOG else ""
IN_OUTPUT = [t for t in D["tests"] if t.status != "not run"]


def mocha_total(word: str) -> int:
    m = re.search(rf"^ +(\d+) {word}", LOG, re.M)
    return int(m.group(1)) if m else 0


def test_every_mapped_test_exists_in_the_run_output():
    # Each ref names at least one test. A named test's title is in the output, or (mocha prints nothing for them)
    # it is declared in its file and its suite's before-all hook failed in this run.
    hooks = D["parsed"]["hooks"]
    for r in D["mapping"]["requirements"]:
        for ref in r["tests"] + [x for neg in r["negatives"] for x in neg["refs"]]:
            got = tbr.resolve(ref, D["tests"])
            assert got, f"requirement {r['n']}: {ref} names no test"
            for t in got:
                if t.status == "not run":
                    assert t.note == tbr.HOOK_NOTE, (ref, t.title, t.note)
                    assert t.title in (tbr.TEST_DIR / f"{t.file}.test.ts").read_text(encoding="utf-8")
                    assert any(h["file"] == t.file and h["title"].startswith('"before all" hook') for h in hooks)
                else:
                    assert t.title in LOG, (ref, t.title)


def test_negative_cases_are_among_the_requirements_tests():
    for r in D["reqs"]:
        for _, got in r["negs"]:
            for _, ts in got:
                assert all(any(t is u for u in r["mapped"]) for t in ts)


def test_every_test_in_the_output_appears_in_the_md_mapped_or_unmapped():
    unmapped = DOC.split("## Tests mapped to no requirement", 1)[1].split("## Index", 1)[0]
    for t in IN_OUTPUT:
        reqs = D["serves"][id(t)]
        row = f"| {t.file} | {tbr.md_escape(t.title)} | {tbr.STATUS_MD[t.status]} | {', '.join(map(str, reqs)) or 'none'} |"
        assert row in DOC, row
        if reqs:
            for n in reqs:   # listed in each of its requirements' sections
                sec = DOC.split(f"\n## {n}. ", 1)[1].split("\n## ", 1)[0]
                assert tbr.md_escape(t.title) in sec or (t.id and f"{t.id} (" in sec), (n, t.title)
        else:
            assert f"| {t.file} | {tbr.md_escape(t.title)} |" in unmapped, t.title
            assert id(t) in D["notes"], f"unmapped without a reason: {t.title}"


def test_counts_match_the_output():
    passing, failing, pending = mocha_total("passing"), mocha_total("failing"), mocha_total("pending")
    assert passing > 0 and SUMMARY
    assert LOG.count("✔ ") == passing                    # one check mark per passed test
    assert sum(t.status == "passed" for t in IN_OUTPUT) == passing
    assert sum(t.status == "failed" for t in IN_OUTPUT) + len(D["parsed"]["hooks"]) == failing
    assert sum(t.status == "pending" for t in IN_OUTPUT) == pending
    assert len(re.findall(r"^ +\d+\) ", LOG[:LOG.index(SUMMARY)], re.M)) == failing
    assert f"| Mocha totals | {passing} passing, {failing} failing, {pending} pending |" in DOC
    notrun = sum(t.status == "not run" for t in D["tests"])
    assert f"| Tests in this document | {len(D['tests'])}: {len(IN_OUTPUT)} in the output, {notrun} declared but not run |" in DOC
    for r in D["reqs"]:
        n = len(r["mapped"])
        assert f"| {r['n']} | {r['name']} | {n} | {tbr.status_line(r['mapped'])} | {len(r['negs'])} |" in DOC
        assert f"**Tests: {n}**" in DOC.split(f"\n## {r['n']}. ", 1)[1]
    unm = [t for t in D["tests"] if not D["serves"][id(t)]]
    assert f"| – | mapped to no requirement | {len(unm)} |" in DOC


def test_every_declared_test_is_accounted_for():
    decls, _ = tbr.parse_sources()
    for d in decls:
        pool = [t for t in D["tests"] if t.file == d.file]
        if d.template:
            assert any(tbr.template_regex(d.title).match(t.title) for t in pool), d.title
        else:
            assert sum(t.title == d.title for t in pool) == 1, d.title


def test_md_is_current():
    assert tbr.render() == DOC


SAMPLE = """Compiled 1 Solidity file

Running Mocha tests


  Top
    ✔ T01 ok (12ms)
    1) T02 bad
console noise from a test
    - T03 skipped
    Nested
      ✔ T04 nested
      2) "before all" hook for "T05 never runs"


  2 passing (1s)
  2 failing
  1 pending

  1) Top
       T02 bad:
     AssertionError: expected 1 to equal 2
      at Context.<anonymous> (test/x.ts:1:1)

  2) Top
       Nested
         "before all" hook for "T05 never runs":
     Error: boom
      at Context.<anonymous> (test/x.ts:2:2)
"""


def test_parse_log_reads_passes_failures_hooks_and_pending():
    p = tbr.parse_log(SAMPLE, {"Top": "X", "Nested": "X"})
    got = [(t.title, t.status, t.ms, t.suites) for t in p["tests"]]
    assert got == [("T01 ok", "passed", 12, ["Top"]), ("T02 bad", "failed", None, ["Top"]),
                   ("T03 skipped", "pending", None, ["Top"]), ("T04 nested", "passed", None, ["Top", "Nested"])]
    assert p["tests"][1].error == "AssertionError: expected 1 to equal 2"
    assert p["hooks"] == [{"file": "X", "suites": ["Top", "Nested"], "title": '"before all" hook for "T05 never runs"', "error": "Error: boom"}]
    assert p["summary"] == {"passing": 2, "duration": "1s", "failing": 2, "pending": 1}
    decls = [tbr.Decl("X", ["Top", "Nested"], "T05 never runs", False), tbr.Decl("X", ["Top"], "T01 ok", False)]
    inv = tbr.inventory(decls, p)
    assert [(t.title, t.status, t.note) for t in inv if t.status == "not run"] == [("T05 never runs", "not run", tbr.HOOK_NOTE)]


def test_template_titles_and_ids():
    assert tbr.template_regex("M01 equals the reference on all ${VECTORS.length} vector cases").match(
        "M01 equals the reference on all 33 vector cases")
    ids = [tbr.Test("F", [], t, "passed").id for t in ("U05 x", "D139a x", "S7 x", "(1) x", "accepts x", "U02 n = 2")]
    assert ids == ["U05", "D139a", "S7", "(1)", "", "U02"]


def test_half_up_rounding():
    assert [tbr.fmt(x) for x in (45820.5, -66.5, 4568.5, 2.4999)] == ["45,821", "-67", "4,569", "2"]
    assert tbr.fmt(2.675, 2) == "2.68" and tbr.fmt(0.125, 2) == "0.13" and tbr.fmt(-0.04, 1) == "0.0"


def test_the_tests_once_on_unmerged_branches_ran_and_say_why_they_serve_no_requirement():
    assert not D["mapping"].get("pending_branches")                      # every task branch is merged into main
    ts = [t for t in D["tests"] if (t.file, t.id) in {("Data1", "D13s"), ("LdpConfigs", "L01"), ("LdpConfigs", "L02"), ("LdpConfigs", "L03")}]
    assert len(ts) == 4
    for t in ts:
        assert t.status == "passed" and t.title in LOG and not D["serves"][id(t)] and id(t) in D["notes"], t.title
    assert "## Tests on unmerged branches" not in DOC
