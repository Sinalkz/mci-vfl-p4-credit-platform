"""Contract test suite mapped to Chapter 4's ten requirements.

Two steps, each rerunnable:

  run     runs the contract suite once (`npx hardhat test mocha` in contracts/),
          saves its full output to results/tests/hardhat-test.txt and the run's
          metadata (commit, machine, Node/Hardhat/mocha/solc versions, wall time,
          exit code) to results/tests/run-meta.json.
  render  parses the saved output, joins it with the hand-written mapping in
          results/tests/requirements-map.json and writes docs/tests-by-requirement.md.

Hardhat 3's `test mocha` task takes no reporter flag (the reporter can only be set
in hardhat.config.ts, which this task does not change), so the saved output is
mocha's default spec reporter. Spec output does not name the test file; each
suite title is matched to the file whose describe(...) carries it.

Every displayed non-integer is rounded half up (decimal.ROUND_HALF_UP).

Usage:  .venv/Scripts/python py/tests_by_requirement.py run [--label server]   (15-60 min)
        .venv/Scripts/python py/tests_by_requirement.py render
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
TEST_DIR = CONTRACTS / "test"
OUT_DIR = ROOT / "results" / "tests"
LOG = OUT_DIR / "hardhat-test.txt"
META = OUT_DIR / "run-meta.json"
MAP = OUT_DIR / "requirements-map.json"
DOC = ROOT / "docs" / "tests-by-requirement.md"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_info  # noqa: E402

MIN_FREE_GB = 2.0


# --------------------------------------------------------------------------- run

def half_up(x, nd: int = 0) -> Decimal:
    """x rounded to nd decimals, halves away from zero; a float is taken at its shortest repr."""
    d = x if isinstance(x, Decimal) else Decimal(repr(x) if isinstance(x, float) else str(x))
    q = d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return q.copy_abs() if q.is_zero() else q


def fmt(x, nd: int = 0) -> str:
    """Thousands-separated, nd decimals, rounded half up."""
    return f"{half_up(x, nd):,.{nd}f}"


def meminfo() -> dict[str, int]:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0]) * 1024
    return out


def free_ram_gb() -> float | None:
    if platform.system() == "Linux":
        return float(half_up(meminfo()["MemAvailable"] / 2**30, 2))
    if platform.system() != "Windows":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return float(half_up(st.ullAvailPhys / 2**30, 2))


def load_average() -> str | None:
    """The 1, 5 and 15 minute load averages (Unix only)."""
    if not hasattr(os, "getloadavg"):
        return None
    return " / ".join(str(half_up(x, 2)) for x in os.getloadavg())


def machine() -> dict[str, str]:
    if platform.system() != "Linux":
        return env_info.hardware()
    cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")), platform.processor())
    return {"OS": f"{platform.system()} {platform.release()}",
            "CPU": f"{cpu} ({os.cpu_count()} logical CPUs)",
            "RAM": f"{half_up(meminfo()['MemTotal'] / 2**30, 1)} GB"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def solc_versions() -> dict[str, str]:
    cfg = (CONTRACTS / "hardhat.config.ts").read_text(encoding="utf-8")
    m = re.search(r'version:\s*"([0-9.]+)"', cfg)
    long_versions = set()
    for f in (CONTRACTS / "artifacts" / "build-info").glob("*.json"):
        if f.name.endswith(".output.json"):
            continue
        try:
            v = json.loads(f.read_text(encoding="utf-8")).get("solcLongVersion")
        except (OSError, json.JSONDecodeError):
            continue
        if v:
            long_versions.add(v)
    return {"configured": m.group(1) if m else "unknown",
            "build-info solcLongVersion": ", ".join(sorted(long_versions)) or "not found"}


def cmd_run(args: argparse.Namespace) -> None:
    free = free_ram_gb()
    if free is not None and free < MIN_FREE_GB and not args.force:
        sys.exit(f"free RAM {free} GB < {MIN_FREE_GB} GB; rerun with --force to start anyway")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npx = shutil.which("npx")
    if npx is None:
        sys.exit("npx not found on PATH")
    command = ["npx", "hardhat", "test", "mocha"]
    meta = {
        "label": args.label,
        "command": " ".join(command),
        "cwd": "contracts/",
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_changes_at_start": git("status", "--porcelain", "--untracked-files=no").splitlines(),
        "machine": machine(),
        "free_ram_gb_at_start": free,
        "load_average_at_start": load_average(),
        "tools": {
            "Node.js": env_info.run(["node", "--version"]),
            "Hardhat": env_info.npm_pkg_version("contracts", "hardhat"),
            "@nomicfoundation/hardhat-mocha": env_info.npm_pkg_version("contracts", "@nomicfoundation/hardhat-mocha"),
            "mocha": env_info.npm_pkg_version("contracts", "mocha"),
            "ethers": env_info.npm_pkg_version("contracts", "ethers"),
            "snarkjs (circuits/)": env_info.npm_pkg_version("circuits", "snarkjs"),
            "Python": sys.version.split()[0],
        },
        "note": args.note,
        "started": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    t0 = time.monotonic()
    with open(LOG, "wb") as fh:
        proc = subprocess.run([npx, *command[1:]], cwd=CONTRACTS, stdout=fh, stderr=subprocess.STDOUT)
    meta["finished"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    meta["wall_seconds"] = float(half_up(time.monotonic() - t0, 1))
    meta["free_ram_gb_at_end"] = free_ram_gb()
    meta["load_average_at_end"] = load_average()
    meta["exit_code"] = proc.returncode
    meta["tools"]["solc"] = solc_versions()
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"exit {proc.returncode} in {meta['wall_seconds']} s; output in {LOG.relative_to(ROOT)}")


# ------------------------------------------------------------------------- parse

ID_RE = re.compile(r"^(?:[A-Z][A-Z0-9]*\d[a-z]?|\(\d+\))$")       # U05, D139a, S7, Q5, (1)
DECL_RE = re.compile(r"""^(\s*)(describe|it)(?:\.(?:skip|only))?\(\s*(["'`])(.*?)(?<!\\)\3""")
PASS_RE = re.compile(r"^( *)[✔√] (.*?)(?: \((\d+)ms\))?$")
FAIL_RE = re.compile(r"^( *)(\d+)\) (.*)$")
PEND_RE = re.compile(r"^( *)- (.*)$")
SUMMARY_RE = re.compile(r"^ +(\d+) (passing|failing|pending)(?: \((.+)\))?$")
HOOK_RE = re.compile(r'^"(before|after) (all|each)" hook')
HOOK_NOTE = "not run: its suite's before-all hook failed"


@dataclass
class Decl:
    """An it(...) declared in a test file."""
    file: str            # file stem: Federation for contracts/test/Federation.test.ts
    suites: list[str]    # the describe path
    title: str           # the literal title, or the template text
    template: bool       # a `...${x}...` title, expanded at run time


@dataclass
class Test:
    file: str
    suites: list[str]
    title: str
    status: str          # passed | failed | pending | not run
    ms: int | None = None
    error: str = ""
    note: str = ""

    @property
    def id(self) -> str:
        tok = self.title.split(" ", 1)[0]
        return tok if ID_RE.match(tok) else ""


def parse_sources(test_dir: Path = TEST_DIR) -> tuple[list[Decl], dict[str, str]]:
    """Every it(...) in contracts/test/*.test.ts, and each describe title's file."""
    decls: list[Decl] = []
    describes: dict[str, str] = {}
    for f in sorted(test_dir.glob("*.test.ts")):
        stem = f.name.removesuffix(".test.ts")
        stack: list[tuple[int, str]] = []
        for line in f.read_text(encoding="utf-8").splitlines():
            m = DECL_RE.match(line)
            if not m:
                continue
            indent, kind, quote, title = len(m.group(1)), m.group(2), m.group(3), m.group(4)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            if kind == "describe":
                if title in describes and describes[title] != stem:
                    raise ValueError(f"describe title in two files: {title!r}")
                describes[title] = stem
                stack.append((indent, title))
            else:
                decls.append(Decl(stem, [t for _, t in stack], title, quote == "`" and "${" in title))
    return decls, describes


def parse_log(text: str, describes: dict[str, str]) -> dict:
    """Tests, hook failures and the summary from mocha's spec output."""
    lines = text.splitlines()
    summary: dict = {}
    first = None
    for i, line in enumerate(lines):
        m = SUMMARY_RE.match(line)
        if m and (first is not None or m.group(2) == "passing"):
            first = i if first is None else first
            summary[m.group(2)] = int(m.group(1))
            if m.group(2) == "passing":
                summary["duration"] = m.group(3)
    body = lines if first is None else lines[:first]
    epilogue = [] if first is None else lines[first:]

    tests: list[Test] = []
    hooks: list[dict] = []
    numbered: dict[int, object] = {}
    stack: list[tuple[int, str]] = []

    def where(indent: int) -> tuple[str, list[str]]:
        suites = [s for ind, s in stack if ind < indent]
        if not suites:
            raise ValueError(f"a test line outside any known suite (indent {indent})")
        return describes[suites[0]], suites

    for line in body:
        indent = len(line) - len(line.lstrip(" "))
        if m := PASS_RE.match(line):
            f, suites = where(indent)
            tests.append(Test(f, suites, m.group(2), "passed", int(m.group(3)) if m.group(3) else None))
        elif m := FAIL_RE.match(line):
            f, suites = where(indent)
            if HOOK_RE.match(m.group(3)):
                h = {"file": f, "suites": suites, "title": m.group(3), "error": ""}
                hooks.append(h)
                numbered[int(m.group(2))] = h
            else:
                t = Test(f, suites, m.group(3), "failed")
                tests.append(t)
                numbered[int(m.group(2))] = t
        elif (m := PEND_RE.match(line)) and stack:
            f, suites = where(indent)
            tests.append(Test(f, suites, m.group(2), "pending"))
        elif line.strip() in describes and indent >= 2:
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, line.strip()))

    # epilogue: "  1) suite / ... / title:" then the error message, then the stack
    blocks: dict[int, list[str]] = {}
    cur = None
    for line in epilogue:
        if m := re.match(r"^  (\d+)\) (.*)$", line):
            cur = int(m.group(1))
            blocks[cur] = [m.group(2)]
        elif cur is not None:
            blocks[cur].append(line)
    for n, blines in blocks.items():
        err: list[str] = []
        title_done = False
        for bl in blines:
            if not title_done:
                title_done = bl.rstrip().endswith(":")
                continue
            s = bl.strip()
            if not s:
                if err:
                    break
                continue
            if s.startswith("at "):
                break
            err.append(s)
            if len(err) == 3:
                break
        target = numbered.get(n)
        if isinstance(target, Test):
            target.error = " ".join(err)[:400]
        elif isinstance(target, dict):
            target["error"] = " ".join(err)[:400]
    return {"tests": tests, "hooks": hooks, "summary": summary}


def template_regex(title: str) -> re.Pattern:
    return re.compile("^" + ".+".join(re.escape(p) for p in re.split(r"\$\{[^}]*\}", title)) + "$")


def inventory(decls: list[Decl], parsed: dict) -> list[Test]:
    """The run's tests, plus every declared test that does not appear in the output (status 'not run')."""
    out = list(parsed["tests"])
    before_hooks = [h for h in parsed["hooks"] if h["title"].startswith('"before')]
    for d in decls:
        pool = [t for t in parsed["tests"] if t.file == d.file]
        rx = template_regex(d.title) if d.template else None
        if any((rx.match(t.title) if rx else t.title == d.title) for t in pool):
            continue
        hooked = any(h["file"] == d.file and d.suites[:len(h["suites"])] == h["suites"] for h in before_hooks)
        out.append(Test(d.file, d.suites, d.title, "not run", note=HOOK_NOTE if hooked else "not run: not in the output"))
    order = {f: i for i, f in enumerate(dict.fromkeys(d.file for d in decls))}
    return sorted(out, key=lambda t: (order.get(t.file, len(order)), t.status == "not run"))


def resolve(ref: str, tests: list[Test]) -> list[Test]:
    file, sel = ref.split(":", 1)
    pool = [t for t in tests if t.file == file]
    if sel == "*":
        return pool
    if sel.startswith("~"):
        return [t for t in pool if t.title.startswith(sel[1:])]
    return [t for t in pool if t.id == sel]


def load_all() -> dict:
    decls, describes = parse_sources()
    parsed = parse_log(LOG.read_text(encoding="utf-8"), describes)
    tests = inventory(decls, parsed)
    mapping = json.loads(MAP.read_text(encoding="utf-8"))
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    reqs = []
    for r in mapping["requirements"]:
        mapped: list[Test] = []
        for ref in r["tests"]:
            got = resolve(ref, tests)
            if not got:
                raise ValueError(f"requirement {r['n']}: {ref} matches no test")
            mapped += [t for t in got if all(t is not u for u in mapped)]
        negs = []
        for neg in r["negatives"]:
            got = []
            for ref in neg["refs"]:
                g = resolve(ref, tests)
                if not g:
                    raise ValueError(f"requirement {r['n']}: negative ref {ref} matches no test")
                if any(all(t is not u for u in mapped) for t in g):
                    raise ValueError(f"requirement {r['n']}: negative ref {ref} is not among its tests")
                got.append((ref, g))
            negs.append((neg["case"], got))
        reqs.append({**r, "mapped": mapped, "negs": negs})
    serves = {id(t): [r["n"] for r in reqs if any(t is u for u in r["mapped"])] for t in tests}
    notes: dict[int, str] = {}
    for ref, note in mapping.get("unmapped_notes", {}).items():
        for t in resolve(ref, tests):
            if serves[id(t)]:
                raise ValueError(f"unmapped note {ref} names a mapped test: {t.title}")
            notes[id(t)] = note
    return {"tests": tests, "parsed": parsed, "reqs": reqs, "serves": serves, "notes": notes,
            "meta": meta, "mapping": mapping}


# ------------------------------------------------------------------------ render

def md_code(s: str) -> str:
    """A table cell's error message as a code span."""
    return "`" + s.replace("`", "'").replace("|", "\\|") + "`" if s else "?"


def md_escape(s: str) -> str:
    return (s.replace("\\", "\\\\").replace("|", "\\|").replace("*", "\\*")
            .replace("<", "&lt;").replace(">", "&gt;"))


def natural(s: str) -> list:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]


STATUS_MD = {"passed": "passed", "failed": "**FAILED**", "pending": "pending", "not run": "not run"}


def status_line(ts: list[Test]) -> str:
    if len(ts) == 1:
        return STATUS_MD[ts[0].status]
    c = {s: [t for t in ts if t.status == s] for s in STATUS_MD}
    if len(c["passed"]) == len(ts):
        return f"all {len(ts)} passed"
    parts = [f"{len(c['passed'])} passed"]
    for s in ("failed", "pending", "not run"):
        if c[s]:
            files = dict.fromkeys(t.file for t in c[s])
            names = [f + " " + ", ".join(sorted({t.id or md_escape(t.title[:40]) for t in c[s] if t.file == f}, key=natural))
                     for f in files]
            parts.append(f"{len(c[s])} {s} ({'; '.join(names)})")
    return ", ".join(parts)


def ref_label(ref: str, got: list[Test]) -> str:
    file, sel = ref.split(":", 1)
    if sel == "*":
        return f"{file} (all {len(got)})"
    if sel.startswith("~"):
        return f"{file} “{md_escape(sel[1:])}…”"
    return f"{file} {sel}" + (f" ({len(got)} cases)" if len(got) > 1 else "")


def group_rows(ts: list[Test]) -> list[tuple[str, str, str, list[Test]]]:
    """(file, label, status, tests): an ID shared by more than three tests (U02) is one row."""
    rows, seen = [], set()
    for t in ts:
        if id(t) in seen:
            continue
        grp = [u for u in ts if u.file == t.file and t.id and u.id == t.id]
        if len(grp) > 3:
            seen.update(id(u) for u in grp)
            head = t.suites[-1] if len(t.suites) > 1 else t.title
            head = head[len(t.id):].lstrip() if head.startswith(t.id + " ") else head
            label = f"{t.id} ({len(grp)} cases): {md_escape(head)}"
            rows.append((t.file, label, status_line(grp), grp))
        else:
            seen.add(id(t))
            rows.append((t.file, md_escape(t.title), STATUS_MD[t.status], [t]))
    return rows


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def gas_table() -> list[str]:
    out = ["| On-chain cost | Varied | Gas | Source |", "|---|---|---|---|"]
    v = json.loads((ROOT / "results" / "m1b" / "verify_gas.json").read_text(encoding="utf-8"))
    pr = v["proofs"]
    out.append("| verifyProof, M1b verifier (N = 64, maxDepth 10) | data size: "
               + " vs ".join(f"\\|I\\| = {p['sizeI']:,}" for p in pr) + " | "
               + " vs ".join(f"{p['verifyProofGasUsed']:,}" for p in pr)
               + " (calldata " + " vs ".join(f"{p['calldataBytes']} B" for p in pr) + ") | `results/m1b/verify_gas.json` |")
    m3 = read_csv(ROOT / "results" / "m3" / "gas.csv")
    cf = [r for r in m3 if r["function"] == "commitFinal"]
    out.append(f"| commitFinal (a 32-byte root, whatever the matrix size) | per passive | "
               + ", ".join(f"{int(r['gas']):,}" for r in cf) + " | `results/m3/gas.csv` |")
    m1 = read_csv(ROOT / "results" / "gas-m1.csv")
    for fn in ("commitAlignment", "register", "create", "happy_path_total"):
        pts = sorted((int(r["n"]), int(r["gas"])) for r in m1 if r["function"] == fn)
        ns, gs = [p[0] for p in pts], [p[1] for p in pts]
        k = len(pts)
        mx, my = sum(ns) / k, sum(gs) / k
        slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in ns)
        resid = max(abs(y - (my + slope * (x - mx))) for x, y in pts)
        val = (f"{gs[0]:,} (every n)" if len(set(gs)) == 1
               else f"{gs[0]:,} → {gs[-1]:,}; {fmt(slope)} per extra passive (least squares), largest residual {fmt(resid)}")
        out.append(f"| {fn} | members: n = {ns[0]}..{ns[-1]} | {val} | `results/gas-m1.csv` |")
    return out


def render() -> str:
    d = load_all()
    tests, parsed, reqs, serves, notes, meta, mapping = (d[k] for k in
        ("tests", "parsed", "reqs", "serves", "notes", "meta", "mapping"))
    summ = parsed["summary"]
    ran = [t for t in tests if t.status != "not run"]
    notrun = [t for t in tests if t.status == "not run"]
    failed = [t for t in tests if t.status == "failed"]
    L: list[str] = []
    w = L.append
    w("# Contract tests by requirement")
    w("")
    w("Generated by `py/tests_by_requirement.py render` from the saved suite output `results/tests/hardhat-test.txt`, "
      "its metadata `results/tests/run-meta.json` and the hand-written mapping `results/tests/requirements-map.json`; "
      "do not edit by hand. The suite is `contracts/test/*.test.ts`; `py/tests_by_requirement.py run` reruns it. "
      "Requirement names are Chapter 4's; the mechanism lines follow `docs/decision-log.md` (D77), `NAMES-MAP.md` "
      "and the M1–M4 briefs. A test may serve several requirements (column “also”).")
    w("")
    w("## The run")
    w("")
    tool = meta.get("tools", {})
    solc = tool.get("solc", {})
    w("| Item | Value |")
    w("|---|---|")
    w(f"| Command | `{meta.get('command', '?')}` in `{meta.get('cwd', 'contracts/')}` |")
    w(f"| Commit | `{meta.get('commit', '?')}` (branch {meta.get('branch', '?')}; tracked changes at start: "
      f"{len(meta.get('tracked_changes_at_start', [])) or 'none'}) |")
    w(f"| Started / finished | {meta.get('started', '?')} / {meta.get('finished', '?')} |")
    w(f"| Where | {meta.get('label', 'laptop')} |")
    wall = meta.get("wall_seconds")
    w(f"| Duration | mocha: {summ.get('duration', '?')}; wall clock incl. compile: {fmt(wall, 1) if wall is not None else '?'} s |")
    w(f"| Exit code | {meta.get('exit_code', '?')} |")
    w(f"| Mocha totals | {summ.get('passing', 0)} passing, {summ.get('failing', 0)} failing, {summ.get('pending', 0)} pending |")
    w(f"| Tests in this document | {len(tests)}: {len(ran)} in the output, {len(notrun)} declared but not run |")
    for k, v in meta.get("machine", {}).items():
        w(f"| {k} | {v} |")
    for k, v in tool.items():
        if k != "solc":
            w(f"| {k} | {v} |")
    if solc:
        w(f"| solc | {solc.get('configured', '?')} configured; build-info: {solc.get('build-info solcLongVersion', '?')} |")
    def gb(k: str) -> str:
        return f"{fmt(meta[k], 2)} GB" if meta.get(k) is not None else "?"
    w(f"| Free RAM (available) at start / end | {gb('free_ram_gb_at_start')} / {gb('free_ram_gb_at_end')} |")
    if meta.get("load_average_at_start"):
        w(f"| Load average (1 / 5 / 15 min) at start; at end | {meta['load_average_at_start']}; {meta.get('load_average_at_end', '?')} |")
    if meta.get("note"):
        w(f"| Note | {meta['note']} |")
    w("")
    if failed or parsed["hooks"]:
        w("**Failures** (mocha counts a failed hook as one failure):")
        w("")
        w("| File | Test or hook | Error |")
        w("|---|---|---|")
        for t in failed:
            w(f"| {t.file} | {md_escape(t.title)} | {md_code(t.error)} |")
        for h in parsed["hooks"]:
            w(f"| {h['file']} | {md_escape(h['title'])} | {md_code(h['error'])} |")
        w("")
    if notrun:
        by_file = {f: [t for t in notrun if t.file == f] for f in dict.fromkeys(t.file for t in notrun)}
        w(f"**Not run** ({len(notrun)}): " + "; ".join(
            f"{f} " + ", ".join(sorted((t.id or md_escape(t.title) for t in ts), key=natural)) for f, ts in by_file.items())
          + ". " + ("They are declared in the source but absent from the output; "
                    + ("their suite's before-all hook failed." if all(t.note == HOOK_NOTE for t in notrun) else "see the index.")))
        w("")
    rn = mapping.get("run_notes")
    if rn:
        stale = rn.get("run_started") != meta.get("started")
        w("**Notes on this run** (hand-written" + (", for ANOTHER run: check them" if stale else "") + "):")
        w("")
        for n in rn["notes"]:
            w(f"- {n}")
        w("")

    w("## Summary")
    w("")
    w("| # | Requirement | Tests | Status | Negative cases | Basis |")
    w("|---|---|---|---|---|---|")
    for r in reqs:
        w(f"| {r['n']} | {r['name']} | {len(r['mapped'])} | {status_line(r['mapped'])} | {len(r['negs'])} | {r.get('basis', '')} |")
    unm = [t for t in tests if not serves[id(t)]]
    w(f"| – | mapped to no requirement | {len(unm)} | {status_line(unm) if unm else '–'} | – | – |")
    w("")
    multi = [t for t in tests if len(serves[id(t)]) > 1]
    w(f"{len(multi)} of the {len(tests)} tests serve two or more requirements (column “also” in each section); "
      f"{len(tests) - len(unm) - len(multi)} serve one; {len(unm)} serve none.")
    w("")

    for r in reqs:
        w(f"## {r['n']}. {r['name'][0].upper() + r['name'][1:]}")
        w("")
        w(f"**Mechanism.** {md_escape(r['mechanism'])}")
        w("")
        files = list(dict.fromkeys(t.file for t in r["mapped"]))
        w(f"**Tests: {len(r['mapped'])}** in {', '.join(f'`{f}.test.ts`' for f in files)}; {status_line(r['mapped'])}.")
        w("")
        w("| File | Test | Status | Also |")
        w("|---|---|---|---|")
        for f, label, st, grp in group_rows(r["mapped"]):
            also = sorted({n for t in grp for n in serves[id(t)] if n != r["n"]})
            w(f"| {f} | {label} | {st} | {', '.join(map(str, also)) or '–'} |")
        w("")
        w("**Key negative cases.**")
        w("")
        for case, got in r["negs"]:
            allg = [t for _, g in got for t in g]
            uniq = [t for i, t in enumerate(allg) if all(t is not u for u in allg[:i])]
            w(f"- {md_escape(case)}: {'; '.join(ref_label(ref, g) for ref, g in got)} ({status_line(uniq)}).")
        w("")
        if r.get("evidence"):
            w("**Evidence outside the test suite.**")
            w("")
            for e in r["evidence"]:
                w(f"- {md_escape(e)}")
            w("")
        if r.get("gas_table"):
            w(*["\n".join(gas_table())])
            w("")
        w(f"**Coverage.** {md_escape(r['coverage'])}")
        w("")

    w("## Tests mapped to no requirement")
    w("")
    w("| File | Test | Status | Why |")
    w("|---|---|---|---|")
    for t in unm:
        w(f"| {t.file} | {md_escape(t.title)} | {STATUS_MD[t.status]} | {notes.get(id(t), '–')} |")
    w("")
    pb = mapping.get("pending_branches")
    if pb and pb.get("tests"):
        w("## Tests on unmerged branches (pending, not in this run)")
        w("")
        w(f"Declared on branches not yet merged into main, as of {pb['as_of']}; they join the mapping when merged.")
        w("")
        w("| Branch | File | Test | Expected requirement |")
        w("|---|---|---|---|")
        for x in pb["tests"]:
            w(f"| {x['branch']} | {x['file']} | {md_escape(x['test'])} | {x['expected']} |")
        w("")
    w("## Index: every test and its requirements")
    w("")
    w("| File | Test | Status | Requirements |")
    w("|---|---|---|---|")
    for t in tests:
        w(f"| {t.file} | {md_escape(t.title)} | {STATUS_MD[t.status]} | {', '.join(map(str, serves[id(t)])) or 'none'} |")
    w("")
    return "\n".join(L)


def cmd_render(args: argparse.Namespace) -> None:
    DOC.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the contract suite once and save its output")
    r.add_argument("--force", action="store_true", help=f"start even with less than {MIN_FREE_GB} GB free RAM")
    r.add_argument("--label", default="laptop", help="where the suite ran, e.g. server")
    r.add_argument("--note", default="", help="a note recorded with the run")
    r.set_defaults(func=cmd_run)
    sub.add_parser("render", help="write docs/tests-by-requirement.md from the saved output").set_defaults(func=cmd_render)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
