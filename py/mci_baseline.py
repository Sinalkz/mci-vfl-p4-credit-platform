"""MCI loan instance, Step 0: the baseline of the frozen thesis code (tag thesis-draft-09) in this worktree.

Two checks, each rerunnable, each writing only under results/mci_loan/baseline/ (hardhat: under --out instead, for a
later recheck of the suite that must leave Step 0's records as they are):

  hardhat  the contract suite (`npx hardhat test mocha` in contracts/), as py/tests_by_requirement.py runs it, in
           parts so that the light files can run on a laptop with little free memory:
             --part light  every test file except M3Scoring.test.ts
             --part heavy  M3Scoring.test.ts alone (it proves with snarkjs at about 6 GB; refused below 7 GB free)
             --part all    the whole suite in one run
           Saves the full output to hardhat-<part>.txt and the run's metadata and mocha totals to hardhat-<part>.json.
  smoke    the repo's off-chain reproducibility check, as contracts/offchain/data1/ldp.mjs runs it before every LDP
           set: E1's C2 run for seed 1 (vfl-run.mjs --mode c2, model config results/data1/configs/e1-s1.json) again,
           whose six figures must equal results/data1/e1/c2-s1.json's. Saves the run to c2-s1.json and the
           comparison to smoke.json.

Usage (from the repo root):
  .venv/Scripts/python py/mci_baseline.py hardhat --part light|heavy|all [--label laptop] [--force] [--out results/...]
  .venv/Scripts/python py/mci_baseline.py smoke [--label laptop]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_info  # noqa: E402
import tests_by_requirement as tbr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
OUT = ROOT / "results" / "mci_loan" / "baseline"
HEAVY = ["M3Scoring.test.ts"]
MIN_FREE_GB = {"light": 2.0, "heavy": 7.0, "all": 7.0}
SMOKE_CONFIG = ROOT / "results" / "data1" / "configs" / "e1-s1.json"
SMOKE_REFERENCE = ROOT / "results" / "data1" / "e1" / "c2-s1.json"


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def base_meta(label: str) -> dict:
    return {
        "label": label,
        "commit": tbr.git("rev-parse", "HEAD"),
        "branch": tbr.git("rev-parse", "--abbrev-ref", "HEAD"),
        "tag_thesis_draft_09": tbr.git("rev-parse", "thesis-draft-09^{commit}"),
        "tracked_changes_at_start": tbr.git("status", "--porcelain", "--untracked-files=no").splitlines(),
        "machine": tbr.machine(),
        "free_ram_gb_at_start": tbr.free_ram_gb(),
        "tools": {"Node.js": env_info.run(["node", "--version"]), "Python": sys.version.split()[0]},
        "started": now(),
    }


def mocha_totals(text: str) -> dict:
    totals = {"passing": 0, "failing": 0, "pending": 0}
    for line in text.splitlines():
        m = tbr.SUMMARY_RE.match(line)
        if m:
            totals[m.group(2)] = int(m.group(1))
    return totals


def test_files(part: str) -> list[str]:
    if part == "all":
        return []
    files = sorted(p.name for p in (CONTRACTS / "test").glob("*.test.ts"))
    chosen = [f for f in files if (f in HEAVY) == (part == "heavy")]
    return [f"test/{f}" for f in chosen]


def cmd_hardhat(args: argparse.Namespace) -> None:
    free = tbr.free_ram_gb()
    need = MIN_FREE_GB[args.part]
    if free is not None and free < need and not args.force:
        sys.exit(f"free RAM {free} GB < {need} GB for --part {args.part}; rerun with --force to start anyway")
    npx = shutil.which("npx")
    if npx is None:
        sys.exit("npx not found on PATH")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log, meta_file = out / f"hardhat-{args.part}.txt", out / f"hardhat-{args.part}.json"
    files = test_files(args.part)
    command = ["npx", "hardhat", "test", "mocha", *files]
    meta = {**base_meta(args.label), "part": args.part, "command": " ".join(command), "cwd": "contracts/", "files": files}
    meta["tools"].update({
        "Hardhat": env_info.npm_pkg_version("contracts", "hardhat"),
        "@nomicfoundation/hardhat-mocha": env_info.npm_pkg_version("contracts", "@nomicfoundation/hardhat-mocha"),
        "mocha": env_info.npm_pkg_version("contracts", "mocha"),
        "snarkjs (circuits/)": env_info.npm_pkg_version("circuits", "snarkjs"),
    })
    t0 = time.monotonic()
    with open(log, "wb") as fh:
        proc = subprocess.run([npx, *command[1:]], cwd=CONTRACTS, stdout=fh, stderr=subprocess.STDOUT)
    meta.update(finished=now(), wall_seconds=float(tbr.half_up(time.monotonic() - t0, 1)), free_ram_gb_at_end=tbr.free_ram_gb(),
                exit_code=proc.returncode, mocha=mocha_totals(log.read_text(encoding="utf-8", errors="replace")))
    meta["tools"]["solc"] = tbr.solc_versions()
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"part": args.part, "exit": proc.returncode, **meta["mocha"], "wall_seconds": meta["wall_seconds"],
                      "out": rel(meta_file)}))


def figures(path: Path) -> dict:
    return {f["name"]: f["F"] for f in json.loads(path.read_text(encoding="utf-8"))["figures"]}


def cmd_smoke(args: argparse.Namespace) -> None:
    node = shutil.which("node")
    if node is None:
        sys.exit("node not found on PATH")
    OUT.mkdir(parents=True, exist_ok=True)
    run = OUT / "c2-s1.json"
    command = ["node", "offchain/data1/vfl-run.mjs", "--mode", "c2", "--seed", "1", "--out", str(run)]
    meta = {**base_meta(args.label), "command": f"VFL_MODEL_CONFIG={rel(SMOKE_CONFIG)} " + " ".join(command[:-1] + [rel(run)]),
            "cwd": "contracts/", "reference": rel(SMOKE_REFERENCE), "rule": "contracts/offchain/data1/ldp.mjs check(): the six figures must be equal"}
    t0 = time.monotonic()
    proc = subprocess.run([node, *command[1:]], cwd=CONTRACTS, env={**os.environ, "VFL_MODEL_CONFIG": str(SMOKE_CONFIG)})
    meta.update(finished=now(), wall_seconds=float(tbr.half_up(time.monotonic() - t0, 1)), exit_code=proc.returncode)
    if proc.returncode == 0:
        got, want = figures(run), figures(SMOKE_REFERENCE)
        meta.update(run=rel(run), figures=got, reference_figures=want, equal=got == want)
    (OUT / "smoke.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"exit": proc.returncode, "equal": meta.get("equal"), "wall_seconds": meta["wall_seconds"]}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hardhat", help="run the contract suite (or a part) and save its output")
    h.add_argument("--part", choices=["light", "heavy", "all"], required=True)
    h.add_argument("--label", default="laptop", help="where the suite ran")
    h.add_argument("--force", action="store_true", help="start even below the part's free-RAM floor")
    h.add_argument("--out", default=rel(OUT), help="output folder, relative to the repo root (default: Step 0's)")
    h.set_defaults(func=cmd_hardhat)
    s = sub.add_parser("smoke", help="rerun E1's C2 seed 1 and compare its six figures with the recorded ones")
    s.add_argument("--label", default="laptop", help="where the check ran")
    s.set_defaults(func=cmd_smoke)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
