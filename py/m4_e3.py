"""M4's E3 run end to end on Besu through settlement: the chain's record against the independent reference.

Reads results/m4/e3/run.json (contracts/offchain/runner3.mjs --run e3 --out-dir ../results/m4/e3) and checks:
- U_fed = max(F_r, F_j) (D139), and the Settled event's E, estate and awards against settlement_reference;
- every party's credits against the reference's owed amounts (the arbiter: D_c for each closed objection, D130);
- every party withdrew exactly what it was credited and nothing is owed at the end;
- lockedForever against the reference, and conservation: deposits = withdrawn + contract balance, where the
  balance left is exactly lockedForever.
Writes results/m4/e3_check.json. Exit code 1 if any check fails.
Usage: python py/m4_e3.py [run.json [out.json]] [--backup-rule backup-q6]
  --backup-rule backup-q6: check a backup settlement against the superseded M4 Q6 rule (E3's first run settled under it;
  D142 replaced it: backup now pays as normal).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import settlement_reference as sr

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "m4" / "e3" / "run.json"
OUT = ROOT / "results" / "m4" / "e3_check.json"
F_R, LOO, BASELINE, F_J = 0, 1, 2, 3  # Federation's figure kinds


def check(run: dict, backup_rule: str = "backup") -> dict:
    ce, s = run["chainEnd"], run["chainEnd"]["settlement"]
    prop = {k: int(ce["proposal"][k]) for k in ("B", "rho", "Dp", "Dc")}
    fig = {(f["kind"], f["memberIndex"]): f["F"] for f in ce["figures"]}
    n = len(ce["members"])
    names = [f"p{i + 1}" for i in range(n)]
    loo = [fig[(LOO, i + 1)] for i in range(n)]
    u_a = fig[(BASELINE, 0)]
    left = [m["left"] for m in ce["members"]]
    eqv = [m["equivocated"] for m in ce["members"]]
    acked = {a["passive"] for a in s["acknowledged"]}
    problems = []

    u_fed = max(fig[(F_R, 0)], fig[(F_J, 0)])
    if ce["uFed"] != u_fed:
        problems.append(f"uFed {ce['uFed']} != max(F_r, F_j) = {u_fed}")
    path = s["path"]
    rule = backup_rule if path == "backup" else path
    ref = sr.settle(rule, B=prop["B"], rho=prop["rho"], Dp=prop["Dp"], u_fed=u_fed, u_a=u_a, loo=loo, left=left, eqv=eqv,
                    acks=[x in acked for x in names])
    if rule == "backup-q6":
        claims = [ref["l"][i] if (not left[i] and names[i] in acked) else 0 for i in range(n)]
        estate = sum(claims) if prop["rho"] * sum(claims) <= prop["B"] else prop["B"] // prop["rho"]
    else:
        estate = ref["Eprime"]
    division = {"E": ref["E"], "Eprime": ref["Eprime"], "estate": estate, "l": ref["l"], "awards": ref["awards"]}

    ev = s["settled"]
    if ev is None:
        problems.append("no Settled event")
    else:
        for k, want in (("E", ref["E"]), ("estate", estate), ("awards", ref["awards"])):
            if ev[k] != want:
                problems.append(f"Settled {k} {ev[k]} != reference {want}")

    closed = sum(1 for m in ce["members"] if m["objection"] != 0)
    expected = {"A": ref["A"], **{names[i]: ref["P"][i] for i in range(n)}, "arbiter": prop["Dc"] * closed}
    credited = {k: int(s["credited"].get(k, 0)) for k in expected}
    withdrawn = {k: 0 for k in expected}
    for w in s["withdrawn"]:
        withdrawn[w["to"]] = withdrawn.get(w["to"], 0) + int(w["amount"])
    for k, want in expected.items():
        if credited[k] != want:
            problems.append(f"{k} credited {credited[k]} != reference {want}")
        if withdrawn[k] != credited[k]:
            problems.append(f"{k} withdrew {withdrawn[k]} != credited {credited[k]}")
        if int(ce["owed"].get(k, 0)) != 0:
            problems.append(f"{k} still owed {ce['owed'][k]}")

    locked, balance = int(s["lockedForever"]), int(s["contractBalance"])
    if locked != ref["locked"]:
        problems.append(f"lockedForever {locked} != reference {ref['locked']}")
    deposits = prop["B"] + prop["Dp"] * n + prop["Dc"] * closed
    total_out = sum(withdrawn.values())
    if balance != locked or total_out + balance != deposits:
        problems.append(f"conservation: deposits {deposits} != withdrawn {total_out} + balance {balance} (locked {locked})")

    return {"ok": not problems, "problems": problems, "path": path, "finalState": run["finalState"], "exitCodes": run["exitCodes"],
            "inputs": {**prop, "F_r": fig[(F_R, 0)], "F_j": fig[(F_J, 0)], "uFed": u_fed, "U_A": u_a, "loo": loo, "left": left,
                       "equivocated": eqv, "acknowledged": sorted(acked)},
            "division": division, "expected": expected, "credited": credited, "withdrawn": withdrawn,
            "deposits": deposits, "withdrawnTotal": total_out, "lockedForever": locked, "contractBalance": balance,
            "machine": run.get("machine"), "chain": run.get("chain")}


def main(argv: list[str]) -> int:
    rule = "backup"
    if "--backup-rule" in argv:
        i = argv.index("--backup-rule")
        rule = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    run_file = Path(argv[1]) if len(argv) > 1 else RUN
    out = Path(argv[2]) if len(argv) > 2 else OUT
    r = check(json.loads(run_file.read_text(encoding="utf-8")), backup_rule=rule)
    r["backupRule"] = rule
    out.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    print(f"path {r['path']}  expected {r['expected']}  locked {r['lockedForever']}  ok {r['ok']}")
    for p in r["problems"]:
        print("PROBLEM", p)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
