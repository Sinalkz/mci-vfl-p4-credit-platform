"""MCI loan instance, Step 5: the five on-chain runs' records, from their files only (nothing is rerun), with the
thesis's own checks: py/m4_e3.py check (the settlement against the independent reference, settlement_reference.py:
U_fed, E, E', the Talmud awards, every credit, every withdrawal, token conservation) and py/m4_e3_ch5.py (figures,
division, gas of every transaction by phase, each proof's measures).

Inputs per run k (results/mci_loan/onchain/run-<k>/, written on the Mac by scenarios/mci_loan/onchain/mac-runs.sh):
run.json (contracts/offchain/runner3.mjs --run mci-loan), replay.json (contracts/offchain/full/replay-full.mjs: the six
figures recomputed centrally, equal to the chain), sample.json, mem.log; also run-<k>-network-check.json, psi-bench.json
and plan.json. The comparison with the electricity scenario reads the thesis's own series, results/full/runs-wait10s/
run-<k>/run.json (the Chapter 5 runs, perBatchWaitMs 10,000 as here), with the same functions.

Outputs (results/mci_loan/onchain/): run-<k>/check.json (m4_e3.check), runs.csv, gas.csv, proofs.csv, payouts.csv,
transactions.csv (every transaction with its hash: the audit trail), events.csv (the settlement events the runner read
from the chain), ranking.csv (each run's passive ranking against off-chain seed k's whole-test-split ranking),
electricity_comparison.csv and summary.json. The schemas are in docs/mci/03-results.md.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/onchain_report.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "py"))
import common as cm  # noqa: E402
import m4_e3  # noqa: E402
import m4_e3_ch5 as ch5  # noqa: E402

OUT = cm.RESULTS / "onchain"
ELEC = cm.ROOT / "results" / "full" / "runs-wait10s"
OFF = cm.RESULTS / "offchain"
RUNS = [1, 2, 3, 4, 5]
FIGS = ["F_r", "F_j", "LOO_1", "LOO_2", "LOO_3", "BASELINE"]
MEMBER = {"A": "Bank", "p1": "MCI", "p2": "E-commerce company", "p3": "Credit bureau", "arbiter": "arbiter"}
PASSIVE = {1: "MCI", 2: "E-commerce company", 3: "Credit bureau"}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def write_csv(name, header, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def ranking(l):
    return tuple(sorted((1, 2, 3), key=lambda i: (-l[i - 1], i)))


def names(r):
    return " > ".join(PASSIVE[i] for i in r)


def measures(run: dict) -> dict:
    """What one run gives the tables: the check, gas by phase, the division, the proofs, the wall clock."""
    chk = m4_e3.check(run)
    g = ch5.gas(run)
    d = ch5.division(run)
    proofs = run["parties"]["A"]["scoring"]
    return {"check": chk, "gas": g, "division": d, "proofs": proofs, "figures": ch5.figures(run)}


def main() -> None:
    plan = load(OUT / "plan.json")
    bench = load(OUT / "psi-bench.json")
    off_loo = {int(r["seed"]): r for r in csv.DictReader((OFF / "loo.csv").open(encoding="utf-8"))}
    rrow, grow, prow, payrow, trow, erow, krow = [], [], [], [], [], [], []
    S = {"what": "the MCI loan instance's five on-chain runs (scenarios/mci_loan/onchain_report.py)", "plan": plan,
         "alignmentBenchmark": {"slowestAlignmentS": bench["slowestAlignmentS"], "alignmentPeriod": bench["alignmentPeriod"],
                                "machine": bench["machine"], "allCorrect": all(x["correct"] for x in bench["reps"])},
         "runs": {}}
    for k in RUNS:
        d_run = OUT / f"run-{k}"
        if not (d_run / "run.json").exists():
            S["runs"][k] = {"missing": True}
            rrow.append([k, "missing"] + [""] * 20)
            continue
        run = load(d_run / "run.json")
        rep = load(d_run / "replay.json") if (d_run / "replay.json").exists() else None
        net = load(OUT / f"run-{k}-network-check.json")
        m = measures(run)
        chk, g, d = m["check"], m["gas"], m["division"]
        cm.write_json(d_run / "check.json", chk, indent=1)
        F = {n: f["F"] for n, f in m["figures"].items()}
        accepted = all(f["accepted"] for f in run["chainEnd"]["figures"])
        st = run["chainEnd"]["settlement"]
        dep = {k2: chk[k2] for k2 in ("deposits", "withdrawnTotal", "contractBalance", "lockedForever")}
        dep["conserved"] = dep["withdrawnTotal"] + dep["contractBalance"] == dep["deposits"]
        rrow.append([k, run["finalState"], st["path"], f"{run['chainEnd']['acceptedFigures']}/{run['chainEnd']['requiredFigures']}",
                     "yes" if accepted else "NO", *[F[n] for n in FIGS], d["U_fed"], *d["l"], d["E"], d["E_prime"],
                     *d["award_tokens"], d["to_A"], "yes" if chk["ok"] else "NO",
                     "yes" if rep and rep["replayEqualsChain"] else ("missing" if rep is None else "NO"),
                     run["parties"]["A"]["sizeI"], round(run["wallClockMs"] / 1000, 1), dep["deposits"], dep["withdrawnTotal"],
                     dep["contractBalance"], "yes" if dep["conserved"] else "NO", "yes" if net["ok"] else "NO"])
        for ph, v in g["phases"].items():
            grow.append([k, ph, v["count"], v["gas"], "yes" if ph in ch5.PHASES else "no (outside the run's total)"])
        grow.append([k, "total (the five phases)", sum(g["phases"][p]["count"] for p in ch5.PHASES), g["total"], "yes"])
        for s in m["proofs"]:
            prow.append([k, s["figure"], s["mask"], s["F"], s["oracleF"], s["prover"], round(s["witnessMs"]), round(s["proveMs"]),
                         round(s["verifyMs"], 1), round(s["peakRssMB"]), round(s["proverPeakRssMB"]) if s.get("proverPeakRssMB") else "",
                         s["cModel"]])
        for who in ("A", "p1", "p2", "p3", "arbiter"):
            cred = st["credited"].get(who, "0")
            wd = sum(int(w["amount"]) for w in st["withdrawn"] if w["to"] == who)
            payrow.append([k, who, MEMBER[who], cred, wd, run["chainEnd"]["owed"].get(who, "0")])
        txs = [{"party": "operator", **t} for t in run["operatorTxs"]] + [{"party": p, **t} for p, v in run["parties"].items() for t in v["txs"]]
        for t in sorted(txs, key=lambda t: (int(t["block"]), t["party"])):
            trow.append([k, t["party"], MEMBER.get(t["party"], t["party"]), t["step"], ch5.phase_of(t["step"]), t["hash"],
                         int(t["block"]), int(t["gasUsed"]), int(t["status"])])
        if st["settled"]:
            erow.append([k, "Settled", st["settled"]["block"], "", f"path {st['settled']['path']}; E {st['settled']['E']}; "
                         f"estate {st['settled']['estate']}; awards {','.join(map(str, st['settled']['awards']))}"])
        for e in st["acknowledged"]:
            erow.append([k, "Acknowledged", e["block"], e["passive"], ""])
        for e in st["objections"]:
            erow.append([k, "AlignmentObjectionUpheld", e["block"], e["passive"], e["listHash"]])
        for who, v in st["credited"].items():
            erow.append([k, "Credited", "", who, v])
        for w in st["withdrawn"]:
            erow.append([k, "Withdrawn", w["block"], w["to"], w["amount"]])
        on_rank = ranking(d["l"])
        o = off_loo[k]
        off_rank = ranking([float(o["l_1"]), float(o["l_2"]), float(o["l_3"])])
        krow.append([k, names(on_rank), names(off_rank), "yes" if on_rank == off_rank else "no", on_rank[0] == off_rank[0],
                     *d["l"], o["l_1"], o["l_2"], o["l_3"]])
        S["runs"][k] = {"finalState": run["finalState"], "path": st["path"], "accepted": accepted, "checkOk": chk["ok"],
                        "replayEqualsChain": bool(rep and rep["replayEqualsChain"]), "figures": F, "division": d,
                        "gas": {p: v["gas"] for p, v in g["phases"].items()}, "gasTotal": g["total"],
                        "proveMs": [s["proveMs"] for s in m["proofs"]], "proverPeakRssMB": [s.get("proverPeakRssMB") for s in m["proofs"]],
                        "wallClockS": run["wallClockMs"] / 1000, "sizeI": run["parties"]["A"]["sizeI"],
                        "conservation": dep, "sameRankingAsOffchainSeed": on_rank == off_rank,
                        "machine": run["machine"], "chain": {k2: run["chain"][k2] for k2 in ("clientVersion", "genesisBlockHash", "launch") if k2 in run["chain"]},
                        "keys": run["keys"], "prover": run["prover"]}
    write_csv("runs.csv", ["run", "final_state", "settlement_path", "figures_accepted", "every_proof_accepted", *[f"F_{n}" for n in FIGS],
                           "U_fed", "l_1", "l_2", "l_3", "E", "E_prime", "award_1_tokens", "award_2_tokens", "award_3_tokens", "to_A",
                           "settlement_check_ok", "replay_equals_chain", "size_I", "wall_clock_s", "deposits_tokens",
                           "withdrawn_tokens", "contract_balance_end", "tokens_conserved", "network_check_ok"], rrow)
    write_csv("gas.csv", ["run", "phase", "transactions", "gas", "in_run_total"], grow)
    write_csv("proofs.csv", ["run", "figure", "mask", "F", "oracle_F", "prover", "witness_ms", "prove_ms", "local_verify_ms",
                             "node_peak_rss_mb", "prover_peak_rss_mb", "c_model"], prow)
    write_csv("payouts.csv", ["run", "party", "member", "credited_tokens", "withdrawn_tokens", "owed_at_end"], payrow)
    write_csv("transactions.csv", ["run", "party", "member", "step", "phase", "tx_hash", "block", "gas_used", "status"], trow)
    write_csv("events.csv", ["run", "event", "block", "party", "detail"], erow)
    write_csv("ranking.csv", ["run", "onchain_ranking", "offchain_seed_k_ranking", "same", "same_top", "l_1_onchain", "l_2_onchain",
                              "l_3_onchain", "l_1_offchain", "l_2_offchain", "l_3_offchain"], krow)

    # the electricity scenario's series (the thesis's Chapter 5 runs), with the same functions
    done = [k for k in RUNS if not S["runs"][k].get("missing")]
    crow = []
    def stats(xs):
        return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)
    elec = {k: measures(load(ELEC / f"run-{k}" / "run.json")) for k in RUNS}
    for ph in [*ch5.PHASES, "total"]:
        lv = [S["runs"][k]["gasTotal"] if ph == "total" else S["runs"][k]["gas"][ph] for k in done]
        ev = [elec[k]["gas"]["total"] if ph == "total" else elec[k]["gas"]["phases"][ph]["gas"] for k in RUNS]
        crow.append([f"gas: {ph}", len(lv), *stats(lv), len(ev), *stats(ev)])
    lp = [s for k in done for s in S["runs"][k]["proveMs"]]
    ep = [s["proveMs"] for k in RUNS for s in elec[k]["proofs"]]
    crow.append(["proof time per figure (ms, rapidsnark)", len(lp), *stats(lp), len(ep), *stats(ep)])
    lm = [x for k in done for x in S["runs"][k]["proverPeakRssMB"] if x]
    em = [s["proverPeakRssMB"] for k in RUNS for s in elec[k]["proofs"] if s.get("proverPeakRssMB")]
    crow.append(["prover peak memory per figure (MB)", len(lm), *stats(lm), len(em), *stats(em)])
    lw = [S["runs"][k]["wallClockS"] for k in done]
    ew = [load(ELEC / f"run-{k}" / "run.json")["wallClockMs"] / 1000 for k in RUNS]
    crow.append(["run wall clock (s)", len(lw), *stats(lw), len(ew), *stats(ew)])
    li = [S["runs"][k]["sizeI"] for k in done]
    ei = [load(ELEC / f"run-{k}" / "run.json")["parties"]["A"]["sizeI"] for k in RUNS]
    crow.append(["|I| (aligned rows)", len(li), *stats(li), len(ei), *stats(ei)])
    write_csv("electricity_comparison.csv", ["quantity", "loan_runs", "loan_mean", "loan_sd", "electricity_runs", "electricity_mean",
                                             "electricity_sd"], crow)
    S["electricityComparison"] = {"source": "results/full/runs-wait10s/run-<k>/run.json (the thesis's Chapter 5 series)",
                                  "rows": [dict(zip(["quantity", "loan_runs", "loan_mean", "loan_sd", "electricity_runs", "electricity_mean", "electricity_sd"], r)) for r in crow]}
    S["allAccepted"] = all(S["runs"][k]["accepted"] for k in done)
    S["allChecksOk"] = all(S["runs"][k]["checkOk"] for k in done)
    S["allReplaysEqual"] = all(S["runs"][k]["replayEqualsChain"] for k in done)
    S["allConserved"] = all(S["runs"][k]["conservation"]["conserved"] for k in done)
    S["runsDone"] = done
    cm.write_json(OUT / "summary.json", S, indent=1)
    print(json.dumps({"runs": done, "allAccepted": S["allAccepted"], "allChecksOk": S["allChecksOk"], "allReplaysEqual": S["allReplaysEqual"],
                      "paths": [S["runs"][k]["path"] for k in done], "gasTotal": [S["runs"][k]["gasTotal"] for k in done]}))


if __name__ == "__main__":
    main()
