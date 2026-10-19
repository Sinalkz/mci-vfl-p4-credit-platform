"""Chapter 5 export of an E3 run on Besu that settled by the normal path (default M4's, results/m4/e3/run.json; the
mpc-setup run: --run-dir results/mpc/e3 --check results/mpc/e3_check.json): the figures it accepted, the division, each
figure's percentile among E1's N = 64 draws, and the gas of every transaction by phase.

- Figures: the six accepted on-chain (F_r, F_j, LOO_1..3, BASELINE) with their proofs' records; acceptance needs the
  deployed Groth16 verifier to return true (Federation._acceptFigure reverts InvalidProof otherwise).
- Division (settlement_reference, checked against the Settled event and every credit by m4_e3.check): U_fed =
  max(F_r, F_j); l_i = U_fed - LOO_i; claims max(0, l_i); E = max(0, U_fed - BASELINE); E' = min(E, floor(B / rho));
  Talmud awards a = talmud(E', claims) in performance units, rho * a_i in tokens, the rest of B to A.
- Percentiles: among E1 C2 seed 1's 1,000 draws of N = 64 evaluation rows (results/data1/e1/n64.json), computed as
  py/data1_analysis.py does for results/data1/e3_vs_e1.csv: 100 * (share below + half the share equal). The run's model
  config is results/data1/configs/e3.json, byte-identical to e1-s1.json; its centralized replay (replay.json) has
  E1 C2 seed 1's config hash. E' and the awards are not ranked: the budget caps E' at floor(B / rho).
- Also: the run's keys and prover (run.json "keys" and "prover"; absent before mpc-setup: the development key and
  snarkjs), each proof's prover time and peak memory, token conservation (deposits = withdrawn + contract balance, from
  m4_e3.check), and the run's wall-clock time.
- Gas: every transaction in run.json, by phase. setup: create, register, acceptArbiter, commitSeed (state Setup) and
  the seed reveals (Revealing); alignment: commitAlignment (Alignment); training: commitFinal, commitFA (Training) and
  the reconciliation's transactions (requestResend, when A asks a passive to resend its final message, and
  finalizeReconciliation); scoring: the six figures; settlement: acknowledge, withdraw. The run's
  total is the five phases. Outside it: the mint and approve calls on the run's test token (MockERC20; a deployment
  would use an existing token), and the operator's deployments (token, verifier, factory), shared by every run.
- full-data runs (Task 2, results/full/runs/run-<k>): --e1-seed k. Run k's model config is results/data1/configs/
  e1-s<k>.json, so its figures are ranked among E1 C2 seed k's own 1,000 draws (results/data1/e1/n64.json for k = 1,
  n64-s<k>.json otherwise); its replay must have c2-s<k>.json's config hash.
Writes <run dir>/ch5-figures.json, with each source file @ the commit that last changed it.
- The rerun with a longer final-round wait (results/full/runs-wait10s): --e1-seed k --model-config FILE. FILE (the run's
  config, results/full/configs/e1-s<k>-wait10s.json) must differ from e1-s<k>.json in perBatchWaitMs only, and be the
  config the run and its replay used; the figures are still ranked among E1 C2 seed k's draws (the wait changes no
  model).
Usage: python py/m4_e3_ch5.py [--run-dir DIR] [--check FILE] [--out FILE] [--e1-seed K [--model-config FILE]]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import half_up as hu
import m4_e3
import settlement_reference as sr

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / "results" / "m4" / "e3"
M4_CHECK = ROOT / "results" / "m4" / "e3_check.json"   # py/m4_e3.py; rechecked here
N64 = ROOT / "results" / "data1" / "e1" / "n64.json"
C2_S1 = ROOT / "results" / "data1" / "e1" / "c2-s1.json"
CONFIG_E3 = ROOT / "results" / "data1" / "configs" / "e3.json"
CONFIG_E1_S1 = ROOT / "results" / "data1" / "configs" / "e1-s1.json"
FEDERATION = ROOT / "contracts" / "contracts" / "Federation.sol"
VERIFIERS = ROOT / "contracts" / "contracts" / "verifiers"

NAMES = {(0, 0): "F_r", (3, 0): "F_j", (1, 1): "LOO_1", (1, 2): "LOO_2", (1, 3): "LOO_3", (2, 0): "BASELINE"}
SCORING_NAME = {"FULL": "F_r", "FULL_J": "F_j", "LOO_1": "LOO_1", "LOO_2": "LOO_2", "LOO_3": "LOO_3",
                "BASELINE": "BASELINE"}
PHASES = ["setup", "alignment", "training", "scoring", "settlement"]
STEP_PHASE = [  # (step prefix in run.json, phase)
    ("deploy ", "infrastructure"),
    ("mint ", "test token"), ("approve ", "test token"),
    ("create", "setup"), ("register", "setup"), ("acceptArbiter", "setup"), ("commitSeed", "setup"),
    ("revealSeed", "setup"),
    ("commitAlignment", "alignment"),
    ("commitFinal", "training"), ("commitFA", "training"), ("requestResend", "training"), ("finalizeReconciliation", "training"),
    ("submitScore ", "scoring"), ("submitJointScore ", "scoring"),
    ("acknowledge", "settlement"), ("withdraw", "settlement"),
]
QUANTITIES = ["F_r", "F_j", "U_fed", "LOO_1", "LOO_2", "LOO_3", "BASELINE", "l_1", "l_2", "l_3", "E"]


def differs_only_in(a: dict, b: dict, field: str) -> bool:
    """True if the two configs differ in `field` and in nothing else."""
    if field not in a or field not in b or a[field] == b[field]:
        return False
    return {k: v for k, v in a.items() if k != field} == {k: v for k, v in b.items() if k != field}


def percentile(values, x) -> float:
    v = np.asarray(values, dtype=float)
    return 100.0 * float((v < x).mean() + 0.5 * (v == x).mean())


def rank_in_draws(values, x) -> dict:
    """x against E1's N = 64 draws of one quantity: its percentile and the draws' mean, SD and percentiles."""
    v = np.asarray(values, dtype=float)
    return {"value": x, "percentile": hu.round_half_up(percentile(v, x), 2),
            "E1_N64_mean": hu.round_half_up(float(v.mean()), 1), "E1_N64_sd": hu.round_half_up(float(v.std(ddof=1)), 1),
            "E1_N64_p2_5": float(np.percentile(v, 2.5)), "E1_N64_p50": float(np.percentile(v, 50)),
            "E1_N64_p97_5": float(np.percentile(v, 97.5))}


def proof_measures(s: dict) -> dict:
    """A's record of one proof: prover time, the prover's and node's peak memory, local verification time."""
    return {"prove_ms": hu.round_half_up(s["proveMs"]),
            "prover_peak_rss_mb": None if s.get("proverPeakRssMB") is None else hu.round_half_up(s["proverPeakRssMB"]),
            "node_peak_rss_mb": hu.round_half_up(s["peakRssMB"]), "local_verify_ms": hu.round_half_up(s["verifyMs"], 1)}


def phase_of(step: str) -> str:
    for prefix, phase in STEP_PHASE:
        if step.startswith(prefix):
            return phase
    raise SystemExit(f"no phase for transaction step {step!r}")


def n64_series(draws: list) -> dict:
    """Per draw: F_r, F_j, U_fed = max(F_r, F_j), LOO_i, BASELINE, l_i = U_fed - LOO_i, E = max(0, U_fed - BASELINE)."""
    F = lambda k: [x["F"][k] for x in draws]  # noqa: E731
    u = [max(a, b) for a, b in zip(F("FULL"), F("FULL_J"))]
    s = {"F_r": F("FULL"), "F_j": F("FULL_J"), "U_fed": u, "BASELINE": F("BASELINE")}
    for i in (1, 2, 3):
        s[f"LOO_{i}"] = F(f"LOO_{i}")
        s[f"l_{i}"] = [a - b for a, b in zip(u, s[f"LOO_{i}"])]
    s["E"] = [max(0, a - b) for a, b in zip(u, s["BASELINE"])]
    return s


def figures(run: dict) -> dict:
    return {NAMES[(f["kind"], f["memberIndex"])]: f for f in run["chainEnd"]["figures"]}


def division(run: dict) -> dict:
    ce = run["chainEnd"]
    prop = {k: int(ce["proposal"][k]) for k in ("B", "rho")}
    F = {n: f["F"] for n, f in figures(run).items()}
    u = max(F["F_r"], F["F_j"])
    loo = [F[f"LOO_{i}"] for i in (1, 2, 3)]
    left = [m["left"] for m in ce["members"]]
    E, e_prime, claims, awards = sr.division(prop["B"], prop["rho"], u, F["BASELINE"], loo, left)
    tokens = [prop["rho"] * a for a in awards]
    regime = ("E' = 0: nothing to divide (U_fed <= BASELINE)" if e_prime == 0
              else "every claim paid in full (sum of claims <= E')" if sum(claims) <= e_prime else "Talmud division of E'")
    return {"B": prop["B"], "rho": prop["rho"], "U_fed": u, "F_j_binds": F["F_j"] > F["F_r"],
            "l": [u - x for x in loo], "claims": claims, "E": E, "E_prime": e_prime,
            "B_over_rho": prop["B"] // prop["rho"], "regime": regime, "awards": awards, "award_tokens": tokens,
            "to_A": prop["B"] - sum(tokens)}


def gas(run: dict) -> dict:
    txs = [{"party": "operator", **t} for t in run["operatorTxs"]]
    txs += [{"party": p, **t} for p, v in run["parties"].items() for t in v["txs"]]
    rows = sorted(({"party": t["party"], "step": t["step"], "block": int(t["block"]), "gasUsed": int(t["gasUsed"]),
                    "status": int(t["status"]), "phase": phase_of(t["step"])} for t in txs),
                  key=lambda t: (t["block"], t["party"]))
    if any(t["status"] != 1 for t in rows):
        raise SystemExit("a transaction failed")
    group = lambda ph: [t for t in rows if t["phase"] == ph]  # noqa: E731
    phases = {ph: {"count": len(group(ph)), "gas": sum(t["gasUsed"] for t in group(ph))}
              for ph in [*PHASES, "test token", "infrastructure"]}
    return {"transactions": rows, "phases": phases, "total": sum(phases[ph]["gas"] for ph in PHASES),
            "total_all_transactions": sum(t["gasUsed"] for t in rows),
            "outside_the_total": {"test token": "mint and approve on the run's test token (MockERC20)",
                                  "infrastructure": "the operator's deployments of the token, verifier and factory, "
                                                    "shared by every run"}}


def commit_of(path: Path, at: str = "HEAD") -> str:
    """The commit that last changed `path` as of `at`; at HEAD the file must also be committed as it is."""
    rel = path.relative_to(ROOT).as_posix()
    out = subprocess.run(["git", "log", "-1", "--format=%h", at, "--", rel], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.strip()
    if at != "HEAD":
        return out
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    if not out or dirty:
        raise SystemExit(f"{rel} is not committed as it is")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=E3)
    ap.add_argument("--check", type=Path, default=M4_CHECK)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--e1-seed", type=int, help="full-data run k: model config e1-s<k>.json, E1 C2 seed k's draws")
    ap.add_argument("--model-config", type=Path, help="the run's config if not e1-s<k>.json (it may differ in perBatchWaitMs only)")
    a = ap.parse_args(argv)
    k = a.e1_seed
    if a.model_config is not None and k is None:
        raise SystemExit("--model-config needs --e1-seed")
    model_config = None if a.model_config is None else (a.model_config if a.model_config.is_absolute() else ROOT / a.model_config)
    n64_file = N64 if k in (None, 1) else ROOT / "results" / "data1" / "e1" / f"n64-s{k}.json"
    c2_file = C2_S1 if k in (None, 1) else ROOT / "results" / "data1" / "e1" / f"c2-s{k}.json"
    config_files = (CONFIG_E3, CONFIG_E1_S1) if k is None else (ROOT / "results" / "data1" / "configs" / f"e1-s{k}.json",)
    if model_config is not None:
        config_files = (*config_files, model_config)
    run_dir = a.run_dir if a.run_dir.is_absolute() else ROOT / a.run_dir
    check_file = a.check if a.check.is_absolute() else ROOT / a.check
    out_file = a.out if a.out else run_dir / "ch5-figures.json"
    RUN, SAMPLE, REPLAY = run_dir / "run.json", run_dir / "sample.json", run_dir / "replay.json"
    run = json.loads(RUN.read_text(encoding="utf-8"))
    keys = run.get("keys", {"setup": "dev", "verifierContract": "MembersVarVerifierN64D10"})
    prover = run.get("prover", "snarkjs")
    VERIFIER = VERIFIERS / f"{keys['verifierContract']}.sol"
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    n64 = json.loads(n64_file.read_text(encoding="utf-8"))
    c2 = json.loads(c2_file.read_text(encoding="utf-8"))
    ce, st = run["chainEnd"], run["chainEnd"]["settlement"]

    # the run: complete, normal path, settled as the reference says
    chk = m4_e3.check(run)
    if not chk["ok"] or st["path"] != "normal" or run["finalState"] != "Settled":
        raise SystemExit(f"not a normal-path settlement that checks: {chk['problems']}")
    committed = json.loads(check_file.read_text(encoding="utf-8"))
    if any(committed[k] != chk[k] for k in ("division", "credited", "withdrawn")):
        raise SystemExit(f"{check_file.name} differs from a fresh check of run.json")
    if chk["deposits"] != chk["withdrawnTotal"] + int(chk["contractBalance"]):
        raise SystemExit("tokens are not conserved")
    figs = figures(run)
    accepted = len(figs) == 6 and all(f["accepted"] for f in figs.values())
    if not accepted or ce["acceptedFigures"] != ce["requiredFigures"]:
        raise SystemExit("not all six figures accepted")
    d = division(run)
    ev = st["settled"]
    if (ev["E"], ev["estate"], ev["awards"], d["U_fed"]) != (d["E"], d["E_prime"], d["awards"], ce["uFed"]):
        raise SystemExit(f"the Settled event {ev} differs from the reference {d}")
    # the model config: E3's = E1 seed 1's, so E1 C2 seed 1's N = 64 draws are the comparison; a full-data run k
    # uses e1-s<k>.json itself, so E1 C2 seed k's draws
    if k is None:
        if CONFIG_E3.read_bytes() != CONFIG_E1_S1.read_bytes() or not sample["modelConfig"].replace("\\", "/").endswith(
                "configs/e3.json") or replay["configHash"] != c2["configHash"] or str(n64["seed"]) != "1":
            raise SystemExit("the run's model config is not E1 seed 1's")
    elif model_config is not None:
        mc = model_config.relative_to(ROOT).as_posix()
        if not differs_only_in(json.loads(config_files[0].read_text(encoding="utf-8")), json.loads(model_config.read_text(encoding="utf-8")),
                               "perBatchWaitMs") or not sample["modelConfig"].replace("\\", "/").endswith(mc)                 or replay["configPath"] != mc or str(n64["seed"]) != str(k) or str(c2["seed"]) != str(k):
            raise SystemExit(f"the run's model config is not E1 seed {k}'s with another perBatchWaitMs ({mc})")
    elif not sample["modelConfig"].replace("\\", "/").endswith(f"configs/e1-s{k}.json") or replay["configHash"] != c2["configHash"]             or str(n64["seed"]) != str(k) or str(c2["seed"]) != str(k):
        raise SystemExit(f"the run's model config is not E1 seed {k}'s")
    if replay["figures"] != {"FULL": figs["F_r"]["F"], "FULL_J": figs["F_j"]["F"],
                             **{k: figs[k]["F"] for k in ("LOO_1", "LOO_2", "LOO_3", "BASELINE")}}:
        raise SystemExit("replay.json does not reproduce the on-chain figures")

    txs = {t["step"]: t for t in run["parties"]["A"]["txs"]}
    scoring = {SCORING_NAME[s["figure"]]: s for s in run["parties"]["A"]["scoring"]}
    proofs = {}
    for n, f in figs.items():
        s = scoring[n]
        t = txs[("submitJointScore " if n == "F_j" else "submitScore ") + s["figure"]]
        proofs[n] = {"F": f["F"], "accepted_on_chain": f["accepted"], "transaction": t["step"],
                     "block": int(t["block"]), "status": int(t["status"]), "gasUsed": int(t["gasUsed"]),
                     "proof_field_elements": len(s["proof"]), "oracle_F_equals_F": s["oracleF"] == f["F"],
                     "prover": s.get("prover", "snarkjs"), **proof_measures(s), "cModel": f["cModel"]}

    series = n64_series(n64["draws"])
    value = {**{n: figs[n]["F"] for n in figs}, "U_fed": d["U_fed"], "E": d["E"],
             **{f"l_{i}": d["l"][i - 1] for i in (1, 2, 3)}}
    ranks = {q: rank_in_draws(series[q], value[q]) for q in QUANTITIES}
    e_cap = d["B_over_rho"]
    rel = lambda p: p.relative_to(ROOT).as_posix()  # noqa: E731
    out = {
        "what": (f"full-data run {k} on Besu" if sample.get("scenario") == "recs-full" else "E3 on Besu")
                + f" ({keys['setup']} keys, {prover} prover), the complete run settled by the normal path (Chapter 5)",
        "sources": {**{rel(p): commit_of(p)
                       for p in (RUN, SAMPLE, REPLAY, check_file, n64_file, c2_file, *config_files)},
                    **{rel(p): commit_of(p, at=commit_of(RUN)) for p in (FEDERATION, VERIFIER)}},
        "sources_note": "each file @ the commit that last changed it; the contracts as of the run's commit",
        "script": "py/m4_e3_ch5.py",
        "run": {"machine": run["machine"]["host"], "besu": run["chain"]["clientVersion"],
                "launch": run["chain"].get("launch"), "wallClockMs": run["wallClockMs"], "keys": keys, "prover": prover,
                "chainId": run["chain"]["chainId"], "federation": run["federation"], "finalState": run["finalState"],
                "path": st["path"], "scoredBlock": st["scoredBlock"], "settledBlock": ev["block"],
                "sizeI": run["parties"]["A"]["sizeI"],
                "evaluationRows": len(run["parties"]["A"]["idx"]),
                "runSeed": run.get("runSeed"), "scenarioSeed": run.get("scenarioSeed"),
                "alignmentPeriod": ce["proposal"].get("alignmentPeriod"), "maxDepth": ce["proposal"]["maxDepth"],
                "distinctEvaluationRows": len(set(map(str, run["parties"]["A"]["idx"])))},
        "figures": {n: figs[n]["F"] for n in ("F_r", "F_j", "LOO_1", "LOO_2", "LOO_3", "BASELINE")},
        "all_six_accepted_on_chain": accepted,
        "proofs": proofs,
        "division": {**{k: d[k] for k in ("U_fed", "F_j_binds", "l", "claims", "E", "B", "rho", "B_over_rho", "E_prime",
                                          "regime", "awards", "award_tokens", "to_A")},
                     "Settled_event": ev, "credited": st["credited"],
                     "deposits_returned_Dp": int(ce["proposal"]["Dp"])},
        "tokens": {"deposits": chk["deposits"], "withdrawn": chk["withdrawnTotal"],
                   "contract_balance": int(chk["contractBalance"]), "locked_forever": int(chk["lockedForever"]),
                   "deposits_equal_withdrawn_plus_balance":
                       chk["deposits"] == chk["withdrawnTotal"] + int(chk["contractBalance"])},
        "independent_recomputation": {"file": rel(REPLAY), "machine": replay.get("machine", {}).get("hostname"),
                                      "figures": replay["figures"], "equals_chain": True},
        "percentiles_in_E1_N64": {
            "draws": len(n64["draws"]), "source": f"E1 C2 seed {k or 1}, N = 64 protocol draws of evaluation rows ({rel(n64_file)})",
            "method": "100 * (share of draws below + half the share equal), as results/data1/e3_vs_e1.csv",
            "model_config": ("results/data1/configs/e3.json = e1-s1.json (byte-identical)" if k is None
                             else f"results/data1/configs/e1-s{k}.json" if model_config is None
                             else f"{model_config.relative_to(ROOT).as_posix()} (e1-s{k}.json with another perBatchWaitMs; replay configHash "
                                  f"{replay['configHash']}, E1 C2 seed {k}'s {c2['configHash']})")
                            + ("" if model_config is not None else f"; replay configHash = c2-s{k or 1} configHash = {c2['configHash']}"),
            "quantities": ranks,
            "not_ranked": f"E' and the awards: E' = min(E, floor(B / rho)) = min(E, {e_cap}); E >= {e_cap} in "
                          f"{sum(x >= e_cap for x in series['E'])} of {len(series['E'])} E1 draws"},
        "gas": gas(run),
    }
    out_file.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    g = out["gas"]
    print(json.dumps({"figures": out["figures"], "U_fed": d["U_fed"], "E": d["E"], "E_prime": d["E_prime"], "l": d["l"],
                      "awards": d["awards"], "tokens": d["award_tokens"], "to_A": d["to_A"],
                      "percentiles": {q: r["percentile"] for q, r in ranks.items()},
                      "gas": {p: g["phases"][p]["gas"] for p in g["phases"]}, "total": g["total"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
