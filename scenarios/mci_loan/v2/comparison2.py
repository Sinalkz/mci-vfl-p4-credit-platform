"""MCI loan instance v2, Step 7: electricity (the thesis), loan v1 (the Bank active) and loan v2 (MCI active), side by
side on the same metrics, costs and stability measures. Collects the three into
results/mci_loan/v2/comparison/scenario-comparison.json (so that every number of the document has a file under
results/mci_loan/) and writes docs/mci/v2/05-scenario-comparison.md from it.

Sources. Electricity and loan v1: results/mci_loan/comparison/scenario-comparison.json (v1's comparison.py, which read
the thesis's files and v1's), unchanged except for the names: under the naming rule (2026-10-19) no real company is
named except MCI, so v1's e-commerce role and the thesis's retailer appear under generic names (RENAME). Loan v2:
results/mci_loan/v2/prepared/manifest.json, results/mci_loan/v2/offchain/summary.json and n64.csv,
results/mci_loan/v2/onchain/summary.json and cost_comparison.csv (which reads the electricity and v1 run files with the
same functions).

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/v2/comparison2.py
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as cm  # noqa: E402
import common2 as c2  # noqa: E402

OUT = c2.RESULTS2 / "comparison" / "scenario-comparison.json"
DOC = c2.DOCS2 / "05-scenario-comparison.md"
V1 = cm.RESULTS / "comparison" / "scenario-comparison.json"
RENAME = [("retailer (Digikala)", "retailer"), ("Digikala", "e-commerce role")]  # applied to every string of v1's file
FORBIDDEN = re.compile(r"digikala|دیجی", re.IGNORECASE)
SIDES = [("electricity", "Electricity (thesis)"), ("loan", "Loan v1 (Bank active)"), ("loan2", "Loan v2 (MCI active)")]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def renamed(o):
    if isinstance(o, dict):
        return {renamed(k): renamed(v) for k, v in o.items()}
    if isinstance(o, list):
        return [renamed(v) for v in o]
    if isinstance(o, str):
        for a, b in RENAME:
            o = o.replace(a, b)
        return o
    return o


def ms(s):
    return {"mean": s["mean"], "sd": s["sd"], "n": s["n"]}


def collect() -> dict:
    v1 = renamed(load(V1))
    m = load(c2.PREPARED2 / "manifest.json")  # delivery: the full manifest is the data's; results/ holds a redacted copy
    lo = load(c2.RESULTS2 / "offchain" / "summary.json")
    lon = load(c2.RESULTS2 / "onchain" / "summary.json")
    n64 = {int(r["seed"]): float(r["share_same_ranking_as_full_test"]) for r in csv.DictReader((c2.RESULTS2 / "offchain" / "n64.csv").open(encoding="utf-8"))}
    cost = {r["quantity"]: r for r in csv.DictReader((c2.RESULTS2 / "onchain" / "cost_comparison.csv").open(encoding="utf-8"))}
    nm = c2.names()
    passives = {str(i): nm[p] for i, p in enumerate(m["passiveOrder"], 1)}
    last = {name: sum(1 for r in lo["ranking"]["rankings"] if r.endswith(f"> {name}")) for name in passives.values()}
    q = lo["configs"]["C2 U_fed = max(F_r, F_j)"]
    tal = lo["talmud"]["B_proposal"]
    loan2 = {
        "quality": {"U_fed": ms(q), "U_A": ms(lo["loo"]["F_C1a"]), "C1b": ms(lo["loo"]["F_C1b"]), "E": ms(lo["loo"]["E"]),
                    "relative_gain": (q["mean"] - lo["loo"]["F_C1a"]["mean"]) / lo["loo"]["F_C1a"]["mean"]},
        "contributions": {f"l_{i}": ms(lo["loo"][f"l_{i}"]) for i in (1, 2, 3)},
        "ranking": {"modal": lo["ranking"]["modal"], "count": lo["ranking"]["count"], "n": lo["ranking"]["n"],
                    "friedman": lo["ranking"]["friedman"], "lastPlace": last},
        "talmudAtB": {"regimes": tal["regimes"], "E_prime": tal["E_prime"]["mean"], "awards_tokens": [a["mean"] for a in tal["awards_tokens"]]},
        "n64SameRankingSeed1": n64[1], "equivalenceBitwise": lo["equivalence"]["all_bitwise"],
        "ldpUtility": {k: {"mean": v["mean"], "sd": v["sd"], "seeds_above": v["seeds_above"]} for k, v in lo["ldp"]["eps"]["utility"].items()},
        "ldpProtocolSigma": {k: ms(v["U_fed"]) for k, v in lo["ldp"]["scales"]["full_test"].items()},
        "onchainRuns": [{"run": int(k), "path": v["path"], "U_fed": v["division"]["U_fed"], "U_A": v["figures"]["BASELINE"], "E": v["division"]["E"],
                         "E_prime": v["division"]["E_prime"], "awards_tokens": v["division"]["award_tokens"], "l": v["division"]["l"],
                         "accepted": v["accepted"]} for k, v in lon["runs"].items()],
        "passiveNames": passives, "activeName": nm["mci"],
    }
    out = {
        "what": "electricity (thesis), loan v1 (Bank active) and loan v2 (MCI active): the same metrics, costs and stability measures "
                "(scenarios/mci_loan/v2/comparison2.py)",
        "names": "v1's file under the naming rule: the thesis's retailer and v1's e-commerce role appear without a company name "
                 "(comparison2.RENAME)",
        "data": {**v1["data"], "loan2": {"rows": m["rows"], "train": len(m["split"]["train_pos"]), "test": len(m["split"]["test_pos"]),
                                         "classes": 2, "passiveWidths": [len(m["parties"][p]) for p in m["passiveOrder"]]}},
        "electricity": {**v1["electricity"], "activeName": "power company"},
        "loan": {**v1["loan"], "activeName": "Bank"},
        "loan2": loan2,
        "onchainCost": {k: {"loan2_mean": float(r["loan_v2_mean"]), "loan2_sd": float(r["loan_v2_sd"]),
                            "loan_mean": float(r["loan_v1_mean"]), "loan_sd": float(r["loan_v1_sd"]),
                            "electricity_mean": float(r["electricity_mean"]), "electricity_sd": float(r["electricity_sd"]),
                            "runs": int(r["loan_v2_runs"])} for k, r in cost.items()},
        "sources": {"electricity and loan v1": [str(V1.relative_to(cm.ROOT)).replace("\\", "/")],
                    "loan v2": ["results/mci_loan/v2/prepared/manifest.json", "results/mci_loan/v2/offchain/summary.json",
                                "results/mci_loan/v2/offchain/n64.csv", "results/mci_loan/v2/onchain/summary.json",
                                "results/mci_loan/v2/onchain/cost_comparison.csv"]},
    }
    return out


def f(x, d=1):
    return f"{x:,.{d}f}"


def pm(s, d=1):
    return f"{s['mean']:,.{d}f} ± {s['sd']:,.{d}f}"


def write_doc(c: dict) -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cm.ROOT, capture_output=True, text=True).stdout.strip()
    src = "`results/mci_loan/v2/comparison/scenario-comparison.json`"
    w = []
    a = w.append
    a("# Electricity (thesis), loan v1 and loan v2: the three instances side by side")
    a("")
    a(f"Generated by `scenarios/mci_loan/v2/comparison2.py` from {src}, which collects the three instances' result files "
      f"(listed under `sources` there); do not edit by hand. Generated on commit `{commit}`. Mean ± SD over 10 seeds unless stated.")
    a("")
    a("**Read with care.** The architecture, contracts, circuit, key, model configuration and protocol are identical; the data, "
      "the label and the roles are not.")
    a("- **Electricity** predicts 5 tariff tiers on RECS 2020 households, with the power company active.")
    a("- **Loan v1** (tag `mci-v1-bank-active`) predicts a binary default label on Kaggle Home Credit Default Risk applicants, with "
      "the Bank active. **Loan v2** has the same applicants and label, with MCI active as the lender (classes 2–4 empty in both).")
    a("- **Raw F is not comparable across the label sets.** Macro-F1 averages over the classes present, so its chance level differs: "
      "about 2,000 for 5 balanced classes, about 5,000 for 2. Compare gains, rankings, stability and costs.")
    a("- **Names.** The loan organisations are generic simulated roles (`docs/mci/v2/02-preregistration.md`, section 0). Under "
      "the naming rule, the thesis's retailer and v1's e-commerce role carry no company name here.")
    a("")
    E, L1, L2 = c["electricity"], c["loan"], c["loan2"]
    head = "| | " + " | ".join(lab for _, lab in SIDES) + " |"
    rule = "|---|" + "---:|" * len(SIDES)

    a("## 1. Data and setup")
    a("")
    d = c["data"]
    a(head)
    a(rule)
    a(f"| Active member | {E['activeName']} | {L1['activeName']} | {L2['activeName']} |")
    a(f"| Active member's rows \\|X_A\\| | {d['electricity']['rows']:,} | {d['loan']['rows']:,} | {d['loan2']['rows']:,} |")
    a(f"| Classes used (of K = 5) | {d['electricity']['classes']} | {d['loan']['classes']} | {d['loan2']['classes']} |")
    a("| Passives (proposal order) | " + " | ".join(", ".join(x["passiveNames"].values()) for x in (E, L1, L2)) + " |")
    a("| Passive input widths (proposal order) | " + " | ".join(" / ".join(map(str, d[k]["passiveWidths"])) for k in ("electricity", "loan", "loan2")) + " |")
    a("")
    a(f"Source: {src}, `data`, `*.passiveNames`.")
    a("")

    a("## 2. Model quality and the federation's gain (off-chain, whole test split)")
    a("")
    a("| Quantity (macro-F1 × 10⁴) | " + " | ".join(lab for _, lab in SIDES) + " |")
    a(rule)
    for k, lab in (("U_fed", "U_fed = max(F_r, F_j)"), ("U_A", "U_A: the active member alone (BASELINE)"),
                   ("C1b", "C1b: the active member trained alone"), ("E", "E = U_fed − U_A")):
        a(f"| {lab} | " + " | ".join(pm(x["quality"][k]) for x in (E, L1, L2)) + " |")
    a("| Relative gain (U_fed − U_A) / U_A | " + " | ".join(f"{100 * x['quality']['relative_gain']:.1f}%" for x in (E, L1, L2)) + " |")
    a("| Split training equals centralized, bit for bit | " + " | ".join("yes" if x["equivalenceBitwise"] else "no" for x in (E, L1, L2)) + " |")
    a("")
    a("Loan v2's active columns are loan v1's (the same 8, `docs/mci/v2/02-preregistration.md`, section 5.1), so C1b is the same in "
      "both. U_A differs because BASELINE reuses the active member's bottom model as trained in its federation, with other "
      f"partners. Source: {src}, `*.quality`.")
    a("")

    a("## 3. Contributions and their stability")
    a("")
    a("| Quantity | " + " | ".join(lab for _, lab in SIDES) + " |")
    a(rule)
    for i in (1, 2, 3):
        a(f"| l_{i} (passive {i}) | " + " | ".join(f"{x['passiveNames'][str(i)]}: {pm(x['contributions'][f'l_{i}'])}" for x in (E, L1, L2)) + " |")
    a("| Modal ranking (largest l first) | " + " | ".join(f"{x['ranking']['modal']} ({x['ranking']['count']} of {x['ranking']['n']})" for x in (E, L1, L2)) + " |")
    a("| Friedman χ² (descriptive), p | " + " | ".join(f"{x['ranking']['friedman']['chi2']:.1f}, {x['ranking']['friedman']['p']:.4f}" for x in (E, L1, L2)) + " |")
    a("| N = 64 draws ranking as the whole test split (seed 1, 1,000 draws) | " + " | ".join(f"{x['n64SameRankingSeed1']:.3f}" for x in (E, L1, L2)) + " |")
    a("")
    lp = L2["ranking"]["lastPlace"]
    a(f"- **The smallest contributor.** The slot-1 member (the telecom, MCI) has the smallest l in {E['ranking']['slot1Last']} of "
      f"10 seeds in electricity and in {L1['ranking']['slot1Last']} of 10 in loan v1. In loan v2, MCI is the active member, and the "
      "smallest partner is " + " or ".join(f"the {k} ({v} of 10)" for k, v in sorted(lp.items(), key=lambda t: -t[1]) if v) + ".")
    a("- **The 64-row evaluation.** It reproduces the whole-test-split ranking in roughly a third of the draws in all three, so "
      f"the on-chain claims carry the same evaluation noise (source: {src}, `*.ranking`, `*.n64SameRankingSeed1`).")
    a("")

    a("## 4. Settlement at the proposal's pool (B = 1,000,003, ρ = 5,000)")
    a("")
    a("| Quantity | " + " | ".join(lab for _, lab in SIDES) + " |")
    a(rule)
    a("| Regime over the 10 seeds | " + " | ".join(", ".join(f"{k} {v}" for k, v in x["talmudAtB"]["regimes"].items()) for x in (E, L1, L2)) + " |")
    a("| E′ = min(E, ⌊B/ρ⌋), mean | " + " | ".join(f(x["talmudAtB"]["E_prime"]) for x in (E, L1, L2)) + " |")
    for i in range(3):
        a(f"| Award to passive {i + 1}, tokens (mean) | " + " | ".join(f"{x['passiveNames'][str(i + 1)]}: {x['talmudAtB']['awards_tokens'][i]:,.0f}" for x in (E, L1, L2)) + " |")
    a("")
    a("The budget binds in every seed of all three (E > ⌊B/ρ⌋ = 200), so each divides 200 performance units by the Talmud rule "
      f"on half-claims (cea). The pool is the active member's: in loan v2, MCI pays its partners. Source: {src}, `*.talmudAtB`.")
    a("")

    a("## 5. Privacy–utility (LDP, σ calibrated to ε per release)")
    a("")
    a("| Condition: U_fed − U_A (seeds above 0) | " + " | ".join(lab for _, lab in SIDES) + " |")
    a(rule)
    for k, lab in (("off", "clipping only"), ("e64", "ε = 64"), ("e16", "ε = 16"), ("e4", "ε = 4"), ("e1", "ε = 1")):
        a(f"| {lab} | " + " | ".join(f"{x['ldpUtility'][k]['mean']:+,.1f} ({x['ldpUtility'][k]['seeds_above']} of 10)" for x in (E, L1, L2)) + " |")
    a("| Protocol σ (x1) against clipping only (x0), U_fed | " + " | ".join(f"{pm(x['ldpProtocolSigma']['x1'])} vs {pm(x['ldpProtocolSigma']['x0'])}" for x in (E, L1, L2)) + " |")
    a("")
    a(f"Source: {src}, `*.ldpUtility`, `*.ldpProtocolSigma`.")
    a("")

    a("## 6. On-chain cost (five runs each, MacBook Air M4, native Besu 26.8.1, rapidsnark)")
    a("")
    oc = c["onchainCost"]
    a("| Quantity | " + " | ".join(lab for _, lab in SIDES) + " |")
    a(rule)
    for q, v in oc.items():
        d0 = 0 if "gas" in q or "|I|" in q else 1
        a(f"| {q.replace('|', chr(92) + '|')} | " + " | ".join(f"{v[f'{k}_mean']:,.{d0}f} ± {v[f'{k}_sd']:,.{d0}f}" for k in ("electricity", "loan", "loan2")) + " |")
    a("")
    a("Gas per phase is the same in all three to within a few hundred gas: the chain sees commitments, proofs and settlement, "
      "never data, so neither the scenario nor which member is active changes the cost (claim 5 of the thesis). Source: "
      f"{src}, `onchainCost` (`results/mci_loan/v2/onchain/cost_comparison.csv`).")
    a("")

    a("## 7. The on-chain runs")
    a("")
    a("| Run | " + " | ".join(f"{lab}: path, U_fed / U_A, E, awards (tokens)" for _, lab in SIDES) + " |")
    a("|---:|" + "---|" * len(SIDES))
    for rs in zip(E["onchainRuns"], L1["onchainRuns"], L2["onchainRuns"]):
        a(f"| {rs[0]['run']} | " + " | ".join(f"{x['path']}, {x['U_fed']:,} / {x['U_A']:,}, {x['E']:,}, {' / '.join(f'{t:,}' for t in x['awards_tokens'])}"
                                         for x in rs) + " |")
    a("")
    zero = [sum(1 for x in s["onchainRuns"] if x["E"] == 0) for s in (E, L1, L2)]
    acc = all(x["accepted"] for s in (E, L1, L2) for x in s["onchainRuns"])
    a(("Every figure was accepted with its proof in all fifteen runs. " if acc else "**Not every figure was accepted in all fifteen runs.** ")
      + "Runs with E = 0 (U_A at or above "
      "U_fed on the run's 64 evaluation rows, so the active member keeps B): " +
      ", ".join(f"{lab} {z} of 5" for (_, lab), z in zip(SIDES, zero)) + f". Source: {src}, `*.onchainRuns`.")
    a("")
    text = "\n".join(w)
    bad = sorted(set(FORBIDDEN.findall(text)))
    if bad:
        sys.exit(f"the naming rule: the document names {bad}")
    DOC.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    c = collect()
    bad = sorted(set(FORBIDDEN.findall(json.dumps(c, ensure_ascii=False))))
    if bad:
        sys.exit(f"the naming rule: the collected data names {bad}")
    cm.write_json(OUT, c, indent=1)
    write_doc(c)
    print(json.dumps({"out": str(OUT.relative_to(cm.ROOT)), "doc": str(DOC.relative_to(cm.ROOT))}))


if __name__ == "__main__":
    main()
