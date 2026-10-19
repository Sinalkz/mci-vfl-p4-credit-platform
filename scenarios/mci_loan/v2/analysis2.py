"""MCI loan instance v2, Step 4: tables (CSV) and a summary (JSON) from the runs of scenarios/mci_loan/v2/offchain/step4v2.mjs.

v1's analysis (scenarios/mci_loan/analysis.py) under v2's roles: MCI is the active member and BASELINE (U_A) is MCI
alone; the passives are the Bank, the E-commerce company and the Credit bureau. The protocol quantities come from the
thesis's own library, unchanged (py/data1_analysis.py: loo_block, ranking, ranking_stability, e1_talmud_rows,
summarize, paired; py/ldp_analysis.py: run_row, draw_rows, seed_n64, condition_tables, noise_rows, check_pairing). The
Talmud awards at the proposal's pool are what MCI pays each partner. The off-chain layer (decision (a): off-chain, not
ZK-verified), on the test split and its thin_file segment, weighted to application_train's default rate:
  - AUC and v1's two business metrics with MCI alone as the baseline (v2/metrics2.py);
  - the credit policy (v2/policy.py): the cross-fitted, prior-corrected PD and the three tiers, per model;
  - the age check: v1's mean margin percentile per age band, and the weighted share of each band in each tier.
v2 has no sensitivity variant. C1b (MCI trained alone) is checked equal to v1's C1b, run for run: MCI's 8 columns are
v1's Bank's, and C1b reads no passive data.

Outputs (results/mci_loan/v2/offchain/; every schema is in docs/mci/v2/03-results.md):
  runs.csv, configs.csv, configs_summary.csv, equivalence.csv, loo.csv, loo_summary.csv, paired.csv, ranking.csv,
  talmud.csv, payments.csv, payments_summary.csv, n64.csv, auc.csv, auc_summary.csv, business.csv,
  business_summary.csv, tiers.csv, tiers_summary.csv, calibration.csv, age_margin.csv, age_margin_summary.csv,
  age_tiers.csv, age_tiers_summary.csv, summary.json; with the LDP runs also ldp_*.csv, ldp_eps_*.csv, ldp_noise.csv,
  ldp_eps_levels.csv, ldp_scores.csv and ldp_scores_summary.csv.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/v2/analysis2.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE.parents[2] / "py")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata  # noqa: E402

import age_check  # noqa: E402
import common as cm  # noqa: E402
import common2 as c2  # noqa: E402
import data1_analysis as da  # noqa: E402
import ldp_analysis as la  # noqa: E402
import metrics2  # noqa: E402
import policy  # noqa: E402
from analysis import Rows, load, scores, summary_rows, unused  # noqa: E402  (v1's role-neutral helpers)

NM = c2.names()
la.PASSIVES = {i: NM[p] for i, p in enumerate(["bank", "ecommerce", "bureau"], 1)}  # v2's names in the thesis's ranking tables
OUT = Path(os.environ["MCI_LOAN_OFFCHAIN"]) if os.environ.get("MCI_LOAN_OFFCHAIN") else c2.RESULTS2 / "offchain"  # a test copy
RUNS = OUT / "runs"
V1_RUNS = cm.RESULTS / "offchain" / "runs"
SEEDS = list(range(1, 11))
MODELS = ["BASELINE", "FULL", "FULL_J", "LOO_1", "LOO_2", "LOO_3"]
LABEL = {"BASELINE": "MCI alone (U_A)", "FULL": "all members (F_r)", "FULL_J": "all members, joint head (F_j)",
         "LOO_1": f"all but the {NM['bank']}", "LOO_2": f"all but the {NM['ecommerce']}", "LOO_3": f"all but the {NM['bureau']}",
         "C1b": "MCI trained alone (C1b)"}
SCALES = [0.0, 1.0, 0.5, 4.0]
OFFCHAIN = "off-chain, not ZK-verified"


def wcsv(name, header, rows):
    return da.write_csv(name, header, rows, OUT)


def with_c1b(models: dict, c1b_run: dict) -> dict:
    return {**models, "C1b": (np.asarray(c1b_run["margins"], dtype=float), np.asarray(c1b_run["preds"]))}


# ---- the off-chain layer -----------------------------------------------------------------------------------

def auc_rows(seed, models, R):
    return [[seed, seg, k, LABEL[k], int(mask.sum()), int(R.y[mask].sum()), metrics2.auc(mg[mask], R.y[mask]), OFFCHAIN]
            for seg, mask in R.segments().items() for k, (mg, _) in models.items()]


BUSINESS_HEADER = ["seed", "segment", "model", "label", "rows", "defaults", "approval_rate_r",
                   "approved_rows", "approved_share", "default_rate_among_approved", "relative_to_baseline",
                   "d_star", "approved_rows_at_d_star", "approved_share_at_d_star", "default_rate_at_d_star",
                   "extra_approval_share", "extra_per_10000", "approved_relative_to_baseline", "baseline_slack", "verified"]


def business_rows(seed, models, R):
    out = []
    for seg, mask in R.segments().items():
        b = metrics2.business({k: (v[0][mask], v[1][mask]) for k, v in models.items()}, R.y[mask], R.w[mask], "BASELINE")
        sec = b["atBaselineDefaultRate"]
        for k in models:
            p, s = b["models"][k], sec["models"][k]
            out.append([seed, seg, k, LABEL[k], b["rows"], int(R.y[mask].sum()), b["approvalRate"],
                        p["approvedRows"], p["approvedWeightShare"], p["defaultRateAmongApproved"], p["relativeToBaseline"],
                        sec["defaultRate"], s["approvedRows"], s["approvedWeightShare"], s["defaultRateAmongApproved"],
                        s["extraApprovalShare"], s["extraPer10000"], s["relativeToBaseline"], sec["baselineShareMinusApprovalRate"],
                        OFFCHAIN])
    return out


TIER_HEADER = ["seed", "segment", "model", "label", "tier", "tier_name", "rows", "defaults", "share", "default_rate", "mean_pd", "verified"]
TIER_KEYS = [T["key"] for T in policy.TIERS] + ["granted", "all"]
TIER_NAME = {**{T["key"]: T["name"] for T in policy.TIERS}, "granted": "credit granted (both tiers)", "all": "all subscribers"}


def tier_rows(seed, pol):
    out, cal = [], []
    for k, v in pol["models"].items():
        for seg, t in (("all", v["overall"]), ("thin_file", v["thinFile"])):
            for key in TIER_KEYS:
                x = t["tiers"][key] if key in t["tiers"] else t[key]
                out.append([seed, seg, k, LABEL[k], key, TIER_NAME[key], x["rows"], x["defaults"], x["share"], x["defaultRate"],
                            x["meanPD"], OFFCHAIN])
        for f in v["calibration"]:
            cal.append([seed, k, LABEL[k], f["fittedOnFold"], f["appliedToFold"], f["fitRows"], f["appliedRows"], f["a"], f["b"],
                        f["marginMean"], f["marginSd"], f["fitDefaultShare"], f["priorShift"]])
    return out, cal


CAL_HEADER = ["seed", "model", "label", "fitted_on_fold", "applied_to_fold", "fit_rows", "applied_rows", "a", "b",
              "margin_mean", "margin_sd", "fit_default_share", "prior_shift"]


# ---- the main set -----------------------------------------------------------------------------------------

def main_set(m, R, S):
    c2r = {s: load(RUNS / f"c2-s{s}.json") for s in SEEDS}
    c4a32 = {s: load(RUNS / f"c4a32-s{s}.json") for s in SEEDS}
    c1b = {s: load(RUNS / f"c1b-s{s}.json") for s in SEEDS}
    n64 = {s: load(RUNS / f"n64-c2-s{s}.json") for s in SEEDS}
    for s in SEEDS:
        if len({c2r[s]["configHash"], c4a32[s]["configHash"], c1b[s]["configHash"]}) != 1:
            raise ValueError(f"seed {s}: C2, C4a-f32 and C1b ran with different model configs")
        if Path(n64[s]["source"]).name != f"c2-s{s}.json":
            raise ValueError(f"n64 seed {s} is not from c2-s{s}.json")
    runs = [("C2", s, c2r[s]) for s in SEEDS] + [("C4a-f32", s, c4a32[s]) for s in SEEDS] + [("C1b", s, c1b[s]) for s in SEEDS]
    S["unusedClassTotal"] = {"main set": sum(unused(r) for _, _, r in runs)}

    # C1b against v1's C1b: the same 8 columns, the same seeds, no passive data
    same = []
    if all((V1_RUNS / f"c1b-s{s}.json").exists() for s in SEEDS):  # delivery: v1's runs are not delivered; checked when present
        for s in SEEDS:
            v1 = load(V1_RUNS / f"c1b-s{s}.json")
            same.append({"seed": s, "F_v2": c1b[s]["F"], "F_v1": v1["F"], "predsEqual": c1b[s]["preds"] == v1["preds"],
                         "marginsEqual": c1b[s]["margins"] == v1["margins"]})
        S["c1bEqualsV1"] = {"all": all(x["predsEqual"] and x["marginsEqual"] and x["F_v2"] == x["F_v1"] for x in same), "seeds": same}
    else:
        S["c1bEqualsV1"] = {"all": None, "seeds": [], "note": "not checked: v1's run files are not in this delivery"}

    # configurations
    blocks = {s: da.loo_block(c2r[s]) for s in SEEDS}
    per = {"C1a (protocol baseline, U_A: MCI alone)": [blocks[s]["F"]["BASELINE"] for s in SEEDS],
           "C1b (MCI trained alone)": [c1b[s]["F"] for s in SEEDS],
           "C2 U_fed = max(F_r, F_j)": [blocks[s]["F"]["FULL"] for s in SEEDS],
           "C2 F_r (retrained FULL)": [blocks[s]["F"]["F_r"] for s in SEEDS],
           "C2 F_j (jointly trained FULL)": [blocks[s]["F"]["F_j"] for s in SEEDS],
           "C4a-f32 (centralized, float32 transport)": [c4a32[s]["F"] for s in SEEDS]}
    wcsv("configs.csv", ["config", *[f"seed_{s}" for s in SEEDS]], [[k, *v] for k, v in per.items()])
    wcsv("configs_summary.csv", da.SUMMARY_HEADER, [da.srow(k, da.summarize(v)) for k, v in per.items()])
    S["configs"] = {k: da.summarize(v) for k, v in per.items()}

    # equivalence: split training against centralized training with float32 transport, bit for bit
    eq = []
    for s in SEEDS:
        l2 = [r["loss"] for r in c2r[s]["training"]["rounds"]]
        l32 = [r["loss"] for r in c4a32[s]["rounds"]]
        fj = da.fig(c2r[s], "FULL_J")
        eq.append([s, max(abs(a - b) for a, b in zip(l2, l32)), fj["F"], c4a32[s]["F"],
                   "yes" if l2 == l32 and fj["preds"] == c4a32[s]["preds"] else "NO"])
    wcsv("equivalence.csv", ["seed", "max_abs_loss_diff_C2_vs_C4a_f32", "F_C2_joint (F_j)", "F_C4a_f32", "C2_equals_C4a_f32_bitwise"], eq)
    S["equivalence"] = {"all_bitwise": all(r[4] == "yes" for r in eq), "seeds": len(eq)}

    # LOO, E, paired comparisons
    rows = [[s, blocks[s]["F"]["FULL"], blocks[s]["F"]["F_r"], blocks[s]["F"]["F_j"], *[blocks[s]["F"][f"LOO_{i}"] for i in (1, 2, 3)],
             blocks[s]["F"]["BASELINE"], c1b[s]["F"], *blocks[s]["l"], *blocks[s]["claims"], blocks[s]["E"],
             "yes" if blocks[s]["binds"] else "no"] for s in SEEDS]
    wcsv("loo.csv", ["seed", "U_fed", "F_r", "F_j", "F_LOO_1", "F_LOO_2", "F_LOO_3", "F_C1a", "F_C1b", "l_1", "l_2", "l_3",
                     "c_1", "c_2", "c_3", "E", "F_j_binds"], rows)
    q = {"U_fed": [blocks[s]["F"]["FULL"] for s in SEEDS], "F_C1a": [blocks[s]["F"]["BASELINE"] for s in SEEDS],
         "F_C1b": [c1b[s]["F"] for s in SEEDS], "E": [blocks[s]["E"] for s in SEEDS],
         **{f"l_{i}": [blocks[s]["l"][i - 1] for s in SEEDS] for i in (1, 2, 3)},
         **{f"F_LOO_{i}": [blocks[s]["F"][f"LOO_{i}"] for s in SEEDS] for i in (1, 2, 3)}}
    wcsv("loo_summary.csv", da.SUMMARY_HEADER, [da.srow(k, da.summarize(v)) for k, v in q.items()])
    pairs = {"U_fed - C1a": da.paired(q["U_fed"], q["F_C1a"]), "U_fed - C1b": da.paired(q["U_fed"], q["F_C1b"]),
             "C1b - C1a": da.paired(q["F_C1b"], q["F_C1a"])}
    for i, j in combinations((1, 2, 3), 2):
        pairs[f"l_{i} - l_{j}"] = da.paired(q[f"l_{i}"], q[f"l_{j}"])
    wcsv("paired.csv", da.SUMMARY_HEADER + ["positive", "zero"], [da.srow(k, v, [v["positive"], v["zero"]]) for k, v in pairs.items()])
    S["loo"] = {k: da.summarize(v) for k, v in q.items()}
    S["paired"] = pairs
    S["F_j_binds"] = sum(1 for s in SEEDS if blocks[s]["binds"])

    # rank stability across seeds (the thesis's ranking_stability and Friedman test, descriptive)
    ls = [blocks[s]["l"] for s in SEEDS]
    st = da.ranking_stability(ls)
    fr = friedmanchisquare(*[[l[i] for l in ls] for i in range(3)])
    wcsv("ranking.csv", ["seed", "l_1", "l_2", "l_3", "ranking (largest l first)"],
         [[s, *blocks[s]["l"], la.rank_names(da.ranking(blocks[s]["l"]))] for s in SEEDS] +
         [["modal", "", "", "", f"{la.rank_names(st['modal'])} in {st['count']} of {st['n']} seeds"]] +
         [[f"share l_{k.replace('>', ' > l_')}", "", "", "", v] for k, v in st["pairwise"].items()] +
         [[f"Friedman chi2 (descriptive, n = {len(SEEDS)})", "", "", "", f"{fr.statistic:.3f}; p = {fr.pvalue:.4f}; "
           f"smallest attainable p: {da.cell(da.friedman_floor(len(SEEDS)))}"]])
    S["ranking"] = {"modal": la.rank_names(st["modal"]), "count": st["count"], "n": st["n"], "pairwise": st["pairwise"],
                    "rankings": [la.rank_names(r) for r in st["rankings"]],
                    "friedman": {"chi2": float(fr.statistic), "p": float(fr.pvalue), "floor_p": da.friedman_floor(len(SEEDS))}}

    # Talmud per seed and pool (10^12: never binds; the proposal's B = 1,000,003, escrowed by MCI; rho * floor(E / 2): binds)
    trow, tal, prow = [], {}, []
    for s in SEEDS:
        b = blocks[s]
        for r in da.e1_talmud_rows(b["E"], b["claims"]):
            trow.append([s, r["pool"], r["E"], r["E_prime"], r["regime"], *r["claims"], *r["awards_perf"], *r["awards_tokens"], r["to_A"]])
            key = {da.POOL_NONBINDING: "1e12", da.B_PROPOSAL: "B_proposal"}.get(r["pool"], "rho*floor(E/2)")
            tal.setdefault(key, []).append(r)
            if r["pool"] == da.B_PROPOSAL:  # what MCI pays each partner, and what stays with MCI
                for i in (1, 2, 3):
                    prow.append([s, i, la.PASSIVES[i], b["l"][i - 1], r["claims"][i - 1], r["awards_tokens"][i - 1], r["pool"], r["regime"]])
                prow.append([s, 0, NM["mci"], "", "", r["to_A"], r["pool"], r["regime"]])
    wcsv("talmud.csv", ["seed", "pool", "E", "E_prime", "regime", "c_1", "c_2", "c_3", "award_1_perf", "award_2_perf",
                        "award_3_perf", "award_1_tokens", "award_2_tokens", "award_3_tokens", "to_A_normal_path"], trow)
    ph = ["seed", "member_index", "member", "l", "claim", "tokens", "pool_B", "regime"]
    wcsv("payments.csv", ph, prow)
    P = pd.DataFrame(prow, columns=ph)
    wcsv("payments_summary.csv", ["member", "quantity", *da.SUMMARY_HEADER[1:]], summary_rows(P, ["member"], ["tokens"]))
    S["talmud"] = {k: {"regimes": dict(Counter(r["regime"] for r in v)), "E_prime": da.summarize([r["E_prime"] for r in v]),
                       "awards_tokens": [da.summarize([r["awards_tokens"][i] for r in v]) for i in range(3)],
                       "to_A": da.summarize([r["to_A"] for r in v])} for k, v in tal.items()}
    S["payments"] = {mem: da.summarize(g["tokens"].astype(float).tolist()) for mem, g in P.groupby("member", sort=False)}

    # N = 64 draws: the agreement of the 64-row LOO with the whole-test-split LOO (decision (b))
    nrow, agree = [], []
    for s in SEEDS:
        dr = la.draw_rows(n64[s])
        full = blocks[s]["l"]
        rank_full = da.ranking(full)
        sgn = {(i, j): np.sign(full[i - 1] - full[j - 1]) for i, j in combinations((1, 2, 3), 2)}
        same_r = float(np.mean([d["ranking"] == rank_full for d in dr]))
        top = float(np.mean([d["ranking"][0] == rank_full[0] for d in dr]))
        pair = float(np.mean([np.mean([np.sign(d[f"l_{i}"] - d[f"l_{j}"]) == v for (i, j), v in sgn.items()]) for d in dr]))
        sn = la.seed_n64(dr)
        agree.append({"seed": s, "same_ranking": same_r, "same_top": top, "pairwise": pair})
        nrow.append([s, la.rank_names(rank_full), len(dr), float(np.mean([x["distinct"] for x in n64[s]["draws"]])), same_r, top, pair,
                     float(np.mean([x["F"]["FULL_J"] > x["F"]["FULL"] for x in n64[s]["draws"]])),
                     *[v for i in (1, 2, 3) for v in (full[i - 1], sn[f"l_{i}"]["mean"], sn[f"l_{i}"]["sd"], sn[f"l_{i}"]["share_le_0"])],
                     blocks[s]["E"], sn["E"]["mean"], sn["E"]["sd"]])
    wcsv("n64.csv", ["seed", "full_test_ranking", "draws", "mean_distinct_rows", "share_same_ranking_as_full_test",
                     "share_same_top_passive", "share_pairwise_order_agreeing", "share_F_j_binds",
                     *[c for i in (1, 2, 3) for c in (f"l_{i}_full_test", f"l_{i}_mean_N64", f"l_{i}_sd_N64", f"l_{i}_share_le_0")],
                     "E_full_test", "E_mean_N64", "E_sd_N64"], nrow)
    S["n64"] = {k: da.summarize([a[k] for a in agree]) for k in ("same_ranking", "same_top", "pairwise")}

    # the off-chain layer: AUC, the business metrics and the credit policy
    arow, brow, trow2, crow = [], [], [], []
    ids = np.asarray(m["ids"])[R.test]
    pi = m["population"]["default_rate"]
    pd_of = {}
    for s in SEEDS:
        models = with_c1b(scores(c2r[s]), c1b[s])
        arow += auc_rows(s, models, R)
        brow += business_rows(s, models, R)
        pol = policy.policy({k: v[0] for k, v in models.items()}, ids, R.y, R.w, pi, R.thin)
        t, c = tier_rows(s, pol)
        trow2 += t
        crow += c
        fold = policy.calibration_folds(ids, R.y)
        pd_of[s] = {k: policy.calibrated_pd(models[k][0], R.y, fold, pi)[0] for k in ("BASELINE", "FULL")}
    ah = ["seed", "segment", "model", "label", "rows", "defaults", "auc", "verified"]
    wcsv("auc.csv", ah, arow)
    A = pd.DataFrame(arow, columns=ah)
    wcsv("auc_summary.csv", ["segment", "model", "quantity", *da.SUMMARY_HEADER[1:]], summary_rows(A, ["segment", "model"], ["auc"]))
    B = pd.DataFrame(brow, columns=BUSINESS_HEADER)
    wcsv("business.csv", BUSINESS_HEADER, brow)
    bq = ["approval_rate_r", "default_rate_among_approved", "relative_to_baseline", "d_star", "approved_share_at_d_star",
          "extra_approval_share", "extra_per_10000", "baseline_slack"]
    wcsv("business_summary.csv", ["segment", "model", "quantity", *da.SUMMARY_HEADER[1:]], summary_rows(B, ["segment", "model"], bq))
    T = pd.DataFrame(trow2, columns=TIER_HEADER)
    wcsv("tiers.csv", TIER_HEADER, trow2)
    wcsv("tiers_summary.csv", ["segment", "model", "tier", "quantity", *da.SUMMARY_HEADER[1:]],
         summary_rows(T, ["segment", "model", "tier"], ["share", "default_rate", "mean_pd", "rows", "defaults"]))
    wcsv("calibration.csv", CAL_HEADER, crow)
    S["auc"] = {f"{k[0]}/{k[1]}": da.summarize(g["auc"].astype(float).tolist())
                for k, g in A.groupby(["segment", "model"], sort=False) if g["auc"].notna().all()}
    S["business"] = {f"{k[0]}/{k[1]}": {q_: da.summarize(g[q_].dropna().astype(float).tolist()) for q_ in bq if g[q_].notna().any()}
                     for k, g in B.groupby(["segment", "model"], sort=False)}
    S["tiers"] = {f"{k[0]}/{k[1]}/{k[2]}": {q_: da.summarize(g[q_].dropna().astype(float).tolist())
                                            for q_ in ("share", "default_rate", "mean_pd", "rows", "defaults") if g[q_].notna().any()}
                  for k, g in T.groupby(["segment", "model", "tier"], sort=False)}
    S["tierRule"] = {"cuts": list(policy.CUTS), "tiers": policy.TIERS, "calibrationSeed": c2.CALIB_SEED,
                     "crossFit": "2 folds of the test rows: calibrate on one, report on the other, swap, pool; every model alike"}
    S["segments"] = {"test_rows": int(len(R.y)), "test_defaults": int(R.y.sum()), "thin_file_test_rows": int(R.thin.sum()),
                     "thin_file_test_defaults": int(R.y[R.thin].sum())}

    # the age check (pre-registration, section 5.5): v1's mean margin percentile per age band, and the weighted share of
    # each band in each tier, for MCI alone (BASELINE) and the federation (FULL)
    grow, gtrow = [], []
    bands = [f"{age_check.AGE_BANDS[k]}-{age_check.AGE_BANDS[k + 1]}" for k in range(len(age_check.AGE_BANDS) - 1)]
    for s in SEEDS:
        models = scores(c2r[s])
        for name in ("BASELINE", "FULL"):
            mg = 100 * rankdata(models[name][0]) / len(models[name][0])
            means = [float(mg[R.band == k].mean()) for k in range(len(bands))]
            tier = policy.tier_of(pd_of[s][name])
            for k, b in enumerate(bands):
                inb = R.band == k
                grow.append([s, name, LABEL[name], b, int(inb.sum()), means[k], all(x > y for x, y in zip(means, means[1:])), OFFCHAIN])
                for t, T_ in enumerate(policy.TIERS):
                    sel = inb & (tier == t)
                    gtrow.append([s, name, LABEL[name], b, T_["key"], T_["name"], int(inb.sum()), int(sel.sum()),
                                  float(R.w[sel].sum() / R.w[inb].sum()), OFFCHAIN])
    G = pd.DataFrame(grow, columns=["seed", "model", "label", "age_band", "rows", "mean_margin_percentile", "falls_with_every_band", "verified"])
    wcsv("age_margin.csv", list(G.columns), grow)
    wcsv("age_margin_summary.csv", ["model", "age_band", "quantity", *da.SUMMARY_HEADER[1:]],
         summary_rows(G, ["model", "age_band"], ["mean_margin_percentile"]))
    gth = ["seed", "model", "label", "age_band", "tier", "tier_name", "band_rows", "rows_in_tier", "share_of_band", "verified"]
    GT = pd.DataFrame(gtrow, columns=gth)
    wcsv("age_tiers.csv", gth, gtrow)
    wcsv("age_tiers_summary.csv", ["model", "age_band", "tier", "quantity", *da.SUMMARY_HEADER[1:]],
         summary_rows(GT, ["model", "age_band", "tier"], ["share_of_band"]))
    oldest = {name: [float(G[(G.seed == s) & (G.model == name)]["mean_margin_percentile"].iloc[-1] -
                           G[(G.seed == s) & (G.model == name)]["mean_margin_percentile"].iloc[0]) for s in SEEDS] for name in ("BASELINE", "FULL")}
    S["age_bands"] = {name: {b: da.summarize(G[(G.model == name) & (G.age_band == b)]["mean_margin_percentile"].tolist()) for b in bands}
                      for name in ("BASELINE", "FULL")}
    S["age"] = {name: {"seeds_falling_with_every_band": int(G[(G.model == name) & (G.age_band == bands[0])]["falls_with_every_band"].sum()),
                       "oldest_minus_youngest_band": da.summarize(oldest[name])} for name in ("BASELINE", "FULL")}
    S["age_tiers"] = {f"{k[0]}/{k[1]}/{k[2]}": da.summarize(g["share_of_band"].tolist())
                      for k, g in GT.groupby(["model", "age_band", "tier"], sort=False)}
    return runs


# ---- the LDP sweep (v1's, with v2's names) ----------------------------------------------------------------

def ldp_set(R, S):
    """The noise-scale sweep and the epsilon curve, with the thesis's condition_tables, and the off-chain scores per
    condition. Requires every run of both sets."""
    tag = la.tag
    runs = {k: {s: load(RUNS / f"ldp-{tag(k)}-s{s}.json") for s in SEEDS} for k in SCALES}
    n64s = {k: {s: load(RUNS / f"n64-{tag(k)}-s{s}.json") for s in SEEDS} for k in SCALES}
    c2r = {s: load(RUNS / f"c2-s{s}.json") for s in SEEDS}
    for s in SEEDS:
        la.check_pairing([runs[k][s] for k in SCALES] + [c2r[s]])
    la.check_n64_sources(n64s, lambda k, s: f"ldp-{tag(k)}-s{s}.json")
    if la.figures_of(OUT / "check-c2-s1.json") != la.figures_of(RUNS / "c2-s1.json"):
        raise ValueError("the scales set's check run does not reproduce C2 seed 1")
    c2row = {s: la.run_row(c2r[s]) for s in SEEDS}
    label = {0.0: "LDP off (clipping only, σ = 0)", 1.0: "LDP on (protocol σ)", 0.5: "weaker noise (σ × 0.5)",
             4.0: "stronger noise (σ × 4)"}
    da.write_csv("ldp_noise.csv", ["condition", "sigma_scale", "passive", "clip", "sigma", "sigma_over_C",
                                   "eps_per_release_analytic", "eps_total_21_releases", "delta_total"],
                 [[label[k], k, f"{i} {la.PASSIVES[i]}", r["clip"], r["sigma"], r["sigma_over_C"], r["eps_per_release_analytic"],
                   r["eps_total"], la.sci(r["delta_total"])] for k in SCALES for i, r in enumerate(la.noise_rows(runs[k][1]), 1)], OUT)
    SS, _, _ = la.condition_tables("ldp", SCALES, label, runs, n64s, 0.0, ("sigma_scale", lambda k: k), tag,
                                   extra=[("LDP off (clipping only) − C2 (no clipping)", 0.0, c2row)], also=[c2row],
                                   seeds=SEEDS, sfx="", out=OUT)
    S["ldp"] = {"scales": SS, "resources": la.resources([runs[k][s] for k in SCALES for s in SEEDS])}

    lv = load(cm.ROOT / "results" / "ldp" / "eps-levels.json")
    ekeys = ["off"] + [f"e{x['eps']:g}" for x in lv["levels"]]
    elabel = {"off": label[0.0], **{f"e{x['eps']:g}": f"ε = {x['eps']:g} per release" for x in lv["levels"]}}
    eps_of = {"off": float("inf"), **{f"e{x['eps']:g}": x["eps"] for x in lv["levels"]}}
    eruns = {"off": runs[0.0], **{k: {s: load(RUNS / f"eps-{k}-s{s}.json") for s in SEEDS} for k in ekeys[1:]}}
    en64 = {"off": n64s[0.0], **{k: {s: load(RUNS / f"n64-eps-{k}-s{s}.json") for s in SEEDS} for k in ekeys[1:]}}
    for s in SEEDS:
        la.check_pairing([eruns[k][s] for k in ekeys])
    la.check_n64_sources({k: en64[k] for k in ekeys[1:]}, lambda k, s: f"eps-{k}-s{s}.json")
    for x in lv["levels"]:
        got = [p["sigma"] for p in eruns[f"e{x['eps']:g}"][1]["noise"]["perPassive"]]
        if any(abs(g - p["sigma"]) > 1e-12 * p["sigma"] for g, p in zip(got, x["passives"])):
            raise ValueError(f"eps {x['eps']}: run sigmas {got} are not the calibrated ones")
    if la.figures_of(OUT / "check-eps-x0-s1.json") != la.figures_of(RUNS / "ldp-x0-s1.json"):
        raise ValueError("the eps set's check run does not reproduce LDP off, seed 1")
    ES, _, err = la.condition_tables("ldp_eps", ekeys, elabel, eruns, en64, "off", ("eps_per_release", eps_of.get),
                                     lambda k: k, seeds=SEEDS, sfx="", out=OUT)
    da.write_csv("ldp_eps_levels.csv", ["eps_per_release", "passive", "clip", "sigma", "sigma_over_C", "eps_check",
                                        "delta_per_release", "releases", "eps_total_gdp", "delta_total_gdp"],
                 [[x["eps"], f"{p['passive']} {la.PASSIVES[p['passive']]}", p["clip"], p["sigma"], p["sigma_over_C"], p["eps_check"],
                   la.sci(lv["delta"]), x["releases"], x["eps_total_gdp"], la.sci(x["delta_total_gdp"])]
                  for x in lv["levels"] for p in x["passives"]], OUT)
    gap = {k: [err[k][s]["U_fed"] - err[k][s]["BASELINE"] for s in SEEDS] for k in ekeys}
    ES["utility"] = {k: {**da.summarize(v), "seeds_above": sum(1 for g in v if g > 0)} for k, v in gap.items()}
    ES["levels_source"] = "results/ldp/eps-levels.json (the thesis's calibration, unchanged)"
    S["ldp"]["eps"] = ES
    S["ldp"]["resources_eps"] = la.resources([eruns[k][s] for k in ekeys[1:] for s in SEEDS])

    # off-chain scores per condition: AUC and the primary and secondary business metrics of every figure
    conds = [(label[k], "sigma_scale", k, runs[k]) for k in SCALES] + [(elabel[k], "eps_per_release", eps_of[k], eruns[k]) for k in ekeys[1:]]
    orow = []
    for name, kind, val, rs in conds:
        for s in SEEDS:
            models = scores(rs[s])
            b = metrics2.business(models, R.y, R.w, "BASELINE")
            for mname in MODELS:
                orow.append([name, kind, val, s, mname, LABEL[mname], metrics2.auc(models[mname][0], R.y), b["approvalRate"],
                             b["models"][mname]["defaultRateAmongApproved"], b["atBaselineDefaultRate"]["models"][mname]["extraPer10000"], OFFCHAIN])
    oh = ["condition", "kind", "value", "seed", "model", "label", "auc", "approval_rate_r", "default_rate_among_approved", "extra_per_10000", "verified"]
    wcsv("ldp_scores.csv", oh, orow)
    O = pd.DataFrame(orow, columns=oh)
    wcsv("ldp_scores_summary.csv", ["condition", "model", "quantity", *da.SUMMARY_HEADER[1:]],
         summary_rows(O, ["condition", "model"], ["auc", "default_rate_among_approved", "extra_per_10000"]))
    S["unusedClassTotal"]["ldp"] = sum(unused(rs[s]) for _, _, _, rs in conds for s in SEEDS)
    return [(f"LDP {name}", s, rs[s]) for name, _, _, rs in conds for s in SEEDS]


def main() -> None:
    m = load(c2.PREPARED2 / "manifest.json")  # delivery: the full manifest is the data's; results/ holds a redacted copy
    R = Rows(m)
    S = {"what": "the MCI loan instance v2's off-chain results (scenarios/mci_loan/v2/analysis2.py)", "seeds": SEEDS,
         "names": {"active": NM["mci"], "passives": la.PASSIVES},
         "meta": {name: load(OUT / f"meta-{name}.json") for name in ("main", "scales", "eps") if (OUT / f"meta-{name}.json").exists()},
         "prepared": {"aSha256": m["a"]["sha256"], "passiveSha256": m["passive"]["sha256"]}}
    runs = main_set(m, R, S)
    have_ldp = all((OUT / f"meta-{x}.json").exists() for x in ("scales", "eps"))
    if have_ldp:
        runs += ldp_set(R, S)
    for label, s, r in runs:
        if "prepared" in r and r["prepared"]["passiveSha256"] != m["passive"]["sha256"]:
            raise ValueError(f"{label} seed {s} ran on other prepared data")
    wcsv("runs.csv", ["config", "seed", "wall_s", "training_s", "peak_rss_mb", "free_ram_mb_start", "free_ram_mb_end",
                      "zero_filled", "hq_saturated_p1", "hq_saturated_p2", "hq_saturated_p3", "rows_in_classes_2_4", "config_hash"],
         [[c, s, r["wallMs"] / 1000, r.get("trainingMs", 0) / 1000, r["peakRssMB"], r["machine"]["freeRamMB"], r.get("freeRamMBAtEnd", ""),
           len(r["training"]["zeroFilled"]) if "training" in r else "", *r["clipping"]["hq"], unused(r), r["configHash"]] for c, s, r in runs])
    S["resources"] = la.resources([r for _, _, r in runs])
    S["machine"] = runs[0][2]["machine"]
    S["ldpIncluded"] = have_ldp
    cm.write_json(OUT / "summary.json", S, indent=1)
    print(json.dumps({"unusedClassTotal": S["unusedClassTotal"], "equivalence": S["equivalence"], "c1bEqualsV1": S["c1bEqualsV1"]["all"],
                      "U_fed": round(S["loo"]["U_fed"]["mean"]), "C1a": round(S["loo"]["F_C1a"]["mean"]),
                      "l": {k: round(S["loo"][k]["mean"], 1) for k in ("l_1", "l_2", "l_3")}, "ranking": S["ranking"]["modal"],
                      "count": S["ranking"]["count"], "n64_same": round(S["n64"]["same_ranking"]["mean"], 3), "ldp": have_ldp}, default=str))


if __name__ == "__main__":
    main()
