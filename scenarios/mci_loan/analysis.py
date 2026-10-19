"""MCI loan instance, Step 4: tables (CSV) and a summary (JSON) from the runs of scenarios/mci_loan/offchain/step4.mjs.

The protocol quantities come from the thesis's own analysis library, unchanged: py/data1_analysis.py (loo_block:
U_fed = max(F_r, F_j), l_i = U_fed - F_LOO_i, c_i = max(0, l_i), E = max(0, U_fed - F_BASELINE); ranking,
ranking_stability, e1_talmud_rows: the contract's rule E' = min(E, floor(pool / rho)), Talmud in performance units,
rho = 5,000; summarize, paired) and py/ldp_analysis.py (run_row, draw_rows, seed_n64, condition_tables, noise_rows,
check_pairing), with the passives named MCI, E-commerce company, Credit bureau. The off-chain scores (decision (a): off-chain,
not ZK-verified) come from scenarios/mci_loan/metrics.py: AUC, and the primary and secondary business metrics,
weighted to application_train's default rate (decision (b)), on the test split and its thin_file segment.

Outputs (results/mci_loan/offchain/; every schema is in docs/mci/03-results.md):
  runs.csv, configs.csv, configs_summary.csv, equivalence.csv, loo.csv, loo_summary.csv, paired.csv, ranking.csv,
  talmud.csv, n64.csv, auc.csv, auc_summary.csv, business.csv, business_summary.csv, age_margin.csv,
  age_margin_summary.csv, sensitivity.csv, sensitivity_summary.csv, summary.json; with the LDP runs also ldp_*.csv,
  ldp_eps_*.csv, ldp_noise.csv, ldp_eps_levels.csv, ldp_scores.csv and ldp_scores_summary.csv.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/analysis.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "py"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata  # noqa: E402

import age_check  # noqa: E402
import common as cm  # noqa: E402
import data1_analysis as da  # noqa: E402
import ldp_analysis as la  # noqa: E402
import metrics  # noqa: E402

la.PASSIVES = {1: "MCI", 2: "E-commerce company", 3: "Credit bureau"}  # the loan's names in the thesis's ranking tables
OUT = Path(os.environ["MCI_LOAN_OFFCHAIN"]) if os.environ.get("MCI_LOAN_OFFCHAIN") else cm.RESULTS / "offchain"  # a test copy
RUNS = OUT / "runs"
SEEDS = list(range(1, 11))
FIGS = ["BASELINE", "FULL", "FULL_J", "LOO_1", "LOO_2", "LOO_3"]
LABEL = {"BASELINE": "Bank alone (U_A)", "FULL": "all members (F_r)", "FULL_J": "all members, joint head (F_j)",
         "LOO_1": "all but MCI", "LOO_2": "all but the E-commerce company", "LOO_3": "all but the Credit bureau",
         "C1b": "Bank trained alone (C1b)"}
SPLITS = {"main": "c2", "mciPhoneOnly": "c2-mcisens"}
SCALES = [0.0, 1.0, 0.5, 4.0]
OFFCHAIN = "off-chain, not ZK-verified"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wcsv(name, header, rows):
    return da.write_csv(name, header, rows, OUT)


def fig(run, name):
    return da.fig(run, name)


def scores(run) -> dict:
    """name -> (margins, preds) on the test rows, for a VFL run's six figures."""
    return {f["name"]: (np.asarray(f["margins"], dtype=float), np.asarray(f["preds"])) for f in run["figures"]}


def unused(run) -> int:
    if "figures" in run:
        return int(sum(f["unusedClass"]["test"] + f["unusedClass"]["train"] for f in run["figures"]))
    return int(run["unusedClass"]["test"] + run["unusedClass"]["train"])


class Rows:
    """The test split: labels, weights, the thin_file mask and the applicants' ages."""

    def __init__(self, m):
        self.test = np.asarray(m["split"]["test_pos"])
        self.y = np.asarray(m["y"])[self.test]
        self.w = np.vectorize(metrics.class_weights(m["population"]["default_rate"]).get)(self.y)
        self.thin = np.isin(self.test, np.asarray(m["thin_file_pos"], dtype=int))
        ids = np.asarray(m["ids"])[self.test]
        app = pd.read_csv(cm.RAW / "application_train.csv", usecols=["SK_ID_CURR", "DAYS_BIRTH"]).set_index("SK_ID_CURR")
        self.band = age_check.bands(-app.loc[ids, "DAYS_BIRTH"].to_numpy() / 365.25)

    def segments(self):
        return {"all": np.ones(len(self.y), dtype=bool), "thin_file": self.thin}


# ---- off-chain scores ----------------------------------------------------------------------------------

def auc_rows(split, seed, models, R):
    out = []
    for seg, mask in R.segments().items():
        for name, (mg, _) in models.items():
            out.append([split, seed, seg, name, LABEL[name], int(mask.sum()), int(R.y[mask].sum()), metrics.auc(mg[mask], R.y[mask]), OFFCHAIN])
    return out


def business_rows(split, seed, models, R):
    out = []
    for seg, mask in R.segments().items():
        b = metrics.business({k: (v[0][mask], v[1][mask]) for k, v in models.items()}, R.y[mask], R.w[mask], "BASELINE")
        sec = b["atBankAloneDefaultRate"]
        for name in models:
            p, s = b["models"][name], sec["models"][name]
            out.append([split, seed, seg, name, LABEL[name], b["rows"], int(R.y[mask].sum()), b["approvalRate"],
                        p["approvedRows"], p["approvedWeightShare"], p["defaultRateAmongApproved"], p["relativeToBankAlone"],
                        sec["defaultRate"], s["approvedRows"], s["approvedWeightShare"], s["defaultRateAmongApproved"],
                        s["extraApprovalShare"], s["extraPer10000"], s["relativeToBankAlone"],
                        sec["bankAloneShareMinusApprovalRate"], OFFCHAIN])
    return out


BUSINESS_HEADER = ["split", "seed", "segment", "model", "label", "rows", "defaults", "approval_rate_r",
                   "approved_rows", "approved_share", "default_rate_among_approved", "relative_to_bank_alone",
                   "d_star", "approved_rows_at_d_star", "approved_share_at_d_star", "default_rate_at_d_star",
                   "extra_approval_share", "extra_per_10000", "approved_relative_to_bank_alone",
                   "bank_alone_slack", "verified"]


def summary_rows(df: pd.DataFrame, keys: list[str], quantities: list[str]):
    out = []
    for k, g in df.groupby(keys, sort=False):
        for q in quantities:
            v = g[q].dropna().astype(float).tolist()
            if v:
                out.append([*(k if isinstance(k, tuple) else (k,)), q, *da.srow("", da.summarize(v))[1:]])
    return out


# ---- the main set ---------------------------------------------------------------------------------------

def main_set(m, R, S):
    c2 = {s: load(RUNS / f"c2-s{s}.json") for s in SEEDS}
    sens = {s: load(RUNS / f"c2-mcisens-s{s}.json") for s in SEEDS}
    c4a32 = {s: load(RUNS / f"c4a32-s{s}.json") for s in SEEDS}
    c1b = {s: load(RUNS / f"c1b-s{s}.json") for s in SEEDS}
    n64 = {s: load(RUNS / f"n64-c2-s{s}.json") for s in SEEDS}
    for s in SEEDS:
        if len({c2[s]["configHash"], c4a32[s]["configHash"], c1b[s]["configHash"]}) != 1:
            raise ValueError(f"seed {s}: C2, C4a-f32 and C1b ran with different model configs")
        if Path(n64[s]["source"]).name != f"c2-s{s}.json":
            raise ValueError(f"n64 seed {s} is not from c2-s{s}.json")
    runs = [("C2", s, c2[s]) for s in SEEDS] + [("C2 MCI phone only", s, sens[s]) for s in SEEDS] + \
           [("C4a-f32", s, c4a32[s]) for s in SEEDS] + [("C1b", s, c1b[s]) for s in SEEDS]
    S["unusedClassTotal"] = {"main set": sum(unused(r) for _, _, r in runs)}

    # configurations
    blocks = {s: da.loo_block(c2[s]) for s in SEEDS}
    per = {"C1a (protocol baseline, U_A)": [blocks[s]["F"]["BASELINE"] for s in SEEDS],
           "C1b (Bank trained alone)": [c1b[s]["F"] for s in SEEDS],
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
        l2 = [r["loss"] for r in c2[s]["training"]["rounds"]]
        l32 = [r["loss"] for r in c4a32[s]["rounds"]]
        fj = fig(c2[s], "FULL_J")
        eq.append([s, max(abs(a - b) for a, b in zip(l2, l32)), fj["F"], c4a32[s]["F"],
                   "yes" if l2 == l32 and fj["preds"] == c4a32[s]["preds"] else "NO"])
    wcsv("equivalence.csv", ["seed", "max_abs_loss_diff_C2_vs_C4a_f32", "F_C2_joint (F_j)", "F_C4a_f32",
                             "C2_equals_C4a_f32_bitwise"], eq)
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

    # Talmud per seed and pool (10^12: never binds; the proposal's B = 1,000,003; rho * floor(E / 2): binds)
    trow, tal = [], {}
    for s in SEEDS:
        b = blocks[s]
        for r in da.e1_talmud_rows(b["E"], b["claims"]):
            trow.append([s, r["pool"], r["E"], r["E_prime"], r["regime"], *r["claims"], *r["awards_perf"], *r["awards_tokens"], r["to_A"]])
            key = {da.POOL_NONBINDING: "1e12", da.B_PROPOSAL: "B_proposal"}.get(r["pool"], "rho*floor(E/2)")
            tal.setdefault(key, []).append(r)
    wcsv("talmud.csv", ["seed", "pool", "E", "E_prime", "regime", "c_1", "c_2", "c_3", "award_1_perf", "award_2_perf",
                        "award_3_perf", "award_1_tokens", "award_2_tokens", "award_3_tokens", "to_A_normal_path"], trow)
    S["talmud"] = {k: {"regimes": dict(Counter(r["regime"] for r in v)), "E_prime": da.summarize([r["E_prime"] for r in v]),
                       "awards_tokens": [da.summarize([r["awards_tokens"][i] for r in v]) for i in range(3)],
                       "to_A": da.summarize([r["to_A"] for r in v])} for k, v in tal.items()}

    # N = 64 draws: the agreement of the 64-row LOO with the whole-test-split LOO (decision (b))
    nrow, agree = [], []
    for s in SEEDS:
        dr = la.draw_rows(n64[s])
        full = blocks[s]["l"]
        rank_full = da.ranking(full)
        sgn = {(i, j): np.sign(full[i - 1] - full[j - 1]) for i, j in combinations((1, 2, 3), 2)}
        same = float(np.mean([d["ranking"] == rank_full for d in dr]))
        top = float(np.mean([d["ranking"][0] == rank_full[0] for d in dr]))
        pair = float(np.mean([np.mean([np.sign(d[f"l_{i}"] - d[f"l_{j}"]) == v for (i, j), v in sgn.items()]) for d in dr]))
        sn = la.seed_n64(dr)
        agree.append({"seed": s, "same_ranking": same, "same_top": top, "pairwise": pair})
        nrow.append([s, la.rank_names(rank_full), len(dr), float(np.mean([x["distinct"] for x in n64[s]["draws"]])), same, top, pair,
                     float(np.mean([x["F"]["FULL_J"] > x["F"]["FULL"] for x in n64[s]["draws"]])),
                     *[v for i in (1, 2, 3) for v in (full[i - 1], sn[f"l_{i}"]["mean"], sn[f"l_{i}"]["sd"], sn[f"l_{i}"]["share_le_0"])],
                     blocks[s]["E"], sn["E"]["mean"], sn["E"]["sd"]])
    wcsv("n64.csv", ["seed", "full_test_ranking", "draws", "mean_distinct_rows", "share_same_ranking_as_full_test",
                     "share_same_top_passive", "share_pairwise_order_agreeing", "share_F_j_binds",
                     *[c for i in (1, 2, 3) for c in (f"l_{i}_full_test", f"l_{i}_mean_N64", f"l_{i}_sd_N64", f"l_{i}_share_le_0")],
                     "E_full_test", "E_mean_N64", "E_sd_N64"], nrow)
    S["n64"] = {k: da.summarize([a[k] for a in agree]) for k in ("same_ranking", "same_top", "pairwise")}

    # off-chain scores: AUC and the business metrics, main split and sensitivity variant
    arow, brow = [], []
    for split, runs_ in (("main", c2), ("mciPhoneOnly", sens)):
        for s in SEEDS:
            models = scores(runs_[s])
            if split == "main":
                models["C1b"] = (np.asarray(c1b[s]["margins"], dtype=float), np.asarray(c1b[s]["preds"]))
            arow += auc_rows(split, s, models, R)
            brow += business_rows(split, s, models, R)
    wcsv("auc.csv", ["split", "seed", "segment", "model", "label", "rows", "defaults", "auc", "verified"], arow)
    A = pd.DataFrame(arow, columns=["split", "seed", "segment", "model", "label", "rows", "defaults", "auc", "verified"])
    wcsv("auc_summary.csv", ["split", "segment", "model", "quantity", *da.SUMMARY_HEADER[1:]],
         summary_rows(A, ["split", "segment", "model"], ["auc"]))
    B = pd.DataFrame(brow, columns=BUSINESS_HEADER)
    wcsv("business.csv", BUSINESS_HEADER, brow)
    bq = ["approval_rate_r", "default_rate_among_approved", "relative_to_bank_alone", "d_star", "approved_share_at_d_star",
          "extra_approval_share", "extra_per_10000", "bank_alone_slack"]
    wcsv("business_summary.csv", ["split", "segment", "model", "quantity", *da.SUMMARY_HEADER[1:]],
         summary_rows(B, ["split", "segment", "model"], bq))
    S["auc"] = {f"{k[0]}/{k[1]}/{k[2]}": da.summarize(g["auc"].astype(float).tolist())
                for k, g in A.groupby(["split", "segment", "model"], sort=False) if g["auc"].notna().all()}
    S["business"] = {f"{k[0]}/{k[1]}/{k[2]}": {q_: da.summarize(g[q_].dropna().astype(float).tolist()) for q_ in bq if g[q_].notna().any()}
                     for k, g in B.groupby(["split", "segment", "model"], sort=False)}
    S["segments"] = {"test_rows": int(len(R.y)), "test_defaults": int(R.y.sum()), "thin_file_test_rows": int(R.thin.sum()),
                     "thin_file_test_defaults": int(R.y[R.thin].sum())}

    # the age check (section 5.5): per age band on the test split, the mean percentile rank of the margin among the seed's
    # test rows (0-100, higher = riskier; the integer margins themselves have no fixed unit across seeds)
    grow = []
    bands = [f"{age_check.AGE_BANDS[k]}-{age_check.AGE_BANDS[k + 1]}" for k in range(len(age_check.AGE_BANDS) - 1)]
    for s in SEEDS:
        models = scores(c2[s])
        for name in ("BASELINE", "FULL"):
            mg = 100 * rankdata(models[name][0]) / len(models[name][0])
            means = [float(mg[R.band == k].mean()) for k in range(len(bands))]
            for k, b in enumerate(bands):
                grow.append([s, name, LABEL[name], b, int((R.band == k).sum()), means[k],
                             all(x > y for x, y in zip(means, means[1:])), OFFCHAIN])
    G = pd.DataFrame(grow, columns=["seed", "model", "label", "age_band", "rows", "mean_margin_percentile", "falls_with_every_band", "verified"])
    wcsv("age_margin.csv", list(G.columns), grow)
    wcsv("age_margin_summary.csv", ["model", "age_band", "quantity", *da.SUMMARY_HEADER[1:]],
         summary_rows(G, ["model", "age_band"], ["mean_margin_percentile"]))
    oldest = {name: [float(G[(G.seed == s) & (G.model == name)]["mean_margin_percentile"].iloc[-1] -
                           G[(G.seed == s) & (G.model == name)]["mean_margin_percentile"].iloc[0]) for s in SEEDS] for name in ("BASELINE", "FULL")}
    S["age_bands"] = {name: {b: da.summarize(G[(G.model == name) & (G.age_band == b)]["mean_margin_percentile"].tolist()) for b in bands}
                      for name in ("BASELINE", "FULL")}
    S["age"] = {name: {"seeds_falling_with_every_band": int(G[(G.model == name) & (G.age_band == bands[0])]["falls_with_every_band"].sum()),
                       "oldest_minus_youngest_band": da.summarize(oldest[name])} for name in ("BASELINE", "FULL")}

    # the sensitivity variant: MCI with its phone and contact columns only, against the main split
    sb = {s: da.loo_block(sens[s]) for s in SEEDS}
    srow = []
    for s in SEEDS:
        for split, b in (("main", blocks[s]), ("mciPhoneOnly", sb[s])):
            tb = next(r for r in da.e1_talmud_rows(b["E"], b["claims"]) if r["pool"] == da.B_PROPOSAL)
            srow.append([split, s, b["F"]["FULL"], b["F"]["F_r"], b["F"]["F_j"], b["F"]["BASELINE"],
                         *[b["F"][f"LOO_{i}"] for i in (1, 2, 3)], *b["l"], *b["claims"], b["E"], tb["E_prime"], tb["regime"],
                         *tb["awards_tokens"], la.rank_names(da.ranking(b["l"]))])
    sh = ["split", "seed", "U_fed", "F_r", "F_j", "F_C1a", "F_LOO_1", "F_LOO_2", "F_LOO_3", "l_1", "l_2", "l_3", "c_1", "c_2",
          "c_3", "E", "E_prime_B", "regime_B", "award_1_tokens_B", "award_2_tokens_B", "award_3_tokens_B", "ranking"]
    wcsv("sensitivity.csv", sh, srow)
    D = pd.DataFrame(srow, columns=sh)
    sq = ["U_fed", "F_C1a", "l_1", "l_2", "l_3", "E", "award_1_tokens_B", "award_2_tokens_B", "award_3_tokens_B"]
    wcsv("sensitivity_summary.csv", ["split", "quantity", *da.SUMMARY_HEADER[1:]], summary_rows(D, ["split"], sq))
    S["sensitivity"] = {split: {q_: da.summarize(g[q_].astype(float).tolist()) for q_ in sq} for split, g in D.groupby("split", sort=False)}
    S["sensitivity"]["paired_main_minus_variant"] = {q_: da.paired(D[D.split == "main"][q_].tolist(), D[D.split == "mciPhoneOnly"][q_].tolist())
                                                     for q_ in ("U_fed", "l_1", "award_1_tokens_B")}
    return runs


# ---- the LDP sweep ---------------------------------------------------------------------------------------

def ldp_set(R, S, runs_all):
    """The noise-scale sweep and the epsilon curve, with the thesis's condition_tables, and the off-chain scores per
    condition. Requires every run of both sets."""
    tag = la.tag
    runs = {k: {s: load(RUNS / f"ldp-{tag(k)}-s{s}.json") for s in SEEDS} for k in SCALES}
    n64s = {k: {s: load(RUNS / f"n64-{tag(k)}-s{s}.json") for s in SEEDS} for k in SCALES}
    c2 = {s: load(RUNS / f"c2-s{s}.json") for s in SEEDS}
    for s in SEEDS:
        la.check_pairing([runs[k][s] for k in SCALES] + [c2[s]])
    la.check_n64_sources(n64s, lambda k, s: f"ldp-{tag(k)}-s{s}.json")
    if la.figures_of(OUT / "check-c2-s1.json") != la.figures_of(RUNS / "c2-s1.json"):
        raise ValueError("the scales set's check run does not reproduce C2 seed 1")
    c2r = {s: la.run_row(c2[s]) for s in SEEDS}
    label = {0.0: "LDP off (clipping only, σ = 0)", 1.0: "LDP on (protocol σ)", 0.5: "weaker noise (σ × 0.5)",
             4.0: "stronger noise (σ × 4)"}
    da.write_csv("ldp_noise.csv", ["condition", "sigma_scale", "passive", "clip", "sigma", "sigma_over_C",
                                   "eps_per_release_analytic", "eps_total_21_releases", "delta_total"],
                 [[label[k], k, f"{i} {la.PASSIVES[i]}", r["clip"], r["sigma"], r["sigma_over_C"], r["eps_per_release_analytic"],
                   r["eps_total"], la.sci(r["delta_total"])] for k in SCALES for i, r in enumerate(la.noise_rows(runs[k][1]), 1)], OUT)
    SS, _, _ = la.condition_tables("ldp", SCALES, label, runs, n64s, 0.0, ("sigma_scale", lambda k: k), tag,
                                   extra=[("LDP off (clipping only) − C2 (no clipping)", 0.0, c2r)], also=[c2r],
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

    # off-chain scores per condition: AUC and the primary business metric of FULL, FULL_J, LOO_1 and BASELINE
    conds = [(label[k], "sigma_scale", k, runs[k]) for k in SCALES] + [(elabel[k], "eps_per_release", eps_of[k], eruns[k]) for k in ekeys[1:]]
    orow = []
    for name, kind, val, rs in conds:
        for s in SEEDS:
            models = {k: v for k, v in scores(rs[s]).items() if k in ("BASELINE", "FULL", "FULL_J", "LOO_1")}
            b = metrics.business(models, R.y, R.w, "BASELINE")
            for mname in models:
                orow.append([name, kind, val, s, mname, metrics.auc(models[mname][0], R.y), b["approvalRate"],
                             b["models"][mname]["defaultRateAmongApproved"], b["atBankAloneDefaultRate"]["models"][mname]["extraPer10000"], OFFCHAIN])
    oh = ["condition", "kind", "value", "seed", "model", "auc", "approval_rate_r", "default_rate_among_approved", "extra_per_10000", "verified"]
    wcsv("ldp_scores.csv", oh, orow)
    O = pd.DataFrame(orow, columns=oh)
    wcsv("ldp_scores_summary.csv", ["condition", "model", "quantity", *da.SUMMARY_HEADER[1:]],
         summary_rows(O, ["condition", "model"], ["auc", "default_rate_among_approved", "extra_per_10000"]))
    S["unusedClassTotal"]["ldp"] = sum(unused(rs[s]) for _, _, _, rs in conds for s in SEEDS)
    return [(f"LDP {name}", s, rs[s]) for name, _, _, rs in conds for s in SEEDS]


def main() -> None:
    m = load(cm.RESULTS / "prepared" / "manifest.json")
    R = Rows(m)
    S = {"what": "the MCI loan instance's off-chain results (scenarios/mci_loan/analysis.py)", "seeds": SEEDS,
         "meta": {name: load(OUT / f"meta-{name}.json") for name in ("main", "scales", "eps") if (OUT / f"meta-{name}.json").exists()},
         "prepared": {"aSha256": m["a"]["sha256"], "passiveSha256": m["passive"]["sha256"]}}
    runs = main_set(m, R, S)
    have_ldp = all((OUT / f"meta-{x}.json").exists() for x in ("scales", "eps"))
    if have_ldp:
        runs += ldp_set(R, S, runs)
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
    print(json.dumps({"unusedClassTotal": S["unusedClassTotal"], "equivalence": S["equivalence"],
                      "U_fed": round(S["loo"]["U_fed"]["mean"]), "C1a": round(S["loo"]["F_C1a"]["mean"]),
                      "l": {k: round(S["loo"][k]["mean"], 1) for k in ("l_1", "l_2", "l_3")}, "ranking": S["ranking"]["modal"],
                      "count": S["ranking"]["count"], "n64_same": round(S["n64"]["same_ranking"]["mean"], 3), "ldp": have_ldp}, default=str))


if __name__ == "__main__":
    main()
