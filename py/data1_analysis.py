"""DATA1 analysis: tables (CSV), statistics and Talmud divisions from the per-run JSON in results/data1/.

Definitions (plan §0.C; D137 + D139, which supersede D118; decision log D54-D56; py/talmud_reference.py):
- F = floor(10^4 macro-F1) on the full test set (3,193 rows), integer path (as the circuit computes it).
- D139: FULL is scored twice: F_r = the retrained FULL (figure FULL) and F_j = the jointly trained fusion model (figure
  FULL_J); U_fed = max(F_r, F_j) is the FULL figure in every table below ("F_j binds" when F_j > F_r). LOO_i and
  BASELINE are retrained; C1a = BASELINE (every passive slot zero).
- l_i = U_fed - F_LOO_i (raw, may be negative); claim c_i = max(0, l_i); E = max(0, U_fed - F_C1a).
- Talmud (E1 and E3): the contract's rule (M4 design answer 5): E' = min(E, floor(pool / rho)), a = talmud(E'; c)
  floored in performance units, paid rho * a_i (rho = 5,000). E1's pools: 10^12 (never binds), the proposal's
  B = 1,000,003, and rho * floor(E / 2) (binds by construction). talmud_rows (divide_capped, token units) is the pre-M4
  rule, kept for its tests.
- Seeds: mean, sample SD (ddof = 1), 95 % t-interval; n = 5 or 10 (descriptive; nothing is called significant).
- E1's seed sets (--e1 writes both): seeds 1-10 (configs e1-s1..e1-s10; seeds 6-10 run later with the runners'
  --seeds 6-10) give the main tables (e1_configs.csv, e1_summary.json, ...); seeds 1-5, the first version, give the same
  tables with the suffix -5seeds (e1_configs-5seeds.csv, ...), byte for byte as first published. C4b's random states
  1..5 are in e1/c4b.json, 6..10 in e1/c4b-s6-10.json. Seed 1's N = 64 draws and bootstrap (e1_n64.csv,
  e1_bootstrap.csv) do not depend on the seed set. --e3 writes E3 against E1 for both sets (e3_vs_e1.csv,
  e3_vs_e1-5seeds.csv). E2, --binding and --uplift stay on seeds 1-5 (their labels say so).

Usage (from the repo root):  .venv/Scripts/python py/data1_analysis.py --e1 [--e2] [--e3] [--binding] [--uplift]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, norm, t as tdist

import half_up as hu
from talmud_reference import divide_capped, regime

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "data1"
RHO = 5000
B_PROPOSAL = 1_000_003
POOL_NONBINDING = 10**12
DELTA = 1e-5
RELEASES = 21  # T = 20 training rounds + the final matrix
FIGS = ["FULL", "LOO_1", "LOO_2", "LOO_3", "BASELINE"]
SEEDS_5 = [1, 2, 3, 4, 5]          # E1's first five seeds: the -5seeds tables
SEEDS_10 = list(range(1, 11))      # all ten: the main tables
C4B = {"c4b.json": SEEDS_5, "c4b-s6-10.json": [6, 7, 8, 9, 10]}  # results/data1/e1 file: the random states it holds


def seed_suffix(seeds):
    """The file-name suffix of a seed set: none for seeds 1-10, -5seeds for 1-5 (-{n}seeds for any other n)."""
    return "" if list(seeds) == SEEDS_10 else f"-{len(seeds)}seeds"


# ---- statistics ------------------------------------------------------------------------------------

def summarize(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(tdist.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "ci95_lo": mean - half, "ci95_hi": mean + half,
            "min": float(v.min()), "max": float(v.max())}


def paired(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {**summarize(d), "positive": int((d > 0).sum()), "zero": int((d == 0).sum())}


def ranking(l):
    """Passive numbers (1..n) by decreasing l; ties broken by the lower number."""
    return tuple(sorted(range(1, len(l) + 1), key=lambda i: (-l[i - 1], i)))


def ranking_stability(ls):
    ranks = [ranking(l) for l in ls]
    modal, count = Counter(ranks).most_common(1)[0]
    n_p = len(ls[0])
    pairwise = {f"{i}>{j}": sum(1 for l in ls if l[i - 1] > l[j - 1]) / len(ls)
                for i in range(1, n_p + 1) for j in range(1, n_p + 1) if i != j}
    return {"modal": modal, "count": count, "n": len(ls), "rankings": ranks, "pairwise": pairwise}


def talmud_rows(E, claims, rho=RHO, pools=None):
    pools = pools or [POOL_NONBINDING, B_PROPOSAL, rho * (E // 2)]
    rows = []
    for pool in pools:
        estate, awards = divide_capped(rho, pool, E, claims)
        rows.append({"pool": pool, "E": E, "E_eff": estate / rho, "estate_tokens": estate,
                     "regime": regime(estate, [rho * c for c in claims]), "claims": list(claims),
                     "awards_tokens": awards, "awards_perf": [a / rho for a in awards]})
    return rows


def talmud_rows_perf(E, claims, rho=RHO, pools=None):
    """M4 (design answer 5; Federation.sol + SettlementMath): E' = min(E, floor(pool / rho)), a = talmud(E'; claims)
    floored in performance units, paid rho * a_i; A keeps pool - rho * sum(a) on the normal path."""
    from talmud_reference import talmud
    rows = []
    for pool in pools or [B_PROPOSAL, POOL_NONBINDING]:
        ep = min(E, pool // rho)
        a = talmud(ep, list(claims))
        rows.append({"pool": pool, "E": E, "E_prime": ep, "regime": regime(ep, list(claims)), "claims": list(claims),
                     "awards_perf": a, "awards_tokens": [rho * x for x in a], "to_A": pool - rho * sum(a)})
    return rows


def e1_talmud_rows(E, claims, rho=RHO):
    """E1's Talmud table with the contract's rule (talmud_rows_perf) for three pools: 10^12 (never binds), the
    proposal's B, and rho * floor(E / 2) (binds by construction)."""
    return talmud_rows_perf(E, claims, rho, pools=[POOL_NONBINDING, B_PROPOSAL, rho * (E // 2)])


def _delta_of_eps(eps, sensitivity, sigma):
    a = sensitivity / (2 * sigma)
    b = eps * sigma / sensitivity
    return float(norm.cdf(a - b) - math.exp(eps + norm.logcdf(-a - b)))


def dp_epsilon(sigma, clip, delta=DELTA, releases=RELEASES):
    """Gaussian mechanism on one row's embedding: L2 clip C, replace-one neighbours (sensitivity 2C), noise sigma.
    eps_classic: Dwork & Roth's bound, which holds only for eps <= 1 -- None where the formula gives more than 1
    (its delta would then exceed the one claimed); eps_analytic: Balle & Wang (2018), by bisection, valid for every
    eps. Composition over `releases` releases of the row: simple (eps and delta add). An ESTIMATE for the direct
    releases only (QUESTIONS-DATA1 Q16)."""
    sens = 2 * clip
    classic = sens * math.sqrt(2 * math.log(1.25 / delta)) / sigma
    classic = classic if classic <= 1 else None
    if _delta_of_eps(0.0, sens, sigma) <= delta:
        analytic = 0.0
    else:
        lo, hi = 0.0, 1.0
        while _delta_of_eps(hi, sens, sigma) > delta:
            hi *= 2
        for _ in range(200):
            mid = (lo + hi) / 2
            if _delta_of_eps(mid, sens, sigma) > delta:
                lo = mid
            else:
                hi = mid
        analytic = hi
    return {"sensitivity": sens, "sigma": sigma, "sigma_over_C": sigma / clip, "delta": delta, "eps_classic": classic,
            "eps_analytic": analytic, "releases": releases, "eps_total_simple": releases * analytic,
            "delta_total": releases * delta}


def pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


# ---- io --------------------------------------------------------------------------------------------

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell(x):
    """Floats to 4 decimals, except a tiny non-zero value, which keeps its magnitude (never written as 0.0000)."""
    if not isinstance(x, float):
        return x
    return f"{x:.3e}" if x != 0 and abs(x) < 5e-5 else f"{x:.4f}"


def write_csv(name, header, rows, out=None):
    out = RESULTS if out is None else Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([cell(x) for x in r])


def fig(run, name):
    return next(f for f in run["figures"] if f["name"] == name)


def u_fed(fr, fj):
    """D139: the contract's FULL figure, the joint model as a floor under the retrained one."""
    return max(fr, fj)


def binding_rows(sets):
    """D139: for each (label, [(F_r, F_j), ...]) the runs, how many have F_j > F_r (the floor binds), the share,
    and the mean F_j - F_r over the runs where it binds ("" if none)."""
    rows = []
    for label, pairs in sets:
        m = [fj - fr for fr, fj in pairs if fj > fr]
        rows.append([label, len(pairs), len(m), len(m) / len(pairs), float(np.mean(m)) if m else ""])
    return rows


def uplift_rows(sets):
    """D139: for each (label, [(F_r, F_j, [F_LOO_1..3], F_C1a), ...]) the runs, how many F_j binds in, and the mean and
    95th percentile (numpy linear) of what the floor adds: uplift = U_fed - F_r, and the resulting change in
    E = max(0, U - C1a), in sum l_i (l_i = U - F_LOO_i) and in sum c_i (c_i = max(0, l_i)), each against the same
    quantity computed with F_r in place of U_fed."""
    rows = []
    for label, runs in sets:
        d = {"up": [], "E": [], "l": [], "c": []}
        for fr, fj, loo, c1a in runs:
            u = u_fed(fr, fj)
            d["up"].append(u - fr)
            d["E"].append(max(0, u - c1a) - max(0, fr - c1a))
            d["l"].append(sum(u - x for x in loo) - sum(fr - x for x in loo))
            d["c"].append(sum(max(0, u - x) for x in loo) - sum(max(0, fr - x) for x in loo))
        stats = [s for k in ("up", "E", "l", "c") for s in (float(np.mean(d[k])), pct(d[k], 95))]
        rows.append([label, len(runs), sum(1 for fr, fj, _, _ in runs if fj > fr), *stats])
    return rows


def uplift_vs_n_rows(noise):
    """D139 uplift against N from the Q17 noise draws (results/data1/q17/noise.json, offchain/data1/q17-noise.mjs):
    variants a (FULL = joint model) and b (FULL = retrained) share the retrained LOO models, so in every draw
    F_j - F_r = l_1(a) - l_1(b) and uplift = max(0, F_j - F_r). Rows: seed, N, draws, F_j binds, uplift mean, uplift p95
    (numpy linear), mean change in sum l_i (= 3 x uplift)."""
    rows = []
    for seed, per_n in noise["seeds"].items():
        for N, v in per_n.items():
            up = [max(0, a[0] - b[0]) for a, b in zip(v["a"], v["b"])]
            rows.append([seed, int(N), len(up), sum(1 for u in up if u > 0), float(np.mean(up)), pct(up, 95),
                         3 * float(np.mean(up))])
    return rows


def loo_block(run, c1a_from=None):
    """F["FULL"] is U_fed = max(F_r, F_j); F["F_r"], F["F_j"] are the two proven FULL figures."""
    fr, fj = fig(run, "FULL")["F"], fig(run, "FULL_J")["F"]
    F = {n: fig(run, n)["F"] for n in FIGS if n != "FULL"}
    F.update({"FULL": u_fed(fr, fj), "F_r": fr, "F_j": fj})
    c1a = F["BASELINE"] if c1a_from is None else c1a_from
    l = [F["FULL"] - F[f"LOO_{i}"] for i in (1, 2, 3)]
    return {"F": F, "l": l, "claims": [max(0, x) for x in l], "E": max(0, F["FULL"] - c1a), "binds": fj > fr}


def srow(label, s, extra=()):
    return [label, s["n"], s["mean"], s["sd"], s["ci95_lo"], s["ci95_hi"], s["min"], s["max"], *extra]


SUMMARY_HEADER = ["quantity", "n", "mean", "sd", "ci95_lo", "ci95_hi", "min", "max"]


# ---- E1 --------------------------------------------------------------------------------------------

def c4b_runs(seeds):
    """C4b's run per random state s of the seed set, read from the file that holds s (C4B)."""
    by = {}
    for name, held in C4B.items():
        if set(held) & set(seeds):
            by |= {r["random_state"]: r for r in load(RESULTS / "e1" / name)["runs"]}
    missing = [s for s in seeds if s not in by]
    if missing:
        raise SystemExit(f"no C4b run for random state(s) {missing}")
    return [by[s] for s in seeds]


def friedman_floor(n, k=3):
    """The smallest Friedman p attainable with n blocks and k treatments (chi2 approximation): chi2 = n (k - 1)."""
    from scipy.stats import chi2
    return float(chi2.sf(n * (k - 1), k - 1))


def e1(seeds=SEEDS_10, out=None):
    """E1's tables over `seeds`, named with seed_suffix(seeds); written to `out` (default results/data1)."""
    d = RESULTS / "e1"
    seeds = list(seeds)
    sfx = seed_suffix(seeds)
    nm = lambda stem, ext=".csv": f"{stem}{sfx}{ext}"  # noqa: E731
    out_dir = RESULTS if out is None else Path(out)
    wcsv = lambda name, header, rows: write_csv(name, header, rows, out_dir)  # noqa: E731
    n_s = len(seeds)
    c2 = {s: load(d / f"c2-s{s}.json") for s in seeds}
    c4a64 = {s: load(d / f"c4a64-s{s}.json") for s in seeds}
    c4a32 = {s: load(d / f"c4a32-s{s}.json") for s in seeds}
    c1b = {s: load(d / f"c1b-s{s}.json") for s in seeds}
    levels = load(d / "noise-levels.json")
    c3 = {lv["name"]: {s: load(d / f"c3-{lv['name']}-s{s}.json") for s in seeds} for lv in levels["levels"]}
    c4b = {"runs": c4b_runs(seeds)}
    out = {"seeds": seeds}

    # per-run resources
    runs = [(f"C2", s, c2[s]) for s in seeds] + [("C4a", s, c4a64[s]) for s in seeds] + [("C4a-f32", s, c4a32[s]) for s in seeds] + \
           [("C1b", s, c1b[s]) for s in seeds] + [(f"C3-{k}", s, v[s]) for k, v in c3.items() for s in seeds]
    wcsv(nm("e1_runs"), ["config", "seed", "wall_s", "training_s", "peak_rss_mb", "free_ram_mb_start", "free_ram_mb_end",
                              "zero_filled", "hq_saturated_p1", "hq_saturated_p2", "hq_saturated_p3", "config_hash"],
              [[c, s, r["wallMs"] / 1000, r.get("trainingMs", 0) / 1000, r["peakRssMB"], r["machine"]["freeRamMB"], r.get("freeRamMBAtEnd", ""),
                len(r["training"]["zeroFilled"]) if "training" in r else "", *r["clipping"]["hq"], r["configHash"]] for c, s, r in runs])

    # configurations
    per = {
        "C1a (protocol baseline)": [fig(c2[s], "BASELINE")["F"] for s in seeds],
        "C1b (A trained alone)": [c1b[s]["F"] for s in seeds],
        "C2 (VFL; U_fed = max(F_r, F_j), D139)": [loo_block(c2[s])["F"]["FULL"] for s in seeds],
        "C2 F_r (retrained FULL)": [fig(c2[s], "FULL")["F"] for s in seeds],
        "C2 F_j (jointly trained FULL)": [fig(c2[s], "FULL_J")["F"] for s in seeds],
        **{f"C3-{k} (VFL + noise; U_fed)": [loo_block(v[s])["F"]["FULL"] for s in seeds] for k, v in c3.items()},
        "C4a (centralized, float64)": [c4a64[s]["F"] for s in seeds],
        "C4a-f32 (centralized, float32 transport)": [c4a32[s]["F"] for s in seeds],
        f"C4b (gradient boosting; random_state {seeds[0]}..{seeds[-1]})": [r["F"] for r in c4b["runs"]],
    }
    wcsv(nm("e1_configs"), ["config", *[f"seed_{s}" for s in seeds]], [[k, *v] for k, v in per.items()])
    wcsv(nm("e1_configs_summary"), SUMMARY_HEADER, [srow(k, summarize(v)) for k, v in per.items()])
    out["configs"] = {k: summarize(v) for k, v in per.items()}

    # equivalence C2 vs C4a
    eq = []
    for s in seeds:
        l2 = [r["loss"] for r in c2[s]["training"]["rounds"]]
        l32 = [r["loss"] for r in c4a32[s]["rounds"]]
        l64 = [r["loss"] for r in c4a64[s]["rounds"]]
        # C4a scores its jointly trained fusion model, so the like-for-like figure of C2 is F_j (FULL_J)
        p2, p32, p64 = fig(c2[s], "FULL_J")["preds"], c4a32[s]["preds"], c4a64[s]["preds"]
        eq.append([s, max(abs(a - b) for a, b in zip(l2, l32)), max(abs(a - b) for a, b in zip(l2, l64)),
                   fig(c2[s], "FULL_J")["F"], c4a32[s]["F"], c4a64[s]["F"], "yes" if l2 == l32 and p2 == p32 else "NO",
                   sum(x != y for x, y in zip(p2, p64))])
    wcsv(nm("e1_equivalence"), ["seed", "max_abs_loss_diff_C2_vs_C4a_f32", "max_abs_loss_diff_C2_vs_C4a_f64", "F_C2_joint (F_j)",
                                     "F_C4a_f32", "F_C4a_f64", "C2_equals_C4a_f32_bitwise", "test_preds_differing_C2_vs_C4a_f64"], eq)
    out["equivalence"] = {"all_bitwise": all(r[6] == "yes" for r in eq), "max_diff_f64": max(r[2] for r in eq),
                          "preds_diff_f64": [r[7] for r in eq]}

    # LOO, E, paired comparisons
    blocks = {s: loo_block(c2[s]) for s in seeds}
    rows = []
    for s in seeds:
        b = blocks[s]
        rows.append([s, *[b["F"][n] for n in FIGS], c1b[s]["F"], *b["l"], *b["claims"], b["E"]])
    q = {"F_FULL": [blocks[s]["F"]["FULL"] for s in seeds], "F_C1a": [blocks[s]["F"]["BASELINE"] for s in seeds],
         "F_C1b": [c1b[s]["F"] for s in seeds], "E": [blocks[s]["E"] for s in seeds],
         **{f"l_{i}": [blocks[s]["l"][i - 1] for s in seeds] for i in (1, 2, 3)},
         **{f"F_LOO_{i}": [blocks[s]["F"][f"LOO_{i}"] for s in seeds] for i in (1, 2, 3)}}
    wcsv(nm("e1_loo"), ["seed", "F_FULL", "F_LOO_1", "F_LOO_2", "F_LOO_3", "F_C1a", "F_C1b", "l_1", "l_2", "l_3",
                             "c_1", "c_2", "c_3", "E"], rows)
    wcsv(nm("e1_loo_summary"), SUMMARY_HEADER, [srow(k, summarize(v)) for k, v in q.items()])
    pairs = {"FULL - C1a": paired(q["F_FULL"], q["F_C1a"]), "FULL - C1b": paired(q["F_FULL"], q["F_C1b"]),
             "C1b - C1a": paired(q["F_C1b"], q["F_C1a"]),
             "C4a (f64) - C2 F_j": paired([c4a64[s]["F"] for s in seeds], [blocks[s]["F"]["F_j"] for s in seeds]),
             "C4b - C2": paired([r["F"] for r in c4b["runs"]], q["F_FULL"])}
    for i, j in combinations((1, 2, 3), 2):
        pairs[f"l_{i} - l_{j}"] = paired(q[f"l_{i}"], q[f"l_{j}"])
    wcsv(nm("e1_paired"), SUMMARY_HEADER + ["positive"], [srow(k, v, [v["positive"]]) for k, v in pairs.items()])
    out["loo"] = {k: summarize(v) for k, v in q.items()}
    out["paired"] = pairs

    # ranking stability, Friedman
    ls = [blocks[s]["l"] for s in seeds]
    st = ranking_stability(ls)
    fr = friedmanchisquare(*[[l[i] for l in ls] for i in range(3)])
    wcsv(nm("e1_ranking"), ["seed", "l_1", "l_2", "l_3", "ranking (largest l first)"],
              [[s, *blocks[s]["l"], "-".join(map(str, ranking(blocks[s]["l"])))] for s in seeds] +
              [["modal", "", "", "", f"{'-'.join(map(str, st['modal']))} in {st['count']} of {st['n']} seeds"]] +
              [[f"share {k}", "", "", "", hu.fmt(v, 2)] for k, v in st["pairwise"].items()] +
              [[f"Friedman chi2 (descriptive, n = {n_s})", "", "", "", f"{hu.fmt(fr.statistic, 3)}; p = {hu.fmt(fr.pvalue, 4)}; "
                f"smallest attainable p with n = {n_s}, k = 3: {cell(friedman_floor(n_s))} (chi2 approximation)"]])
    out["ranking"] = {**st, "friedman": {"chi2": float(fr.statistic), "p": float(fr.pvalue)}}

    # Talmud per seed and pool
    trow = []
    tal = {}
    for s in seeds:
        b = blocks[s]
        for r in e1_talmud_rows(b["E"], b["claims"]):  # the contract's rule (M4 design answer 5), as E3
            trow.append([s, r["pool"], r["E"], r["E_prime"], r["regime"], *r["claims"], *r["awards_perf"], *r["awards_tokens"], r["to_A"]])
            tal.setdefault(r["pool"] if r["pool"] in (POOL_NONBINDING, B_PROPOSAL) else "rho*floor(E/2)", []).append(r)
    wcsv(nm("e1_talmud"), ["seed", "pool", "E", "E_prime", "regime", "c_1", "c_2", "c_3", "award_1_perf", "award_2_perf",
                                "award_3_perf", "award_1_tokens", "award_2_tokens", "award_3_tokens", "to_A_normal_path"], trow)
    out["talmud"] = {str(k): {"regimes": Counter(r["regime"] for r in v), "E_prime": summarize([r["E_prime"] for r in v]),
                              "awards_perf": [summarize([r["awards_perf"][i] for r in v]) for i in range(3)]} for k, v in tal.items()}

    # C3 noise levels, DP estimate
    crow = []
    out["c3"] = {}
    for lv in levels["levels"]:
        runs3 = c3[lv["name"]]
        bl = {s: loo_block(runs3[s]) for s in seeds}
        dp = dp_epsilon(lv["sigma"], lv["clip"])
        sm = {"F_FULL": summarize([bl[s]["F"]["FULL"] for s in seeds]), "F_C1a": summarize([bl[s]["F"]["BASELINE"] for s in seeds]),
              "E": summarize([bl[s]["E"] for s in seeds]), **{f"l_{i}": summarize([bl[s]["l"][i - 1] for s in seeds]) for i in (1, 2, 3)}}
        out["c3"][lv["name"]] = {"clip": lv["clip"], "sigma": lv["sigma"], "mult": lv["mult"], **sm, "dp": dp,
                                 "zero_filled": sum(len(runs3[s]["training"]["zeroFilled"]) for s in seeds)}
        crow.append([lv["name"], lv["clip"], lv["sigma"], lv["mult"], sm["F_FULL"]["mean"], sm["F_FULL"]["sd"], sm["F_C1a"]["mean"],
                     sm["F_C1a"]["sd"], sm["E"]["mean"], sm["E"]["sd"], *[x for i in (1, 2, 3) for x in (sm[f"l_{i}"]["mean"], sm[f"l_{i}"]["sd"])],
                     "n/a (bound needs eps <= 1)" if dp["eps_classic"] is None else dp["eps_classic"],
                     dp["eps_analytic"], dp["eps_total_simple"], dp["delta_total"],
                     "estimate, direct releases only (Q16)"])
    wcsv(nm("e1_c3"), ["level", "clip_C", "sigma", "sigma_over_C", "F_FULL_mean", "F_FULL_sd", "F_C1a_mean", "F_C1a_sd",
                            "E_mean", "E_sd", "l1_mean", "l1_sd", "l2_mean", "l2_sd", "l3_mean", "l3_sd",
                            "eps_per_release_classic", "eps_per_release_analytic", "eps_total_21_releases", "delta_total", "note"], crow)
    out["noise_levels"] = levels

    # N = 64
    n64 = load(d / "n64.json")
    dr = n64["draws"]
    # every draw's U_fed = max(F_r, F_j) on the same 64 rows (the contract's rule)
    for x in dr + [{"F": n64["fullTest"]}]:
        x["F"]["F_r"], x["F"]["F_j"] = x["F"]["FULL"], x["F"]["FULL_J"]
        x["F"]["FULL"] = u_fed(x["F"]["F_r"], x["F"]["F_j"])
    series = {n: [x["F"][n] for x in dr] for n in [*FIGS, "F_r", "F_j"]}
    series.update({f"l_{i}": [x["F"]["FULL"] - x["F"][f"LOO_{i}"] for x in dr] for i in (1, 2, 3)})
    series["E"] = [max(0, x["F"]["FULL"] - x["F"]["BASELINE"]) for x in dr]
    ft = n64["fullTest"]
    full_ref = {**ft, **{f"l_{i}": ft["FULL"] - ft[f"LOO_{i}"] for i in (1, 2, 3)}, "E": max(0, ft["FULL"] - ft["BASELINE"])}
    rank_full = ranking([full_ref[f"l_{i}"] for i in (1, 2, 3)])
    same_rank = sum(1 for x in dr if ranking([x["F"]["FULL"] - x["F"][f"LOO_{i}"] for i in (1, 2, 3)]) == rank_full) / len(dr)
    binds64 = sum(1 for x in dr if x["F"]["F_j"] > x["F"]["F_r"]) / len(dr)
    nrow = []
    for k, v in series.items():
        s = summarize(v)
        nrow.append([k, full_ref[k], s["mean"], s["sd"], pct(v, 2.5), pct(v, 50), pct(v, 97.5),
                     (sum(1 for x in v if x <= 0) / len(v)) if k.startswith("l_") or k == "E" else ""])
    wcsv("e1_n64.csv", ["quantity", "full_test_seed1", "mean_N64", "sd_N64", "p2_5", "p50", "p97_5", "share_le_0"], nrow)
    out["n64"] = {"draws": len(dr), "mean_distinct_rows": float(np.mean([x["distinct"] for x in dr])), "F_j_binds_share": binds64,
                  "same_ranking_as_full_test": same_rank, "full_test_ranking": rank_full,
                  **{k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "p2_5": pct(v, 2.5), "p97_5": pct(v, 97.5),
                         "share_le_0": sum(1 for x in v if x <= 0) / len(v)} for k, v in series.items()}}

    # full-test bootstrap (seed 1)
    bs = load(d / "bootstrap.json")
    sm = bs["samples"]
    uf = np.maximum(np.array(sm["FULL"]), np.array(sm["FULL_J"]))  # U_fed per resample (D139)
    ftb = bs["fullTest"]
    uref = u_fed(ftb["FULL"], ftb["FULL_J"])
    derived = {"U_fed": list(uf), "F_r": sm["FULL"], "F_j": sm["FULL_J"], **{k: sm[k] for k in sm if k not in ("FULL", "FULL_J")},
               **{f"l_{i}": list(uf - np.array(sm[f"LOO_{i}"])) for i in (1, 2, 3)},
               "E": list(np.maximum(0, uf - np.array(sm["BASELINE"]))), "U_fed - C1b": list(uf - np.array(sm["C1b"]))}
    ref = {"U_fed": uref, "F_r": ftb["FULL"], "F_j": ftb["FULL_J"], **{k: v for k, v in ftb.items() if k not in ("FULL", "FULL_J")},
           **{f"l_{i}": uref - ftb[f"LOO_{i}"] for i in (1, 2, 3)}, "E": max(0, uref - ftb["BASELINE"]), "U_fed - C1b": uref - ftb["C1b"]}
    wcsv("e1_bootstrap.csv", ["quantity", "full_test_seed1", "sd", "p2_5", "p97_5"],
              [[k, ref[k], float(np.std(v, ddof=1)), pct(v, 2.5), pct(v, 97.5)] for k, v in derived.items()])
    out["bootstrap"] = {k: {"value": ref[k], "sd": float(np.std(v, ddof=1)), "p2_5": pct(v, 2.5), "p97_5": pct(v, 97.5)}
                        for k, v in derived.items()}

    # D139: in which honest runs does F_j bind (F_j > F_r)?
    brow = []
    for label, runs in [("C2", c2), *[(f"C3-{k}", v) for k, v in c3.items()]]:
        for s in seeds:
            b = loo_block(runs[s])
            brow.append([label, s, b["F"]["F_r"], b["F"]["F_j"], b["F"]["FULL"], b["F"]["F_j"] - b["F"]["F_r"], "yes" if b["binds"] else "no"])
    wcsv(nm("e1_full_binding"), ["config", "seed", "F_r", "F_j", "U_fed", "F_j - F_r", "F_j binds"], brow)
    out["binding"] = {c: sum(1 for r in brow if r[0] == c and r[6] == "yes") for c in dict.fromkeys(r[0] for r in brow)}
    out["full_gap"] = summarize([blocks[s]["F"]["F_r"] - blocks[s]["F"]["F_j"] for s in seeds])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / nm("e1_summary", ".json")).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"seeds": seeds, "configs": {k: hu.round_half_up(v["mean"]) for k, v in out["configs"].items()}, "equivalence": out["equivalence"],
                      "l": {f"l_{i}": hu.round_half_up(out["loo"][f"l_{i}"]["mean"], 1) for i in (1, 2, 3)},
                      "ranking": out["ranking"]["modal"], "count": out["ranking"]["count"]}, default=str))
    return out


# ---- E2 --------------------------------------------------------------------------------------------

def e2():
    d = RESULTS / "e2"
    seeds = [1, 2, 3, 4, 5]
    parts = load(d / "partitions.json")
    c1b = {s: load(RESULTS / "e1" / f"c1b-s{s}.json")["F"] for s in seeds}
    e1c2 = {s: load(RESULTS / "e1" / f"c2-s{s}.json") for s in seeds}
    summary_rows, out = [], {}

    def analyse(label, runs, widths, varcounts, kept):
        blocks = {s: loo_block(runs[s]) for s in seeds}
        q = {"F_FULL": [blocks[s]["F"]["FULL"] for s in seeds], "F_C1a": [blocks[s]["F"]["BASELINE"] for s in seeds],
             "F_C1b": [c1b[s] for s in seeds], "E": [blocks[s]["E"] for s in seeds],
             **{f"l_{i}": [blocks[s]["l"][i - 1] for s in seeds] for i in (1, 2, 3)}}
        ls = [blocks[s]["l"] for s in seeds]
        st = ranking_stability(ls)
        fr = friedmanchisquare(*[[l[i] for l in ls] for i in range(3)])
        p_a, p_b = paired(q["F_FULL"], q["F_C1a"]), paired(q["F_FULL"], q["F_C1b"])
        means = [summarize(q[f"l_{i}"])["mean"] for i in (1, 2, 3)]
        sds = [summarize(q[f"l_{i}"])["sd"] for i in (1, 2, 3)]
        pair_l = {f"l_{i} - l_{j}": paired(q[f"l_{i}"], q[f"l_{j}"]) for i, j in combinations((1, 2, 3), 2)}
        res = {"widths": widths, "variables": varcounts, "kept_from_original": kept, "binds": sum(1 for s in seeds if blocks[s]["binds"]),
               **{k: summarize(v) for k, v in q.items()}, "FULL - C1a": p_a, "FULL - C1b": p_b, "pairs_l": pair_l,
               "l_spread_of_means": max(means) - min(means), "mean_seed_sd_of_l": float(np.mean(sds)),
               "ranking": st, "friedman": {"chi2": float(fr.statistic), "p": float(fr.pvalue)}}
        rows = [[s, *[blocks[s]["F"][n] for n in FIGS], c1b[s], *blocks[s]["l"], blocks[s]["E"],
                 "-".join(map(str, ranking(blocks[s]["l"])))] for s in seeds]
        return res, rows

    items = [("original (E1)", e1c2, [load(RESULTS / "e1" / "c2-s1.json")["widths"]][0], [20, 67, 101], None)]
    for p in parts:
        runs = {s: load(d / f"p{p['k']}-c2-s{s}.json") for s in seeds}
        items.append((f"partition {p['k']}", runs, p["widths"], [len(p["vars"][n]) for n in ("telecom", "retailer", "device")],
                      p["kept_from_original"]))
    for label, runs, widths, varcounts, kept in items:
        res, rows = analyse(label, runs, widths, varcounts, kept)
        out[label] = res
        tag = "original" if label.startswith("original") else label.split()[-1]
        write_csv(f"e2_partition_{tag}.csv", ["seed", "F_FULL", "F_LOO_1", "F_LOO_2", "F_LOO_3", "F_C1a", "F_C1b", "l_1", "l_2",
                                               "l_3", "E", "ranking"],
                  rows + [["mean", *[res[k]["mean"] for k in ("F_FULL",)], "", "", "", res["F_C1a"]["mean"], res["F_C1b"]["mean"],
                           res["l_1"]["mean"], res["l_2"]["mean"], res["l_3"]["mean"], res["E"]["mean"], ""],
                          ["sd", res["F_FULL"]["sd"], "", "", "", res["F_C1a"]["sd"], res["F_C1b"]["sd"], res["l_1"]["sd"],
                           res["l_2"]["sd"], res["l_3"]["sd"], res["E"]["sd"], ""]])
        summary_rows.append([label, "/".join(map(str, widths)), "/".join(map(str, varcounts)),
                             "/".join(map(str, kept.values())) if kept else "20/67/101",
                             res["F_FULL"]["mean"], res["F_FULL"]["sd"], res["F_C1a"]["mean"], res["F_C1a"]["sd"],
                             res["FULL - C1a"]["mean"], res["FULL - C1a"]["sd"], f"{res['FULL - C1a']['positive']}/5",
                             res["FULL - C1b"]["mean"], f"{res['FULL - C1b']['positive']}/5",
                             res["l_1"]["mean"], res["l_1"]["sd"], res["l_2"]["mean"], res["l_2"]["sd"], res["l_3"]["mean"], res["l_3"]["sd"],
                             res["l_spread_of_means"], res["mean_seed_sd_of_l"],
                             f"{'-'.join(map(str, res['ranking']['modal']))} ({res['ranking']['count']}/5)", res["friedman"]["p"],
                             f"{res['binds']}/5"])
    write_csv("e2_summary.csv", ["partition", "widths", "variables", "variables kept from the original party", "U_fed_mean",
                                 "U_fed_sd", "F_C1a_mean", "F_C1a_sd", "FULL_minus_C1a_mean", "FULL_minus_C1a_sd",
                                 "FULL_gt_C1a_seeds", "FULL_minus_C1b_mean", "FULL_gt_C1b_seeds", "l1_mean", "l1_sd", "l2_mean",
                                 "l2_sd", "l3_mean", "l3_sd", "spread_of_mean_l", "mean_seed_sd_of_l", "modal_ranking",
                                 "friedman_p (descriptive)", "F_j binds (seeds)"], summary_rows)
    (RESULTS / "e2_summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for r in summary_rows:
        print(r[0], "FULL-C1a", hu.round_half_up(r[8], 1), r[10], "l", [hu.round_half_up(x, 1) for x in (r[13], r[15], r[17])], r[21])
    return out


# ---- E3 --------------------------------------------------------------------------------------------

def e3(out=None):
    """E3's tables (written to `out`, default results/data1); E3 against E1 for both of E1's seed sets."""
    d = RESULTS / "e3"
    out_dir = RESULTS if out is None else Path(out)
    run = load(d / "run.json")
    a = load(d / "A" / "summary.json")
    sample = load(d / "sample.json")
    names = {(0, 0): "FULL", (3, 0): "FULL_J", (1, 1): "LOO_1", (1, 2): "LOO_2", (1, 3): "LOO_3", (2, 0): "BASELINE"}
    figs = {names[(f["kind"], f["memberIndex"])]: f for f in run["chainEnd"]["figures"]}
    Fs = {n: f["F"] for n, f in figs.items()}
    accepted = all(f["accepted"] for f in figs.values()) and len(figs) == 6
    U = u_fed(Fs["FULL"], Fs["FULL_J"])
    if U != run["chainEnd"]["uFed"]:
        raise SystemExit(f"E3: max(F_r, F_j) = {U} but the contract's uFed = {run['chainEnd']['uFed']}")
    F = {**{n: Fs[n] for n in FIGS if n != "FULL"}, "FULL": U}
    l = [U - F[f"LOO_{i}"] for i in (1, 2, 3)]
    claims = [max(0, x) for x in l]
    E = max(0, U - F["BASELINE"])
    idx = a.get("idx", [])
    order = ["FULL", "FULL_J", "LOO_1", "LOO_2", "LOO_3", "BASELINE"]
    label = {"FULL": "F_r (FULL, retrained)", "FULL_J": "F_j (FULL_J, joint, gammaG)"}
    write_csv("e3_figures.csv", ["figure", "F_on_chain", "cModel"],
              [[label.get(n, n), Fs[n], figs[n]["cModel"]] for n in order] +
              [["U_fed = max(F_r, F_j) (contract)", run["chainEnd"]["uFed"], ""], ["F_j binds", "yes" if Fs["FULL_J"] > Fs["FULL"] else "no", ""],
               ["l_1", l[0], ""], ["l_2", l[1], ""], ["l_3", l[2], ""], ["E", E, ""],
               ["gammaG (committed at commitFA)", "", run["chainEnd"].get("gammaG", "")],
               ["final state", run["finalState"], ""], ["all 6 accepted", "yes" if accepted else "NO", ""],
               ["|I|", a.get("sizeI", sample["sizeI"]), ""], ["|I^tr|", a.get("sizeI", sample["sizeI"]) - len(set(idx)) if idx else "", ""],
               ["distinct evaluation rows of 64", len(set(idx)) if idx else "", ""]], out_dir)
    trows = talmud_rows_perf(E, claims, pools=[B_PROPOSAL, POOL_NONBINDING])  # M4: floored in performance units
    write_csv("e3_talmud.csv", ["pool", "E", "E_prime", "regime", "c_1", "c_2", "c_3", "award_1_perf", "award_2_perf", "award_3_perf",
                                "award_1_tokens", "award_2_tokens", "award_3_tokens", "to_A_normal_path"],
              [[r["pool"], r["E"], r["E_prime"], r["regime"], *r["claims"], *r["awards_perf"], *r["awards_tokens"], r["to_A"]] for r in trows],
              out_dir)
    replay = load(d / "replay.json") if (d / "replay.json").exists() else None  # offchain/data1/e3-replay.mjs
    if replay and (replay["figures"] != Fs or replay["uFed"] != U):
        raise SystemExit("e3/replay.json does not reproduce the on-chain figures")
    n64 = load(RESULTS / "e1" / "n64.json")["draws"]
    ufd = [u_fed(x["F"]["FULL"], x["F"]["FULL_J"]) for x in n64]
    series = {n: [x["F"][n] for x in n64] for n in FIGS if n != "FULL"}
    series["FULL"] = ufd
    series.update({f"l_{i}": [u - x["F"][f"LOO_{i}"] for u, x in zip(ufd, n64)] for i in (1, 2, 3)})
    series["E"] = [max(0, u - x["F"]["BASELINE"]) for u, x in zip(ufd, n64)]
    e3v = {**F, **{f"l_{i}": l[i - 1] for i in (1, 2, 3)}, "E": E}
    for sfx in (seed_suffix(SEEDS_5), seed_suffix(SEEDS_10)):  # E1's full-test mean and SD over that seed set
        e1s = load(RESULTS / f"e1_summary{sfx}.json")
        e1full = {"FULL": e1s["loo"]["F_FULL"], **{f"LOO_{i}": e1s["loo"][f"F_LOO_{i}"] for i in (1, 2, 3)},
                  "BASELINE": e1s["loo"]["F_C1a"], **{f"l_{i}": e1s["loo"][f"l_{i}"] for i in (1, 2, 3)}, "E": e1s["loo"]["E"]}
        crow = []
        for k in [*FIGS, "l_1", "l_2", "l_3", "E"]:
            v = np.asarray(series[k], dtype=float)
            crow.append(["U_fed" if k == "FULL" else k, e1full[k]["mean"], e1full[k]["sd"], float(v.mean()), float(v.std(ddof=1)),
                         e3v[k], 100.0 * float((v < e3v[k]).mean() + 0.5 * (v == e3v[k]).mean())])
        write_csv(f"e3_vs_e1{sfx}.csv", ["quantity", "E1_full_test_mean_C2", "E1_full_test_sd", "E1_N64_mean_seed1", "E1_N64_sd",
                                         "E3_value", "E3_percentile_in_E1_N64"], crow, out_dir)
    out = {"F": F, "Fs": Fs, "uFed": U, "binds": Fs["FULL_J"] > Fs["FULL"], "l": l, "claims": claims, "E": E, "talmud": trows,
           "accepted": accepted, "finalState": run["finalState"], "replay": replay,
           "sizeI": a.get("sizeI"), "distinctEvalRows": len(set(idx)) if idx else None, "wallClockMs": run["wallClockMs"],
           "machine": run["machine"], "scoring": a.get("scoring")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e3_summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"F": Fs, "uFed": U, "l": l, "E": E, "E_prime": trows[0]["E_prime"], "awards": trows[0]["awards_tokens"]}))
    return out


# ---- D139: where does the joint floor bind? ---------------------------------------------------------

def binding():
    seeds = [1, 2, 3, 4, 5]
    fr_fj = lambda run: (fig(run, "FULL")["F"], fig(run, "FULL_J")["F"])  # noqa: E731
    e1 = RESULTS / "e1"
    levels = [lv["name"] for lv in load(e1 / "noise-levels.json")["levels"]]
    parts = load(RESULTS / "e2" / "partitions.json")
    run3 = load(RESULTS / "e3" / "run.json")["chainEnd"]["figures"]
    e3 = (next(f["F"] for f in run3 if f["kind"] == 0), next(f["F"] for f in run3 if f["kind"] == 3))
    n64 = load(e1 / "n64.json")["draws"]
    sets = [("E1 C2 (5 seeds, full test)", [fr_fj(load(e1 / f"c2-s{s}.json")) for s in seeds]),
            ("E1 C3 (3 noise levels x 5 seeds, full test)", [fr_fj(load(e1 / f"c3-{k}-s{s}.json")) for k in levels for s in seeds]),
            ("E2 (5 partitions x 5 seeds, full test)", [fr_fj(load(RESULTS / "e2" / f"p{p['k']}-c2-s{s}.json")) for p in parts for s in seeds]),
            ("E3 (on-chain, 1 run, 64 draws)", [e3]),
            ("E1 C2 seed 1, N = 64 protocol draws (1,000)", [(x["F"]["FULL"], x["F"]["FULL_J"]) for x in n64])]
    rows = binding_rows(sets)
    write_csv("d139_binding.csv", ["honest runs", "runs", "F_j binds (F_j > F_r)", "share", "mean F_j - F_r where it binds"], rows)
    print(json.dumps([[r[0], r[2], r[1]] for r in rows]))
    return rows


def uplift():
    """D139 uplift in honest runs; the N = 64 draws per E1 seed come from offchain/data1/n64.mjs (seed 1: n64.json)."""
    seeds = [1, 2, 3, 4, 5]
    e1 = RESULTS / "e1"
    run4 = lambda run: (fig(run, "FULL")["F"], fig(run, "FULL_J")["F"],  # noqa: E731
                        [fig(run, f"LOO_{i}")["F"] for i in (1, 2, 3)], fig(run, "BASELINE")["F"])
    draw4 = lambda x: (x["F"]["FULL"], x["F"]["FULL_J"], [x["F"][f"LOO_{i}"] for i in (1, 2, 3)], x["F"]["BASELINE"])  # noqa: E731
    levels = [lv["name"] for lv in load(e1 / "noise-levels.json")["levels"]]
    parts = load(RESULTS / "e2" / "partitions.json")
    draws = {s: [draw4(x) for x in load(e1 / ("n64.json" if s == 1 else f"n64-s{s}.json"))["draws"]] for s in seeds}
    sets = [("E1 C2 (5 seeds, full test)", [run4(load(e1 / f"c2-s{s}.json")) for s in seeds]),
            ("E1 C3 (3 noise levels x 5 seeds, full test)", [run4(load(e1 / f"c3-{k}-s{s}.json")) for k in levels for s in seeds]),
            ("E2 (5 partitions x 5 seeds, full test)", [run4(load(RESULTS / "e2" / f"p{p['k']}-c2-s{s}.json")) for p in parts for s in seeds]),
            *[(f"E1 C2 seed {s}, N = 64 draws (1,000)", draws[s]) for s in seeds],
            ("E1 C2 seeds 1-5, N = 64 draws pooled (5,000)", [r for s in seeds for r in draws[s]])]
    rows = uplift_rows(sets)
    write_csv("d139_uplift.csv", ["honest runs", "runs", "F_j binds", "uplift mean", "uplift p95", "dE mean", "dE p95",
                                  "d sum l mean", "d sum l p95", "d sum c mean", "d sum c p95"], rows)
    print(json.dumps([[r[0], hu.round_half_up(r[3], 1), r[4]] for r in rows]))
    vs_n = uplift_vs_n_rows(load(RESULTS / "q17" / "noise.json"))
    for r in vs_n:  # N = 64 uses the same draws (sampleIndices, j = 1..1000) as the per-seed rows above
        if r[1] == 64:
            [u] = [x for x in rows if x[0] == f"E1 C2 seed {r[0][1:]}, N = 64 draws (1,000)"]
            if (r[3], r[4], r[5]) != (u[2], u[3], u[4]):
                raise SystemExit(f"{r[0]} N = 64: q17 noise draws disagree with n64 draws")
    write_csv("d139_uplift_vs_N.csv", ["seed", "N", "draws", "F_j binds", "uplift mean", "uplift p95", "d sum l mean"], vs_n)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", action="store_true")
    ap.add_argument("--e2", action="store_true")
    ap.add_argument("--e3", action="store_true")
    ap.add_argument("--binding", action="store_true")
    ap.add_argument("--uplift", action="store_true")
    a = ap.parse_args()
    if a.e1:
        e1(SEEDS_5)
        e1(SEEDS_10)
    if a.e2:
        e2()
    if a.e3:
        e3()
    if a.binding:
        binding()
    if a.uplift:
        uplift()
