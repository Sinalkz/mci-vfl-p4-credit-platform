"""LDP with vs without: tables (CSV) and RESULTS-LDP.md from the runs of contracts/offchain/data1/ldp.mjs.

Setting: DATA1's (E1's rows, 12,772 training / 3,193 test households; model configs e1-s1..e1-s5, e1-s1 being E3's
config), the passives' protocol noise setting (LOCAL_NOISE, party-m3.mjs) with sigma x 0 (LDP off: the same clipping,
no noise), x 1 (LDP on), x 0.5 and x 4; within a seed every condition has the same config, test rows and seeded normal
draws (check_pairing), so the conditions differ in sigma only.
Definitions as DATA1 (py/data1_analysis.py): F = floor(10^4 macro-F1) on the full test set, integer path; D139:
U_fed = max(F_r, F_j), F_r the retrained FULL, F_j the jointly trained fusion model; l_i = U_fed - F_LOO_i (raw, may be
negative); E = max(0, U_fed - F_BASELINE), BASELINE = C1a (every passive slot zero). Seeds: mean, sample SD, 95 %
t-interval over n = 5 or 10 (descriptive; nothing is called significant). N = 64: per run, 1,000 draws of 64 test rows
with replacement (n64.mjs, the protocol's sampling, j = 1..1,000: the same rows in every condition), each figure's
predictions indexed at the drawn rows; U_fed, l_i and E per draw as above.
Seed sets (each run writes both): seeds 1-10 (configs e1-s1..e1-s10; seeds 6-10 run later with ldp.mjs --seeds 6-10,
meta-s6-10.json / meta-eps-s6-10.json and their own check runs) give the main tables (ldp_summary.csv, summary.json,
...); seeds 1-5, the first version, the same tables with the suffix -5seeds (ldp_summary-5seeds.csv,
summary-5seeds.json, ...), byte for byte as first published. ldp_noise.csv and ldp_eps_levels.csv hold no seed.
RESULTS-LDP.md reports the five seeds; it is written only with --report.
Usage (repo root): .venv/Scripts/python py/ldp_analysis.py [--report]   (results/ldp/*.csv, summary*.json[, RESULTS-LDP.md])
"""
from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

import data1_analysis as da
import half_up as hu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ldp"
RUNS = OUT / "runs"
E1 = ROOT / "results" / "data1" / "e1"
SEEDS_5 = [1, 2, 3, 4, 5]         # the first five seeds: the -5seeds tables and RESULTS-LDP.md
SEEDS_10 = list(range(1, 11))     # all ten: the main tables
# per file, the seeds whose runs it covers (ldp.mjs: a seed set other than 1-5 gets its own meta and check files)
META = {"meta.json": SEEDS_5, "meta-s6-10.json": [6, 7, 8, 9, 10]}
META_EPS = {"meta-eps.json": SEEDS_5, "meta-eps-s6-10.json": [6, 7, 8, 9, 10]}
CHECK_C2 = {"check-c2-s1.json": SEEDS_5, "check-c2-s1-s6-10.json": [6, 7, 8, 9, 10]}
CHECK_EPS = {"check-eps-x0-s1.json": SEEDS_5, "check-eps-x0-s1-s6-10.json": [6, 7, 8, 9, 10]}
REPORT = ROOT / "RESULTS-LDP.md"
FIVE = "-5seeds"  # the report's tables are the five-seed ones
OFF, ON = 0.0, 1.0
SCALES = [OFF, ON, 0.5, 4.0]
LABEL = {OFF: "LDP off (clipping only, σ = 0)", ON: "LDP on (configured σ)", 0.5: "weaker noise (σ × 0.5)",
         4.0: "stronger noise (σ × 4)"}
PASSIVES = {1: "telecom", 2: "retailer", 3: "device maker"}
QUANTS = ["F_j", "F_r", "U_fed", "BASELINE", "l_1", "l_2", "l_3", "E"]
DRAW_QUANTS = ["U_fed", "l_1", "l_2", "l_3", "E"]
EPS_TARGETS = [1, 4, 16, 64]   # per-release epsilon of the privacy-utility curve
DELTA_CONVENTIONAL = 1e-6      # a conventional total delta, for the per-household totals beside delta = 1/n


# ---- per run and per draw ---------------------------------------------------------------------------

def _derived(F):
    u = da.u_fed(F["FULL"], F["FULL_J"])
    l = [u - F[f"LOO_{i}"] for i in (1, 2, 3)]
    return {"U_fed": u, "l_1": l[0], "l_2": l[1], "l_3": l[2], "E": max(0, u - F["BASELINE"]), "ranking": da.ranking(l)}


def run_row(run):
    """One run's figures: F_r, F_j, U_fed (D139), LOO_i, BASELINE, l_i, E and the ranking of the passives by l."""
    F = {f["name"]: f["F"] for f in run["figures"]}
    return {"F_r": F["FULL"], "F_j": F["FULL_J"], "F_j_binds": F["FULL_J"] > F["FULL"], "BASELINE": F["BASELINE"],
            **{f"LOO_{i}": F[f"LOO_{i}"] for i in (1, 2, 3)}, **_derived(F)}


def draw_rows(n64):
    """Each N = 64 draw with the contract's rule applied to its own figures."""
    return [{"j": x.get("j"), **_derived(x["F"])} for x in n64["draws"]]


def seed_n64(draws):
    """Mean and SD over one run's draws; for l_i and E also the share of draws with a value <= 0."""
    out = {}
    for q in DRAW_QUANTS:
        v = [d[q] for d in draws]
        out[q] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
        if q != "U_fed":
            out[q]["share_le_0"] = sum(1 for x in v if x <= 0) / len(v)
    return out


def same_ranking_share(a, b):
    """Share of draws (paired by index: the same 64 rows) on which two conditions rank the passives alike."""
    if len(a) != len(b):
        raise ValueError(f"{len(a)} draws against {len(b)}")
    return sum(1 for x, y in zip(a, b) if tuple(x) == tuple(y)) / len(a)


def noise_rows(run):
    """Each passive's clip and sigma in a run, with the Gaussian-mechanism estimate of DATA1 (da.dp_epsilon: one row's
    direct releases, T + 1 = 21 of them, delta 1e-5 each; an estimate, not a guarantee, QUESTIONS-DATA1 Q16). No noise:
    no epsilon."""
    rows = []
    for i, nz in enumerate(run["noise"]["perPassive"], 1):
        e = da.dp_epsilon(nz["sigma"], nz["clip"]) if nz["sigma"] > 0 else None
        rows.append({"passive": i, "clip": nz["clip"], "sigma": nz["sigma"], "sigma_over_C": nz["sigma"] / nz["clip"],
                     "eps_per_release_analytic": e["eps_analytic"] if e else None,
                     "eps_total": e["eps_total_simple"] if e else None, "delta_total": e["delta_total"] if e else None})
    return rows


def check_pairing(runs):
    """The runs of one seed must share the model config, the test rows and (where noise is on) the normal draws."""
    for key, get in [("config", lambda r: r["configHash"]), ("test rows", lambda r: r["rows"]["testPosSha256"]),
                     ("noise seeds", lambda r: tuple(r["noise"]["noiseSeeds"]) if isinstance(r["noise"], dict) else None)]:
        vals = {get(r) for r in runs} - {None}
        if len(vals) > 1:
            raise ValueError(f"runs differ in {key}: {sorted(map(str, vals))}")


# ---- formal epsilon ---------------------------------------------------------------------------------

def calibrate_sigma(eps, clip, delta):
    """The sigma at which the Gaussian mechanism on one clipped row (L2 clip `clip`, replace-one neighbours, so L2
    sensitivity 2 * clip) is exactly (eps, delta)-DP by the analytic Gaussian mechanism (Balle & Wang 2018): bisection
    on log sigma, since delta(eps) falls as sigma grows."""
    sens = 2 * clip
    lo, hi = 1e-9 * sens, 1e6 * sens
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if da._delta_of_eps(eps, sens, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def composed_eps(sigma, clip, releases, delta):
    """A household's epsilon over `releases` Gaussian releases of its row at total delta, by Gaussian DP (Dong, Roth &
    Su 2019): k releases with sensitivity 2C and noise sigma are (sqrt(k) * 2C / sigma)-GDP, i.e. one Gaussian
    mechanism with sensitivity sqrt(k) * 2C; exact for Gaussian mechanisms, also under adaptive composition."""
    return da.dp_epsilon(sigma, clip * math.sqrt(releases), delta)["eps_analytic"]


def eps_levels(targets, clips, delta, releases):
    """Per target per-release epsilon: each passive's sigma for its clip bound, the epsilon it gives (check), and a
    training household's total over its releases by Gaussian DP and by simple composition."""
    out = []
    for eps in targets:
        ps = []
        for i, c in enumerate(clips, 1):
            s = calibrate_sigma(eps, c, delta)
            ps.append({"passive": i, "clip": c, "sigma": s, "sigma_over_C": s / c,
                       "eps_check": da.dp_epsilon(s, c, delta)["eps_analytic"]})
        out.append({"eps": eps, "passives": ps, "releases": releases,
                    "eps_total_gdp": composed_eps(ps[0]["sigma"], clips[0], releases, delta), "delta_total_gdp": delta,
                    "eps_total_simple": releases * eps, "delta_total_simple": releases * delta})
    return out

# ---- io ---------------------------------------------------------------------------------------------

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sci(x):
    """A small probability for the CSVs and tables, without the 4-decimal rounding of da.cell (7.83e-05, not 0.0001)."""
    return None if x is None else f"{x:.4e}"


def tag(k):
    return f"x{k:g}"


def write_csv(name, header, rows, out=None):
    out = OUT if out is None else Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([da.cell(x) for x in r])
    return header, rows


def md(table, fmt=None):
    """A Markdown table from (header, rows); numbers with thousands separators, floats to 1 decimal unless fmt says."""
    header, rows = table
    fmt = fmt or {}

    def c(x, col):
        if x is None or x == "":
            return ""
        if isinstance(x, bool):
            return "yes" if x else "no"
        if isinstance(x, int):
            return f"{x:,}"
        if isinstance(x, float):
            return hu.format_spec(fmt.get(col, "{:,.1f}"), x)  # rounded half up
        return str(x)
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(c(x, h) for x, h in zip(r, header)) + " |" for r in rows]
    return "\n".join(out)


def rank_names(r):
    return " > ".join(PASSIVES[i] for i in r)


# ---- tables -----------------------------------------------------------------------------------------

def condition_tables(prefix, keys, label, runs, n64s, ref, col, tagf, extra=(), also=(), seeds=SEEDS_5, sfx="", out=None):
    """The tables of one set of conditions (keys; label[k] its name; runs[k][s] and n64s[k][s] per seed) against the
    reference condition ref (LDP off): {prefix}_runs, _summary, _paired, _n64_seeds, _n64 and _ranking.csv. col: the
    name and value of the column that identifies a condition; extra: further paired comparisons (name, key, rows of
    another set per seed); also: run rows outside the set that count for the telecom-last tally; seeds: the seed set,
    sfx: its file-name suffix, out: the directory written to. Returns the summary, the tables shown in RESULTS-LDP.md
    and the run rows."""
    SEEDS = seeds  # noqa: N806
    wcsv = lambda stem, header, rows: write_csv(f"{stem}{sfx}.csv", header, rows, out)  # noqa: E731
    rr = {k: {s: run_row(runs[k][s]) for s in SEEDS} for k in keys}
    dr = {k: {s: draw_rows(n64s[k][s]) for s in SEEDS} for k in keys}
    S, T = {}, {}
    run_cols = ["F_r", "F_j", "U_fed", "F_j_binds", "LOO_1", "LOO_2", "LOO_3", "BASELINE", "l_1", "l_2", "l_3", "E"]
    wcsv(f"{prefix}_runs", ["condition", col[0], "seed", *run_cols, "ranking"],
              [[label[k], col[1](k), s, *[rr[k][s][c] for c in run_cols], rank_names(rr[k][s]["ranking"])]
               for k in keys for s in SEEDS])

    srows, S["full_test"] = [], {}
    for k in keys:
        S["full_test"][tagf(k)] = {}
        for q in QUANTS:
            st = da.summarize([rr[k][s][q] for s in SEEDS])
            S["full_test"][tagf(k)][q] = st
            srows.append([label[k], q, st["n"], st["mean"], st["sd"], st["ci95_lo"], st["ci95_hi"], st["min"], st["max"]])
    T["summary"] = wcsv(f"{prefix}_summary", ["condition", "quantity", *da.SUMMARY_HEADER[1:]], srows)

    prows, S["paired"] = [], {}
    comps = [(f"{label[k]} − LDP off", rr[k], rr[ref]) for k in keys if k != ref]
    comps += [(name, rr[k], rows) for name, k, rows in extra]
    for name, a, b in comps:
        S["paired"][name] = {}
        for q in QUANTS:
            p = da.paired([a[s][q] for s in SEEDS], [b[s][q] for s in SEEDS])
            S["paired"][name][q] = p
            prows.append([name, q, p["n"], p["mean"], p["sd"], p["ci95_lo"], p["ci95_hi"], p["min"], p["max"],
                          p["positive"], p["zero"]])
    T["paired"] = wcsv(f"{prefix}_paired", ["comparison", "quantity", "n", "mean", "sd", "ci95_lo", "ci95_hi",
                                                     "min", "max", "positive", "zero"], prows)

    nrows_seed, nrows, S["n64"] = [], [], {}
    for k in keys:
        per = {s: seed_n64(dr[k][s]) for s in SEEDS}
        S["n64"][tagf(k)] = {}
        for q in DRAW_QUANTS:
            for s in SEEDS:
                nrows_seed.append([label[k], s, q, rr[k][s][q], per[s][q]["mean"], per[s][q]["sd"],
                                   per[s][q].get("share_le_0", "")])
            means = [per[s][q]["mean"] for s in SEEDS]
            pooled = [d[q] for s in SEEDS for d in dr[k][s]]
            agg = {"mean_of_seed_means": float(np.mean(means)), "sd_of_seed_means": float(np.std(means, ddof=1)),
                   "mean_within_seed_sd": float(np.mean([per[s][q]["sd"] for s in SEEDS])),
                   "pooled_sd": float(np.std(pooled, ddof=1)),
                   "share_le_0": (sum(1 for x in pooled if x <= 0) / len(pooled)) if q != "U_fed" else None}
            S["n64"][tagf(k)][q] = agg
            nrows.append([label[k], q, len(pooled), *agg.values()])
    wcsv(f"{prefix}_n64_seeds", ["condition", "seed", "quantity", "full_test", "mean_N64", "sd_N64",
                                          "share_le_0"], nrows_seed)
    T["n64"] = wcsv(f"{prefix}_n64", ["condition", "quantity", "draws", "mean_of_seed_means",
                                               "sd_of_seed_means", "mean_within_seed_sd", "pooled_sd", "share_le_0"], nrows)

    krows, S["ranking"] = [], {}
    for k in keys:
        same_full = [rr[k][s]["ranking"] == rr[ref][s]["ranking"] for s in SEEDS]
        own = [same_ranking_share([d["ranking"] for d in dr[k][s]], [rr[k][s]["ranking"]] * len(dr[k][s])) for s in SEEDS]
        vs_off = [same_ranking_share([d["ranking"] for d in dr[k][s]], [d["ranking"] for d in dr[ref][s]]) for s in SEEDS]
        S["ranking"][tagf(k)] = {"full_test_same_as_off": sum(same_full), "n64_same_as_own_full_test": float(np.mean(own)),
                                 "n64_same_as_off_same_draw": float(np.mean(vs_off)),
                                 "full_test_rankings": [rank_names(rr[k][s]["ranking"]) for s in SEEDS]}
        for s, sf, o, v in zip(SEEDS, same_full, own, vs_off):
            krows.append([label[k], s, rank_names(rr[k][s]["ranking"]), sf, o, v])
    last = [rr[k][s]["ranking"][-1] for k in keys for s in SEEDS] + [r["ranking"][-1] for rows in also for r in rows.values()]
    S["ranking"]["telecom_last"] = {"runs": len(last), "telecom_last": sum(1 for x in last if x == 1)}
    T["ranking"] = wcsv(f"{prefix}_ranking", ["condition", "seed", "full_test_ranking",
                                                       "same_as_LDP_off_full_test", "N64_share_same_as_own_full_test",
                                                       "N64_share_same_as_LDP_off_same_draw"], krows)
    return S, T, rr


def resources(rs):
    return {"runs": len(rs), "wall_s_total": sum(r["wallMs"] for r in rs) / 1000,
            "peak_rss_mb_max": max(r["peakRssMB"] for r in rs),
            "free_ram_mb_min_at_start": min(r["machine"]["freeRamMB"] for r in rs)}


def figures_of(path):
    return {f["name"]: f["F"] for f in load(path)["figures"]}


def check_n64_sources(n64s, name):
    for k, by in n64s.items():
        for s, n in by.items():
            if n["source"].replace("\\", "/").split("/")[-1] != name(k, s):
                raise ValueError(f"n64 {k} s{s} is not from its run")


def held(files, seeds):
    """The files (of META, CHECK_C2, ...) that cover some of `seeds`."""
    return [name for name, covered in files.items() if set(covered) & set(seeds)]


def main():
    if "--levels" in sys.argv:
        return write_levels()
    analyse(SEEDS_5, report=REPORT if "--report" in sys.argv else None)
    analyse(SEEDS_10)


def analyse(seeds, out=None, report=None):
    """The tables over `seeds`, named with the suffix of the seed set (da.seed_suffix: none for 1-10, -5seeds for 1-5),
    written to `out` (default results/ldp); with `report` (a path, seeds 1-5 only) also RESULTS-LDP.md."""
    SEEDS = list(seeds)  # noqa: N806
    sfx = da.seed_suffix(SEEDS)
    if report is not None and SEEDS != SEEDS_5:
        raise ValueError("RESULTS-LDP.md reports seeds 1-5")
    wcsv = lambda name, header, rows: write_csv(name, header, rows, out)  # noqa: E731
    out_dir = OUT if out is None else Path(out)
    meta = load(OUT / "meta.json")
    more = {name: load(OUT / name) for name in held(META, SEEDS) if name != "meta.json"}
    runs = {k: {s: load(RUNS / f"ldp-{tag(k)}-s{s}.json") for s in SEEDS} for k in SCALES}
    n64s = {k: {s: load(RUNS / f"n64-{tag(k)}-s{s}.json") for s in SEEDS} for k in SCALES}
    c2 = {s: load(E1 / f"c2-s{s}.json") for s in SEEDS}  # E1's C2: no clipping, no noise (34e33ac)
    for s in SEEDS:
        check_pairing([runs[k][s] for k in SCALES] + [c2[s]])
    check_n64_sources(n64s, lambda k, s: f"ldp-{tag(k)}-s{s}.json")
    e1_c2s1 = figures_of(E1 / "c2-s1.json")
    for name in held(CHECK_C2, SEEDS):  # each set of runs reran E1's C2 seed 1 first
        chk = figures_of(OUT / name)
        if chk != e1_c2s1:
            raise ValueError(f"check run {name} {chk} differs from E1's C2 seed 1 {e1_c2s1}")
    c2r = {s: run_row(c2[s]) for s in SEEDS}
    summary = {"meta": meta, "check_c2_s1_equals_e1": True}
    if SEEDS != SEEDS_5:  # the first version's keys stay as they were
        summary |= {"seeds": SEEDS, "meta_other_seeds": more}

    t_noise = wcsv("ldp_noise.csv", ["condition", "sigma_scale", "passive", "clip", "sigma", "sigma_over_C",
                                          "eps_per_release_analytic", "eps_total_21_releases", "delta_total"],
                        [[LABEL[k], k, f"{i} {PASSIVES[i]}", r["clip"], r["sigma"], r["sigma_over_C"],
                          r["eps_per_release_analytic"], r["eps_total"], sci(r["delta_total"])]
                         for k in SCALES for i, r in enumerate(noise_rows(runs[k][1]), 1)])
    S, T, _ = condition_tables("ldp", SCALES, LABEL, runs, n64s, OFF, ("sigma_scale", lambda k: k), tag,
                               extra=[("LDP off (clipping only) − E1 C2 (no clipping)", OFF, c2r)], also=[c2r],
                               seeds=SEEDS, sfx=sfx, out=out)
    summary.update(S)
    summary["resources"] = resources([runs[k][s] for k in SCALES for s in SEEDS])

    # privacy-utility curve: sigma calibrated to a per-release epsilon, against the same LDP-off runs
    lv = load(OUT / "eps-levels.json")
    ekeys = ["off"] + [f"e{x['eps']:g}" for x in lv["levels"]]
    elabel = {"off": LABEL[OFF], **{f"e{x['eps']:g}": f"ε = {x['eps']:g} per release" for x in lv["levels"]}}
    eps_of = {"off": float("inf"), **{f"e{x['eps']:g}": x["eps"] for x in lv["levels"]}}
    eruns = {"off": runs[OFF], **{k: {s: load(RUNS / f"eps-{k}-s{s}.json") for s in SEEDS} for k in ekeys[1:]}}
    en64 = {"off": n64s[OFF], **{k: {s: load(RUNS / f"n64-eps-{k}-s{s}.json") for s in SEEDS} for k in ekeys[1:]}}
    for s in SEEDS:
        check_pairing([eruns[k][s] for k in ekeys])
    check_n64_sources({k: en64[k] for k in ekeys[1:]}, lambda k, s: f"eps-{k}-s{s}.json")
    for x in lv["levels"]:  # the runs used the calibrated sigmas
        got = [p["sigma"] for p in eruns[f"e{x['eps']:g}"][1]["noise"]["perPassive"]]
        if any(abs(g - p["sigma"]) > 1e-12 * p["sigma"] for g, p in zip(got, x["passives"])):
            raise ValueError(f"eps {x['eps']}: run sigmas {got} are not the calibrated ones")
    for name in held(CHECK_EPS, SEEDS):
        if figures_of(OUT / name) != figures_of(RUNS / "ldp-x0-s1.json"):
            raise ValueError(f"the eps-set check run {name} does not reproduce LDP off, seed 1")
    E, ET, err = condition_tables("ldp_eps", ekeys, elabel, eruns, en64, "off", ("eps_per_release", eps_of.get),
                                  lambda k: k, seeds=SEEDS, sfx=sfx, out=out)
    t_levels = wcsv("ldp_eps_levels.csv", ["eps_per_release", "passive", "clip", "sigma", "sigma_over_C",
                                                "eps_check", "delta_per_release", "releases", "eps_total_gdp",
                                                "eps_total_gdp_delta_1e-6", "delta_total_gdp", "eps_total_simple",
                                                "delta_total_simple", "eps_total_gdp_3_passives",
                                                "eps_total_gdp_3_passives_delta_1e-6"],
                         [[x["eps"], f"{p['passive']} {PASSIVES[p['passive']]}", p["clip"], p["sigma"], p["sigma_over_C"],
                           p["eps_check"], sci(lv["delta"]), x["releases"], x["eps_total_gdp"],
                           composed_eps(p["sigma"], p["clip"], x["releases"], DELTA_CONVENTIONAL), sci(x["delta_total_gdp"]),
                           x["eps_total_simple"], sci(x["delta_total_simple"]),
                           composed_eps(p["sigma"], p["clip"], 3 * x["releases"], lv["delta"]),
                           composed_eps(p["sigma"], p["clip"], 3 * x["releases"], DELTA_CONVENTIONAL)]
                          for x in lv["levels"] for p in x["passives"]])
    E["totals_delta_1e-6"] = {f"{x['eps']:g}": {
        "per_passive": composed_eps(x["passives"][0]["sigma"], x["passives"][0]["clip"], x["releases"], DELTA_CONVENTIONAL),
        "three_passives": composed_eps(x["passives"][0]["sigma"], x["passives"][0]["clip"], 3 * x["releases"],
                                       DELTA_CONVENTIONAL)} for x in lv["levels"]}
    urows, E["utility"] = [], {}
    for k in ekeys:
        gap = [err[k][s]["U_fed"] - err[k][s]["BASELINE"] for s in SEEDS]
        st = da.summarize(gap)
        E["utility"][k] = {**st, "seeds_above": sum(1 for g in gap if g > 0)}
        urows.append([elabel[k], eps_of[k], st["mean"], st["sd"], st["min"], st["max"], E["utility"][k]["seeds_above"]])
    t_util = wcsv(f"ldp_eps_utility{sfx}.csv", ["condition", "eps_per_release", "FULL_minus_BASELINE_mean", "sd", "min",
                                               "max", "seeds_FULL_above_BASELINE"], urows)
    # the lowest tested epsilon from which on (it and every larger one) FULL > BASELINE in every seed
    stays = [k for k in ekeys[1:]
             if all(E["utility"][j]["seeds_above"] == len(SEEDS) for j in ekeys[1:] if eps_of[j] >= eps_of[k])]
    E["lowest_eps_full_above_baseline"] = min((eps_of[k] for k in stays), default=None)
    E["levels"], E["meta"] = lv, load(OUT / "meta-eps.json")
    E["resources"] = resources([eruns[k][s] for k in ekeys[1:] for s in SEEDS])
    if SEEDS != SEEDS_5:
        E |= {"seeds": SEEDS, "meta_other_seeds": {name: load(OUT / name) for name in held(META_EPS, SEEDS)
                                                   if name != "meta-eps.json"}}
    summary["eps"] = E
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"summary{sfx}.json").write_text(json.dumps(summary, indent=1, default=list) + "\n", encoding="utf-8")
    t_sat = wcsv(f"ldp_eps_saturation{sfx}.csv", ["condition", "eps_per_release", "passive", "share_at_int16_limit"],
                      [[elabel[k], eps_of[k], f"{i + 1} {PASSIVES[i + 1]}",
                        float(np.mean([eruns[k][s]["clipping"]["hq"][i] / eruns[k][s]["clipping"]["entries"] for s in SEEDS]))]
                       for k in ekeys for i in range(3)])
    if report is not None:
        write_report(summary, runs, t_noise, T, ET, t_levels, t_util, t_sat, runs[ON][1]["machine"], Path(report))
    print(f"seeds {SEEDS[0]}-{SEEDS[-1]}: wrote {out_dir}/*{sfx}.csv, summary{sfx}.json" + (f", {Path(report).name}" if report else ""))


def write_levels():
    """results/ldp/eps-levels.json: per target epsilon, each passive's sigma for its current clip bound (LOCAL_NOISE,
    as the LDP runs recorded it), with delta = 1 / (training households) and the totals over a training household's
    releases (T + 1)."""
    ref = load(RUNS / "ldp-x0-s1.json")
    n_train = ref["rows"]["train"]
    clips = [p["clip"] for p in ref["noise"]["perPassive"]]
    releases = load(ROOT / "results" / "data1" / "configs" / "e1-s1.json")["T"] + 1
    out = {"delta": 1 / n_train, "n_train": n_train, "releases_training_household": releases,
           "releases_test_household": 1,
           "mechanism": "Gaussian mechanism on one household's released embedding row (L2 clip C, N(0, sigma^2) per "
                        "entry); replace-one neighbours, L2 sensitivity 2C; analytic Gaussian mechanism (Balle & Wang 2018)",
           "accountant": "Gaussian DP (Dong, Roth & Su 2019): k releases at sensitivity 2C and noise sigma compose to one "
                         "Gaussian mechanism with sensitivity sqrt(k) * 2C (exact); simple composition (k eps, k delta) "
                         "for comparison",
           "levels": eps_levels(EPS_TARGETS, clips, 1 / n_train, releases)}
    (OUT / "eps-levels.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps([{"eps": x["eps"], "sigmas": [hu.round_half_up(p["sigma"], 4) for p in x["passives"]],
                       "eps_total_gdp": hu.round_half_up(x["eps_total_gdp"], 2)} for x in out["levels"]]))


def write_report(S, runs, t_noise, T, ET, t_levels, t_util, t_sat, m0, path=REPORT):
    """RESULTS-LDP.md from the five-seed results; its tables are the -5seeds files (ldp_noise.csv and
    ldp_eps_levels.csv hold no seed)."""
    ft, pr, n6, rk = S["full_test"], S["paired"], S["n64"], S["ranking"]
    on, off = ft[tag(ON)], ft[tag(OFF)]
    d = pr[f"{LABEL[ON]} − LDP off"]
    clip = pr["LDP off (clipping only) − E1 C2 (no clipping)"]
    cfg = load(ROOT / "results" / "data1" / "configs" / "e1-s1.json")
    nz = runs[ON][1]["noise"]["perPassive"]
    pm = lambda st: f"{hu.fmt(st['mean'])} ± {hu.fmt(st['sd'])}"  # noqa: E731  (all numbers: half up)
    lower = lambda p: p["n"] - p["positive"] - p["zero"]  # noqa: E731
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    eps = noise_rows(runs[ON][1])
    eps_lo, eps_hi = min(r["eps_per_release_analytic"] for r in eps), max(r["eps_per_release_analytic"] for r in eps)
    tot_lo, tot_hi = min(r["eps_total"] for r in eps), max(r["eps_total"] for r in eps)
    releases = cfg["T"] + 1
    worked = "; ".join(f"{PASSIVES[i]}: C = {r['clip']:g}, σ = {r['sigma']:g}, Δ/σ = {hu.fmt(2 * r['clip'] / r['sigma'])}, "
                       f"ε = {hu.fmt(r['eps_per_release_analytic'])}" for i, r in enumerate(eps, 1))
    E, lv = S["eps"], S["eps"]["levels"]
    ekeys = ["off"] + [f"e{x['eps']:g}" for x in lv["levels"]]
    eft, eu, ern, en6 = E["full_test"], E["utility"], E["ranking"], E["n64"]
    low = E["lowest_eps_full_above_baseline"]
    sat = {}
    for _, e_, _, share in t_sat[1]:
        sat.setdefault(e_, []).append(share)
    pct = lambda e_: f"{hu.fmt(100 * hu.dec(min(sat[e_])))}–{hu.fmt(100 * hu.dec(max(sat[e_])))} %"  # noqa: E731
    cap = (f"The ε = 1 and ε = 4 points are also affected by the int16 cap on the final matrix (±8 at scale 2¹²): {pct(1)} "
           f"(ε = 1) and {pct(4)} (ε = 4) of its entries are cut to the limit (`ldp_eps_saturation{FIVE}.csv`), so their figures "
           f"reflect that cap as well as the noise.")
    curve = "; ".join(f"ε = {x['eps']:g}: FULL {pm(eft[k]['U_fed'])}, BASELINE {pm(eft[k]['BASELINE'])}, FULL − BASELINE "
                      f"{hu.fmt(eu[k]['mean'], sign=True)} ± {hu.fmt(eu[k]['sd'])} (above in {eu[k]['seeds_above']} of 5 seeds)"
                      for x, k in zip(lv["levels"], ekeys[1:]))
    text = f"""# RESULTS-LDP — local differential privacy, with vs without (DATA1 setting)

Generated by `py/ldp_analysis.py` from the runs in `results/ldp/runs/` (written by
`contracts/offchain/data1/ldp.mjs`); do not edit by hand. Every number below is read from a file in `results/ldp/`,
and every table is also there as CSV (named above it). Two sets of runs: the LDP runs (σ × 0, 1, 0.5, 4) used the
code at commit **{S['meta']['commit']}** ({'clean tree' if S['meta']['clean'] else 'UNCOMMITTED CHANGES in contracts/offchain'}),
started {S['meta']['started']}; the privacy–utility runs (σ calibrated to ε) used **{E['meta']['commit']}**
({'clean tree' if E['meta']['clean'] else 'UNCOMMITTED CHANGES'}), started {E['meta']['started']}. Report generated at commit
{git('rev-parse', '--short', 'HEAD')}.

**Seeds.** This is the report of the first runs, seeds 1–5; its tables are the `-5seeds` files (`ldp_noise.csv` and
`ldp_eps_levels.csv` hold no seed). The runs were later extended to seeds 1–10 (`ldp.mjs --seeds 6-10`): those tables
keep the main names (`ldp_summary.csv`, ...) and are the ones Chapter 5 cites (`CH5-EVIDENCE.md`).

**Machine.** {m0['cpu']}, {m0['cores']} logical cores, {m0['totalRamMB']:,} MB RAM, {m0['platform']}; free RAM at the
start of each run is in its JSON (lowest {min(S['resources']['free_ram_mb_min_at_start'], E['resources']['free_ram_mb_min_at_start']):,} MB).
**Tools.** Node {m0['node']}; Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}.
**Run time.** LDP runs: {S['resources']['runs']} training runs, {hu.fmt(S['resources']['wall_s_total'] / 60, 1)} min; privacy–utility runs:
{E['resources']['runs']} runs, {hu.fmt(E['resources']['wall_s_total'] / 60, 1)} min (one at a time; peak RSS
{max(S['resources']['peak_rss_mb_max'], E['resources']['peak_rss_mb_max']):,} MB).

**Rerun** (from the repo root; needs `data/recs/prepared/`, see RESULTS-DATA1.md §2):

```bash
cd contracts && node offchain/data1/ldp.mjs && cd ..              # LDP runs: check run, 20 runs, N = 64 draws
.venv/Scripts/python py/ldp_analysis.py --levels                   # sigma per epsilon -> results/ldp/eps-levels.json
cd contracts && node offchain/data1/ldp.mjs --set eps && cd ..    # privacy-utility runs: check run, 20 runs, N = 64
.venv/Scripts/python py/ldp_analysis.py --report                   # these tables (seeds 1-5 and 1-10) and this file
```

## Configuration

- **Data.** DATA1 (RECS 2020): all 15,965 households, split as E1 (seed 20260927): 12,772 training, 3,193 test
  rows. Active party A: 8 columns; passives telecom / retailer / device maker (RESULTS-DATA1.md §1–2).
- **Model and training.** The thesis's split training and scoring code (`training.mjs`, `scoring.mjs`) over
  in-process sessions, as E1 (no chain, no proofs; E3 showed the proven figures equal this off-chain oracle). Model
  configs `results/data1/configs/e1-s1.json` … `e1-s5.json`: E3's config is `e1-s1` (identical), and the five
  differ only in initSeed and shuffleSeed. T = {cfg['T']} rounds, batch {cfg['batchSize']}, SGD, learning rate {cfg['learningRate']};
  a passive backpropagates A's gradient straight through its clipping and noise.
- **Noise.** Each passive's own protocol setting, `LOCAL_NOISE` in `contracts/offchain/party-m3.mjs`: every
  released row of its embedding (each training batch and the final matrix) is clipped to L2 norm C and gets
  N(0, σ²) per entry. **LDP on:** C = {' / '.join(f"{x['clip']:g}" for x in nz)}, σ = {' / '.join(f"{x['sigma']:g}" for x in nz)}
  (telecom / retailer / device maker). **LDP off:** the same C, σ = 0. Optional levels: σ × 0.5 and σ × 4, same C.
  Within a seed every condition uses the same normal draws (noise seeds 20263000 + 10·seed + i, as E1's C3), scaled by
  σ. E1's C2 (no clipping, no noise; `results/data1/e1/`) is the reference for the clipping alone.
- **Seeds and repetitions.** Seeds 1–5 per condition, one run each (20 runs). N = 64 (the protocol's evaluation
  size): per run, 1,000 draws of 64 test rows (j = 1..1,000, the same rows in every condition). Before the runs, C2
  seed 1 was rerun and reproduced E1's six figures exactly (`results/ldp/check-c2-s1.json`), so the current code is
  E1's.
- **Definitions.** F = ⌊10⁴ × macro-F1⌋ on the 5 tariff tiers (integer path). F_j = the jointly trained model (the
  "joint model"), F_r = the retrained FULL, FULL = U_fed = max(F_r, F_j) (D139). BASELINE = C1a (all passive slots
  zero). l_i = U_fed − F_LOO_i (the leave-one-out share, raw); E = max(0, U_fed − F_BASELINE). n = 5 seeds:
  descriptive only, nothing is called significant.

## Headline

- **Accuracy.** Joint model F_j: {pm(on['F_j'])} with LDP, {pm(off['F_j'])} without. FULL (U_fed): {pm(on['U_fed'])}
  with, {pm(off['U_fed'])} without; paired difference {hu.fmt(d['U_fed']['mean'], sign=True)} ± {hu.fmt(d['U_fed']['sd'])}, lower with LDP in
  {lower(d['U_fed'])} of 5 seeds. BASELINE: {pm(on['BASELINE'])} with, {pm(off['BASELINE'])} without.
- **Shares (full test).** With LDP: telecom {pm(on['l_1'])}, retailer {pm(on['l_2'])}, device maker {pm(on['l_3'])}; without:
  {pm(off['l_1'])}, {pm(off['l_2'])}, {pm(off['l_3'])}. E: {pm(on['E'])} with, {pm(off['E'])} without.
- **Shares (N = 64).** Mean over seeds of the draw means, with LDP: {', '.join(hu.fmt(n6[tag(ON)][f'l_{i}']['mean_of_seed_means']) for i in (1, 2, 3))};
  without: {', '.join(hu.fmt(n6[tag(OFF)][f'l_{i}']['mean_of_seed_means']) for i in (1, 2, 3))}. The spread of one draw
  (pooled SD) is {', '.join(hu.fmt(n6[tag(ON)][f'l_{i}']['pooled_sd']) for i in (1, 2, 3))} with LDP and
  {', '.join(hu.fmt(n6[tag(OFF)][f'l_{i}']['pooled_sd']) for i in (1, 2, 3))} without.
- **Ranking.** Full test: the LDP-on ranking equals the LDP-off ranking in {rk[tag(ON)]['full_test_same_as_off']} of 5 seeds
  (LDP off: {'; '.join(rk[tag(OFF)]['full_test_rankings'])}; LDP on: {'; '.join(rk[tag(ON)]['full_test_rankings'])}). At N = 64, on the
  same 64 rows, the two conditions rank the passives alike in {hu.fmt_pct(rk[tag(ON)]['n64_same_as_off_same_draw'], 1)} of the draws.
  Telecom ranks last in {rk['telecom_last']['telecom_last']} of {rk['telecom_last']['runs']} full-test runs (these 20 and E1's 5 C2
  runs){', so every change of ranking is a swap of retailer and device maker' if rk['telecom_last']['telecom_last'] == rk['telecom_last']['runs'] else ''}.
- **Clipping alone** (LDP off against E1's C2, which has no clipping): U_fed {hu.fmt(clip['U_fed']['mean'], sign=True)} ± {hu.fmt(clip['U_fed']['sd'])},
  l = {', '.join(hu.fmt(clip[f'l_{i}']['mean'], sign=True) for i in (1, 2, 3))} (paired means over the 5 seeds).
- **Privacy–utility curve** (σ calibrated to a per-release ε, δ = 1/{lv['n_train']:,}; section below): {curve}. The lowest
  tested ε at which FULL stays above BASELINE in every seed (at it and at every larger tested ε) is
  **{'none of the tested levels' if low is None else f'ε = {low:g} per release'}**{'' if low is None else f", {hu.fmt(E['levels']['levels'][[x['eps'] for x in lv['levels']].index(low)]['eps_total_gdp'])} at δ = 1/{lv['n_train']:,} or {hu.fmt(E['totals_delta_1e-6'][f'{low:g}']['per_passive'])} at δ = 10⁻⁶ for a training household over its {releases} releases (Gaussian DP)"}.
  {cap}

## How ε is computed

The ε values in `ldp_noise.csv` (per release {hu.fmt(eps_lo)}–{hu.fmt(eps_hi)} at the configured σ) and in the privacy–utility
section come from one calculation, `dp_epsilon` in `py/data1_analysis.py` (DATA1's, QUESTIONS-DATA1 Q16):

- **One release** is one transmission of one household's embedding row (E = 4 numbers) from a passive to A. In each
  of the T = {cfg['T']} training rounds the passive sends f_i(x) for the rows of each batch A asks for, and the batches of a
  round cover every training household exactly once (`batchesFor`, `training.mjs`); after the last round it sends
  the final matrix, which covers every household of I. So a training household's row is released T + 1 = {releases} times,
  a test household's once (the final matrix only).
- **Mechanism.** Before every release the row is clipped to L2 norm C and N(0, σ²) is added to each of its 4 entries
  (`clipAndNoise`, `model.mjs`): the Gaussian mechanism. Quantizing the final matrix to int16 (scale 2¹², range ±8)
  is post-processing and does not change ε.
- **Sensitivity.** Local DP compares any two values of the household's own row; after clipping both lie in the
  L2 ball of radius C, so they differ by at most Δ = 2C (replace-one).
- **δ.** 10⁻⁵ per release for `ldp_noise.csv` (DATA1's choice); 1/{lv['n_train']:,} = {hu.fmt_e(lv['delta'], 3)} per release for the
  privacy–utility curve, as asked.
- **ε.** The smallest ε with δ(ε) ≤ δ on the exact privacy curve of the Gaussian mechanism (the analytic Gaussian
  mechanism, Balle & Wang 2018, Theorem 8): δ(ε) = Φ(Δ/(2σ) − εσ/Δ) − e^ε · Φ(−Δ/(2σ) − εσ/Δ), solved by bisection.
  It depends only on Δ/σ and δ. At the configured setting: {worked}. The classic bound
  σ ≥ Δ·√(2 ln(1.25/δ))/ε holds only for ε ≤ 1 and does not apply here.
- **Totals.** `ldp_noise.csv` adds up the {releases} releases (simple composition: {releases} ε and {releases} δ). The
  privacy–utility section also uses the exact Gaussian-DP composition (below).
- **What ε does not cover.** Only the direct releases of the household's own row. The passive trains f_i on all of
  its households with A's gradients, so one household's data also shapes f_i and, through it, every row released
  later; that path gets no per-household noise (that would need DP-SGD on f_i). So ε is not a privacy guarantee for
  the whole training (Q16).

## Tables

`ldp_noise.csv` — the noise per condition and passive. ε is DATA1's estimate for one row's direct releases
(T + 1 = 21 releases, δ = 10⁻⁵ each; analytic Gaussian mechanism, Balle & Wang 2018), not a DP guarantee for the
model or the training (QUESTIONS-DATA1 Q16). σ = 0 has no ε: it is not differentially private.

{md(t_noise, {"clip": "{:g}", "sigma": "{:g}", "sigma_over_C": "{:.4f}", "eps_per_release_analytic": "{:,.1f}",
              "eps_total_21_releases": "{:,.1f}", "delta_total": "{:.5f}", "sigma_scale": "{:g}"})}

`ldp_summary{FIVE}.csv` — full test set, per condition (5 seeds)

{md(T['summary'])}

`ldp_paired{FIVE}.csv` — paired by seed (same config and noise draws): mean difference, and in how many seeds it is
positive or zero

{md(T['paired'])}

`ldp_n64{FIVE}.csv` — N = 64: per condition, over 5 seeds × 1,000 draws; per seed in `ldp_n64_seeds{FIVE}.csv`

{md(T['n64'], {"share_le_0": "{:.3f}"})}

`ldp_ranking{FIVE}.csv` — the passives' ranking by l on the full test set per seed, and at N = 64 the share of draws
ranking as the run's own full test or as LDP off on the same rows

{md(T['ranking'], {"N64_share_same_as_own_full_test": "{:.3f}", "N64_share_same_as_LDP_off_same_draw": "{:.3f}"})}

Per run: `ldp_runs{FIVE}.csv`; raw runs and draws: `results/ldp/runs/`; log: `results/ldp/ldp.log`.

## Privacy–utility curve with formal ε

**Setup.** As the LDP runs (same rows, configs, seeds 1–5 and normal draws), with each passive's σ calibrated so
that one release is (ε, δ)-DP by the analytic Gaussian mechanism, for ε ∈ {{{', '.join(f"{x['eps']:g}" for x in lv['levels'])}}} per release, at the
current clip bounds (C = {' / '.join(f"{p['clip']:g}" for p in lv['levels'][0]['passives'])}) and δ = 1/{lv['n_train']:,} (the training households)
per release (`results/ldp/eps-levels.json`, `py/ldp_analysis.py --levels`). σ/C is the same for every passive at a
given ε (σ is proportional to Δ = 2C). The LDP-off runs above are the reference (ε = ∞, the same clipping, no
noise); before the runs, LDP off seed 1 was rerun and reproduced its committed figures exactly
(`results/ldp/check-eps-x0-s1.json`). 20 runs, each with 1,000 N = 64 draws.

**Accountant.** A training household's row is released {lv['releases_training_household']} times (T = {cfg['T']} rounds + the final matrix), a test
household's once. Its total ε over the training run is computed with **Gaussian differential privacy** (Dong, Roth &
Su 2019): k releases with sensitivity Δ and noise σ compose, exactly and also when each release depends on the
earlier ones, to √k·Δ/σ-GDP, i.e. to one Gaussian mechanism with sensitivity √k·Δ; that is converted to ε at a total
δ of 1/{lv['n_train']:,} with the same analytic formula, and also at the conventional total δ = 10⁻⁶ (columns ending in
_delta_1e-6). Simple composition ({lv['releases_training_household']} ε at {lv['releases_training_household']} δ) is shown for
comparison; it is looser. These totals are per passive (each protects its own columns of the household); a
household is a customer of all three passives, and over the three its rows are released 3 × {lv['releases_training_household']} times
(column eps_total_gdp_3_passives, the same composition with k = {3 * lv['releases_training_household']}). The caveat above applies: these ε cover
the household's own released rows, not its influence on f_i.

`ldp_eps_levels.csv` — σ per passive and level; ε_check is the per-release ε that σ gives (the target); totals for
a training household

{md(t_levels, {"eps_per_release": "{:g}", "clip": "{:g}", "sigma": "{:.4f}", "sigma_over_C": "{:.4f}", "eps_check": "{:.4f}",
               "delta_per_release": "{:.3e}", "eps_total_gdp": "{:,.1f}", "delta_total_gdp": "{:.3e}",
               "eps_total_simple": "{:,.0f}", "delta_total_simple": "{:.3e}", "eps_total_gdp_3_passives": "{:,.1f}",
               "eps_total_gdp_delta_1e-6": "{:,.1f}", "eps_total_gdp_3_passives_delta_1e-6": "{:,.1f}"})}

`ldp_eps_utility{FIVE}.csv` — FULL − BASELINE per level (5 seeds) and in how many seeds FULL is above BASELINE

{md(t_util, {"eps_per_release": "{:g}"})}

`ldp_eps_summary{FIVE}.csv` — full test set, per level (5 seeds)

{md(ET['summary'])}

`ldp_eps_paired{FIVE}.csv` — each level against LDP off, paired by seed

{md(ET['paired'])}

`ldp_eps_n64{FIVE}.csv` — N = 64, per level, over 5 seeds × 1,000 draws; per seed in `ldp_eps_n64_seeds{FIVE}.csv`

{md(ET['n64'], {"share_le_0": "{:.3f}"})}

`ldp_eps_ranking{FIVE}.csv` — ranking stability against LDP off: full-test ranking per seed, and at N = 64 the share of draws
ranking as LDP off on the same rows. Telecom ranks last in {ern['telecom_last']['telecom_last']} of {ern['telecom_last']['runs']} of these full-test runs.

{md(ET['ranking'], {"N64_share_same_as_own_full_test": "{:.3f}", "N64_share_same_as_LDP_off_same_draw": "{:.3f}"})}

`ldp_eps_saturation{FIVE}.csv` — share of the final matrix's entries at the int16 limit (±8 at scale 2¹²), mean over the
5 seeds: with large σ the noisy values are cut to ±8 when the matrix is quantized (post-processing: ε is unchanged,
but the matrix A uses for scoring carries even less of the signal)

{md(t_sat, {"share_at_int16_limit": "{:.3f}"})}

Per run: `ldp_eps_runs{FIVE}.csv`; raw runs and draws: `results/ldp/runs/eps-*.json`.

## Relation to E1's C3

E1's C3 (RESULTS-DATA1.md, `results/data1/e1_c3.csv`, commit 34e33ac) added noise at σ = 0.1, 0.5 and 2 × C with
C = 4.066 for every passive and compared it with C2, which has no clipping: there, clipping and noise change
together, and the levels are DATA1's choice (QUESTIONS-DATA1 Q6), not the protocol's setting. This experiment
keeps the clipping fixed, uses the protocol's own per-passive setting, and adds N = 64. The training code is
unchanged since 34e33ac (checked by the C2 rerun above).

## Limits

- n = 5 seeds per condition: descriptive.
- ε covers the direct releases of one household's row only, not its influence on the passive's model (see How ε is
  computed). At the configured σ it is {hu.fmt(eps_lo)}–{hu.fmt(eps_hi)} per release ({hu.fmt(tot_lo)}–{hu.fmt(tot_hi)} over the 21 releases, simple
  composition).
- Off-chain runs (as E1): the protocol's chain, proofs and settlement do not take part; E3 showed that the proven
  figures equal the off-chain oracle for the same models.
"""
    path.write_text(text, encoding="utf-8")

if __name__ == "__main__":
    sys.exit(main())
