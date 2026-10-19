"""MCI loan instance v2, Step 4: writes docs/mci/v2/03-results.md from results/mci_loan/v2/offchain/
(scenarios/mci_loan/v2/analysis2.py), so that every number in it has a source file, named next to it. The schema of every
results file is documented here: the columns are read from each CSV's header, and the text says what each one holds.
The naming rule (2026-10-19): the document names no real company except MCI; the run fails if it does.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/v2/results_doc2.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as cm  # noqa: E402
import common2 as c2  # noqa: E402
import policy  # noqa: E402

TEST = os.environ.get("MCI_LOAN_OFFCHAIN")  # a test copy (analysis2.py's override); the doc then goes there too
OUT = Path(TEST) if TEST else c2.RESULTS2 / "offchain"
ON = c2.RESULTS2 / "onchain"
PO = "results/mci_loan/v2/onchain"
DOC = OUT / "03-results.md" if TEST else c2.DOCS2 / "03-results.md"
P = "results/mci_loan/v2/offchain"
NM = c2.names()
PASSIVE = {1: NM["bank"], 2: NM["ecommerce"], 3: NM["bureau"]}
NAMES = {f"l_{i}": v for i, v in PASSIVE.items()}
FORBIDDEN = re.compile(r"digikala|دیجی", re.IGNORECASE)
MODEL_LABEL = [("BASELINE", "(a) MCI alone (BASELINE, U_A)"), ("C1b", "MCI trained alone (C1b)"), ("FULL", "(b) All members (F_r)"),
               ("FULL_J", "(b′) All members (F_j)"), ("LOO_1", f"(c) All but the {NM['bank']}"),
               ("LOO_2", f"(c) All but the {NM['ecommerce']}"), ("LOO_3", f"(c) All but the {NM['bureau']}")]

# file -> (what one row is, {column: meaning}); columns not listed are explained by the common terms below
SCHEMA = {
    "runs.csv": ("one run (process) of step4v2.mjs", {
        "config": "C2, C4a-f32, C1b, or LDP <condition>", "wall_s": "wall time of the process",
        "training_s": "training part", "peak_rss_mb": "peak resident memory", "free_ram_mb_start": "free RAM at start (MB)",
        "free_ram_mb_end": "free RAM at the end", "zero_filled": "zero-filled training batches (late embeddings; must be 0)",
        "hq_saturated_p1": "passive 1's final int16 entries at the limits (clipping evidence); p2, p3 likewise",
        "rows_in_classes_2_4": "rows, training and test, that any figure's head puts in classes 2-4 (must be 0)",
        "config_hash": "keccak256 of the model config (the run's modelConfigHash)"}),
    "configs.csv": ("one configuration; one column per seed", {"config": "the configuration", "seed_<s>": "F on the test split"}),
    "configs_summary.csv": ("one configuration over the 10 seeds", {}),
    "equivalence.csv": ("one seed", {"max_abs_loss_diff_C2_vs_C4a_f32": "largest per-round training-loss difference",
                                     "F_C2_joint (F_j)": "C2's jointly trained fusion model", "F_C4a_f32": "centralized, float32 transport",
                                     "C2_equals_C4a_f32_bitwise": "losses and test predictions identical"}),
    "loo.csv": ("one seed's C2 run", {"U_fed": "max(F_r, F_j), the contract's FULL figure", "F_C1a": "BASELINE (U_A): MCI alone",
                                      "F_C1b": "MCI trained alone", "l_i": "U_fed - F_LOO_i (may be negative)",
                                      "c_i": "max(0, l_i), the claim", "E": "max(0, U_fed - F_C1a), the estate in performance units",
                                      "F_j_binds": "F_j > F_r"}),
    "loo_summary.csv": ("one quantity of loo.csv over the 10 seeds", {}),
    "paired.csv": ("one paired difference over the 10 seeds", {"positive": "seeds with a positive difference", "zero": "seeds with zero"}),
    "ranking.csv": ("one seed's ranking of the partners by l_i; then the modal ranking, the pairwise shares (share of seeds "
                    "with l_i > l_j) and the Friedman test (descriptive)", {}),
    "talmud.csv": ("one seed and pool: the contract's settlement rule", {
        "pool": "B: 10^12 (never binds), 1,000,003 (the proposal's, escrowed by MCI), rho * floor(E / 2) (binds by construction)",
        "E_prime": "min(E, floor(pool / rho))", "regime": "shortfall / surplus as talmud_reference.regime names it",
        "award_i_perf": "Talmud award in performance units", "award_i_tokens": "rho x award (token units)",
        "to_A_normal_path": "pool - sum of the awards: what MCI keeps"}),
    "payments.csv": ("one seed and member at the proposal's pool B: what MCI pays each partner, and what stays with MCI", {
        "member_index": "proposal index (0 = MCI)", "l": "the partner's l_i", "claim": "c_i = max(0, l_i)",
        "tokens": "the award in tokens (for MCI: the pool less the awards)", "regime": "the Talmud regime"}),
    "payments_summary.csv": ("one member over the 10 seeds", {}),
    "n64.csv": ("one seed: 1,000 protocol draws of 64 test rows (with replacement) from the C2 run", {
        "share_same_ranking_as_full_test": "draws whose l ranking equals the whole-test-split ranking (decision (b))",
        "share_same_top_passive": "draws with the same largest-l partner", "share_pairwise_order_agreeing": "mean share of the 3 pairs ordered alike",
        "share_F_j_binds": "draws with F_j > F_r", "l_i_full_test": "l_i on the whole test split",
        "l_i_mean_N64": "mean over the draws", "l_i_sd_N64": "SD over the draws", "l_i_share_le_0": "draws with l_i <= 0"}),
    "auc.csv": ("one seed, segment and model", {"segment": "all test rows, or thin_file", "auc": "ROC-AUC of the margin z_1 - z_0",
                                                "verified": "always off-chain, not ZK-verified"}),
    "auc_summary.csv": ("one segment and model over the 10 seeds", {}),
    "business.csv": ("one seed, segment and model; weighted to application_train's default rate", {
        "approval_rate_r": "MCI alone's (BASELINE's) weighted share predicted repaid: the fixed approval rate",
        "approved_share": "the model's weighted approved share at rate r (lowest margin first)",
        "default_rate_among_approved": "primary metric: weighted default rate among the approved",
        "relative_to_baseline": "(rate - MCI alone's rate) / MCI alone's rate",
        "d_star": "secondary metric's fixed default rate: MCI alone's primary value",
        "approved_share_at_d_star": "the longest run from the lowest margin with default rate <= d*",
        "extra_approval_share": "approved_share_at_d_star - MCI alone's, under the same rule",
        "extra_per_10000": "the same per 10,000 applicants", "approved_relative_to_baseline": "share / MCI alone's share - 1",
        "baseline_slack": "MCI alone's share at d* minus r (>= 0)", "verified": "always off-chain, not ZK-verified"}),
    "business_summary.csv": ("one segment, model and quantity of business.csv over the 10 seeds", {}),
    "tiers.csv": ("one seed, segment, model and tier (v2/policy.py); weighted to application_train's default rate", {
        "tier": "credit4m (PD < 4%), credit1m (4% <= PD < 12%), none (PD >= 12%), granted (both credit tiers), all",
        "share": "the weighted share of the segment's subscribers in the tier",
        "default_rate": "the weighted default rate in the tier", "mean_pd": "the weighted mean calibrated PD in the tier",
        "verified": "always off-chain, not ZK-verified"}),
    "tiers_summary.csv": ("one segment, model, tier and quantity of tiers.csv over the 10 seeds", {}),
    "calibration.csv": ("one seed, model and calibration fold: the Platt fit on one half of the test rows, applied to the other", {
        "fitted_on_fold": "the half the calibrator was fitted on", "applied_to_fold": "the half it was applied to (never the same)",
        "a": "intercept on the standardized margin", "b": "slope", "margin_mean": "the fit half's mean margin (standardization)",
        "margin_sd": "the fit half's margin SD", "fit_default_share": "s, the fit half's default share",
        "prior_shift": "logit(pi) - logit(s), the prior correction"}),
    "age_margin.csv": ("one seed, model (BASELINE, FULL) and age band of the test split", {
        "mean_margin_percentile": "mean percentile rank (0-100) of the margin z_1 - z_0 among the seed's test rows, in the band (higher = riskier)",
        "falls_with_every_band": "the seed's band means fall from each band to the next"}),
    "age_margin_summary.csv": ("one model and age band over the 10 seeds", {}),
    "age_tiers.csv": ("one seed, model (BASELINE, FULL), age band and tier", {
        "share_of_band": "the weighted share of the band's test rows in the tier", "rows_in_tier": "unweighted rows"}),
    "age_tiers_summary.csv": ("one model, age band and tier over the 10 seeds", {}),
    "ldp_runs.csv": ("one sigma-scale condition and seed (py/ldp_analysis.condition_tables)", {}),
    "ldp_summary.csv": ("one condition and quantity over the 10 seeds", {}),
    "ldp_paired.csv": ("one condition against LDP off (and LDP off against C2), paired by seed", {}),
    "ldp_n64_seeds.csv": ("one condition, seed and quantity over the 1,000 draws", {}),
    "ldp_n64.csv": ("one condition and quantity over the draws of all seeds", {}),
    "ldp_ranking.csv": ("one condition and seed: the ranking, and its agreement with LDP off", {}),
    "ldp_noise.csv": ("one sigma-scale condition and passive: clip, sigma and the per-release epsilon estimate (delta 1e-5)", {}),
    "ldp_eps_runs.csv": ("as ldp_runs.csv, for the epsilon curve", {}), "ldp_eps_summary.csv": ("as ldp_summary.csv", {}),
    "ldp_eps_paired.csv": ("as ldp_paired.csv", {}), "ldp_eps_n64_seeds.csv": ("as ldp_n64_seeds.csv", {}),
    "ldp_eps_n64.csv": ("as ldp_n64.csv", {}), "ldp_eps_ranking.csv": ("as ldp_ranking.csv", {}),
    "ldp_eps_levels.csv": ("one epsilon level and passive: the thesis's calibrated sigma (results/ldp/eps-levels.json, unchanged)", {
        "eps_total_gdp": "a training applicant's epsilon over its 21 releases (Gaussian DP)"}),
    "ldp_scores.csv": ("one LDP condition, seed and model: off-chain scores", {}),
    "ldp_scores_summary.csv": ("one condition, model and quantity over the 10 seeds", {}),
}
# the on-chain records (scenarios/mci_loan/v2/onchain_report2.py), documented when Step 5 has run
SCHEMA_ON = {
    "runs.csv": ("one on-chain run", {
        "final_state": "the Federation contract's final state", "settlement_path": "normal or backup",
        "figures_accepted": "figures accepted with a valid proof / figures required", "F_<figure>": "each accepted figure",
        "award_i_tokens": "partner i's Talmud award (tokens)", "to_A": "what stays with MCI", "settlement_check_ok":
        "py/m4_e3.py check: every quantity against the independent reference", "replay_equals_chain":
        "the six figures recomputed centrally (replay-full.mjs) equal the chain's", "tokens_conserved":
        "withdrawn + contract balance = deposits", "network_check_ok": "the genesis and validator check before the run"}),
    "gas.csv": ("one run and phase (py/m4_e3_ch5.py's phases)", {"in_run_total": "whether the phase counts in the run's total"}),
    "proofs.csv": ("one run and figure: its Groth16 proof", {"oracle_F": "the figure recomputed by the Python oracle",
                                                            "prove_ms": "rapidsnark's proving time", "c_model": "the model commitment"}),
    "payouts.csv": ("one run and party (A = MCI, p1-p3 the partners, the arbiter)", {
        "credited_tokens": "awards plus returned deposits", "withdrawn_tokens": "pulled by the party", "owed_at_end": "left in the contract"}),
    "transactions.csv": ("one transaction: the audit trail", {"tx_hash": "the chain's transaction hash", "phase": "py/m4_e3_ch5.py's phase"}),
    "events.csv": ("one settlement event the runner read from the chain", {}),
    "ranking.csv": ("one run: its partner ranking by l_i against off-chain seed k's whole-test-split ranking", {}),
    "cost_comparison.csv": ("one quantity: v2's five runs against v1's and the electricity scenario's five", {}),
}
COMMON = ("**Common terms.** F = floor(10⁴ × macro-F1) on the whole test split (3,193 rows), through the circuit's integer "
          f"path. FULL = F_r (the retrained FULL), FULL_J = F_j (the jointly trained fusion model), LOO_i = all but passive i "
          f"(1 {PASSIVE[1]}, 2 {PASSIVE[2]}, 3 {PASSIVE[3]}), BASELINE = C1a = U_A = MCI alone (every passive slot zero). "
          "Summary rows: n, mean, sd (sample, ddof 1), ci95_lo / ci95_hi (95 % t-interval), min, max; descriptive only, "
          "nothing is called significant.")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def rows(name):
    with (OUT / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def header(name):
    with (OUT / name).open(encoding="utf-8") as f:
        return next(csv.reader(f))


def f1(x, d=1):
    return f"{float(x):,.{d}f}"


def pct(x, d=2):
    return f"{100 * float(x):.{d}f}%"


def ms(s, d=1, scale=1.0, unit=""):
    return f"{scale * s['mean']:,.{d}f}{unit} (SD {scale * s['sd']:,.{d}f})"


def tier_cell(v) -> str:
    """'share (SD) / default rate' of one tier over the seeds; the default rate's seed count when the tier was empty in some."""
    if not v:
        return "–"
    dr = v.get("default_rate")
    rate = "–" if not dr else pct(dr["mean"], 1) + (f" (n {dr['n']})" if dr["n"] < 10 else "")
    return f"{pct(v['share']['mean'], 1)} (SD {pct(v['share']['sd'], 1)}) / {rate}"


def main() -> None:
    S = load("summary.json")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cm.ROOT, capture_output=True, text=True).stdout.strip()
    L = []
    w = L.append
    w("# MCI loan instance v2: off-chain results (Step 4, MCI as the active member)")
    w("")
    w(f"Generated by `scenarios/mci_loan/v2/results_doc2.py` from `{P}/` (written by `scenarios/mci_loan/v2/analysis2.py` from "
      f"the runs of `scenarios/mci_loan/v2/offchain/step4v2.mjs`); do not edit by hand. Generated on commit `{commit}`. "
      "The runs follow the pre-registration (`docs/mci/v2/02-preregistration.md`, tag `mci-prereg-v2`).")
    w("")
    w("**The organisations are simulated** (`docs/mci/v2/02-preregistration.md`, section 0). Every partner is a generic "
      "simulated role over one public dataset (Kaggle Home Credit Default Risk); no real company's data is used, MCI's "
      "included. Every number describes the role assignment over that dataset. AUC, business-metric, PD and tier numbers are "
      "**off-chain, not ZK-verified** (decision (a)); macro-F1 figures are what the circuit proves on-chain (Step 5).")
    w("")

    # ---- 1. runs and checks
    m = S["meta"]
    w("## 1. Runs and checks")
    w("")
    nr = len(rows("runs.csv"))
    w(f"- **Runs:** {nr} training and scoring processes (`{P}/runs.csv`), plus one N = 64 draw process per C2 and LDP run, one "
      f"at a time, on {S['machine']['cpu']} ({S['machine']['cores']} threads, {S['machine']['totalRamMB']:,} MB), Node "
      f"{S['machine']['node']}. Code commits: " +
      ", ".join(f"{k} `{v['commit']}` (clean: {'yes' if v['clean'] else 'no'})" for k, v in m.items()) + f" (`{P}/meta-*.json`).")
    w(f"- **No prediction in classes 2–4:** {sum(S['unusedClassTotal'].values())} rows over all runs and figures "
      f"(`{P}/summary.json`, `unusedClassTotal`; per run in `runs.csv`, `rows_in_classes_2_4`).")
    w(f"- **Split training equals centralized training bit for bit** (C2 against C4a-f32, losses and test predictions): "
      f"{'yes' if S['equivalence']['all_bitwise'] else 'NO'} in {S['equivalence']['seeds']} of {S['equivalence']['seeds']} seeds "
      f"(`{P}/equivalence.csv`).")
    ce = S["c1bEqualsV1"]
    if ce["all"] is None:  # delivery: the check needs v1's run files
        w(f"- **C1b against v1's C1b:** {ce['note']} (`{P}/summary.json`, `c1bEqualsV1`).")
    else: w(f"- **C1b equals v1's C1b** (MCI's 8 columns are v1's Bank's; C1b reads no passive data): "
      f"{'yes' if ce['all'] else 'NO'}, F, test predictions and margins, in "
      f"{sum(1 for x in ce['seeds'] if x['predsEqual'] and x['marginsEqual'])} of {len(ce['seeds'])} seeds "
      f"(`{P}/summary.json`, `c1bEqualsV1`; v1: `results/mci_loan/offchain/runs/c1b-s<s>.json`).")
    zf = sum(int(r["zero_filled"] or 0) for r in rows("runs.csv"))
    w(f"- **Zero-filled training batches:** {zf} (`{P}/runs.csv`).")
    if S.get("ldpIncluded"):
        w("- **Reproducibility checks:** C2 seed 1 rerun before the σ-scale set, and LDP-off seed 1 rerun before the ε set, "
          f"each equal to the first run in all six figures and their test predictions (`{P}/check-c2-s1.json`, "
          f"`{P}/check-eps-x0-s1.json`; `{P}/step4.log`).")
    w("")

    # ---- 2. model quality
    c = S["configs"]
    w("## 2. Model quality: macro-F1 × 10⁴ on the test split (mean over 10 seeds)")
    w("")
    w(f"Source: `{P}/configs_summary.csv` (per seed: `{P}/configs.csv`).")
    w("")
    w("| Configuration | Mean | SD | Min | Max |")
    w("|---|---:|---:|---:|---:|")
    for k, s in c.items():
        w(f"| {k} | {f1(s['mean'])} | {f1(s['sd'])} | {s['min']:,.0f} | {s['max']:,.0f} |")
    w("")
    a = S["auc"]
    w(f"ROC-AUC, off-chain, not ZK-verified (`{P}/auc_summary.csv`; per seed `{P}/auc.csv`):")
    w("")
    w("| Model | All test rows | thin_file |")
    w("|---|---:|---:|")
    for mdl, lab in MODEL_LABEL:
        x, y = a.get(f"all/{mdl}"), a.get(f"thin_file/{mdl}")
        w(f"| {lab} | {ms(x, 3) if x else '–'} | {ms(y, 3) if y else '–'} |")
    seg = S["segments"]
    w("")
    w(f"Test split: {seg['test_rows']:,} rows ({seg['test_defaults']:,} defaults); thin_file: {seg['thin_file_test_rows']:,} rows "
      f"({seg['thin_file_test_defaults']:,} defaults) (`{P}/summary.json`, `segments`).")
    w("")

    # ---- 3. contributions
    lo = S["loo"]
    w("## 3. What each partner adds: leave-one-out contributions and rank stability")
    w("")
    w(f"Source: `{P}/loo_summary.csv` (per seed `{P}/loo.csv`), `{P}/ranking.csv`, `{P}/paired.csv`.")
    w("")
    w("| Quantity | Mean | SD | 95 % CI | Min | Max |")
    w("|---|---:|---:|---|---:|---:|")
    for k in ("U_fed", "F_C1a", "F_C1b", "E", "l_1", "l_2", "l_3"):
        s = lo[k]
        lab = {"U_fed": "U_fed = max(F_r, F_j)", "F_C1a": "U_A: MCI alone (BASELINE)", "F_C1b": "C1b: MCI trained alone",
               "E": "E = U_fed − U_A"}.get(k, f"{k} ({NAMES.get(k, '')})")
        w(f"| {lab} | {f1(s['mean'])} | {f1(s['sd'])} | [{f1(s['ci95_lo'])}, {f1(s['ci95_hi'])}] | {s['min']:,.0f} | {s['max']:,.0f} |")
    w("")
    rk = S["ranking"]
    w(f"- **Modal ranking** (largest l first): {rk['modal']}, in {rk['count']} of {rk['n']} seeds. Per seed: "
      + "; ".join(f"s{s} {r}" for s, r in zip(S["seeds"], rk["rankings"])) + ".")
    w("- **Pairwise** (share of seeds): " + ", ".join(f"l_{k.replace('>', ' > l_')} {v:.1f}" for k, v in rk["pairwise"].items()) + ".")
    fr = rk["friedman"]
    w(f"- **Friedman** (descriptive, n = 10, k = 3): χ² = {fr['chi2']:.3f}, p = {fr['p']:.4f}.")
    w(f"- **F_j binds** (F_j > F_r) in {S['F_j_binds']} of 10 seeds (`{P}/loo.csv`).")
    w("")

    # ---- 4. Talmud: what MCI pays each partner
    t = S["talmud"]
    w("## 4. What MCI pays each partner: the Talmudic division (the contract's rule)")
    w("")
    w(f"Source: `{P}/talmud.csv`, `{P}/payments_summary.csv` (per seed `{P}/payments.csv`). Rule: E′ = min(E, ⌊B/ρ⌋), "
      "Talmud on the claims c_i in performance units, ρ = 5,000 (`py/data1_analysis.py` `talmud_rows_perf`, "
      "`py/talmud_reference.py`). MCI escrows the pool B when it creates the run; what the awards leave stays with MCI.")
    w("")
    w("Regimes (`talmud_reference.regime`, C = Σ c_i): surplus = E′ ≥ C (every claim paid); cea = E′ ≤ C/2 (equal awards "
      "on the half-claims); cel = in between (half-claims, then equal losses). Token amounts are means over the 10 seeds.")
    w("")
    w(f"| Pool | E′ (mean) | Regimes | {PASSIVE[1]} tokens | {PASSIVE[2]} tokens | {PASSIVE[3]} tokens | MCI keeps |")
    w("|---|---:|---|---:|---:|---:|---:|")
    for k, lab in (("B_proposal", "B = 1,000,003 (the proposal's)"), ("1e12", "10¹² (never binds)"), ("rho*floor(E/2)", "ρ⌊E/2⌋ (binds)")):
        v = t[k]
        w(f"| {lab} | {f1(v['E_prime']['mean'])} | {', '.join(f'{r} {n}' for r, n in v['regimes'].items())} | "
          + " | ".join(f"{x['mean']:,.0f}" for x in v["awards_tokens"]) + f" | {v['to_A']['mean']:,.0f} |")
    w("")
    py_ = S["payments"]
    w("At the proposal's pool, per member over the 10 seeds (`payments_summary.csv`): " +
      "; ".join(f"{mem} {ms(v, 0)} tokens (min {v['min']:,.0f}, max {v['max']:,.0f})" for mem, v in py_.items()) + ".")
    w("")

    # ---- 5. N = 64
    n6 = S["n64"]
    w("## 5. The 64-row evaluation against the whole test split (decision (b))")
    w("")
    w(f"Source: `{P}/n64.csv`: per seed, 1,000 draws of 64 test rows exactly as the protocol draws them.")
    w("")
    w(f"- Share of draws whose partner ranking equals the whole-test-split ranking: {ms(n6['same_ranking'], 3)} over the 10 seeds.")
    w(f"- Same top partner: {ms(n6['same_top'], 3)}; pairs ordered alike: {ms(n6['pairwise'], 3)}.")
    w("- Step 5 compares each on-chain run's ranking with off-chain seed k's whole-test-split ranking.")
    w("")

    # ---- 6. business metrics
    b = S["business"]
    w("## 6. Business metrics (off-chain, not ZK-verified)")
    w("")
    w(f"Source: `{P}/business_summary.csv` (per seed `{P}/business.csv`); weighted to application_train's default rate. "
      "Primary: approval rate fixed at MCI alone's, default rate among the approved. Secondary: default rate fixed at "
      "MCI alone's, extra approvals per 10,000 applicants.")
    w("")
    w("| Segment | Model | Approval rate r | Default rate among approved | Change vs MCI alone | Extra approvals per 10,000 |")
    w("|---|---|---:|---:|---:|---:|")
    for segm in ("all", "thin_file"):
        for mdl, lab in MODEL_LABEL:
            v = b.get(f"{segm}/{mdl}")
            if not v:
                continue
            rel = v.get("relative_to_baseline")
            w(f"| {segm} | {lab} | {pct(v['approval_rate_r']['mean'])} | {pct(v['default_rate_among_approved']['mean'])} "
              f"(SD {pct(v['default_rate_among_approved']['sd'])}) | {pct(rel['mean'], 1) if rel else '–'} | "
              f"{v['extra_per_10000']['mean']:,.0f} (SD {v['extra_per_10000']['sd']:,.0f}) |")
    w("")
    w(f"MCI alone's own slack under the secondary rule (its share at d* minus r), all rows: "
      f"{pct(b['all/BASELINE']['baseline_slack']['mean'])} (SD {pct(b['all/BASELINE']['baseline_slack']['sd'])}) "
      f"(`{P}/business_summary.csv`, `baseline_slack`).")
    w("")

    # ---- 7. credit tiers
    tr = S["tiers"]
    lo_, hi_ = S["tierRule"]["cuts"]
    w("## 7. Credit policy: the three tiers (off-chain, not ZK-verified)")
    w("")
    w(f"Source: `{P}/tiers_summary.csv` (per seed `{P}/tiers.csv`; the calibrations `{P}/calibration.csv`). Rule "
      f"(`scenarios/mci_loan/v2/policy.py`, pre-registration section 9): the PD is calibrated on one half of the test rows "
      f"and reported on the other, then the halves are swapped and pooled (seed {S['tierRule']['calibrationSeed']}), prior-"
      f"corrected to application_train's default rate. 4-month credit if PD < {pct(lo_, 0)}, 1-month credit if PD < "
      f"{pct(hi_, 0)}, no credit otherwise. Shares and default rates are weighted to the population; mean over 10 seeds (SD).")
    w("")
    for segm, slab in (("all", "All test rows"), ("thin_file", "thin_file (no bureau record)")):
        w(f"**{slab}**")
        w("")
        w("| Model | " + " | ".join(f"{T['name']}: share / default rate" for T in policy.TIERS) + " | Credit granted: share / default rate |")
        w("|---|" + "---:|" * (len(policy.TIERS) + 1))
        for mdl, lab in MODEL_LABEL:
            cells = [tier_cell(tr.get(f"{segm}/{mdl}/{key}")) for key in [T["key"] for T in policy.TIERS] + ["granted"]]
            w(f"| {lab} | " + " | ".join(cells) + " |")
        w("")
    ca = {k: tr[f"all/{k}/all"] for k in ("BASELINE", "FULL")}
    w("**The calibration check** (all test rows): weighted mean PD against the weighted default rate: " +
      "; ".join(f"{dict(MODEL_LABEL)[k]} {pct(v['mean_pd']['mean'])} against {pct(v['default_rate']['mean'])}" for k, v in ca.items()) +
      f" (`{P}/tiers_summary.csv`, tier `all`).")
    w("")
    w("A tier's default rate is a mean over the seeds in which the tier is not empty; its n is shown when that is fewer than "
      "10 seeds.")
    w("")

    # ---- 8. age check
    g = S["age"]
    w("## 8. The age check (pre-registration, section 5.5)")
    w("")
    w(f"Source: `{P}/age_margin_summary.csv` (per seed `{P}/age_margin.csv`): per age band of the test split, the mean "
      "percentile rank (0–100) of the margin z₁ − z₀ among the seed's test rows; higher = scored riskier. Mean over 10 seeds:")
    w("")
    ab = S["age_bands"]
    bands = list(ab["BASELINE"])
    w("| Model | " + " | ".join(bands) + " | Seeds falling with every band |")
    w("|---|" + "---:|" * (len(bands) + 1))
    for mdl, v in g.items():
        w(f"| {dict(MODEL_LABEL)[mdl]} | " + " | ".join(f1(ab[mdl][b_]['mean']) for b_ in bands) + f" | {v['seeds_falling_with_every_band']} of 10 |")
    w("")
    w("Oldest band minus youngest band (percentile points): " + "; ".join(f"{dict(MODEL_LABEL)[mdl]} {ms(v['oldest_minus_youngest_band'], 1)}" for mdl, v in g.items())
      + f" (`{P}/summary.json`, `age`).")
    w("")
    at = S["age_tiers"]
    w(f"Each band's weighted share in each tier, mean over 10 seeds (`{P}/age_tiers_summary.csv`; per seed `{P}/age_tiers.csv`):")
    w("")
    w("| Model | Tier | " + " | ".join(bands) + " |")
    w("|---|---|" + "---:|" * len(bands))
    for mdl in ("BASELINE", "FULL"):
        for T in policy.TIERS:
            w(f"| {dict(MODEL_LABEL)[mdl]} | {T['name']} | " + " | ".join(pct(at[f"{mdl}/{b_}/{T['key']}"]["mean"], 1) for b_ in bands) + " |")
    w("")

    # ---- 9. LDP
    if S.get("ldpIncluded"):
        ld = S["ldp"]
        w("## 9. LDP: the noise-scale sweep and the privacy–utility curve")
        w("")
        w(f"Sources: `{P}/ldp_summary.csv`, `{P}/ldp_eps_summary.csv`, `{P}/ldp_ranking.csv`, `{P}/ldp_eps_ranking.csv`, "
          f"`{P}/ldp_eps_levels.csv`, `{P}/ldp_scores_summary.csv`. No fixed-budget configurations (the student's v1 review, change 3).")
        w("")
        w(f"| Condition | U_fed | U_A (MCI alone) | l_1 ({PASSIVE[1]}) | l_2 ({PASSIVE[2]}) | l_3 ({PASSIVE[3]}) | Seeds ranked as LDP off |")
        w("|---|---:|---:|---:|---:|---:|---:|")
        for set_ in ("scales", "eps"):
            src = ld[set_]
            for cond, q in src["full_test"].items():
                same = src["ranking"][cond]["full_test_same_as_off"]
                w(f"| {cond} | {f1(q['U_fed']['mean'])} | {f1(q['BASELINE']['mean'])} | {f1(q['l_1']['mean'])} | "
                  f"{f1(q['l_2']['mean'])} | {f1(q['l_3']['mean'])} | {same} of 10 |")
        w("")
        w("Conditions: x1 = LDP on (protocol σ: 0.05 / 0.1 / 0.05 at clips 4 / 4 / 3, for passives 1 / 2 / 3), x0 = LDP off "
          "(clipping only), x0.5 and x4 the weaker and stronger noise; e1 … e64 = σ calibrated to ε = 1 … 64 per release "
          "(`ldp_eps_levels.csv`). U_A is BASELINE retrained in the same noisy federation, so it moves with the condition too.")
        w("")
        nz = [r for r in rows("ldp_noise.csv") if r["sigma_scale"] == "1.0000"]
        w("- **The protocol's own σ** (x1) is an estimate of ε per release of " +
          ", ".join(f"{float(r['eps_per_release_analytic']):,.0f} ({r['passive'][2:]})" for r in nz) +
          f" (δ = 10⁻⁵ per release, `{P}/ldp_noise.csv`).")
        ut = ld["eps"]["utility"]
        w("- **Utility on the ε curve** (U_fed − U_A, mean over 10 seeds, and the seeds where the federation beats MCI alone; "
          f"`{P}/summary.json`, `ldp.eps.utility`): " +
          "; ".join(f"{'LDP off' if k == 'off' else 'ε = ' + k[1:]} {v['mean']:+,.1f} ({v['seeds_above']} of 10)" for k, v in ut.items()) + ".")
        sc = {(r["condition"], r["model"], r["quantity"]): r for r in rows("ldp_scores_summary.csv")}
        conds = list(dict.fromkeys(r["condition"] for r in rows("ldp_scores_summary.csv")))
        w(f"- **Off-chain scores per condition** (FULL; off-chain, not ZK-verified; `{P}/ldp_scores_summary.csv`): " +
          "; ".join(f"{c_}: AUC {float(sc[(c_, 'FULL', 'auc')]['mean']):.3f}, default rate among approved "
                    f"{pct(sc[(c_, 'FULL', 'default_rate_among_approved')]['mean'])}" for c_ in conds) + ".")
        w("")

    # ---- 10. on-chain
    if (ON / "summary.json").exists():
        O = json.loads((ON / "summary.json").read_text(encoding="utf-8"))
        done = O["runsDone"]
        w("## 10. The on-chain runs (Step 5)")
        w("")
        w(f"Source: `{PO}/runs.csv`, `{PO}/summary.json` (`scenarios/mci_loan/v2/onchain_report2.py`, from the run files under "
          f"`{PO}/run-<k>/`). Five runs on the 4-node Besu QBFT network on the MacBook Air, as planned before run 1 (`{PO}/plan.json`): "
          f"model config `loan2-s<k>-wait10s.json`, run seed k, the maxDepth 14 multi-party key, rapidsnark. The alignment "
          f"period: {O['plan']['runs'][0]['alignmentPeriodBlocks']} blocks, at least twice the slowest of the benchmark's "
          f"alignments ({O['alignmentBenchmark']['slowestAlignmentS']:.1f} s, `{PO}/psi-bench.json`).")
        w("")
        w(f"| Run | State, path | Figures accepted | U_fed | U_A (MCI alone) | E | E′ | {PASSIVE[1]} | {PASSIVE[2]} | {PASSIVE[3]} | MCI keeps | Check, replay, conserved |")
        w("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for k in done:
            r = O["runs"][str(k)]
            d = r["division"]
            w(f"| {k} | {r['finalState']}, {r['path']} | {'all' if r['accepted'] else 'NOT ALL'} | {d['U_fed']:,} | {r['figures']['BASELINE']:,} | "
              f"{d['E']:,} | {d['E_prime']:,} | " + " | ".join(f"{t:,}" for t in d["award_tokens"]) + f" | {d['to_A']:,} | "
              f"{'yes' if r['checkOk'] else 'NO'}, {'yes' if r['replayEqualsChain'] else 'NO'}, {'yes' if r['conservation']['conserved'] else 'NO'} |")
        w("")
        same = sum(1 for k in done if O["runs"][str(k)]["sameRankingAsOffchainSeed"])
        top = sum(1 for k in done if O["runs"][str(k)]["sameTopAsOffchainSeed"])
        w(f"- **Every figure accepted with its proof:** {'yes' if O['allAccepted'] else 'NO'}; settlement checks: "
          f"{'all ok' if O['allChecksOk'] else 'NOT ALL'}; replays equal to the chain: {'all' if O['allReplaysEqual'] else 'NOT ALL'}; "
          f"tokens conserved: {'all' if O['allConserved'] else 'NOT ALL'} (`{PO}/summary.json`).")
        w(f"- **Ranking against off-chain seed k** (`{PO}/ranking.csv`): the same partner ranking in {same} of {len(done)} runs, the "
          f"same top partner in {top} of {len(done)}. Each run scores 64 evaluation rows (section 5).")
        cc = {r["quantity"]: r for r in csv.DictReader((ON / "cost_comparison.csv").open(encoding="utf-8"))}
        g, pt = cc["gas: total"], cc["proof time per figure (ms, rapidsnark)"]
        w(f"- **Cost** (`{PO}/cost_comparison.csv`): gas per run {float(g['loan_v2_mean']):,.0f} (v1 {float(g['loan_v1_mean']):,.0f}, "
          f"electricity {float(g['electricity_mean']):,.0f}); proof time per figure {float(pt['loan_v2_mean']):,.1f} ms (v1 "
          f"{float(pt['loan_v1_mean']):,.1f}, electricity {float(pt['electricity_mean']):,.1f}).")
        w("")

    # ---- schema
    w("## Schema of every results file")
    w("")
    w(COMMON)
    w("")
    for name, (what, cols) in SCHEMA.items():
        if not (OUT / name).exists():
            continue
        hd = header(name)
        w(f"### `{P}/{name}`")
        w("")
        w(f"One row: {what}. Columns: " + ", ".join(f"`{h}`" for h in hd) + ".")
        if cols:
            w("")
            for k, v in cols.items():
                w(f"- `{k}`: {v}")
        w("")
    if (ON / "summary.json").exists():
        for name, (what, cols) in SCHEMA_ON.items():
            hd = next(csv.reader((ON / name).open(encoding="utf-8")))
            w(f"### `{PO}/{name}`")
            w("")
            w(f"One row: {what}. Columns: " + ", ".join(f"`{h}`" for h in hd) + ".")
            if cols:
                w("")
                for k, v in cols.items():
                    w(f"- `{k}`: {v}")
            w("")
        w(f"### `{PO}/summary.json`, `plan.json`, `psi-bench.json`, `run-<k>/`")
        w("")
        w("`summary.json`: every run's record (state, path, figures, division, gas by phase, proof times, conservation, "
          "machine, chain, keys), the cost comparison and the four all-runs checks. `plan.json`: the five runs fixed before run 1. "
          "`psi-bench.json`: the alignment benchmark on the Mac. `run-<k>/`: the runner's files (run.json, each process's "
          "summary and log, sample.json, replay.json, check.json, mem.log) and `run-<k>-network-check.json`.")
        w("")
    w(f"### `{P}/summary.json`")
    w("")
    w("Every summary above as JSON: `names`, `meta` (code commit per set), `c1bEqualsV1`, `configs`, `equivalence`, `loo`, "
      "`paired`, `ranking`, `talmud`, `payments`, `n64`, `auc`, `business`, `tiers`, `tierRule`, `segments`, `age`, "
      "`age_bands`, `age_tiers`, `ldp` (the thesis's condition_tables summaries), `unusedClassTotal`, `resources`, `machine`.")
    w("")
    w(f"### `{P}/meta-<set>.json`, `check-*.json`, `step4.log`")
    w("")
    w("`meta-<set>.json` (main, scales, eps): the code commit the set ran on and whether the code tree was clean, its seeds, "
      "the prepared folder and start time (step4v2.mjs). `check-c2-s1.json`, `check-eps-x0-s1.json`: the reproducibility "
      "reruns, in the run format. `step4.log`: every process's start, free RAM, exit code and duration.")
    w("")
    w(f"### `{P}/runs/`")
    w("")
    w("The raw run files: `c2-s<s>.json` (vfl-run.mjs: every figure's F, test predictions, margins, classes 2–4 counts, "
      "training record), `c4a32-s<s>.json`, `c1b-s<s>.json` (central-run.mjs), `n64-*.json` (n64.mjs: every draw's six F "
      "values), `ldp-x<k>-s<s>.json`, `eps-e<ε>-s<s>.json`. Format as v1's `results/mci_loan/offchain/runs/`.")
    w("")
    text = "\n".join(L)
    bad = sorted(set(FORBIDDEN.findall(text)))
    if bad:
        sys.exit(f"the naming rule: the document names {bad}")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"doc": str(DOC), "lines": len(L)}))


if __name__ == "__main__":
    main()
