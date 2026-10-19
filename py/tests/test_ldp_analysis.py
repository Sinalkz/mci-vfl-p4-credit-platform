"""Tests for py/ldp_analysis.py (LDP with vs without)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data1_analysis as da  # noqa: E402
import ldp_analysis as la  # noqa: E402


def run(F, seed=1, scale=1.0, noise=None):
    return {"seed": seed, "configHash": "0xc", "rows": {"testPosSha256": "ab"},
            "noise": noise or {"sigmaScale": scale, "perPassive": [{"clip": 4, "sigma": 0.05 * scale}] * 3,
                               "noiseSeeds": [1, 2, 3]},
            "figures": [{"name": k, "F": v} for k, v in F.items()]}


def test_run_row_takes_u_fed_as_the_larger_full_and_l_from_it():
    r = la.run_row(run({"FULL": 100, "FULL_J": 120, "LOO_1": 110, "LOO_2": 130, "LOO_3": 90, "BASELINE": 70}))
    assert (r["F_r"], r["F_j"], r["U_fed"], r["F_j_binds"]) == (100, 120, 120, True)
    assert (r["l_1"], r["l_2"], r["l_3"], r["E"]) == (10, -10, 30, 50)
    assert r["ranking"] == (3, 1, 2)


def test_run_row_floors_e_at_zero():
    r = la.run_row(run({"FULL": 100, "FULL_J": 90, "LOO_1": 100, "LOO_2": 100, "LOO_3": 100, "BASELINE": 140}))
    assert r["U_fed"] == 100 and r["E"] == 0 and not r["F_j_binds"]


def test_draw_rows_apply_the_contract_rule_on_each_draw():
    n64 = {"draws": [{"j": 1, "F": {"FULL": 50, "FULL_J": 60, "LOO_1": 40, "LOO_2": 70, "LOO_3": 55, "BASELINE": 65}},
                     {"j": 2, "F": {"FULL": 80, "FULL_J": 20, "LOO_1": 81, "LOO_2": 60, "LOO_3": 70, "BASELINE": 30}}]}
    d = la.draw_rows(n64)
    assert [x["U_fed"] for x in d] == [60, 80]
    assert [(x["l_1"], x["l_2"], x["l_3"]) for x in d] == [(20, -10, 5), (-1, 20, 10)]
    assert [x["E"] for x in d] == [0, 50]
    assert [x["ranking"] for x in d] == [(1, 3, 2), (2, 3, 1)]


def test_seed_n64_summarises_over_draws():
    draws = [{"U_fed": u, "l_1": a, "l_2": 0, "l_3": 1, "E": e, "ranking": (3, 1, 2)}
             for u, a, e in [(10, -1, 0), (20, 1, 5), (30, 3, 10)]]
    s = la.seed_n64(draws)
    assert s["l_1"]["mean"] == 1 and s["l_1"]["sd"] == pytest.approx(2.0) and s["l_1"]["share_le_0"] == pytest.approx(1 / 3)
    assert s["l_2"]["share_le_0"] == 1 and s["E"]["share_le_0"] == pytest.approx(1 / 3)
    assert "share_le_0" not in s["U_fed"]


def test_same_ranking_share_pairs_draws_by_index():
    on = [(1, 2, 3), (3, 2, 1), (3, 2, 1), (2, 1, 3)]
    off = [(1, 2, 3), (3, 2, 1), (1, 2, 3), (2, 1, 3)]
    assert la.same_ranking_share(on, off) == 0.75
    with pytest.raises(ValueError):
        la.same_ranking_share(on, off[:3])


def test_noise_rows_give_no_epsilon_without_noise_and_the_gaussian_estimate_with_it():
    off = la.noise_rows(run({}, scale=0.0))
    assert all(r["sigma"] == 0 and r["eps_per_release_analytic"] is None for r in off)
    on = la.noise_rows(run({}, scale=1.0))
    want = da.dp_epsilon(0.05, 4)
    assert on[0]["eps_per_release_analytic"] == pytest.approx(want["eps_analytic"])
    assert on[0]["eps_total"] == pytest.approx(want["eps_total_simple"])
    assert [r["passive"] for r in on] == [1, 2, 3]


def test_check_pairing_rejects_a_run_from_another_config_or_noise_draw():
    a = run({}, scale=0.0)
    la.check_pairing([a, run({}, scale=1.0)])  # same config, test rows and noise seeds: fine
    b = run({}, scale=1.0)
    b["configHash"] = "0xd"
    with pytest.raises(ValueError):
        la.check_pairing([a, b])
    c = run({}, scale=1.0)
    c["noise"]["noiseSeeds"] = [4, 5, 6]
    with pytest.raises(ValueError):
        la.check_pairing([a, c])


# ---- formal epsilon: calibration and composition ----------------------------------------------------

@pytest.mark.parametrize("eps", [1, 4, 16, 64])
def test_calibrate_sigma_gives_the_target_per_release_epsilon(eps):
    delta = 1 / 12772
    s = la.calibrate_sigma(eps, 4, delta)
    assert da.dp_epsilon(s, 4, delta)["eps_analytic"] == pytest.approx(eps, rel=1e-6)


def test_calibrated_sigma_scales_with_the_clip_bound():
    assert la.calibrate_sigma(16, 3, 1e-4) / la.calibrate_sigma(16, 4, 1e-4) == pytest.approx(0.75, rel=1e-9)


def test_gdp_composition_of_k_releases():
    from scipy.stats import norm
    import math
    sigma, clip, k, delta = 2.0, 4.0, 21, 1 / 12772
    per = la.composed_eps(sigma, clip, 1, delta)
    assert per == pytest.approx(da.dp_epsilon(sigma, clip, delta)["eps_analytic"])
    tot = la.composed_eps(sigma, clip, k, delta)
    mu = math.sqrt(k) * 2 * clip / sigma  # Dong, Roth & Su (2019): k Gaussian releases are sqrt(k) * (sensitivity / sigma)-GDP
    d = norm.cdf(-tot / mu + mu / 2) - math.exp(tot) * norm.cdf(-tot / mu - mu / 2)
    assert d == pytest.approx(delta, rel=1e-6)
    assert per < tot < k * per  # tighter than simple composition


def test_eps_levels_calibrate_each_passive_to_its_own_clip():
    lv = la.eps_levels([1, 64], clips=[4, 4, 3], delta=1 / 12772, releases=21)
    assert [x["eps"] for x in lv] == [1, 64]
    one = lv[0]["passives"]
    assert [p["clip"] for p in one] == [4, 4, 3]
    assert one[2]["sigma"] == pytest.approx(0.75 * one[0]["sigma"])
    assert all(p["eps_check"] == pytest.approx(1, rel=1e-6) for p in one)
    assert lv[0]["eps_total_gdp"] == pytest.approx(la.composed_eps(one[0]["sigma"], 4, 21, 1 / 12772))
    assert lv[0]["eps_total_simple"] == 21 and lv[0]["delta_total_simple"] == pytest.approx(21 / 12772)


def test_a_smaller_total_delta_gives_a_larger_total_epsilon():
    s, c = la.calibrate_sigma(16, 4, 1 / 12772), 4
    at_1_over_n, at_1e6 = la.composed_eps(s, c, 21, 1 / 12772), la.composed_eps(s, c, 21, la.DELTA_CONVENTIONAL)
    assert la.DELTA_CONVENTIONAL == 1e-6 and at_1e6 > at_1_over_n
    assert la.composed_eps(s, c, 63, la.DELTA_CONVENTIONAL) > at_1e6  # three passives' releases



# ---- the seed set: seeds 1-5 give the first tables (now *-5seeds), seeds 1-10 the main ones ---------------------------
import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import ch5_stats as cs  # noqa: E402

SEED_TABLES = [f"{p}_{t}" for p in ("ldp", "ldp_eps") for t in ("runs", "summary", "paired", "n64_seeds", "n64", "ranking")] + \
    ["ldp_eps_utility", "ldp_eps_saturation"]
SEED_FREE = ["ldp_noise.csv", "ldp_eps_levels.csv"]  # the noise settings and sigma per epsilon: no seed in them


def stored(path):
    """A file's bytes as git stores them (the repo normalizes line ends to LF; Windows writes CRLF)."""
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def table(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


@pytest.fixture(scope="module")
def seed_sets(tmp_path_factory):
    five, ten = tmp_path_factory.mktemp("ldp-5seeds"), tmp_path_factory.mktemp("ldp-10seeds")
    la.analyse(la.SEEDS_5, out=five)
    la.analyse(la.SEEDS_10, out=ten)
    return five, ten


def test_the_seed_sets():
    assert la.SEEDS_5 == [1, 2, 3, 4, 5] and la.SEEDS_10 == list(range(1, 11))


def test_seeds_1_to_5_reproduce_the_five_seed_tables_byte_for_byte(seed_sets):
    five, _ = seed_sets
    written = sorted(x.name for x in five.iterdir())
    assert written == sorted([f"{t}-5seeds.csv" for t in SEED_TABLES] + ["summary-5seeds.json", *SEED_FREE])
    for name in written:
        assert stored(five / name) == stored(la.OUT / name), name


def test_seeds_1_to_10_write_the_main_tables(seed_sets):
    five, ten = seed_sets
    written = sorted(x.name for x in ten.iterdir())
    assert written == sorted([f"{t}.csv" for t in SEED_TABLES] + ["summary.json", *SEED_FREE])
    for name in written:
        assert stored(ten / name) == stored(la.OUT / name), name
    for name in SEED_FREE:
        assert stored(five / name) == stored(ten / name), name


def test_the_ten_seed_tables_extend_the_five_seed_ones(seed_sets):
    five, ten = seed_sets
    r5, r10 = table(five / "ldp_eps_runs-5seeds.csv"), table(ten / "ldp_eps_runs.csv")
    assert [r for r in r10 if int(r["seed"]) <= 5] == r5 and {r["seed"] for r in r10} == {str(s) for s in range(1, 11)}
    assert all(r["n"] == "10" for r in table(ten / "ldp_summary.csv") + table(ten / "ldp_eps_summary.csv"))
    s10 = json.loads((ten / "summary.json").read_text(encoding="utf-8"))
    assert s10["seeds"] == la.SEEDS_10 and s10["eps"]["seeds"] == la.SEEDS_10
    assert sorted(s10["meta_other_seeds"]) == ["meta-s6-10.json"] and sorted(s10["eps"]["meta_other_seeds"]) == ["meta-eps-s6-10.json"]
    s5 = json.loads((five / "summary-5seeds.json").read_text(encoding="utf-8"))
    assert "seeds" not in s5 and "meta_other_seeds" not in s5    # the first version's keys only


def test_the_ten_seed_rows_agree_with_ch5_stats(seed_sets):
    _, ten = seed_sets
    want = {r["id"]: r for r in cs.rows(seeds=cs.SEEDS)}
    paired = {(r["comparison"], r["quantity"]): r for r in table(ten / "ldp_paired.csv")}
    for rid, comp in [("clipping_cost", "LDP off (clipping only) − E1 C2 (no clipping)"),
                      ("noise_cost", "LDP on (configured σ) − LDP off")]:
        got = paired[(comp, "U_fed")]
        assert (float(got["mean"]), float(got["sd"])) == (pytest.approx(want[rid]["mean"], abs=1e-3),
                                                          pytest.approx(want[rid]["sd"], abs=1e-3)), rid
    utility = {r["eps_per_release"]: r for r in table(ten / "ldp_eps_utility.csv")}
    for eps in (1, 4, 16, 64):
        got, w = utility[str(eps)], want[f"ldp_eps{eps}"]
        assert (float(got["FULL_minus_BASELINE_mean"]), float(got["sd"])) == (pytest.approx(w["mean"], abs=1e-3),
                                                                             pytest.approx(w["sd"], abs=1e-3)), eps
        assert int(got["seeds_FULL_above_BASELINE"]) == w["seeds_expected_sign"], eps


def test_the_report_is_written_only_when_asked_from_the_five_seeds(tmp_path):
    with pytest.raises(ValueError):
        la.analyse(la.SEEDS_10, out=tmp_path / "t10", report=tmp_path / "R10.md")
    assert not (tmp_path / "R10.md").exists()
    la.analyse(la.SEEDS_5, out=tmp_path / "t5", report=tmp_path / "R.md")
    new = (tmp_path / "R.md").read_text(encoding="utf-8").splitlines()
    old = (la.ROOT / "RESULTS-LDP.md").read_text(encoding="utf-8").splitlines()
    # the committed milestone report is the five-seed one as this code writes it: only the commit it was generated at
    # may differ (a line of its own, "abc1234.")
    assert len(new) == len(old)
    differ = [(a, b) for a, b in zip(old, new) if a != b]
    assert all(re.fullmatch("[0-9a-f]{7}[.]", a) and re.fullmatch("[0-9a-f]{7}[.]", b) for a, b in differ), differ
    text = "\n".join(new)
    assert "**Seeds.**" in text and "`ldp_summary-5seeds.csv`" in text and "`ldp_summary.csv`" in text  # note names both
    assert not any(ln.startswith("`ldp_summary.csv`") for ln in new)                                   # tables: 5 seeds


def test_the_report_tables_round_halves_up():
    table = (["a", "b", "c"], [[1234.25, 0.2125, float("inf")]])
    assert la.md(table, {"b": "{:.3f}"}).splitlines()[2] == "| 1,234.3 | 0.213 | inf |"
