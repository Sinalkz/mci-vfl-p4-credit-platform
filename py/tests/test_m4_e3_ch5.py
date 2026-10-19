"""Chapter 5 export of M4's E3 run (py/m4_e3_ch5.py): percentiles, the gas phases and the division."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import m4_e3_ch5 as ch5  # noqa: E402

RUN = Path(__file__).resolve().parents[2] / "results" / "m4" / "e3" / "run.json"


def test_percentile_is_the_midrank_share_below():
    assert ch5.percentile([1, 2, 3, 4], 0) == 0.0
    assert ch5.percentile([1, 2, 3, 4], 5) == 100.0
    assert ch5.percentile([1, 2, 3, 4], 3) == 62.5          # 2 below, 1 tie: (2 + 0.5) / 4
    assert ch5.percentile([2, 2, 2, 2], 2) == 50.0


def test_the_rank_against_e1s_draws_rounds_halves_up():
    r = ch5.rank_in_draws([0, 0, 0, 1], 1)
    assert (r["value"], r["percentile"], r["E1_N64_mean"], r["E1_N64_sd"]) == (1, 87.5, 0.3, 0.5)  # mean 0.25 -> 0.3


def test_the_proof_measures_round_halves_up():
    m = ch5.proof_measures({"proveMs": 2.5, "proverPeakRssMB": 4568.5, "peakRssMB": 0.5, "verifyMs": 0.25})
    assert m == {"prove_ms": 3, "prover_peak_rss_mb": 4569, "node_peak_rss_mb": 1, "local_verify_ms": 0.3}
    assert ch5.proof_measures({"proveMs": 1, "peakRssMB": 1, "verifyMs": 1})["prover_peak_rss_mb"] is None


def test_every_step_has_a_phase_and_an_unknown_one_stops():
    assert ch5.phase_of("submitJointScore FULL_J") == "scoring"
    assert ch5.phase_of("revealSeed A") == "setup"
    assert ch5.phase_of("commitAlignment") == "alignment"
    assert ch5.phase_of("mint Dp") == "test token"
    assert ch5.phase_of("finalizeReconciliation") == "training"
    assert ch5.phase_of("deploy FederationFactory") == "infrastructure"
    with pytest.raises(SystemExit):
        ch5.phase_of("selfdestruct")


def test_n64_series_derive_u_fed_l_and_e_per_draw():
    draws = [{"F": {"FULL": 10, "FULL_J": 12, "LOO_1": 11, "LOO_2": 13, "LOO_3": 5, "BASELINE": 20}}]
    s = ch5.n64_series(draws)
    assert (s["F_r"], s["F_j"], s["U_fed"]) == ([10], [12], [12])
    assert (s["l_1"], s["l_2"], s["l_3"]) == ([1], [-1], [7])
    assert s["E"] == [0]                                   # max(0, U_fed - BASELINE)


@pytest.mark.skipif(not RUN.exists(), reason="results/m4/e3/run.json not present")
def test_the_committed_run_divides_and_sums_as_recorded():
    run = json.loads(RUN.read_text(encoding="utf-8"))
    d = ch5.division(run)
    assert (d["U_fed"], d["E"], d["E_prime"]) == (2082, 605, 200)
    assert d["l"] == [-45, -191, 123] and d["claims"] == [0, 0, 123] and d["awards"] == [0, 0, 123]
    assert d["award_tokens"] == [0, 0, 615000] and d["to_A"] == 385003
    g = ch5.gas(run)
    assert sum(t["gasUsed"] for t in g["transactions"]) == g["total_all_transactions"]
    assert g["total"] == sum(g["phases"][p]["gas"] for p in ch5.PHASES)
    assert {p: (g["phases"][p]["count"], g["phases"][p]["gas"]) for p in g["phases"]} == {
        "setup": (13, 6626572), "alignment": (1, 120500), "training": (5, 409639), "scoring": (6, 2445454),
        "settlement": (7, 596439), "test token": (8, 407452), "infrastructure": (3, 9234668)}
    assert g["total"] == 10198604


def test_a_rerun_config_may_differ_from_e1s_only_in_the_wait():
    a = {"T": 20, "perBatchWaitMs": 2000, "initSeed": 1}
    assert ch5.differs_only_in(a, {**a, "perBatchWaitMs": 10000}, "perBatchWaitMs")
    assert not ch5.differs_only_in(a, {**a, "perBatchWaitMs": 10000, "initSeed": 2}, "perBatchWaitMs")
    assert not ch5.differs_only_in(a, dict(a), "perBatchWaitMs")       # no change at all is not a rerun config
    assert not ch5.differs_only_in(a, {"T": 20, "initSeed": 1}, "perBatchWaitMs")


def test_model_config_option_needs_the_e1_seed():
    with pytest.raises(SystemExit):
        ch5.main(["--run-dir", "results/full/runs/run-1", "--model-config", "results/full/configs/e1-s1-wait10s.json"])
