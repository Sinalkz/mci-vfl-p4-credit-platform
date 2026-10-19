"""M4's E3 run: the chain's settlement record against the independent reference (py/m4_e3.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import m4_e3  # noqa: E402


def e3_like():
    """A run.json shaped as runner3 writes it, with DATA1's E3 figures, settled on the normal path."""
    fig = lambda kind, j, F: {"kind": kind, "memberIndex": j, "accepted": True, "F": F}  # noqa: E731
    return {
        "run": "e3", "finalState": "Settled", "exitCodes": {"A": 0, "p1": 0, "p2": 0, "p3": 0, "arbiter": 0},
        "machine": {"host": "MacBook Air M4"},
        "chain": {"clientVersion": "besu/v26.8.1/osx-aarch_64/openjdk-java-21", "genesisBlockHash": "0x" + "ab" * 32,
                  "genesisFileSha256": "cd" * 32, "launch": "native"},
        "chainEnd": {
            "proposal": {"B": "1000003", "rho": "5000", "Dp": "1001", "Dc": "501", "K": 3, "deltaA": 30},
            "uFed": 3364,
            "figures": [fig(0, 0, 3364), fig(3, 0, 2998), fig(1, 1, 3365), fig(1, 2, 3571), fig(1, 3, 3005), fig(2, 0, 2096)],
            "members": [{"left": False, "equivocated": False, "objection": 0} for _ in range(3)],
            "owed": {"A": "0", "p1": "0", "p2": "0", "p3": "0", "arbiter": "0"},
            "settlement": {
                "path": "normal",
                "settled": {"block": 100, "path": 1, "E": 1268, "estate": 200, "awards": [0, 0, 200]},
                "acknowledged": [{"block": 98, "passive": "p1"}, {"block": 99, "passive": "p2"}, {"block": 100, "passive": "p3"}],
                "objections": [],
                "credited": {"A": "3", "p1": "1001", "p2": "1001", "p3": "1001001"},
                "withdrawn": [{"block": 101, "to": "A", "amount": "3"}, {"block": 101, "to": "p1", "amount": "1001"},
                              {"block": 102, "to": "p2", "amount": "1001"}, {"block": 102, "to": "p3", "amount": "1001001"}],
                "lockedForever": "0", "contractBalance": "0",
            },
        },
    }


def test_e3_normal_path_matches_the_reference_and_conserves_tokens():
    r = m4_e3.check(e3_like())
    assert r["ok"], r["problems"]
    assert r["path"] == "normal"
    assert r["expected"] == {"A": 3, "p1": 1001, "p2": 1001, "p3": 1_001_001, "arbiter": 0}
    assert r["deposits"] == 1_000_003 + 3 * 1001
    assert r["division"] == {"E": 1268, "Eprime": 200, "estate": 200, "l": [0, 0, 359], "awards": [0, 0, 200]}


def test_a_credit_off_by_one_is_reported():
    run = e3_like()
    run["chainEnd"]["settlement"]["credited"]["p3"] = "1001000"
    r = m4_e3.check(run)
    assert not r["ok"] and any("p3" in p and "credited" in p for p in r["problems"])


def test_a_withdrawal_that_differs_from_the_credit_is_reported():
    run = e3_like()
    run["chainEnd"]["settlement"]["withdrawn"].pop()
    r = m4_e3.check(run)
    assert not r["ok"] and any("p3" in p and "withdrew" in p for p in r["problems"])


def test_tokens_left_in_the_contract_beyond_the_locked_amount_are_reported():
    run = e3_like()
    run["chainEnd"]["settlement"]["contractBalance"] = "5"
    r = m4_e3.check(run)
    assert not r["ok"] and any("conservation" in p for p in r["problems"])


def test_a_settled_event_that_disagrees_with_the_reference_is_reported():
    run = e3_like()
    run["chainEnd"]["settlement"]["settled"]["awards"] = [0, 1, 199]
    r = m4_e3.check(run)
    assert not r["ok"] and any("awards" in p for p in r["problems"])


def test_u_fed_must_be_the_larger_of_f_r_and_f_j():
    run = e3_like()
    run["chainEnd"]["uFed"] = 2998
    r = m4_e3.check(run)
    assert not r["ok"] and any("uFed" in p for p in r["problems"])


def test_e3_run1_is_rechecked_under_the_backup_rule_it_settled_by():
    # E3's first run settled by backup under M4 Q6 (superseded by D142): checked with that rule, it passes; checked with
    # D142 it must not, because the chain applied the old rule
    import json
    run1 = json.loads((Path(__file__).resolve().parents[2] / "results" / "m4" / "e3-run1-ack-out-of-gas" / "run.json").read_text(encoding="utf-8"))
    assert m4_e3.check(run1, backup_rule="backup-q6")["ok"]
    assert not m4_e3.check(run1)["ok"]
