"""The M4 settlement reference: hand-checked cases and conservation on random cases."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import settlement_reference as sr  # noqa: E402

BASE = dict(B=1_000_003, rho=5_000, Dp=1_001)


def test_division_caps_the_estate_at_floor_B_over_rho_and_zeroes_left_claims():
    E, ep, l, a = sr.division(1_000_003, 5_000, 3364, 2096, [3365, 3571, 3005], [False, False, False])
    assert (E, ep, l) == (1268, 200, [0, 0, 359])            # E3's figures: E' = floor(1,000,003 / 5,000) = 200
    assert a == [0, 0, 200]
    assert sr.division(10**9, 1, 4000, 3000, [3500, 3600, 3700], [False, True, False])[2] == [500, 0, 300]


def test_normal_path_by_hand():
    # l = (600, 400, 200), E = 1000 capped at floor(B/rho) = 200; talmud(200; 600, 400, 200): CEA on half-claims gives
    # 66.67 each, floored to 66 (the flooring keeps 2 of E' back for A)
    r = sr.settle("normal", **BASE, u_fed=4000, u_a=3000, loo=[3400, 3600, 3800], left=[False] * 3, eqv=[False] * 3)
    assert r["Eprime"] == 200 and r["a"] == [66, 66, 66]
    assert r["P"] == [5_000 * 66 + 1_001] * 3
    assert r["A"] == 1_000_003 - 5_000 * 198 and r["locked"] == 0


def test_equivocator_gets_no_deposit_and_no_share_left_passive_gets_nothing():
    r = sr.settle("normal", **BASE, u_fed=4000, u_a=3900, loo=[3950, 3960, 3990], left=[True, False, False], eqv=[False, True, False])
    assert r["P"][0] == 0                                     # left
    assert r["P"][1] == 5_000 * r["a"][1]                     # equivocator: award only
    assert r["P"][2] == 5_000 * r["a"][2] + 1_001 + 2 * 1_001  # the two forfeited deposits
    total = r["A"] + sum(r["P"]) + r["locked"]
    assert total == 1_000_003 + 3 * 1_001


def test_backup_settles_exactly_like_the_normal_path_d142():
    # D142: after the window, silence counts as acceptance: the same payouts as the normal path, nothing locked for it
    r = sr.settle("backup", B=10**7, rho=1_000, Dp=7, u_fed=4000, u_a=3000, loo=[3900, 3800, 3700], left=[False] * 3,
                  eqv=[False] * 3, acks=[True, False, True])
    assert r == sr.settle("normal", B=10**7, rho=1_000, Dp=7, u_fed=4000, u_a=3000, loo=[3900, 3800, 3700], left=[False] * 3,
                          eqv=[False] * 3)
    assert r["P"] == [1_000 * a + 7 for a in r["a"]] and r["A"] == 10**7 - 1_000 * sum(r["a"]) and r["locked"] == 0
    rng = random.Random(142)
    for _ in range(2000):
        B, rho, Dp = rng.randint(1, 10**7), rng.randint(1, 20_000), rng.randint(1, 5_000)
        left = [rng.random() < 0.2 for _ in range(3)]
        eqv = [(not left[i]) and rng.random() < 0.2 for i in range(3)]
        kw = dict(B=B, rho=rho, Dp=Dp, u_fed=rng.randint(0, 10_000), u_a=rng.randint(0, 10_000),
                  loo=[rng.randint(0, 10_000) for _ in range(3)], left=left, eqv=eqv)
        assert sr.settle("backup", **kw, acks=[rng.random() < 0.5 for _ in range(3)]) == sr.settle("normal", **kw)


def test_the_superseded_backup_rule_is_kept_for_e3_run1():
    # M4 Q6 (superseded by D142), under which E3's first run on Besu settled: responders only, the rest of B locked
    r = sr.settle("backup-q6", B=10**7, rho=1_000, Dp=7, u_fed=4000, u_a=3000, loo=[3900, 3800, 3700], left=[False] * 3,
                  eqv=[False] * 3, acks=[True, False, True])
    assert r["awards"] == [100, 0, 300]
    assert r["P"] == [100_000 + 7 + 4, 0, 300_000 + 7 + 3]    # non-responder's D_p split 4 / 3 (remainder to the first)
    assert r["A"] == 0 and r["locked"] == 10**7 - 400_000


def test_every_path_conserves_tokens_on_random_cases():
    rng = random.Random(4)
    for _ in range(3000):
        n = 3
        B, rho, Dp = rng.randint(1, 10**7), rng.randint(1, 20_000), rng.randint(1, 5_000)
        left = [rng.random() < 0.2 for _ in range(n)]
        eqv = [(not left[i]) and rng.random() < 0.2 for i in range(n)]
        u_fed, u_a = rng.randint(0, 10_000), rng.randint(0, 10_000)
        loo = [rng.randint(0, 10_000) for _ in range(n)]
        acks = [rng.random() < 0.6 for _ in range(n)]
        for path in ("normal", "objection", "backup", "backup-q6"):
            r = sr.settle(path, B=B, rho=rho, Dp=Dp, u_fed=u_fed, u_a=u_a, loo=loo, left=left, eqv=eqv, acks=acks)
            assert r["A"] + sum(r["P"]) + r["locked"] == B + n * Dp, (path, B, rho, Dp, left, eqv, acks)
            assert min(r["P"]) >= 0 and r["A"] >= 0 and r["locked"] >= 0


def test_backup_with_no_remaining_passive_returns_B_to_A_and_locks_the_forfeits_d141():
    r = sr.settle("backup", **BASE, u_fed=3364, u_a=2096, loo=[3365, 3571, 3005], left=[True] * 3, eqv=[False] * 3,
                  acks=[False] * 3)
    assert r["A"] == 1_000_003 and r["P"] == [0, 0, 0] and r["locked"] == 3 * 1_001
    assert r["awards"] == [0, 0, 0]
