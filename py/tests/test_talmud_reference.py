"""Tests for py/talmud_reference.py.

Besides the published values, every case is checked against the rule's
*definition* rather than against the algorithm: in the CEA regime there must be
one level lam with award_i == min(c_i/2, lam) for all i and sum == estate; in
the CEL regime the same holds for the losses c_i - award_i. Since
lam -> sum_i min(h_i, lam) is strictly increasing wherever it matters, these two
conditions determine the awards uniquely.
"""
from __future__ import annotations

import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from talmud_reference import CEA, CEL, SURPLUS, divide_capped, regime, talmud, talmud_exact  # noqa: E402

VECTORS = Path(__file__).resolve().parents[2] / "contracts" / "test" / "vectors" / "talmud_vectors.json"


def check_definition(estate: int, claims: list[int]) -> None:
    exact = talmud_exact(estate, claims)
    r = regime(estate, claims)
    total = sum(claims)
    if r == SURPLUS:
        assert exact == [Fraction(c) for c in claims]
        return
    assert sum(exact) == estate
    halves = [Fraction(c, 2) for c in claims]
    parts = exact if r == CEA else [c - a for c, a in zip(claims, exact)]
    level = max(parts)
    assert all(p == min(h, level) for p, h in zip(parts, halves))
    if r == CEL:
        assert sum(parts) == total - estate


def check_properties(estate: int, claims: list[int], awards: list[int]) -> None:
    n, total = len(claims), sum(claims)
    assert sum(awards) <= estate
    assert min(estate, total) - sum(awards) < max(n, 1)
    if estate >= total:
        assert sum(awards) == total
    for c, a in zip(claims, awards):
        assert 0 <= a <= c
        if c == 0:
            assert a == 0
    for i in range(n):
        for j in range(n):
            if claims[i] == claims[j]:
                assert awards[i] == awards[j]


@pytest.mark.parametrize("estate,expected", [
    (100, [Fraction(100, 3)] * 3),
    (200, [50, 75, 75]),
    (300, [50, 100, 150]),
])
def test_aumann_maschler_table(estate, expected):
    assert talmud_exact(estate, [100, 200, 300]) == expected


def test_contested_garment():
    assert talmud_exact(100, [50, 100]) == [25, 75]


def test_self_duality():
    rng = random.Random(1)
    for _ in range(500):
        claims = [rng.randint(0, 50) for _ in range(rng.randint(1, 8))]
        total = sum(claims)
        e = rng.randint(0, total)
        loss = talmud_exact(total - e, claims)
        assert talmud_exact(e, claims) == [c - l for c, l in zip(claims, loss)]


def test_definition_and_properties_random():
    rng = random.Random(2)
    for _ in range(3000):
        n = rng.randint(1, 10)
        claims = [0 if rng.random() < 0.15 else rng.randint(1, rng.choice([5, 100, 10**6])) for _ in range(n)]
        estate = rng.randint(0, 2 * sum(claims) + 1)
        check_definition(estate, claims)
        check_properties(estate, claims, talmud(estate, claims))


def test_capped_is_homogeneous():
    rho = 10**15
    cl = [420, 180, 75]
    for e in range(0, 1000, 7):
        for pool_perf in (100, 400, 675, 10_000):
            estate, awards = divide_capped(rho, rho * pool_perf, e, cl)
            exact = talmud_exact(min(e, pool_perf), cl)
            assert estate == rho * min(e, pool_perf)
            assert awards == [int(rho * a) for a in exact]


@pytest.mark.skipif(not VECTORS.exists(), reason="run py/talmud_vectors.py first")
def test_vector_file_matches_reference():
    data = json.loads(VECTORS.read_text(encoding="utf-8"))
    for c in data["cases"] + data["gas"]:
        estate, claims = int(c["estate"]), [int(x) for x in c["claims"]]
        assert [int(x) for x in c["expected"]] == talmud(estate, claims), c["id"]
        check_definition(estate, claims)
        check_properties(estate, claims, talmud(estate, claims))
