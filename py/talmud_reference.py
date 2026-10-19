"""Reference implementation of the Talmud division rule (Aumann & Maschler, 1985).

Exact rational arithmetic with fractions.Fraction; each award is floored only at
the very end. This is the oracle the Solidity contract TalmudDivision.sol is
checked against.

Rule, with C = sum(claims):
  estate >= C      : award_i = claim_i                       (surplus: residual stays with the active party)
  estate <= C / 2  : award_i = min(claim_i / 2, lam)         (CEA on half-claims)
  otherwise        : award_i = claim_i - min(claim_i / 2, mu) (half-claims plus CEL on half-claims)
where lam and mu are chosen so that the awards sum exactly to the estate.
"""
from __future__ import annotations

from fractions import Fraction
from math import floor
from typing import Sequence

SURPLUS, CEA, CEL = "surplus", "cea", "cel"


def regime(estate: int, claims: Sequence[int]) -> str:
    total = sum(claims)
    if estate >= total:
        return SURPLUS
    if 2 * estate <= total:
        return CEA
    return CEL


def cea_level(amount: Fraction, caps: Sequence[Fraction]) -> Fraction:
    """Return lam with sum_i min(cap_i, lam) == amount. Requires amount <= sum(caps)."""
    if amount > sum(caps, Fraction(0)):
        raise ValueError("amount exceeds the sum of caps")
    ordered = sorted(caps)
    n = len(ordered)
    remaining = Fraction(amount)
    for k, cap in enumerate(ordered):
        share = remaining / (n - k)
        if share <= cap:
            return share
        remaining -= cap
    raise AssertionError("unreachable: amount <= sum(caps)")


def talmud_exact(estate: int, claims: Sequence[int]) -> list[Fraction]:
    """Exact (unrounded) Talmud awards."""
    if estate < 0 or any(c < 0 for c in claims):
        raise ValueError("estate and claims must be non-negative")
    total = sum(claims)
    r = regime(estate, claims)
    if r == SURPLUS:
        return [Fraction(c) for c in claims]
    halves = [Fraction(c, 2) for c in claims]
    if r == CEA:
        lam = cea_level(Fraction(estate), halves)
        return [min(h, lam) for h in halves]
    mu = cea_level(Fraction(total - estate), halves)
    return [c - min(h, mu) for c, h in zip(claims, halves)]


def talmud(estate: int, claims: Sequence[int]) -> list[int]:
    """Talmud awards in integer units, each floored."""
    return [floor(a) for a in talmud_exact(estate, claims)]


def divide_capped(rho: int, pool: int, estate_perf: int,
                  claims_perf: Sequence[int]) -> tuple[int, list[int]]:
    """Budget-capped division in token units (decision log D56/D65).

    estate_tokens = min(rho * estate_perf, pool); claim_i in tokens = rho * claim_i.
    Because the Talmud rule is homogeneous (T(rho*E, rho*c) = rho*T(E, c)), this is
    exactly E_eff = min(E, pool/rho) in performance units, with flooring at token
    granularity.
    """
    estate = min(rho * estate_perf, pool)
    return estate, talmud(estate, [rho * c for c in claims_perf])


if __name__ == "__main__":
    # Aumann & Maschler (1985), Table 1: claims 100/200/300.
    for e in (100, 200, 300):
        print(e, [str(a) for a in talmud_exact(e, [100, 200, 300])], talmud(e, [100, 200, 300]))
