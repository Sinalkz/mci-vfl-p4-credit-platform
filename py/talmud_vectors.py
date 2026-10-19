"""Generate S4 test vectors with expected awards from the exact reference.

Writes contracts/test/vectors/talmud_vectors.json, consumed by the Hardhat test
(contracts/test/TalmudDivision.test.ts) and the measurement script
(contracts/scripts/s4-talmud.ts). All integers are serialised as decimal
strings so that values beyond 2^53 survive JSON.

Usage:
  .venv/Scripts/python py/talmud_vectors.py [--count 100] [--nmin 3] [--nmax 10] [--seed 20260923]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from talmud_reference import CEA, CEL, SURPLUS, divide_capped, regime, talmud

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "contracts" / "test" / "vectors" / "talmud_vectors.json"

# Claim magnitudes: tiny values stress rounding, F1-scale values are what the
# protocol measures (10^-4 units), token-scale values are rho * F1 with an
# 18-decimal token.
SCALES = {"tiny": 20, "f1": 10_000, "token": 10_000 * 10**18}
RHO_TOKEN = 10**15  # token units per 10^-4 F1 used for gas vectors


def _claims(rng: random.Random, n: int, scale: int, p_zero: float = 0.1) -> list[int]:
    return [0 if rng.random() < p_zero else rng.randint(1, scale) for _ in range(n)]


def _estate_for(rng: random.Random, reg: str, total: int) -> int | None:
    half = total // 2  # largest estate with 2E <= C
    if reg == SURPLUS:
        return rng.randint(total, 2 * total + 1)
    if reg == CEA:
        return rng.randint(0, half) if total > 0 else None
    lo, hi = half + 1, total - 1  # C/2 < E < C
    return rng.randint(lo, hi) if lo <= hi else None


def case(cid: str, kind: str, estate: int, claims: list[int]) -> dict:
    claims = sorted(claims)  # TalmudDivision.divide takes claims in non-decreasing order
    return {
        "id": cid,
        "kind": kind,
        "regime": regime(estate, claims),
        "estate": str(estate),
        "claims": [str(c) for c in claims],
        "expected": [str(a) for a in talmud(estate, claims)],
    }


def random_cases(rng: random.Random, count: int, nmin: int, nmax: int) -> list[dict]:
    regimes = (SURPLUS, CEA, CEL)
    scales = list(SCALES.values())
    out = []
    i = 0
    while len(out) < count:
        reg = regimes[len(out) % 3]  # round-robin so every regime is covered
        n = rng.randint(nmin, nmax)
        claims = _claims(rng, n, scales[i % len(scales)])
        i += 1
        estate = _estate_for(rng, reg, sum(claims))
        if estate is None:
            continue
        c = case(f"rand-{len(out):03d}", "random", estate, claims)
        assert c["regime"] == reg
        out.append(c)
    return out


def edge_cases() -> list[dict]:
    e = []
    add = lambda cid, estate, claims: e.append(case(cid, "edge", estate, claims))
    add("all-zero-estate-0", 0, [0, 0, 0, 0])
    add("all-zero-estate-100", 100, [0, 0, 0, 0])
    add("one-zero-surplus", 1000, [0, 300, 200, 100])
    add("one-zero-cea", 150, [0, 300, 200, 100])
    add("one-zero-cel", 450, [0, 300, 200, 100])
    add("equal-surplus", 100, [7, 7, 7])
    add("equal-cea", 10, [7, 7, 7])
    add("equal-cel", 17, [7, 7, 7])
    add("equal-ten", 12_345, [4_000] * 10)
    add("estate-half-exact", 300, [100, 200, 300])          # C = 600, E = C/2
    add("estate-half-exact-uneven", 5_000, [1, 2, 9_997])   # C = 10000, E = C/2
    add("odd-total-floor-half", 3, [1, 2, 4])                # C = 7, E = floor(C/2)
    add("odd-total-ceil-half", 4, [1, 2, 4])                 # C = 7, E = ceil(C/2)
    add("estate-equals-total", 600, [100, 200, 300])
    add("estate-total-minus-1", 599, [100, 200, 300])
    add("estate-zero", 0, [100, 200, 300])
    add("estate-one", 1, [100, 200, 300])
    add("single-claimant", 5, [9])                            # n = 1
    add("dominant-claim", 1_000, [1, 1, 10**6])
    add("large-values", 3 * 10**30 + 7, [10**30, 2 * 10**30 + 1, 3 * 10**30 + 3, 5])
    return e


def textbook_cases() -> list[dict]:
    t = []
    # Aumann & Maschler (1985), Table 1 (the Talmud's three estates).
    for estate in (100, 200, 300):
        t.append(case(f"am85-{estate}", "textbook", estate, [100, 200, 300]))
    # Contested garment (Mishna Baba Metzia): claims 1/2 and 1 of a garment worth 1.
    t.append(case("garment", "textbook", 100, [50, 100]))
    return t


def capped_cases() -> list[dict]:
    """Budget-capped scenarios in token units (D56/D65): 3 passive parties."""
    rho = 10**15
    scenarios = [
        # id, pool, E (10^-4 F1), claims (10^-4 F1)
        ("no-cap-surplus", 10**19, 900, [420, 180, 75]),      # E >= C, pool not binding
        ("no-cap-value-bankruptcy", 10**19, 500, [420, 180, 75]),
        ("budget-bankruptcy", 4 * 10**17, 900, [420, 180, 75]),  # pool/rho = 400 < C
        ("budget-and-value", 2 * 10**17, 500, [420, 180, 75]),   # pool/rho = 200 < E < C
        ("budget-exactly-C", 675 * 10**15, 900, [420, 180, 75]),
        ("budget-odd-remainder", 333_333_333_333_333_337, 900, [420, 180, 75]),
        ("zero-estate", 10**19, 0, [420, 180, 75]),
    ]
    out = []
    for cid, pool, e, cl in scenarios:
        cl = sorted(cl)
        estate, awards = divide_capped(rho, pool, e, cl)
        out.append({
            "id": cid, "rho": str(rho), "pool": str(pool),
            "estatePerf": str(e), "claimsPerf": [str(c) for c in cl],
            "expectedEstate": str(estate), "expected": [str(a) for a in awards],
            "regime": regime(estate, [rho * c for c in cl]),
        })
    return out


def gas_cases(rng: random.Random, sizes=(3, 5, 10), per_cell: int = 20) -> list[dict]:
    """Vectors for gas measurement: per n and regime, `per_cell` random claim sets.

    The draws are the same as when divide sorted on-chain (same seed), so gas is
    comparable vector by vector; the claims are now passed sorted (see case()).
    """
    out = []
    for n in sizes:
        for reg in (SURPLUS, CEA, CEL):
            made = 0
            while made < per_cell:
                claims = [RHO_TOKEN * c for c in _claims(rng, n, 5_000, p_zero=0.0)]
                estate = _estate_for(rng, reg, sum(claims))
                if estate is None:
                    continue
                c = case(f"gas-n{n}-{reg}-{made:02d}", "gas", estate, claims)
                c["n"] = n
                out.append(c)
                made += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--nmin", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = {
        "meta": {"generator": "py/talmud_vectors.py", "seed": args.seed, "count": args.count,
                 "nmin": args.nmin, "nmax": args.nmax},
        "cases": random_cases(rng, args.count, args.nmin, args.nmax) + edge_cases() + textbook_cases(),
        "capped": capped_cases(),
        "gas": gas_cases(rng),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    by_regime = {}
    for c in data["cases"]:
        by_regime[c["regime"]] = by_regime.get(c["regime"], 0) + 1
    print(f"wrote {args.out.relative_to(ROOT)}: {len(data['cases'])} division cases {by_regime}, "
          f"{len(data['capped'])} capped, {len(data['gas'])} gas vectors")


if __name__ == "__main__":
    main()
