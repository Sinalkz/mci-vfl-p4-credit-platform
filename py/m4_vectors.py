"""M4 test vectors: settlement scenarios for contracts/test/M4Settlement.test.ts, expected values from
py/settlement_reference.py. n = 3 passives (the scoring circuit's slots).

Each case: name, path (normal | objection | backup), B, rho, Dp, figures (Fr, Fj, loo[3], base), left[3], eqv[3],
acks[3] (who acknowledges; for normal every remaining passive), objector (objection path), expected
{E, Eprime, l, a, awards, owedA, owedP[3], locked}.

Usage (from the repo root):  .venv/Scripts/python py/m4_vectors.py   -> contracts/test/vectors/m4_settlement.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import settlement_reference as sr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "contracts" / "test" / "vectors" / "m4_settlement.json"
E3 = dict(B=1_000_003, rho=5_000, Dp=1_001, Fr=3364, Fj=2998, loo=[3365, 3571, 3005], base=2096)
NO = [False, False, False]


def case(name, path, *, B, rho, Dp, Fr, Fj, loo, base, left=NO, eqv=NO, acks=None, objector=None):
    remaining = [not x for x in left]
    if path == "normal":
        acks = remaining
    acks = list(acks) if acks is not None else [False] * 3
    if path == "objection":
        objector = objector if objector is not None else remaining.index(True)
        assert remaining[objector] and not acks[objector]
    for i in range(3):
        assert not (acks[i] and left[i]), name
    if path == "backup":  # every remaining passive acknowledging would settle on the normal path instead
        assert any(remaining[i] and not acks[i] for i in range(3)), name
    r = sr.settle(path, B=B, rho=rho, Dp=Dp, u_fed=max(Fr, Fj), u_a=base, loo=loo, left=list(left), eqv=list(eqv), acks=acks)
    return {"name": name, "path": path, "B": B, "rho": rho, "Dp": Dp,
            "figures": {"Fr": Fr, "Fj": Fj, "loo": loo, "base": base},
            "left": list(left), "eqv": list(eqv), "acks": acks, "objector": objector,
            "expected": {"E": r["E"], "Eprime": r["Eprime"], "l": r["l"], "a": r["a"], "awards": r["awards"],
                         "owedA": r["A"], "owedP": r["P"], "locked": r["locked"]}}


def cases():
    out = [
        case("E3 figures, normal", "normal", **E3),
        case("E3 figures, objection by p1 after p2 acknowledged", "objection", **E3, acks=[False, True, False], objector=0),
        case("E3 figures, backup, p3 silent", "backup", **E3, acks=[True, True, False]),
        case("E = 0 (U_fed below U_A)", "normal", B=1_000_003, rho=5_000, Dp=1_001, Fr=2000, Fj=1900, loo=[1800, 1700, 1600], base=2500),
        case("every l = 0", "normal", B=1_000_003, rho=5_000, Dp=1_001, Fr=4000, Fj=3900, loo=[4000, 4100, 4200], base=3000),
        case("E' capped by floor(B/rho), B not a multiple of rho", "normal", B=12_345, rho=100, Dp=77, Fr=4000, Fj=3000, loo=[3500, 3700, 3900], base=3000),
        case("large B: E' = E", "normal", B=10**9, rho=1_000, Dp=1_001, Fr=4500, Fj=4600, loo=[4400, 4300, 4200], base=3000),
        case("p3 left, p1 equivocated: normal", "normal", **E3, left=[False, False, True], eqv=[True, False, False]),
        case("p3 left, p1 equivocated: objection by p2", "objection", **E3, left=[False, False, True], eqv=[True, False, False], objector=1),
        case("p3 left, p1 equivocated: backup, only the equivocator responds", "backup", **E3, left=[False, False, True], eqv=[True, False, False], acks=[True, False, False]),
        case("only an equivocator remains: forfeits locked", "normal", **E3, left=[True, True, False], eqv=[False, False, True]),
        case("backup: sum rho*l = B exactly (p3 silent)", "backup", B=300_000, rho=1_000, Dp=11, Fr=4000, Fj=3900, loo=[3900, 3800, 3700], base=3000, acks=[True, True, False]),
        case("backup: B short, Talmud on floor(B/rho) (p3 silent)", "backup", B=150_500, rho=1_000, Dp=11, Fr=4000, Fj=3900, loo=[3900, 3800, 3700], base=3000, acks=[True, True, False]),
        case("backup: no responders", "backup", **E3, acks=[False, False, False]),
        case("backup: equivocator responds, gets no D_p", "backup", **E3, eqv=[False, True, False], acks=[False, True, True]),
        case("ties in the claims", "normal", B=1_000_003, rho=2_000, Dp=1_001, Fr=5000, Fj=4000, loo=[4800, 4800, 4800], base=4000),
        case("claims in decreasing proposal order", "normal", B=10**8, rho=1_000, Dp=1_001, Fr=5000, Fj=4000, loo=[4100, 4500, 4900], base=3000),
        case("objection with every other passive acknowledged", "objection", **E3, acks=[True, False, True], objector=1),
    ]
    rng = random.Random(2026)
    for k in range(15):
        path = ["normal", "objection", "backup"][k % 3]
        left = [rng.random() < 0.2 for _ in range(3)]
        if all(left):
            left[k % 3] = False
        eqv = [(not left[i]) and rng.random() < 0.2 for i in range(3)]
        remaining = [i for i in range(3) if not left[i]]
        acks = [(not left[i]) and rng.random() < 0.6 for i in range(3)]
        objector = None
        if path == "objection":
            objector = rng.choice(remaining)
            acks[objector] = False
        if path == "backup" and all(acks[i] for i in remaining):
            acks[rng.choice(remaining)] = False
        base = rng.randint(0, 6000)
        out.append(case(f"random {k}", path, B=rng.randint(10_000, 5 * 10**6), rho=rng.choice([1, 7, 100, 1_000, 5_000]),
                        Dp=rng.randint(1, 3_000), Fr=rng.randint(0, 9999), Fj=rng.randint(0, 9999),
                        loo=[rng.randint(0, 9999) for _ in range(3)], base=base, left=left, eqv=eqv, acks=acks, objector=objector))
    return out


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {"source": "py/m4_vectors.py (py/settlement_reference.py)", "cases": cases()}
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(data['cases'])} cases")
