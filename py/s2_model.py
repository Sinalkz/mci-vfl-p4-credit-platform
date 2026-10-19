"""S2 data, model and exact reference for the macro-F1 circuit.

Produces circuits/inputs/model_eval.json containing:
  - a quantised MLP fusion head (D=16 -> H=8 ReLU -> K=5), integer weights;
  - an evaluation set of N_max rows (integer inputs, labels 0..4);
  - for every N in --sizes (prefixes of the evaluation set): predictions,
    confusion counts and F = floor(10^4 * macro-F1), computed with exact
    integer / Fraction arithmetic that mirrors the circuit bit for bit.

Model quality is not a thesis claim; the head is trained only so that the
F1 figure is not degenerate.

Fixed-point convention (also enforced by range checks in the circuit):
  x   : scale 2^12, int16   (|x|  < 2^15)
  W1  : scale 2^12, int16
  b1  : scale 2^24, int32   (the scale of W1.x)
  W2  : scale 2^12, int16
  b2  : scale 2^36, int48   (the scale of W2.relu(W1.x + b1))
No rescaling is done between layers: argmax is scale-invariant, and the
bounds give |z1| < 2^35 and |z2| < 2^54, far below the field size.

Empty classes: a class with 2TP + FP + FN = 0 (absent from labels and from
predictions) has undefined F1 and is excluded from the mean. This is the
default of sklearn.metrics.f1_score(average="macro").

Usage: .venv/Scripts/python py/s2_model.py [--sizes 64 128 256] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import floor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "inputs" / "model_eval.json"

D, H, K = 16, 8, 5
MEMBERS = ["power_company", "mci", "digikala", "device_maker"]  # 4 dims each
SX = SW = 12                       # fractional bits of inputs and weights
INT16 = (-(1 << 15), (1 << 15) - 1)
B1_BITS, B2_BITS = 32, 48
F1_SCALE = 10_000

# Tariff tiers 0..4: sparse high tiers (decision log D26).
TIER_PRIOR = [0.35, 0.28, 0.20, 0.12, 0.05]
# How much class signal each member's embedding block carries (synthetic).
MEMBER_SIGNAL = [1.0, 0.6, 0.45, 0.8]


def synthetic_embeddings(rng: np.random.Generator, n: int, centroids: np.ndarray):
    y = rng.choice(K, size=n, p=TIER_PRIOR)
    x = centroids[y] + rng.normal(0.0, 1.0, size=(n, D))
    return x, y


def train_mlp(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, epochs=600, lr=0.2):
    """Full-batch gradient descent on softmax cross-entropy. Float64."""
    w1 = rng.normal(0, 1 / np.sqrt(D), size=(H, D)); b1 = np.zeros(H)
    w2 = rng.normal(0, 1 / np.sqrt(H), size=(K, H)); b2 = np.zeros(K)
    onehot = np.eye(K)[y]
    n = len(y)
    for _ in range(epochs):
        z1 = x @ w1.T + b1; h = np.maximum(z1, 0)
        z2 = h @ w2.T + b2
        p = np.exp(z2 - z2.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
        g2 = (p - onehot) / n
        gw2 = g2.T @ h; gb2 = g2.sum(0)
        g1 = (g2 @ w2) * (z1 > 0)
        gw1 = g1.T @ x; gb1 = g1.sum(0)
        w1 -= lr * gw1; b1 -= lr * gb1; w2 -= lr * gw2; b2 -= lr * gb2
    return w1, b1, w2, b2


def q(v: np.ndarray, frac_bits: int, bits: int) -> list:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(np.rint(v * (1 << frac_bits)), lo, hi).astype(np.int64).tolist()


def forward_int(model: dict, x: list[int]) -> tuple[int, list[int]]:
    """Exact integer forward pass; returns (predicted class, logits). Mirrors the circuit."""
    z1 = [sum(model["W1"][h][d] * x[d] for d in range(D)) + model["b1"][h] for h in range(H)]
    hid = [z if z >= 0 else 0 for z in z1]
    z2 = [sum(model["W2"][k][h] * hid[h] for h in range(H)) + model["b2"][k] for k in range(K)]
    assert all(abs(z) < 1 << 35 for z in z1) and all(abs(z) < 1 << 54 for z in z2)
    best = 0
    for k in range(1, K):
        if z2[k] > z2[best]:   # strict: ties go to the smallest index
            best = k
    return best, z2


def macro_f1_exact(labels: list[int], preds: list[int]) -> dict:
    tp = [0] * K; pc = [0] * K; lc = [0] * K
    for y, p in zip(labels, preds):
        pc[p] += 1; lc[y] += 1
        if y == p:
            tp[y] += 1
    denom = [pc[c] + lc[c] for c in range(K)]           # = 2TP + FP + FN
    present = [c for c in range(K) if denom[c] > 0]
    macro = sum((Fraction(2 * tp[c], denom[c]) for c in present), Fraction(0)) / len(present)
    return {
        "TP": tp, "FP": [pc[c] - tp[c] for c in range(K)], "FN": [lc[c] - tp[c] for c in range(K)],
        "emptyClasses": [c for c in range(K) if denom[c] == 0],
        "macroF1": f"{macro.numerator}/{macro.denominator}",
        "F": floor(macro * F1_SCALE),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", type=int, default=4000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    scale = np.repeat(MEMBER_SIGNAL, 4)
    centroids = rng.normal(0, 1.0, size=(K, D)) * scale
    xtr, ytr = synthetic_embeddings(rng, args.train, centroids)
    w1, b1, w2, b2 = train_mlp(xtr, ytr, rng)

    model = {
        "W1": q(w1, SW, 16), "b1": q(b1, SX + SW, B1_BITS),
        "W2": q(w2, SW, 16), "b2": q(b2, SX + 2 * SW, B2_BITS),  # scale 2^36
    }
    nmax = max(args.sizes)
    xev, yev = synthetic_embeddings(rng, nmax, centroids)
    rows = q(xev, SX, 16)
    labels = yev.astype(int).tolist()
    preds_all = [forward_int(model, r)[0] for r in rows]

    # Float model on the same rows, for information only.
    zf = np.maximum(xev @ w1.T + b1, 0) @ w2.T + b2
    float_preds = zf.argmax(1).tolist()

    per_n = {}
    for n in sorted(args.sizes):
        res = macro_f1_exact(labels[:n], preds_all[:n])
        res["preds"] = preds_all[:n]
        res["floatModelAgreement"] = sum(a == b for a, b in zip(preds_all[:n], float_preds[:n])) / n
        per_n[str(n)] = res
        print(f"N={n:4d}  F={res['F']:5d}  TP={res['TP']} FP={res['FP']} FN={res['FN']} "
              f"empty={res['emptyClasses']}  quantised-vs-float agreement={res['floatModelAgreement']:.3f}")

    data = {
        "meta": {"generator": "py/s2_model.py", "seed": args.seed, "D": D, "H": H, "K": K,
                 "members": MEMBERS, "fracBits": {"x": SX, "W1": SW, "b1": SX + SW, "W2": SW, "b2": SX + 2 * SW},
                 "labelPrior": TIER_PRIOR, "f1Scale": F1_SCALE,
                 "emptyClassRule": "excluded from the mean (sklearn average='macro' default)"},
        "model": model,
        "rows": rows,
        "labels": labels,
        "perN": per_n,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
