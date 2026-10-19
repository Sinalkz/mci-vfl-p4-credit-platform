"""S2b data, models and exact reference: macro-F1 over rows bound to member commitments.

Produces circuits/inputs/s2b_data.json with
  - the aligned table over I = {0, ..., I_MAX-1}:
      power company: label y (5 tariff tiers) and 8 raw features x_A,
      each passive party i: its final-round embedding h_i (4 dims);
    the |I| = 1,000 case uses the first 1,000 IDs;
  - the power company's bottom model f_A (8 -> 4, ReLU), frozen after joint training;
  - five fusion heads (16 -> 8 -> 5), one per measurement model, keyed by the
    passive-party mask (bit p set = passive party p+1 included):
      7 = fused, 6/5/3 = leave-one-out (party 1/2/3 left out), 0 = baseline;
    a left-out party's embedding is zero-padded, so all five share one
    architecture and one circuit;
  - f_A' (one weight changed) for the "different C_fA" negative test.

Fixed point (mirrored and range-checked by circuits/src/macro_f1_members.circom):
  x_A, W0, h_i, W1, W2 : int16 at scale 2^12
  b0, b1               : int32 at scale 2^24
  b2                   : int48 at scale 2^36
  e_A = floor(max(W0.x_A + b0, 0) / 2^12)        (back to scale 2^12, e_A < 2^22)
  input = [e_A, m_1*h_1, m_2*h_2, m_3*h_3]; z1 = W1.input + b1; z2 = W2.relu(z1) + b2
  prediction = first argmax of z2; F = floor(10^4 * macro-F1), empty classes excluded.

Model quality is not a thesis claim; heads are trained only so the figures differ.
Training uses a separate synthetic sample, not the aligned table.

Usage: .venv/Scripts/python py/s2b_model.py [--seed 11] [--rows 10000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s2_model import macro_f1_exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "inputs" / "s2b_data.json"

FA, E, P, H, K = 8, 4, 3, 8, 5          # active features, embedding dims, passive parties, hidden, classes
D = E * (P + 1)                          # fusion input width (16)
SX = 12                                  # fractional bits of x, weights and embeddings
TIER_PRIOR = [0.35, 0.28, 0.20, 0.12, 0.05]
ACTIVE_SIGNAL = 1.0
PASSIVE_SIGNAL = [0.6, 0.45, 0.8]        # MCI, the e-commerce company, device maker
MASKS = {"fused": 7, "loo1": 6, "loo2": 5, "loo3": 3, "baseline": 0}


def generate(rng, n, cent_a, cent_p):
    y = rng.choice(K, size=n, p=TIER_PRIOR)
    xa = cent_a[y] + rng.normal(0, 1, size=(n, FA))
    hp = [cent_p[p][y] + rng.normal(0, 1, size=(n, E)) for p in range(P)]
    return y, xa, hp


def softmax_ce_grad(z, onehot):
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return (p - onehot) / len(z)


def train(y, xa, hp, rng, mask=7, fa=None, epochs=800, lr=0.2):
    """Train f_A jointly with the head (fa=None), or only a head on top of a frozen f_A."""
    onehot = np.eye(K)[y]
    w0 = rng.normal(0, 1 / np.sqrt(FA), (E, FA)) if fa is None else fa[0]
    b0 = np.zeros(E) if fa is None else fa[1]
    w1 = rng.normal(0, 1 / np.sqrt(D), (H, D)); b1 = np.zeros(H)
    w2 = rng.normal(0, 1 / np.sqrt(H), (K, H)); b2 = np.zeros(K)
    m = np.concatenate([np.ones(E)] + [np.full(E, (mask >> p) & 1) for p in range(P)])
    for _ in range(epochs):
        z0 = xa @ w0.T + b0; ea = np.maximum(z0, 0)
        inp = np.concatenate([ea] + hp, axis=1) * m
        z1 = inp @ w1.T + b1; h = np.maximum(z1, 0)
        z2 = h @ w2.T + b2
        g2 = softmax_ce_grad(z2, onehot)
        g1 = (g2 @ w2) * (z1 > 0)
        w2 -= lr * g2.T @ h; b2 -= lr * g2.sum(0)
        if fa is None:
            gin = (g1 @ w1)[:, :E] * (z0 > 0)
            w0 -= lr * gin.T @ xa; b0 -= lr * gin.sum(0)
        w1 -= lr * g1.T @ inp; b1 -= lr * g1.sum(0)
    return (w0, b0), (w1, b1, w2, b2)


def q(v, frac_bits, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(np.rint(np.asarray(v) * (1 << frac_bits)), lo, hi).astype(np.int64).tolist()


def quant_fa(fa):
    return {"W0": q(fa[0], SX, 16), "b0": q(fa[1], 2 * SX, 32)}


def quant_head(head):
    w1, b1, w2, b2 = head
    return {"W1": q(w1, SX, 16), "b1": q(b1, 2 * SX, 32), "W2": q(w2, SX, 16), "b2": q(b2, 3 * SX, 48)}


def forward_int(fa: dict, head: dict, mask: int, xa: list[int], hp: list[list[int]]) -> int:
    """Exact integer forward pass; returns the predicted class. Mirrors the circuit."""
    z0 = [sum(fa["W0"][e][f] * xa[f] for f in range(FA)) + fa["b0"][e] for e in range(E)]
    assert all(abs(z) < 1 << 34 for z in z0)
    ea = [max(z, 0) >> SX for z in z0]
    inp = ea + [v * ((mask >> p) & 1) for p in range(P) for v in hp[p]]
    z1 = [sum(head["W1"][h][d] * inp[d] for d in range(D)) + head["b1"][h] for h in range(H)]
    hid = [max(z, 0) for z in z1]
    z2 = [sum(head["W2"][k][h] * hid[h] for h in range(H)) + head["b2"][k] for k in range(K)]
    assert all(abs(z) < 1 << 40 for z in z1) and all(abs(z) < 1 << 59 for z in z2)
    best = 0
    for k in range(1, K):
        if z2[k] > z2[best]:
            best = k
    return best


def evaluate(fa: dict, head: dict, mask: int, rows: list[dict]) -> dict:
    preds = [forward_int(fa, head, mask, r["xa"], r["hp"]) for r in rows]
    res = macro_f1_exact([r["y"] for r in rows], preds)
    res["preds"] = preds
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--rows", type=int, default=10_000)
    ap.add_argument("--train", type=int, default=4_000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cent_a = rng.normal(0, 1, (K, FA)) * ACTIVE_SIGNAL
    cent_p = [rng.normal(0, 1, (K, E)) * s for s in PASSIVE_SIGNAL]
    ytr, xatr, hptr = generate(rng, args.train, cent_a, cent_p)
    fa, fused = train(ytr, xatr, hptr, rng)
    heads = {"fused": fused}
    for name, mask in MASKS.items():
        if name != "fused":
            heads[name] = train(ytr, xatr, hptr, rng, mask=mask, fa=fa)[1]

    y, xa, hp = generate(rng, args.rows, cent_a, cent_p)
    table = {
        "y": y.astype(int).tolist(),
        "xa": q(xa, SX, 16),
        "hp": [q(hp[p], SX, 16) for p in range(P)],
    }
    fa_q = quant_fa(fa)
    fa_alt = json.loads(json.dumps(fa_q))
    fa_alt["W0"][0][0] += 1  # a different bottom model -> a different C_fA

    # Sanity report on the first 256 rows (not the sampled evaluation set).
    rows = [{"y": table["y"][j], "xa": table["xa"][j], "hp": [table["hp"][p][j] for p in range(P)]} for j in range(256)]
    heads_q = {name: {"mask": MASKS[name], **quant_head(h)} for name, h in heads.items()}
    for name, h in heads_q.items():
        r = evaluate(fa_q, h, h["mask"], rows)
        print(f"{name:8s} mask={h['mask']}  F(first 256 rows)={r['F']}")

    data = {
        "meta": {"generator": "py/s2b_model.py", "seed": args.seed, "rows": args.rows, "FA": FA, "E": E,
                 "P": P, "H": H, "K": K, "fracBits": SX, "masks": MASKS, "labelPrior": TIER_PRIOR},
        "table": table,
        "fA": fa_q,
        "fAalt": fa_alt,
        "heads": heads_q,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
