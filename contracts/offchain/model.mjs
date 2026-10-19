// M3 §2: the split model the M1b circuit evaluates, its configuration, and its fixed-point form.
//   f_A    (A)          8 -> 4, ReLU                  evaluated in the circuit
//   f_i    (passive i)  6 -> 8 ReLU -> 4              outside the circuit; its output h_i is committed
//   fusion (A)          16 -> 8 ReLU -> 5             evaluated in the circuit
// Training is in floating point (plain SGD, manual backprop: the layers are tiny); the trained
// f_A and fusion model are then quantized exactly as py/s2b_model.py does, and forwardInt /
// macroF1 mirror the circuit (and py/s2b_model.forward_int / s2_model.macro_f1_exact) exactly.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { keccak256 } from "ethers";

import { gauss, prng } from "./data.mjs";

const DEFAULT_CONFIG_PATH = resolve(fileURLToPath(import.meta.url), "..", "model-config.json");
/** The model configuration file: model-config.json, or the file named by VFL_MODEL_CONFIG (DATA1: one file per run). */
export const CONFIG_PATH = process.env.VFL_MODEL_CONFIG ? resolve(process.env.VFL_MODEL_CONFIG) : DEFAULT_CONFIG_PATH;
export const CONFIG_BYTES = readFileSync(CONFIG_PATH);
/** The model configuration; modelConfigHash = keccak256 of the exact file bytes. */
export const CONFIG = JSON.parse(CONFIG_BYTES.toString("utf8"));
export const modelConfigHash = () => keccak256(CONFIG_BYTES);

export const K = 5;
export const E = 4;
export const FRAC = 12;
export const SCALE = 1 << FRAC;
export const INIT_TAG = { fA: 0, fusion: 100, passive: (i) => i }; // i = 1..n

// ---- A dense layer with SGD ------------------------------------------------------

class Dense {
  constructor(inDim, outDim, act, rng) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.act = act;
    this.W = Float64Array.from({ length: outDim * inDim }, () => gauss(rng) / Math.sqrt(inDim));
    this.b = new Float64Array(outDim);
  }

  forward(X, n) {
    const { inDim: I, outDim: O, W, b } = this;
    const Z = new Float64Array(n * O);
    for (let r = 0; r < n; r++) {
      for (let o = 0; o < O; o++) {
        let z = b[o];
        for (let i = 0; i < I; i++) z += W[o * I + i] * X[r * I + i];
        Z[r * O + o] = z;
      }
    }
    const A = this.act === "relu" ? Z.map((z) => (z > 0 ? z : 0)) : Z;
    this.cache = { X, Z, n };
    return A;
  }

  /** dA: gradient w.r.t. this layer's output. Returns dX (with the pre-update weights), then steps. */
  backward(dA, lr) {
    const { inDim: I, outDim: O, W, b } = this;
    const { X, Z, n } = this.cache;
    const dZ = this.act === "relu" ? dA.map((g, k) => (Z[k] > 0 ? g : 0)) : dA;
    const dX = new Float64Array(n * I);
    const dW = new Float64Array(O * I);
    const db = new Float64Array(O);
    for (let r = 0; r < n; r++) {
      for (let o = 0; o < O; o++) {
        const g = dZ[r * O + o];
        if (g === 0) continue;
        db[o] += g;
        for (let i = 0; i < I; i++) { dW[o * I + i] += g * X[r * I + i]; dX[r * I + i] += g * W[o * I + i]; }
      }
    }
    for (let k = 0; k < W.length; k++) W[k] -= lr * dW[k];
    for (let o = 0; o < O; o++) b[o] -= lr * db[o];
    return dX;
  }
}

export class MLP {
  /** spec: { in, layers: [{ out, act }] } from the configuration; tag selects the init stream. */
  constructor(spec, tag) {
    const rng = prng(BigInt(CONFIG.initSeed) + BigInt(tag));
    let d = spec.in;
    this.inDim = spec.in;
    this.layers = spec.layers.map((l) => { const x = new Dense(d, l.out, l.act, rng); d = l.out; return x; });
    this.outDim = d;
  }

  forward(X, n) { let A = X; for (const l of this.layers) A = l.forward(A, n); return A; }

  backward(dOut, lr = CONFIG.learningRate) {
    let g = dOut;
    for (let i = this.layers.length - 1; i >= 0; i--) g = this.layers[i].backward(g, lr);
    return g;
  }

  clone() {
    const c = Object.create(MLP.prototype);
    c.inDim = this.inDim;
    c.outDim = this.outDim;
    c.layers = this.layers.map((l) => Object.assign(Object.create(Dense.prototype), l, { W: l.W.slice(), b: l.b.slice(), cache: null }));
    return c;
  }
}

export const newFA = () => new MLP(CONFIG.architecture.fA, INIT_TAG.fA);
export const newFusion = () => new MLP(CONFIG.architecture.fusion, INIT_TAG.fusion);
/** f_i's layer spec for passive i (1..n): architecture.fi.in is one width for every passive, or one per passive. */
export const fiSpec = (i) => {
  const s = CONFIG.architecture.fi;
  return { ...s, in: Array.isArray(s.in) ? s.in[i - 1] : s.in };
};

/** Dense.forward does not check input lengths, so a row of the wrong width would train silently: refuse it. */
export function assertFiWidth(i, width) {
  if (fiSpec(i).in !== width) throw new Error(`passive ${i}: row width ${width} != fi.in ${fiSpec(i).in} (${CONFIG_PATH})`);
}

/** DATA1: e3 runs only on an explicit config whose fi widths are the RECS data's; the M3 runs (i6-i9) only on the
 * default config. Checked before the runner deletes or writes anything. */
export function checkRunConfig({ run, configEnv, fiIn, dataWidths }) {
  if (run === "e3") {
    if (!configEnv) throw new Error("--run e3 needs VFL_MODEL_CONFIG (results/data1/configs/e3.json)");
    if (fiIn.some((w, k) => w !== dataWidths[k])) throw new Error(`fi width ${fiIn} != RECS data width ${dataWidths}`);
  } else if (configEnv) {
    throw new Error(`--run ${run} uses the default model config; unset VFL_MODEL_CONFIG (${configEnv})`);
  }
}
export const newFi = (i) => new MLP(fiSpec(i), INIT_TAG.passive(i));

/** Mean softmax cross-entropy over the batch; returns { loss, dZ }. */
export function softmaxCE(Z, labels, n) {
  const dZ = new Float64Array(n * K);
  let loss = 0;
  for (let r = 0; r < n; r++) {
    let m = -Infinity;
    for (let k = 0; k < K; k++) m = Math.max(m, Z[r * K + k]);
    let s = 0;
    for (let k = 0; k < K; k++) s += Math.exp(Z[r * K + k] - m);
    for (let k = 0; k < K; k++) {
      const p = Math.exp(Z[r * K + k] - m) / s;
      dZ[r * K + k] = (p - (labels[r] === k ? 1 : 0)) / n;
      if (labels[r] === k) loss -= Math.log(p);
    }
  }
  return { loss: loss / n, dZ };
}

/** The batches of one round (or scoring epoch) over positions 0..m-1: a seeded shuffle, then slices. */
export function batchesFor(m, round) {
  const rng = prng(BigInt(CONFIG.shuffleSeed) * 1000n + BigInt(round));
  const order = Array.from({ length: m }, (_, i) => i);
  for (let i = m - 1; i > 0; i--) { const j = Number(rng.u64() % BigInt(i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const out = [];
  for (let s = 0; s < m; s += CONFIG.batchSize) out.push(order.slice(s, s + CONFIG.batchSize));
  return out;
}

// ---- Embedding noise (a passive's local setting) ----------------------------------

/** Clips each row of H (rows of E) to L2 norm <= clip, then adds N(0, sigma^2) per value (in place). */
export function clipAndNoise(H, n, { clip, sigma }, normal) {
  for (let r = 0; r < n; r++) {
    let s = 0;
    for (let e = 0; e < E; e++) s += H[r * E + e] ** 2;
    const f = Math.sqrt(s) > clip ? clip / Math.sqrt(s) : 1;
    for (let e = 0; e < E; e++) H[r * E + e] = H[r * E + e] * f + sigma * normal();
  }
  return H;
}

// ---- Fixed point (as py/s2b_model.py) ---------------------------------------------

export function q(v, frac, bits) {
  const lo = -(2 ** (bits - 1));
  const hi = 2 ** (bits - 1) - 1;
  return Math.max(lo, Math.min(hi, Math.round(v * 2 ** frac)));
}
const qb = (v, frac, bits) => BigInt(q(v, frac, bits)); // int48 values exceed 2^31 but not 2^53

/** f_A as integers: { W0 [4][8] int16 @2^12, b0 [4] int32 @2^24 }. */
export function quantFA(fa) {
  const L = fa.layers[0];
  return {
    W0: Array.from({ length: L.outDim }, (_, o) => Array.from({ length: L.inDim }, (_, i) => q(L.W[o * L.inDim + i], FRAC, 16))),
    b0: Array.from(L.b, (v) => q(v, 2 * FRAC, 32)),
  };
}

/** The fusion model as integers: { W1 int16 @2^12, b1 int32 @2^24, W2 int16 @2^12, b2 int48 @2^36 }. */
export function quantHead(fusion) {
  const [L1, L2] = fusion.layers;
  const mat = (L) => Array.from({ length: L.outDim }, (_, o) => Array.from({ length: L.inDim }, (_, i) => q(L.W[o * L.inDim + i], FRAC, 16)));
  return {
    W1: mat(L1), b1: Array.from(L1.b, (v) => q(v, 2 * FRAC, 32)),
    W2: mat(L2), b2: Array.from(L2.b, (v) => Number(qb(v, 3 * FRAC, 48))),
  };
}

/** e_A = floor(max(W0 xA + b0, 0) / 2^12) for one row of int16 features (exact integers). */
export function eAInt(fa, xa) {
  return fa.W0.map((row, e) => {
    let z = BigInt(fa.b0[e]);
    for (let f = 0; f < row.length; f++) z += BigInt(row[f]) * BigInt(xa[f]);
    if (z >= 1n << 34n || z <= -(1n << 34n)) throw new Error("z0 out of the circuit's range");
    return Number((z > 0n ? z : 0n) >> BigInt(FRAC));
  });
}

/** The predicted class for one row, exactly as the circuit (first argmax). hp: 3 rows of 4 int16. */
export function forwardInt(fa, head, mask, xa, hp) {
  const inp = [...eAInt(fa, xa), ...hp.flatMap((h, p) => h.map((v) => ((mask >> p) & 1 ? v : 0)))].map(BigInt);
  const hid = head.W1.map((row, h) => {
    let z = BigInt(head.b1[h]);
    for (let d = 0; d < row.length; d++) z += BigInt(row[d]) * inp[d];
    return z > 0n ? z : 0n;
  });
  let best = 0;
  let bestZ = null;
  head.W2.forEach((row, k) => {
    let z = BigInt(head.b2[k]);
    for (let h = 0; h < row.length; h++) z += BigInt(row[h]) * hid[h];
    if (bestZ === null || z > bestZ) { best = k; bestZ = z; }
  });
  return best;
}

/** F = floor(10^4 * macro-F1) over the classes present in labels or predictions, exact (as s2_model.macro_f1_exact). */
export function macroF1(labels, preds) {
  const tp = Array(K).fill(0), pc = Array(K).fill(0), lc = Array(K).fill(0);
  labels.forEach((y, r) => { pc[preds[r]]++; lc[y]++; if (y === preds[r]) tp[y]++; });
  const present = [...Array(K).keys()].filter((c) => pc[c] + lc[c] > 0);
  // sum_c 2 tp_c / d_c over present classes, divided by |present|, as one fraction num/den.
  let num = 0n, den = 1n;
  for (const c of present) {
    const d = BigInt(pc[c] + lc[c]);
    num = num * d + 2n * BigInt(tp[c]) * den;
    den *= d;
  }
  den *= BigInt(present.length);
  return { F: Number((num * 10000n) / den), TP: tp, FP: pc.map((v, c) => v - tp[c]), FN: lc.map((v, c) => v - tp[c]) };
}
