// Shared helpers for the S2 pipeline: paths, commitments, circuit input.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildPoseidon } from "circomlibjs";

export const CIRCUITS = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const ROOT = resolve(CIRCUITS, "..");
export const BUILD = resolve(CIRCUITS, "build");
export const MODEL_EVAL = resolve(CIRCUITS, "inputs", "model_eval.json");

export const buildDir = (n, merkle = false) => resolve(BUILD, `N${n}${merkle ? "-merkle" : ""}`);
export const membersBuildDir = (n, isize) => resolve(BUILD, `members-N${n}-I${isize}`);

export function loadModelEval() {
  return JSON.parse(readFileSync(MODEL_EVAL, "utf8"));
}

let poseidonP;
async function poseidon() {
  poseidonP ??= buildPoseidon();
  return poseidonP;
}

/** Field element of a (possibly negative) integer. */
function fe(P, v) {
  const x = BigInt(v) % P;
  return x < 0n ? x + P : x;
}

/** C_model: Poseidon chain over W1, b1, W2, b2 in chunks of 15 (mirrors the circuit). */
export async function modelCommitment(model, D, H, K) {
  const h = await poseidon();
  const P = h.F.p;
  const flat = [...model.W1.flat(), ...model.b1, ...model.W2.flat(), ...model.b2].map((v) => fe(P, v));
  const CH = 15;
  while (flat.length % CH !== 0) flat.push(0n);
  let acc = BigInt(D * 65536 + H * 256 + K);
  for (let c = 0; c < flat.length / CH; c++) {
    acc = h.F.toObject(h([acc, ...flat.slice(c * CH, (c + 1) * CH)]));
  }
  return acc;
}

/** Packs one row into field elements: D slots (x + 2^15) then the label, 15 16-bit slots per element. */
export function packRow(x, y) {
  const slots = [...x.map((v) => BigInt(v) + 32768n), BigInt(y)];
  const np = Math.ceil(slots.length / 15);
  const out = Array(np).fill(0n);
  slots.forEach((s, i) => { out[Math.floor(i / 15)] += s << BigInt(16 * (i % 15)); });
  return out;
}

/** C_eval as a hash chain acc_i = Poseidon(acc_{i-1}, packed row i), acc_0 = N. */
export async function evalCommitmentChain(rows, labels) {
  const h = await poseidon();
  let acc = BigInt(rows.length);
  for (let i = 0; i < rows.length; i++) acc = h.F.toObject(h([acc, ...packRow(rows[i], labels[i])]));
  return acc;
}

/** C_eval as a Poseidon Merkle root over packed rows (N a power of two). */
export async function evalCommitmentMerkle(rows, labels) {
  const h = await poseidon();
  let level = rows.map((r, i) => h.F.toObject(h(packRow(r, labels[i]))));
  while (level.length > 1) {
    const next = [];
    for (let i = 0; i < level.length; i += 2) next.push(h.F.toObject(h([level[i], level[i + 1]])));
    level = next;
  }
  return level[0];
}

/** Full circuit input for the first n evaluation rows. */
export async function circuitInput(n, { merkle = false } = {}) {
  const me = loadModelEval();
  const { D, H, K } = me.meta;
  const rows = me.rows.slice(0, n);
  const labels = me.labels.slice(0, n);
  const expected = me.perN[String(n)];
  if (!expected) throw new Error(`no reference for N=${n}; rerun py/s2_model.py with --sizes including ${n}`);
  const C_model = await modelCommitment(me.model, D, H, K);
  const C_eval = merkle ? await evalCommitmentMerkle(rows, labels) : await evalCommitmentChain(rows, labels);
  const s = (v) => String(v);
  return {
    input: {
      C_model: s(C_model),
      C_eval: s(C_eval),
      F: s(expected.F),
      W1: me.model.W1.map((r) => r.map(s)),
      b1: me.model.b1.map(s),
      W2: me.model.W2.map((r) => r.map(s)),
      b2: me.model.b2.map(s),
      X: rows.map((r) => r.map(s)),
      Y: labels.map(s),
    },
    expected,
    meta: me.meta,
  };
}
