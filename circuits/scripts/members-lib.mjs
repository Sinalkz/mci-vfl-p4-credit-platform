// S2b helpers, mirroring circuits/src/macro_f1_members.circom exactly:
// member Merkle roots, index derivation from the seed, model commitments,
// circuit input. Also used by the contract tests (S3 signs these roots).
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { buildPoseidon } from "circomlibjs";

import { CIRCUITS, ROOT } from "./lib.mjs";

export const S2B_DATA = resolve(CIRCUITS, "inputs", "s2b_data.json");
export const FIELD_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;
// Public scoring seed used in the experiments. In the protocol it is generated
// jointly in Phase A (D41; mechanism out of scope) and fixed in the run contract.
export const DEFAULT_SEED = 20260923n;

let poseidonP;
async function poseidon() {
  poseidonP ??= buildPoseidon();
  return poseidonP;
}
const fe = (v) => ((BigInt(v) % FIELD_P) + FIELD_P) % FIELD_P;

export const depthFor = (isize) => { let d = 0; while ((1 << d) < isize) d++; return Math.max(d, 1); };

/** gammaA leaf: 8 features (x + 2^15, 16 bits each) then the label at bit 128. */
export function leafA(xa, y) {
  let v = BigInt(y) << BigInt(16 * xa.length);
  xa.forEach((x, f) => { v += (BigInt(x) + 32768n) << BigInt(16 * f); });
  return v;
}
/** gammaP leaf: embedding values (h + 2^15, 16 bits each). */
export function leafP(h) {
  let v = 0n;
  h.forEach((x, e) => { v += (BigInt(x) + 32768n) << BigInt(16 * e); });
  return v;
}

/** Binary Poseidon Merkle tree over `leaves`, zero-padded to 2^depth. */
export async function buildTree(leaves, depth) {
  const h = await poseidon();
  const levels = [[...leaves, ...Array((1 << depth) - leaves.length).fill(0n)]];
  for (let l = 0; l < depth; l++) {
    const prev = levels[l];
    const next = new Array(prev.length / 2);
    for (let i = 0; i < prev.length; i += 2) next[i / 2] = h.F.toObject(h([prev[i], prev[i + 1]]));
    levels.push(next);
  }
  return { depth, levels, root: levels[depth][0] };
}
export function siblings(tree, idx) {
  const out = [];
  let i = idx;
  for (let l = 0; l < tree.depth; l++) { out.push(tree.levels[l][i ^ 1]); i >>= 1; }
  return out;
}

/** Root of a final-round embedding matrix (rows of E values): gamma_i, as signed in S3. */
export async function embeddingRoot(matrix, depth = depthFor(matrix.length)) {
  return (await buildTree(matrix.map(leafP), depth)).root;
}

/** idx_k = (Poseidon(seed, k) mod 2^128) mod isize, k = 0..n-1 (with replacement). */
export async function sampleIndices(seed, n, isize) {
  const h = await poseidon();
  const low = (1n << 128n) - 1n;
  return Array.from({ length: n }, (_, k) =>
    Number((h.F.toObject(h([BigInt(seed), BigInt(k)])) & low) % BigInt(isize)));
}

async function paramChain(values, tag) {
  const h = await poseidon();
  const flat = values.map(fe);
  while (flat.length % 15 !== 0) flat.push(0n);
  let acc = BigInt(tag);
  for (let c = 0; c < flat.length / 15; c++) acc = h.F.toObject(h([acc, ...flat.slice(c * 15, c * 15 + 15)]));
  return acc;
}
export const faCommitment = (fa) => paramChain([...fa.W0.flat(), ...fa.b0], 8 * 65536 + 4 * 256);
export const headCommitment = (hd) =>
  paramChain([...hd.W1.flat(), ...hd.b1, ...hd.W2.flat(), ...hd.b2], 16 * 65536 + 8 * 256 + 5);

export function loadS2b() {
  return JSON.parse(readFileSync(S2B_DATA, "utf8"));
}

const treeCache = new WeakMap(); // data object -> (isize -> trees)
/** The four member trees over the first `isize` aligned IDs. */
export async function memberTrees(data, isize) {
  if (!treeCache.has(data)) treeCache.set(data, new Map());
  const cache = treeCache.get(data);
  if (cache.has(isize)) return cache.get(isize);
  const depth = depthFor(isize);
  const t = data.table;
  const A = await buildTree(t.y.slice(0, isize).map((y, j) => leafA(t.xa[j], y)), depth);
  const P = [];
  for (const hp of t.hp) P.push(await buildTree(hp.slice(0, isize).map(leafP), depth));
  const trees = { depth, A, P };
  cache.set(isize, trees);
  return trees;
}

const PY = resolve(ROOT, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
export function oracle(req) {
  return JSON.parse(execFileSync(PY, [resolve(ROOT, "py", "s2b_oracle.py")], {
    input: JSON.stringify(req), cwd: resolve(ROOT, "py"), encoding: "utf8", maxBuffer: 1 << 26,
  }));
}

/**
 * Full circuit input for N rows sampled from the first `isize` aligned IDs.
 * mask: which passive parties are included (7 fused, 6/5/3 leave-one-out, 0 baseline).
 * faKey: "fA" (the registered bottom model) or "fAalt" (a different one, negative tests).
 */
export async function membersInput(n, isize, { mask = 7, faKey = "fA", seed = DEFAULT_SEED, data = loadS2b() } = {}) {
  const head = Object.values(data.heads).find((h) => h.mask === mask);
  if (!head) throw new Error(`no head for mask ${mask}`);
  const fa = data[faKey];
  const trees = await memberTrees(data, isize);
  const idx = await sampleIndices(seed, n, isize);
  const t = data.table;
  const rows = idx.map((j) => ({ y: t.y[j], xa: t.xa[j], hp: t.hp.map((hp) => hp[j]) }));
  const ref = oracle({ fA: fa, head, mask, rows });
  const s = (v) => String(v);
  const input = {
    C_model: s(await headCommitment(head)),
    C_fA: s(await faCommitment(fa)),
    gammaA: s(trees.A.root),
    gammaP: trees.P.map((tr) => s(tr.root)),
    seed: s(seed),
    mask: s(mask),
    F: s(ref.F),
    W0: fa.W0.map((r) => r.map(s)), b0: fa.b0.map(s),
    W1: head.W1.map((r) => r.map(s)), b1: head.b1.map(s),
    W2: head.W2.map((r) => r.map(s)), b2: head.b2.map(s),
    XA: rows.map((r) => r.xa.map(s)),
    Y: rows.map((r) => s(r.y)),
    HP: rows.map((r) => r.hp.map((v) => v.map(s))),
    sibA: idx.map((j) => siblings(trees.A, j).map(s)),
    sibP: idx.map((j) => trees.P.map((tr) => siblings(tr, j).map(s))),
  };
  return { input, ref, idx, distinctRows: new Set(idx).size, depth: trees.depth };
}
