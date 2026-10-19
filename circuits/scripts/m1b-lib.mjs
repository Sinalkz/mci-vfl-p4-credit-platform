// M1b helpers for circuits/src/macro_f1_members_var.circom: sizeI is a public input and
// every member tree has depth maxDepth (D106). Everything else is the S2b helpers in
// members-lib.mjs, imported, not copied.
import { createHash } from "node:crypto";
import { closeSync, openSync, readSync } from "node:fs";
import { resolve } from "node:path";

import { BUILD } from "./lib.mjs";
import {
  DEFAULT_SEED, buildTree, faCommitment, headCommitment, leafA, leafP, loadS2b, oracle, sampleIndices, siblings,
} from "./members-lib.mjs";

export const varBuildDir = (n, maxDepth) => resolve(BUILD, `members-var-N${n}-D${maxDepth}`);
export const varVerifierName = (n, maxDepth) => `MembersVarVerifierN${n}D${maxDepth}`;

/** sha256 (hex) of a file read `chunk` bytes at a time: readFileSync refuses files over 2 GiB (the N 512 r1cs is 2.64 GB). */
export function fileSha256(path, chunk = 1 << 26) {
  const h = createHash("sha256");
  const buf = Buffer.alloc(chunk);
  const fd = openSync(path, "r");
  try {
    for (let n; (n = readSync(fd, buf, 0, chunk, null)) > 0;) h.update(buf.subarray(0, n));
  } finally { closeSync(fd); }
  return h.digest("hex");
}

const treeCache = new WeakMap(); // data object -> ("sizeI:maxDepth" -> trees)
/** The four member trees over the first sizeI aligned IDs, at depth maxDepth (zero-padded). */
export async function varTrees(data, sizeI, maxDepth) {
  if (sizeI < 1 || sizeI > 2 ** maxDepth) throw new Error(`sizeI ${sizeI} outside [1, 2^${maxDepth}]`);
  if (!treeCache.has(data)) treeCache.set(data, new Map());
  const cache = treeCache.get(data);
  const key = `${sizeI}:${maxDepth}`;
  if (cache.has(key)) return cache.get(key);
  const t = data.table;
  const A = await buildTree(t.y.slice(0, sizeI).map((y, j) => leafA(t.xa[j], y)), maxDepth);
  const P = [];
  for (const hp of t.hp) P.push(await buildTree(hp.slice(0, sizeI).map(leafP), maxDepth));
  const trees = { depth: maxDepth, A, P };
  cache.set(key, trees);
  return trees;
}

/**
 * Circuit input for N draws from the first sizeI aligned IDs, trees at depth maxDepth.
 * `rowsOverride(rows, idx, trees)` may replace rows/positions (negative tests); F is then
 * recomputed by the oracle on the rows actually used, so a rejection is due to the
 * membership or index constraints, not to F.
 */
export async function membersVarInput(n, sizeI, maxDepth,
  { mask = 7, faKey = "fA", seed = DEFAULT_SEED, data = loadS2b(), positions = null } = {}) {
  const head = Object.values(data.heads).find((h) => h.mask === mask);
  if (!head) throw new Error(`no head for mask ${mask}`);
  const fa = data[faKey];
  const trees = await varTrees(data, sizeI, maxDepth);
  const idx = await sampleIndices(seed, n, sizeI);
  const pos = positions ?? idx; // tree positions whose rows are used
  const t = data.table;
  const PAD = { y: 0, xa: Array(8).fill(-32768), hp: t.hp.map(() => Array(4).fill(-32768)) }; // packs to leaf 0
  const rows = pos.map((j) => (j < sizeI ? { y: t.y[j], xa: t.xa[j], hp: t.hp.map((hp) => hp[j]) } : PAD));
  const ref = oracle({ fA: fa, head, mask, rows });
  const s = (v) => String(v);
  const input = {
    C_model: s(await headCommitment(head)),
    C_fA: s(await faCommitment(fa)),
    gammaA: s(trees.A.root),
    gammaP: trees.P.map((tr) => s(tr.root)),
    seed: s(seed),
    sizeI: s(sizeI),
    mask: s(mask),
    F: s(ref.F),
    W0: fa.W0.map((r) => r.map(s)), b0: fa.b0.map(s),
    W1: head.W1.map((r) => r.map(s)), b1: head.b1.map(s),
    W2: head.W2.map((r) => r.map(s)), b2: head.b2.map(s),
    XA: rows.map((r) => r.xa.map(s)),
    Y: rows.map((r) => s(r.y)),
    HP: rows.map((r) => r.hp.map((v) => v.map(s))),
    sibA: pos.map((j) => siblings(trees.A, j).map(s)),
    sibP: pos.map((j) => trees.P.map((tr) => siblings(tr, j).map(s))),
  };
  return { input, ref, idx, distinctRows: new Set(idx).size, depth: maxDepth };
}
