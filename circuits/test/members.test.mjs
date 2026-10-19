// Soundness tests of the S2b circuit (member roots + seed-derived indices) at N = 8, |I| = 16.
// Usage (from circuits/): node scripts/build.mjs --n 8 --members --isize 16 --compile-only
//                         node --test test/members.test.mjs
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { after, describe, it } from "node:test";
import * as snarkjs from "snarkjs";

import { membersBuildDir } from "../scripts/lib.mjs";
import { DEFAULT_SEED, faCommitment, loadS2b, memberTrees, membersInput, oracle, siblings }
  from "../scripts/members-lib.mjs";

const N = 8;
const I = 16;
const dir = membersBuildDir(N, I);
const wasm = resolve(dir, "main_js", "main.wasm");
const r1cs = resolve(dir, "main.r1cs");
const base = loadS2b();
const clone = (o) => JSON.parse(JSON.stringify(o));

let seq = 0;
async function ok(input) {
  const wtns = resolve(dir, `t_${seq++}.wtns`);
  try {
    await snarkjs.wtns.calculate(input, wasm, wtns);
  } catch {
    return false;
  }
  return snarkjs.wtns.check(r1cs, wtns, { info() {}, warn() {}, error() {} });
}
after(async () => { await globalThis.curve_bn128?.terminate(); });

describe("S2b members circuit (N = 8, |I| = 16)", { skip: !existsSync(wasm) && "build first" }, () => {
  it("fused, leave-one-out and baseline: witness satisfies the R1CS and F equals the oracle", async () => {
    for (const mask of [7, 6, 5, 3, 0]) {
      const { input, ref, distinctRows } = await membersInput(N, I, { mask, data: base });
      assert.equal(input.F, String(ref.F));
      assert.equal(await ok(input), true, `mask ${mask}`);
      if (mask === 7) assert.ok(distinctRows < N, "this seed draws a duplicate index at |I| = 16 (kept by design)");
    }
  });

  it("rejects F +/- 1", async () => {
    const { input } = await membersInput(N, I, { data: base });
    for (const d of [-1n, 1n]) assert.equal(await ok({ ...input, F: String(BigInt(input.F) + d) }), false);
  });

  it("rejects a row that is in the tree but not at the seed-derived index", async () => {
    const { input, idx } = await membersInput(N, I, { data: base });
    const trees = await memberTrees(base, I);
    const j = (idx[0] + 1) % I; // a genuine row with a genuine path, at the wrong position
    const bad = clone(input);
    bad.XA[0] = base.table.xa[j].map(String);
    bad.Y[0] = String(base.table.y[j]);
    bad.HP[0] = base.table.hp.map((hp) => hp[j].map(String));
    bad.sibA[0] = siblings(trees.A, j).map(String);
    bad.sibP[0] = trees.P.map((t) => siblings(t, j).map(String));
    assert.equal(await ok(bad), false);
  });

  it("rejects a tampered sibling, feature, label or embedding value", async () => {
    const { input } = await membersInput(N, I, { data: base });
    const muts = [
      (x) => { x.sibA[3][1] = String(BigInt(x.sibA[3][1]) + 1n); },
      (x) => { x.sibP[2][1][0] = String(BigInt(x.sibP[2][1][0]) + 1n); },
      (x) => { x.XA[5][2] = String(BigInt(x.XA[5][2]) + 1n); },
      (x) => { x.Y[4] = String((Number(x.Y[4]) + 1) % 5); },
      (x) => { x.HP[6][0][3] = String(BigInt(x.HP[6][0][3]) + 1n); },
    ];
    for (const m of muts) { const bad = clone(input); m(bad); assert.equal(await ok(bad), false); }
  });

  it("rejects a different seed or root for the same private data", async () => {
    const { input } = await membersInput(N, I, { data: base });
    assert.equal(await ok({ ...input, seed: String(DEFAULT_SEED + 1n) }), false);
    assert.equal(await ok({ ...input, gammaA: input.gammaP[0] }), false);
    assert.equal(await ok({ ...input, gammaP: [input.gammaP[1], input.gammaP[0], input.gammaP[2]] }), false);
  });

  it("binds the bottom model: C_fA must match the weights used", async () => {
    const { input } = await membersInput(N, I, { data: base });
    const altC = String(await faCommitment(base.fAalt));
    assert.equal(await ok({ ...input, C_fA: altC }), false);             // wrong commitment
    const alt = await membersInput(N, I, { faKey: "fAalt", data: base });  // consistent, other model
    assert.equal(alt.input.C_fA, altC);
    assert.equal(await ok(alt.input), true);                              // valid on its own terms
  });

  it("the mask changes the computation: fused head with mask 0 is a different statement", async () => {
    const head = Object.values(base.heads).find((h) => h.mask === 7);
    // Deterministically find a seed whose rows are classified differently once
    // the passive parties are masked out (on few rows both F can be 10000).
    let input;
    let f0;
    for (let s = DEFAULT_SEED + 1n; s < DEFAULT_SEED + 200n; s++) {
      ({ input } = await membersInput(N, I, { mask: 7, seed: s, data: base }));
      const rows = input.Y.map((y, n) => ({ y: Number(y), xa: input.XA[n].map(Number), hp: input.HP[n].map((v) => v.map(Number)) }));
      f0 = oracle({ fA: base.fA, head, mask: 0, rows }).F;
      if (f0 !== Number(input.F)) break;
    }
    assert.notEqual(f0, Number(input.F), "no seed found where masking changes F");
    assert.equal(await ok(input), true);
    assert.equal(await ok({ ...input, mask: "0" }), false);
    assert.equal(await ok({ ...input, mask: "0", F: String(f0) }), true);
  });

  it("rejects out-of-range features even with honestly recomputed roots", async () => {
    const { idx } = await membersInput(N, I, { data: base });
    const d = clone(base);
    d.table.xa[idx[0]][0] = 32768; // int16 max is 32767
    const { input } = await membersInput(N, I, { data: d });
    assert.equal(await ok(input), false);
  });
});
