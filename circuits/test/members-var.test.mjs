// M1b soundness tests of macro_f1_members_var.circom (sizeI public, depth maxDepth) at
// N = 8, maxDepth = 5 (sizeI up to 32).
// Usage (from circuits/): node scripts/m1b-build.mjs --n 8 --maxdepth 5 --compile-only
//                         node --test test/members-var.test.mjs
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { after, describe, it } from "node:test";
import * as snarkjs from "snarkjs";

import { DEFAULT_SEED, loadS2b, sampleIndices, siblings } from "../scripts/members-lib.mjs";
import { membersVarInput, varBuildDir, varTrees } from "../scripts/m1b-lib.mjs";

const N = 8;
const D = 5;
const dir = varBuildDir(N, D);
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

describe("M1b members circuit, sizeI public (N = 8, maxDepth = 5)", { skip: !existsSync(wasm) && "build first" }, () => {
  it("one circuit accepts every sizeI from 1 to 2^maxDepth; F equals the oracle", async () => {
    for (const sizeI of [1, 7, 20, 31, 32]) {
      const { input, ref } = await membersVarInput(N, sizeI, D, { data: base });
      assert.equal(input.F, String(ref.F));
      assert.equal(await ok(input), true, `sizeI ${sizeI}`);
    }
  });

  it("fused, leave-one-out and baseline at sizeI = 20", async () => {
    for (const mask of [7, 6, 5, 3, 0]) {
      const { input } = await membersVarInput(N, 20, D, { mask, data: base });
      assert.equal(await ok(input), true, `mask ${mask}`);
    }
  });

  it("the circuit's indices are the S2b rule with modulus sizeI", async () => {
    // A witness exists only with the rows at these positions (see the next tests).
    const { idx } = await membersVarInput(N, 20, D, { data: base });
    assert.deepEqual(idx, await sampleIndices(DEFAULT_SEED, N, 20));
    assert.ok(idx.every((j) => j >= 0 && j < 20));
  });

  it("rejects a public sizeI other than the one the rows were drawn with", async () => {
    const { input } = await membersVarInput(N, 20, D, { data: base });
    for (const s of ["19", "21"]) assert.equal(await ok({ ...input, sizeI: s }), false, `sizeI ${s}`);
  });

  it("rejects sizeI = 0 and sizeI = 2^maxDepth + 1", async () => {
    const { input } = await membersVarInput(N, 32, D, { data: base });
    assert.equal(await ok({ ...input, sizeI: "0" }), false);
    assert.equal(await ok({ ...input, sizeI: "33" }), false);
  });

  it("rejects a zero padding leaf at idx + sizeI claimed as a row", async () => {
    // Positions >= sizeI hold leaf 0, which is also the packing of a row with every
    // value at -2^15 and label 0. Here the witness generator still computes r = idx, so
    // the Merkle path check fails; that a forged r = idx + sizeI is stopped by r < sizeI
    // is tested on an edited witness in sample-index-var.test.mjs.
    const sizeI = 20;
    const idx = await sampleIndices(DEFAULT_SEED, N, sizeI);
    const k = idx.findIndex((j) => j + sizeI < 2 ** D);
    assert.ok(k >= 0, "some draw has idx + sizeI inside the tree");
    const positions = [...idx];
    positions[k] = idx[k] + sizeI; // same low bits mod sizeI, a padding position
    const { input } = await membersVarInput(N, sizeI, D, { data: base, positions });
    assert.equal(input.XA[k][0], "-32768");
    assert.equal(await ok(input), false);
  });

  it("rejects a genuine row at a position other than the seed-derived index", async () => {
    const sizeI = 20;
    const idx = await sampleIndices(DEFAULT_SEED, N, sizeI);
    const positions = [...idx];
    positions[0] = (idx[0] + 1) % sizeI;
    const { input } = await membersVarInput(N, sizeI, D, { data: base, positions });
    assert.equal(await ok(input), false);
  });

  it("rejects a tampered sibling, and roots of a tree of another depth", async () => {
    const { input } = await membersVarInput(N, 20, D, { data: base });
    const bad = clone(input);
    bad.sibA[2][4] = String(BigInt(bad.sibA[2][4]) + 1n);
    assert.equal(await ok(bad), false);
    // Roots over the same 20 rows at depth ceil(log2 20) = 5 coincide with maxDepth here;
    // use maxDepth 6 trees with this circuit's depth-5 paths: the roots differ.
    const deeper = await varTrees(base, 20, 6);
    assert.notEqual(String(deeper.A.root), input.gammaA);
    assert.equal(await ok({ ...input, gammaA: String(deeper.A.root) }), false);
    assert.equal(siblings(deeper.A, 0).length, 6);
  });
});
