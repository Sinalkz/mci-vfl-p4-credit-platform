// Correctness and soundness smoke tests of the macro-F1 circuit at N = 4.
// Usage (from circuits/): node scripts/build.mjs --n 4 --compile-only && node --test test/
//
// Positive cases must produce a witness that satisfies the R1CS, with F equal
// to the Python oracle. Negative cases (wrong F, tampered data, out-of-range
// values with *recomputed* commitments) must fail witness generation.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { after, describe, it } from "node:test";
import * as snarkjs from "snarkjs";

import { ROOT, buildDir, evalCommitmentChain, loadModelEval, modelCommitment } from "../scripts/lib.mjs";

const N = 4;
const dir = buildDir(N);
const wasm = resolve(dir, "main_js", "main.wasm");
const r1cs = resolve(dir, "main.r1cs");
const PY = resolve(ROOT, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const me = loadModelEval();
const { D, H, K } = me.meta;

function oracle(model, rows, labels) {
  const out = execFileSync(PY, [resolve(ROOT, "py", "s2_oracle.py")], {
    input: JSON.stringify({ model, rows, labels }), cwd: resolve(ROOT, "py"), encoding: "utf8",
  });
  return JSON.parse(out);
}

async function makeInput(model, rows, labels, F) {
  const s = (v) => String(v);
  return {
    C_model: s(await modelCommitment(model, D, H, K)),
    C_eval: s(await evalCommitmentChain(rows, labels)),
    F: s(F),
    W1: model.W1.map((r) => r.map(s)), b1: model.b1.map(s),
    W2: model.W2.map((r) => r.map(s)), b2: model.b2.map(s),
    X: rows.map((r) => r.map(s)), Y: labels.map(s),
  };
}

let seq = 0;
async function witnessOk(input) {
  const wtns = resolve(dir, `test_${seq++}.wtns`);
  try {
    await snarkjs.wtns.calculate(input, wasm, wtns);
  } catch {
    return false;
  }
  return snarkjs.wtns.check(r1cs, wtns, { info() {}, warn() {}, error() {} });
}

// snarkjs caches a multi-threaded curve; stop its workers so the process can exit.
after(async () => { await globalThis.curve_bn128?.terminate(); });

const clone = (o) => JSON.parse(JSON.stringify(o));
const baseRows = me.rows.slice(0, N);
const baseLabels = me.labels.slice(0, N);

describe("macro-F1 circuit (N = 4)", { skip: !existsSync(wasm) && "build N=4 first" }, () => {
  it("real rows: witness satisfies the R1CS and F equals the oracle", async () => {
    const ref = oracle(me.model, baseRows, baseLabels);
    assert.equal(ref.F, me.perN["4"].F);
    assert.equal(await witnessOk(await makeInput(me.model, baseRows, baseLabels, ref.F)), true);
  });

  it("all logits tied: argmax takes the smallest index (class 0)", async () => {
    const m = clone(me.model);
    m.W2 = m.W2.map((r) => r.map(() => 0));
    m.b2 = m.b2.map(() => 12345);
    const labels = [0, 1, 0, 3];
    const ref = oracle(m, baseRows, labels);
    assert.deepEqual(ref.preds, [0, 0, 0, 0]);
    assert.equal(await witnessOk(await makeInput(m, baseRows, labels, ref.F)), true);
    assert.equal(await witnessOk(await makeInput(m, baseRows, labels, ref.F + 1)), false);
  });

  it("empty classes are excluded from the mean", async () => {
    const labels = [1, 1, 1, 1];
    const ref = oracle(me.model, baseRows, labels);
    assert.ok(ref.emptyClasses.length >= 1);
    assert.equal(await witnessOk(await makeInput(me.model, baseRows, labels, ref.F)), true);
  });

  it("rejects a wrong F (one unit up or down)", async () => {
    const F = me.perN["4"].F;
    for (const bad of [F - 1, F + 1]) {
      assert.equal(await witnessOk(await makeInput(me.model, baseRows, baseLabels, bad)), false);
    }
  });

  it("rejects weights or rows that do not match the public commitments", async () => {
    const good = await makeInput(me.model, baseRows, baseLabels, me.perN["4"].F);
    const w = clone(good); w.W1[3][5] = String(BigInt(w.W1[3][5]) + 1n);
    const x = clone(good); x.X[2][7] = String(BigInt(x.X[2][7]) + 1n);
    const y = clone(good); y.Y[0] = String((Number(y.Y[0]) + 1) % K);
    for (const bad of [w, x, y]) assert.equal(await witnessOk(bad), false);
  });

  it("rejects out-of-range values even with honestly recomputed commitments", async () => {
    const rows = clone(baseRows); rows[0][0] = 32768;          // x must be int16
    const m = clone(me.model); m.W1[0][0] = -32769;            // W1 must be int16
    const m2 = clone(me.model); m2.b2[0] = 2 ** 47;             // b2 must be int48
    const labels = clone(baseLabels); labels[1] = 5;           // label must be < K
    const F = me.perN["4"].F;
    assert.equal(await witnessOk(await makeInput(me.model, rows, baseLabels, F)), false);
    assert.equal(await witnessOk(await makeInput(m, baseRows, baseLabels, F)), false);
    assert.equal(await witnessOk(await makeInput(m2, baseRows, baseLabels, F)), false);
    assert.equal(await witnessOk(await makeInput(me.model, baseRows, labels, F)), false);
  });
});
