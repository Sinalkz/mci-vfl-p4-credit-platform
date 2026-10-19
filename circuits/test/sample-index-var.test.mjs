// M1b: the one new kind of constraint, r < sizeI in SampleIndexVar, tested in isolation.
// The honest witness generator always computes r = low mod sizeI, so a malicious r can
// only be tested by editing a witness. This test compiles SampleIndexVar(5, K) unoptimised
// (every signal keeps its name), takes an honest witness, substitutes the forgery
// q' = q - 1, r' = r + sizeI (same low, a padding position of the tree), recomputes every
// signal that depends on q or r, and evaluates the R1CS constraint by constraint.
// Expected: the only violated constraint is lt.out === 1.
// Usage (from circuits/): node --test test/sample-index-var.test.mjs   (compiles itself, ~seconds)
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { after, before, describe, it } from "node:test";
import * as snarkjs from "snarkjs";

import { BUILD, CIRCUITS } from "../scripts/lib.mjs";
import { FIELD_P, sampleIndices } from "../scripts/members-lib.mjs";

const D = 5;
const K = 3;
const SIZE = 20n;
const dir = resolve(BUILD, "probe-sample-index-var");
const mod = (x) => ((x % FIELD_P) + FIELD_P) % FIELD_P;
let r1cs, sym, seed;

after(async () => { await globalThis.curve_bn128?.terminate(); });

before(async () => {
  mkdirSync(dir, { recursive: true });
  const main = resolve(dir, "main.circom");
  writeFileSync(main, `pragma circom 2.1.9;\ninclude "macro_f1_members_var.circom";\n` +
    `component main {public [seed, sizeI]} = SampleIndexVar(${D}, ${K});\n`);
  execFileSync(process.env.CIRCOM ?? "circom", [main, "--r1cs", "--wasm", "--sym", "--O0", "-o", dir,
    "-l", resolve(CIRCUITS, "src"), "-l", resolve(CIRCUITS, "node_modules")], { encoding: "utf8" });
  r1cs = await snarkjs.r1cs.exportJson(resolve(dir, "main.r1cs"), { info() {}, warn() {}, error() {}, debug() {} });
  sym = new Map();
  for (const line of readFileSync(resolve(dir, "main.sym"), "utf8").trim().split(/\r?\n/)) {
    const [, w, , name] = line.split(",");
    if (Number(w) >= 0) sym.set(name, Number(w));
  }
  // A seed whose draw K lands low enough that r + sizeI is still a tree position (< 2^D).
  for (let s = 1n; ; s++) {
    if ((await sampleIndices(s, K + 1, SIZE))[K] + Number(SIZE) < 2 ** D) { seed = s; break; }
  }
});

async function witness(input) {
  const f = resolve(dir, "w.wtns");
  await snarkjs.wtns.calculate(input, resolve(dir, "main_js", "main.wasm"), f);
  return (await snarkjs.wtns.exportJson(f)).map(BigInt);
}

/** Indices of the constraints the witness violates. */
function violated(w) {
  const lc = (m) => Object.entries(m).reduce((acc, [i, c]) => mod(acc + BigInt(c) * w[Number(i)]), 0n);
  return r1cs.constraints.map(([a, b, c], i) => (mod(lc(a) * lc(b) - lc(c)) === 0n ? -1 : i)).filter((i) => i >= 0);
}

function set(w, name, value) {
  const i = sym.get(name);
  assert.ok(i !== undefined, `signal ${name} not in the witness`);
  w[i] = mod(value);
}
const get = (w, name) => w[sym.get(name)];
const setBits = (w, prefix, value, n) => { for (let i = 0; i < n; i++) set(w, `${prefix}[${i}]`, (value >> BigInt(i)) & 1n); };

describe("M1b SampleIndexVar: r < sizeI is what stops a padding position", () => {
  it("the honest witness satisfies every constraint and outputs the S2b index", async () => {
    const w = await witness({ seed: String(seed), sizeI: String(SIZE) });
    assert.deepEqual(violated(w), []);
    const idx = (await sampleIndices(seed, K + 1, SIZE))[K];
    let out = 0;
    for (let i = 0; i < D; i++) out += Number(get(w, `main.bits[${i}]`)) << i;
    assert.equal(out, idx);
  });

  it("q - 1, r + sizeI (consistently propagated) violates exactly lt.out === 1", async () => {
    const w = await witness({ seed: String(seed), sizeI: String(SIZE) });
    const q = get(w, "main.q");
    const r = get(w, "main.r");
    const q2 = q - 1n;
    const r2 = r + SIZE;
    assert.ok(r2 < 2n ** BigInt(D), "the forged index is a tree position");
    set(w, "main.q", q2);
    set(w, "main.r", r2);
    set(w, "main.qs", q2 * SIZE);                 // low === qs + r still holds
    set(w, "main.qRange.in", q2);
    setBits(w, "main.qRange.out", q2, 128);
    set(w, "main.rBits.in", r2);
    setBits(w, "main.rBits.out", r2, D);
    setBits(w, "main.bits", r2, D);               // the path bits would select position r + sizeI
    set(w, "main.lt.in[0]", r2);
    const t = r2 + (1n << BigInt(D + 1)) - SIZE;  // LessThan(D+1): n2b.in = in[0] + 2^(D+1) - in[1]
    set(w, "main.lt.n2b.in", t);
    setBits(w, "main.lt.n2b.out", t, D + 2);
    set(w, "main.lt.out", 1n - ((t >> BigInt(D + 1)) & 1n)); // = 0, since r' >= sizeI
    const bad = violated(w);
    assert.equal(bad.length, 1, `violated: ${bad}`);
    const [a, b, c] = r1cs.constraints[bad[0]];
    const ltOut = String(sym.get("main.lt.out"));
    assert.ok(ltOut in a || ltOut in b || ltOut in c, "the violated constraint is the one on lt.out");
    // Setting lt.out to 1 instead moves the violation into LessThan's own definition
    // (out = 1 - n2b.out[D+1]): no assignment of lt.out satisfies both.
    set(w, "main.lt.out", 1n);
    assert.ok(violated(w).length >= 1);
  });
});
