// The multi-party ceremony's steps (scripts/mpc-step.mjs) on a tiny circuit: setup, two contributions with their own
// entropy, verify against the r1cs and the ptau (a tampered key must fail), export of a verifier that
// accepts a proof made with the final key. Uses the smallest phase-1 file present (2^19). Also the ceremony's
// parametrization (scripts/mpc-ceremony.mjs: circuit, folders, contract name, the verify memory guard) and its fit check,
// which must agree with the domain snarkjs's own setup chose.
// Usage (from circuits/):  node --test test/mpc-step.test.mjs
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { after, describe, it } from "node:test";
import * as snarkjs from "snarkjs";

import { BUILD, CIRCUITS, ROOT } from "../scripts/lib.mjs";
import { ceremonyConfig, groth16Fit, r1csHeader } from "../scripts/mpc-ceremony.mjs";

const DIR = resolve(BUILD, `test-mpc-step-${process.pid}`); // per process: build/ may be shared by the repo's worktrees
const PTAU = resolve(CIRCUITS, "ptau", "ppot_0080_19.ptau");
const step = (...a) => {
  const r = spawnSync(process.execPath, [resolve(CIRCUITS, "scripts", "mpc-step.mjs"), ...a], { cwd: CIRCUITS, encoding: "utf8" });
  return { status: r.status, out: r.stdout.trim() ? JSON.parse(r.stdout.trim().split("\n").pop()) : null };
};

describe("mpc-step", () => {
  rmSync(DIR, { recursive: true, force: true });
  mkdirSync(DIR, { recursive: true });
  writeFileSync(resolve(DIR, "main.circom"),
    "pragma circom 2.1.9;\ntemplate M() { signal input a; signal input b; signal output c; c <== a * b; }\ncomponent main = M();\n");
  const c = spawnSync("circom", [resolve(DIR, "main.circom"), "--r1cs", "--wasm", "-o", DIR], { encoding: "utf8" });
  after(async () => { rmSync(DIR, { recursive: true, force: true }); await globalThis.curve_bn128?.terminate(); });
  const r1cs = resolve(DIR, "main.r1cs");
  const z = (k) => resolve(DIR, `main_${k}.zkey`);

  it("compiles the tiny circuit", () => assert.equal(c.status, 0, c.stderr));

  it("runs setup, two contributions, verify and export", async () => {
    const s = step("setup", r1cs, PTAU, z(0));
    assert.equal(s.status, 0);
    assert.ok(s.out.outputBytes > 0 && s.out.peakRssMB > 0 && s.out.ms > 0);
    const h1 = step("contribute", z(0), z(1), "power");
    const h2 = step("contribute", z(1), z(2), "mci");
    assert.equal(h1.status, 0);
    assert.equal(h2.status, 0);
    assert.match(h1.out.contributionHash, /^[0-9a-f]{128}$/);
    assert.notEqual(h1.out.contributionHash, h2.out.contributionHash);
    assert.ok(!JSON.stringify(h1.out).includes("entropy\":\"0x"), "no entropy in the output");
    const v = step("verify", r1cs, PTAU, z(2));
    assert.equal(v.status, 0);
    assert.equal(v.out.verified, true);
    assert.ok(v.out.transcript.some((l) => /power/.test(l)) && v.out.transcript.some((l) => /mci/.test(l)));
    const sol = resolve(DIR, "V.sol");
    const e = step("export", z(2), resolve(DIR, "vk.json"), sol, "TinyVerifier");
    assert.equal(e.status, 0);
    assert.match(readFileSync(sol, "utf8"), /contract TinyVerifier/);
    const { proof, publicSignals } = await snarkjs.groth16.fullProve({ a: 3, b: 11 }, resolve(DIR, "main_js", "main.wasm"), z(2));
    assert.equal(publicSignals[0], "33");
    assert.ok(await snarkjs.groth16.verify(JSON.parse(readFileSync(resolve(DIR, "vk.json"), "utf8")), publicSignals, proof));
  });

  it("verify fails for a tampered key", () => {
    assert.equal(step("setup", r1cs, PTAU, z("x")).status, 0);
    assert.equal(step("contribute", z("x"), z("y"), "power").status, 0);
    const v = step("verify", r1cs, PTAU, z("y"));
    assert.equal(v.status, 0, "an independent setup + contribution still verifies against the r1cs and ptau");
    // tampering: overwrite part of the contributed key; verification must fail
    const buf = readFileSync(z("y"));
    buf.fill(7, buf.length - 512, buf.length - 256);
    writeFileSync(z("y"), buf);
    assert.notEqual(step("verify", r1cs, PTAU, z("y")).status, 0);
  });

  it("fit check: snarkjs's rule gives the domain its own setup chose", async () => {
    const h = r1csHeader(r1cs);
    assert.deepEqual([h.nConstraints, h.nPubInputs, h.nOutputs], [1, 0, 1]);
    const fit = groth16Fit(h, 19);
    const zk = await snarkjs.zKey.exportJson(z(0));
    assert.equal(fit.domain, zk.domainSize);
    assert.equal(fit.cirPower, zk.power);
    assert.ok(fit.fits);
  });
});

describe("mpc-ceremony parametrization", () => {
  it("defaults are the maxDepth 10 run's circuit, folders and contract", () => {
    const c = ceremonyConfig([]);
    assert.equal(c.r1cs, resolve(BUILD, "members-var-N64-D10", "main.r1cs"));
    assert.equal(c.keys, resolve(BUILD, "members-var-N64-D10-mpc"));
    assert.equal(c.contract, "MembersVarVerifierN64D10Mpc");
    assert.equal(c.out, resolve(ROOT, "results", "mpc", "setup"));
    assert.equal(c.guard, null);
  });

  it("maxDepth 14: its own r1cs, keys folder and contract; --out required", () => {
    assert.throws(() => ceremonyConfig(["--maxdepth", "14"]), /--out is required/);
    const c = ceremonyConfig(["--maxdepth", "14", "--out", resolve(ROOT, "results", "full", "setup"), "--verify-guard", "1.31:512"]);
    assert.equal(c.r1cs, resolve(BUILD, "members-var-N64-D14", "main.r1cs"));
    assert.equal(c.keys, resolve(BUILD, "members-var-N64-D14-mpc"));
    assert.equal(c.contract, "MembersVarVerifierN64D14Mpc");
    assert.equal(c.out, resolve(ROOT, "results", "full", "setup"));
    assert.deepEqual(c.guard, { ratio: 1.31, marginMB: 512 });
    assert.throws(() => ceremonyConfig(["--maxdepth", "14", "--out", "x", "--verify-guard", "1.31"]), /verify-guard/);
  });

  it("fit rule: nConstraints + nPublic + 1 <= 2^power", () => {
    const d14 = groth16Fit({ nConstraints: 1_020_525, nPubInputs: 10, nOutputs: 0 }, 20); // results/full/setup/compile-N64-D14.json
    assert.deepEqual([d14.fits, d14.cirPower, d14.rows, d14.marginRows], [true, 20, 1_020_536, 28_040]);
    assert.equal(groth16Fit({ nConstraints: 2 ** 20 - 11, nPubInputs: 10, nOutputs: 0 }, 20).fits, true);  // rows = 2^20
    assert.equal(groth16Fit({ nConstraints: 2 ** 20 - 10, nPubInputs: 10, nOutputs: 0 }, 20).fits, false); // one row more
    assert.equal(groth16Fit({ nConstraints: 1_206_003, nPubInputs: 10, nOutputs: 0 }, 20).fits, false);    // maxDepth 17
  });
});
