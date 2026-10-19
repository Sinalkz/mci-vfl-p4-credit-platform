// Prove and verify the macro-F1 statement for one N; print measurements as JSON.
// Run in a fresh process per N so that peak memory is attributable to that N.
// Usage (from circuits/): node scripts/prove.mjs --n 64 [--reps 3]
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import * as snarkjs from "snarkjs";

import { ROOT, buildDir, circuitInput } from "./lib.mjs";

const { values: args } = parseArgs({ options: { n: { type: "string" }, reps: { type: "string", default: "3" } } });
const N = Number(args.n);
const reps = Number(args.reps);
const dir = buildDir(N);
const wasm = resolve(dir, "main_js", "main.wasm");
const zkey = resolve(dir, "main_final.zkey");
const vkey = JSON.parse(readFileSync(resolve(dir, "verification_key.json"), "utf8"));
const ms = (t0) => performance.now() - t0;

let t0 = performance.now();
const { input, expected } = await circuitInput(N);
const commitmentsMs = ms(t0);
writeFileSync(resolve(dir, "input.json"), JSON.stringify(input));

const runs = [];
let proof, publicSignals;
for (let i = 0; i < reps; i++) {
  const wtns = resolve(dir, "witness.wtns");
  t0 = performance.now();
  await snarkjs.wtns.calculate(input, wasm, wtns);
  const witnessMs = ms(t0);
  t0 = performance.now();
  ({ proof, publicSignals } = await snarkjs.groth16.prove(zkey, wtns));
  const proveMs = ms(t0);
  t0 = performance.now();
  const ok = await snarkjs.groth16.verify(vkey, publicSignals, proof);
  const verifyMs = ms(t0);
  if (!ok) throw new Error("proof did not verify");
  runs.push({ witnessMs, proveMs, verifyMs });
}

// Public signals are [C_model, C_eval, F] in declaration order.
if (publicSignals[2] !== String(expected.F)) throw new Error("public F differs from the reference");
const tampered = [...publicSignals];
tampered[2] = String(BigInt(tampered[2]) + 1n);
const tamperedRejected = !(await snarkjs.groth16.verify(vkey, tampered, proof));

const calldata = JSON.parse(`[${await snarkjs.groth16.exportSolidityCallData(proof, publicSignals)}]`);
const outDir = resolve(ROOT, "results", "s2", "proofs");
mkdirSync(outDir, { recursive: true });
writeFileSync(resolve(outDir, `N${N}.json`), JSON.stringify({ N, proof, publicSignals, calldata }, null, 1) + "\n");

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const pick = (k) => runs.map((r) => r[k]);
const result = {
  N,
  reps,
  F: Number(publicSignals[2]),
  commitmentsMs,
  witnessMs: { median: median(pick("witnessMs")), all: pick("witnessMs") },
  proveMs: { median: median(pick("proveMs")), all: pick("proveMs") },
  verifyMs: { median: median(pick("verifyMs")), all: pick("verifyMs") },
  peakRssMB: process.resourceUsage().maxRSS / 1024,
  proofJsonBytes: Buffer.byteLength(JSON.stringify(proof)),
  proofCalldataBytes: 8 * 32, // A (2), B (2x2), C (2) field elements of 32 bytes
  publicSignalsBytes: publicSignals.length * 32,
  tamperedFRejected: tamperedRejected,
  zkeyMB: statSync(zkey).size / 2 ** 20,
  wasmMB: statSync(wasm).size / 2 ** 20,
  threads: cpus().length,
};
console.log(JSON.stringify(result));
await globalThis.curve_bn128?.terminate();
process.exit(0);
