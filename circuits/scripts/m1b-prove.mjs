// M1b: prove and verify the sizeI-public statement for one sizeI; print measurements as JSON.
// Run in a fresh process per sizeI so that peak memory is attributable (m1b-measure.mjs does).
// Usage (from circuits/):  node scripts/m1b-prove.mjs --n 64 --maxdepth 10 --isize 700 [--reps 3] [--proof-dir <dir>]
// Saves the fused proof to results/m1b/proofs/N<N>-D<D>-I<sizeI>.json (with Solidity calldata).
// --key-dir <dir>: prove with the key in <dir> (main_final.zkey, verification_key.json) instead of the build's own
// (mpc-setup: build/members-var-N64-D10-mpc, the multi-party key); the circuit (wasm) is the build's either way.
// --prover rapidsnark: prove with rapidsnark (RAPIDSNARK = its prover binary; native-prover.mjs), same zkey and witness;
// the output then also holds the prover process's own peak (proverPeakRssMB). Default snarkjs, as M1b and the proof-cost
// tables were measured.
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import * as snarkjs from "snarkjs";

import { ROOT } from "./lib.mjs";
import { loadS2b } from "./members-lib.mjs";
import { membersVarInput, varBuildDir } from "./m1b-lib.mjs";
import { rapidsnarkProve } from "./native-prover.mjs";
import { checkRapidsnark } from "./prover-config.mjs";

const { values: args } = parseArgs({ options: {
  n: { type: "string" }, maxdepth: { type: "string" }, isize: { type: "string" }, reps: { type: "string", default: "3" },
  "proof-dir": { type: "string" }, // default results/m1b/proofs
  "key-dir": { type: "string" },
  prover: { type: "string", default: "snarkjs" },
} });
if (!["snarkjs", "rapidsnark"].includes(args.prover)) throw new Error("--prover snarkjs|rapidsnark");
const rapidsnark = args.prover === "rapidsnark" ? checkRapidsnark(process.env.RAPIDSNARK) : null;
const N = Number(args.n);
const D = Number(args.maxdepth);
const I = Number(args.isize);
const reps = Number(args.reps);
const dir = varBuildDir(N, D);
const wasm = resolve(dir, "main_js", "main.wasm");
const keyDir = args["key-dir"] ? resolve(args["key-dir"]) : dir;
const zkey = resolve(keyDir, "main_final.zkey");
const vkey = JSON.parse(readFileSync(resolve(keyDir, "verification_key.json"), "utf8"));
const outDir = args["proof-dir"] ? resolve(args["proof-dir"]) : resolve(ROOT, "results", "m1b", "proofs");
mkdirSync(outDir, { recursive: true });
const ms = (t0) => performance.now() - t0;

// Public signal positions: C_model, C_fA, gammaA, gammaP[3], seed, sizeI, mask, F
const SIZE_I = 7;
const F_POS = 9;

let t0 = performance.now();
const { input, idx, distinctRows } = await membersVarInput(N, I, D, { data: loadS2b() });
const inputBuildMs = ms(t0);
const runs = [];
let proof, publicSignals;
for (let i = 0; i < reps; i++) {
  const wtns = resolve(dir, "witness.wtns");
  t0 = performance.now();
  await snarkjs.wtns.calculate(input, wasm, wtns);
  const witnessMs = ms(t0);
  let proveMs, proverPeakRssMB = null;
  if (rapidsnark) {
    ({ proof, publicSignals, proveMs, peakRssMB: proverPeakRssMB } = rapidsnarkProve(rapidsnark, zkey, wtns));
  } else {
    t0 = performance.now();
    ({ proof, publicSignals } = await snarkjs.groth16.prove(zkey, wtns));
    proveMs = ms(t0);
  }
  t0 = performance.now();
  if (!(await snarkjs.groth16.verify(vkey, publicSignals, proof))) throw new Error("proof did not verify");
  runs.push({ witnessMs, proveMs, verifyMs: ms(t0), proverPeakRssMB });
}
if (publicSignals[SIZE_I] !== String(I) || publicSignals[F_POS] !== input.F) throw new Error("public signals differ from the input");

// The same proof must fail for any other public sizeI or F.
const tamper = async (pos, v) => { const ps = [...publicSignals]; ps[pos] = String(v); return !(await snarkjs.groth16.verify(vkey, ps, proof)); };
const rejected = {
  sizeIplus1: await tamper(SIZE_I, I + 1),
  sizeIminus1: await tamper(SIZE_I, I - 1),
  sizeIother: await tamper(SIZE_I, I === 1000 ? 700 : 1000),
  Fplus1: await tamper(F_POS, BigInt(input.F) + 1n),
};

const calldata = JSON.parse(`[${await snarkjs.groth16.exportSolidityCallData(proof, publicSignals)}]`);
writeFileSync(resolve(outDir, `N${N}-D${D}-I${I}.json`), JSON.stringify({
  N, maxDepth: D, sizeI: I, mask: 7, F: Number(input.F), indices: idx, proof, publicSignals, calldata,
}, null, 1) + "\n");

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const pick = (k) => runs.map((r) => r[k]);
console.log(JSON.stringify({
  N, maxDepth: D, sizeI: I, reps, F: Number(input.F), distinctRows, duplicateDraws: N - distinctRows,
  inputBuildMs,
  witnessMs: { median: median(pick("witnessMs")), all: pick("witnessMs") },
  proveMs: { median: median(pick("proveMs")), all: pick("proveMs") },
  verifyMs: { median: median(pick("verifyMs")), all: pick("verifyMs") },
  peakRssMB: process.resourceUsage().maxRSS / 1024,
  prover: args.prover, ...(rapidsnark ? { proverPeakRssMB: Math.max(...pick("proverPeakRssMB")), rapidsnark } : {}),
  publicSignals: publicSignals.length, tamperedRejected: rejected,
  zkeyMB: statSync(zkey).size / 2 ** 20, threads: cpus().length,
}));
await globalThis.curve_bn128?.terminate();
process.exit(0);
