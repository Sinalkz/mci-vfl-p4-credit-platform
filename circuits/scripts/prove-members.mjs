// Prove and verify the S2b statement for one (N, |I|); print measurements as JSON.
// Run in a fresh process per configuration so that peak memory is attributable.
// Usage (from circuits/):
//   node scripts/prove-members.mjs --n 64 --isize 1000 [--reps 3] [--export-all]
// --export-all also proves the leave-one-out and baseline figures and one proof with
// a different bottom model (fAalt), for the contract tests; all go to
// results/s2b/proofs/N<N>-I<I>/.
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import * as snarkjs from "snarkjs";

import { ROOT, membersBuildDir } from "./lib.mjs";
import { loadS2b, membersInput } from "./members-lib.mjs";

const { values: args } = parseArgs({ options: {
  n: { type: "string" }, isize: { type: "string" }, reps: { type: "string", default: "3" },
  "export-all": { type: "boolean", default: false },
} });
const N = Number(args.n);
const I = Number(args.isize);
const reps = Number(args.reps);
const dir = membersBuildDir(N, I);
const wasm = resolve(dir, "main_js", "main.wasm");
const zkey = resolve(dir, "main_final.zkey");
const vkey = JSON.parse(readFileSync(resolve(dir, "verification_key.json"), "utf8"));
const outDir = resolve(ROOT, "results", "s2b", "proofs", `N${N}-I${I}`);
mkdirSync(outDir, { recursive: true });
const ms = (t0) => performance.now() - t0;
const data = loadS2b();

async function proveOnce(input) {
  const wtns = resolve(dir, "witness.wtns");
  let t0 = performance.now();
  await snarkjs.wtns.calculate(input, wasm, wtns);
  const witnessMs = ms(t0);
  t0 = performance.now();
  const { proof, publicSignals } = await snarkjs.groth16.prove(zkey, wtns);
  const proveMs = ms(t0);
  t0 = performance.now();
  const ok = await snarkjs.groth16.verify(vkey, publicSignals, proof);
  const verifyMs = ms(t0);
  if (!ok) throw new Error("proof did not verify");
  return { proof, publicSignals, witnessMs, proveMs, verifyMs };
}
async function save(name, r, extra) {
  const calldata = JSON.parse(`[${await snarkjs.groth16.exportSolidityCallData(r.proof, r.publicSignals)}]`);
  writeFileSync(resolve(outDir, `${name}.json`),
    JSON.stringify({ N, isize: I, ...extra, proof: r.proof, publicSignals: r.publicSignals, calldata }, null, 1) + "\n");
}

// ---- fused figure, timed -------------------------------------------------------
let t0 = performance.now();
const fused = await membersInput(N, I, { mask: 7, data });
const inputMs = ms(t0); // trees over |I| rows x 4 parties, indices, commitments, oracle
const runs = [];
let last;
for (let i = 0; i < reps; i++) {
  last = await proveOnce(fused.input);
  runs.push({ witnessMs: last.witnessMs, proveMs: last.proveMs, verifyMs: last.verifyMs });
}
if (last.publicSignals[8] !== fused.input.F) throw new Error("public F differs from the oracle");
const tampered = [...last.publicSignals];
tampered[8] = String(BigInt(tampered[8]) + 1n);
const tamperedRejected = !(await snarkjs.groth16.verify(vkey, tampered, last.proof));
await save("fused", last, { mask: 7, fa: "fA", F: Number(fused.input.F) });

// ---- the other figures of a run, and a proof about a different bottom model --------
const figures = { fused: Number(fused.input.F) };
if (args["export-all"]) {
  for (const [name, mask] of [["loo1", 6], ["loo2", 5], ["loo3", 3], ["baseline", 0]]) {
    const { input } = await membersInput(N, I, { mask, data });
    await save(name, await proveOnce(input), { mask, fa: "fA", F: Number(input.F) });
    figures[name] = Number(input.F);
  }
  const alt = await membersInput(N, I, { mask: 7, faKey: "fAalt", data });
  await save("fused-other-fA", await proveOnce(alt.input), { mask: 7, fa: "fAalt", F: Number(alt.input.F) });
}

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const pick = (k) => runs.map((r) => r[k]);
const result = {
  N, isize: I, depth: fused.depth, reps,
  distinctRows: fused.distinctRows, duplicateDraws: N - fused.distinctRows,
  figures,
  inputBuildMs: inputMs,
  witnessMs: { median: median(pick("witnessMs")), all: pick("witnessMs") },
  proveMs: { median: median(pick("proveMs")), all: pick("proveMs") },
  verifyMs: { median: median(pick("verifyMs")), all: pick("verifyMs") },
  peakRssMB: process.resourceUsage().maxRSS / 1024,
  proofCalldataBytes: 8 * 32,
  publicSignals: last.publicSignals.length,
  tamperedFRejected: tamperedRejected,
  zkeyMB: statSync(zkey).size / 2 ** 20,
  threads: cpus().length,
};
console.log(JSON.stringify(result));
await globalThis.curve_bn128?.terminate();
process.exit(0);
