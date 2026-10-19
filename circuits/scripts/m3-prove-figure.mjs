// M3 §5.1: one Groth16 proof of the M1b circuit (N = 64, maxDepth 10, or 14 for the full data), in its own process so
// that its peak memory is attributable. Usage (from anywhere): node circuits/scripts/m3-prove-figure.mjs <input.json> <out.json>
// Called by contracts/offchain/scoring.mjs (proveInChild).
// Writes { proof8, publicSignals, witnessMs, proveMs, verifyMs, peakRssMB, prover, proverPeakRssMB, maxDepth }; proof8 is the
// Solidity calldata order [a0, a1, b00, b01, b10, b11, c0, c1].
// Key and prover (scripts/prover-config.mjs): by default the multi-party key (build/members-var-N64-D10-mpc) and
// rapidsnark (RAPIDSNARK = its prover binary; native-prover.mjs), same zkey and witness format as snarkjs;
// proverPeakRssMB is then that process's peak, else null. VFL_KEY_DIR names another key folder (the development key:
// build/members-var-N64-D10), VFL_PROVER=snarkjs proves with snarkjs. A missing rapidsnark binary fails at once.
// The circuit follows the input: maxDepth = the number of Merkle siblings per row (sibA), so the witness comes from
// build/members-var-N64-D<maxDepth>/main_js/main.wasm, and without VFL_KEY_DIR the key is that circuit's multi-party key.
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import * as snarkjs from "snarkjs";

import { varBuildDir } from "./m1b-lib.mjs";
import { rapidsnarkProve } from "./native-prover.mjs";
import { checkKeyDir, proverFromEnv } from "./prover-config.mjs";

const [inFile, outFile] = process.argv.slice(2);
proverFromEnv(process.env); // a missing rapidsnark binary fails here, before the input is read
const input = JSON.parse(readFileSync(inFile, "utf8"));
const maxDepth = input.sibA[0].length;
const dir = varBuildDir(64, maxDepth);
if (!existsSync(resolve(dir, "main_js", "main.wasm"))) throw new Error(`no circuit for maxDepth ${maxDepth}: ${dir}/main_js/main.wasm (scripts/m1b-build.mjs)`);
const { keyDir, prover, rapidsnark } = proverFromEnv(process.env, { maxDepth });
checkKeyDir(keyDir);
const wtns = outFile.replace(/\.json$/, ".wtns");
let t0 = performance.now();
await snarkjs.wtns.calculate(input, resolve(dir, "main_js", "main.wasm"), wtns);
const witnessMs = performance.now() - t0;
let proof, publicSignals, proveMs, proverPeakRssMB = null;
if (prover === "rapidsnark") {
  ({ proof, publicSignals, proveMs, peakRssMB: proverPeakRssMB } =
    rapidsnarkProve(rapidsnark, resolve(keyDir, "main_final.zkey"), wtns));
} else {
  t0 = performance.now();
  ({ proof, publicSignals } = await snarkjs.groth16.prove(resolve(keyDir, "main_final.zkey"), wtns));
  proveMs = performance.now() - t0;
}
rmSync(wtns, { force: true }); // tens of MB; not needed after proving
t0 = performance.now();
const vkey = JSON.parse(readFileSync(resolve(keyDir, "verification_key.json"), "utf8"));
if (!(await snarkjs.groth16.verify(vkey, publicSignals, proof))) throw new Error("proof does not verify");
const verifyMs = performance.now() - t0;
const cd = JSON.parse(`[${await snarkjs.groth16.exportSolidityCallData(proof, publicSignals)}]`);
const proof8 = [...cd[0], ...cd[1][0], ...cd[1][1], ...cd[2]].map((x) => BigInt(x).toString());
writeFileSync(outFile, JSON.stringify({ proof8, publicSignals, witnessMs, proveMs, verifyMs,
  peakRssMB: process.resourceUsage().maxRSS / 1024, prover, proverPeakRssMB, maxDepth }) + "\n");
await globalThis.curve_bn128?.terminate();
process.exit(0);
