// RESULTS-PROOFCOST: one timed operation in its own process, so that its peak memory is attributable.
//   node [--import ./scripts/threads-preload.mjs] scripts/prove-one.mjs witness <input.json> <main.wasm> <out.wtns>
//   node [--import ./scripts/threads-preload.mjs] scripts/prove-one.mjs prove <main_final.zkey> <in.wtns> <out.proof.json>
// Prints one JSON line: { op, ms, peakRssMB, workers } (workers = the snarkjs thread pool actually used; 0 = none built).
// prove writes { proof, publicSignals } for verification by the caller.
import { readFileSync, writeFileSync } from "node:fs";
import * as snarkjs from "snarkjs";

const [op, a, b, out] = process.argv.slice(2);
const t0 = performance.now();
if (op === "witness") {
  await snarkjs.wtns.calculate(JSON.parse(readFileSync(a, "utf8")), b, out);
} else if (op === "prove") {
  const { proof, publicSignals } = await snarkjs.groth16.prove(a, b);
  writeFileSync(out, JSON.stringify({ proof, publicSignals }));
} else throw new Error("op: witness | prove");
const ms = performance.now() - t0;
const tm = globalThis.curve_bn128?.tm;
console.log(JSON.stringify({ op, ms, peakRssMB: process.resourceUsage().maxRSS / 1024, workers: tm ? (tm.singleThread ? 1 : tm.concurrency) : 0 }));
await globalThis.curve_bn128?.terminate();
process.exit(0);
