// Task 4 item 3: may a complete run (runner3.mjs --run e3 --prover snarkjs, Besu by docker-compose as the earlier laptop
// runs) start on this laptop now? The student's rule: only if (1) at least 7 GB is free (7,000 MB, the reading the
// orchestrator applied: 7,062 MB "meets" it) and (2) the estimate fits in what is free:
//   Besu   4 nodes x (the heap cap of chain/docker-compose.yml's BESU_OPTS + 300 MB of JVM non-heap, an assumption);
//          Docker Desktop's WSL 2 VM adds its own overhead, not counted
//   parties  5 processes (A, 3 passives, arbiter) x the largest peak RSS of this task's E3-sized processes
//          (results/overhead/raw: train-bench e3 holds all four parties in one process; psi-bench subset)
//   prover   one snarkjs proof at a time: results/proofcost/prove-N64-D10.json's peak RSS (the deployed circuit,
//          N = 64, maxDepth 10, this laptop)
//   margin   1 GB
// Free RAM = the lowest of 5 os.freemem() samples, one a second. Writes results/overhead/item3-check.json; starts nothing.
//   node offchain/overhead/item3-check.mjs
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { freemem } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { machine } from "../data1/machine.mjs";

const ROOT = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "..");
const RAW = resolve(ROOT, "results", "overhead", "raw");
const json = (...p) => JSON.parse(readFileSync(resolve(ROOT, ...p), "utf8"));
export const MIN_FREE_MB = 7000;
export const JVM_NON_HEAP_MB = 300;
export const MARGIN_MB = 1024;

const prove = json("results", "proofcost", "prove-N64-D10.json");
const e3 = json("results", "data1", "e3", "A", "summary.json");
const compose = readFileSync(resolve(ROOT, "chain", "docker-compose.yml"), "utf8");
const heapMB = Number(/-Xmx(\d+)g/.exec(compose)[1]) * 1024;
const raw = readdirSync(RAW).filter((f) => /^(train-e3-|psi-subset-)/.test(f)).map((f) => JSON.parse(readFileSync(resolve(RAW, f), "utf8")));
const partyPeaks = raw.flatMap((r) => (r.item === 1 ? [r.peakRssMB] : [r.A.peakRssMB, ...r.passives.map((p) => p.peakRssMB)]));
const partyMB = Math.max(...partyPeaks);
const earlier = ["m3/i6", "m3/i7", "m3/i8", "m3/i9", "data1/e3"].map((d) => {
  const r = json("results", ...d.split("/"), "run.json");
  return { run: `results/${d}`, freeRamMBAtRunnerStart: r.machine.freeRamMBAtStart, besu: r.chain.clientVersion,
    launch: r.chain.launch ?? "docker-compose (chain/docker-compose.yml)" };
});

const samples = [];
for (let k = 0; k < 5; k++) { samples.push(Math.round(freemem() / 2 ** 20)); await new Promise((ok) => setTimeout(ok, 1000)); }
const freeMB = Math.min(...samples);
const parts = { besuMB: 4 * (heapMB + JVM_NON_HEAP_MB), partiesMB: 5 * partyMB, proverMB: Math.round(prove.result.peakRssMB), marginMB: MARGIN_MB };
const neededMB = parts.besuMB + parts.partiesMB + parts.proverMB + parts.marginMB;
const cond1 = freeMB >= MIN_FREE_MB;
const cond2 = neededMB <= freeMB;
const out = {
  item: 3, at: new Date().toISOString(), machine: machine(), freeRamSamplesMB: samples, freeRamMB: freeMB,
  rule: { minFreeMB: MIN_FREE_MB, estimateMustFit: true }, estimate: { ...parts, neededMB,
    neededWithoutBesuMB: neededMB - parts.besuMB,
    sources: { besu: `chain/docker-compose.yml BESU_OPTS -Xmx${heapMB / 1024}g x 4 nodes, + ${JVM_NON_HEAP_MB} MB JVM non-heap per node (assumed)`,
      parties: `5 x ${partyMB} MB, the largest peak RSS among ${raw.length} E3-sized measurements of this task`,
      prover: "results/proofcost/prove-N64-D10.json (snarkjs 0.7.6, 12 workers, this laptop)" },
    e3LaptopProverPeaksMB: e3.scoring.map((x) => Math.round(x.peakRssMB)) },
  earlierLaptopRuns: earlier,
  conditions: { atLeast7GBFree: cond1, estimateFits: cond2 }, decision: cond1 && cond2 ? "run" : "not run",
};
writeFileSync(resolve(ROOT, "results", "overhead", "item3-check.json"), JSON.stringify(out, null, 1) + "\n");
console.log(JSON.stringify({ freeMB, neededMB, ...out.conditions, decision: out.decision }));
