// full-data planning: the time of one Poseidon Merkle tree as the protocol builds it (members-lib.mjs buildTree: every
// one of the 2^depth leaves is hashed in, zero padding included), at maxDepth 10 (the subset) and 14 (the full data),
// median of --reps builds after one warm-up. Prints one JSON line.
// Usage (from circuits/):  node scripts/tree-time.mjs [--reps 5] [--size10 910] [--size14 14243]
import { cpus, platform, release, totalmem } from "node:os";
import { parseArgs } from "node:util";

import { buildTree, leafP } from "./members-lib.mjs";

const { values: a } = parseArgs({ options: { reps: { type: "string", default: "5" }, size10: { type: "string", default: "910" },
  size14: { type: "string", default: "14243" } } });
const reps = Number(a.reps);
await buildTree([1n, 2n], 1); // Poseidon set-up outside the timing
const rows = [];
for (const [depth, size] of [[10, Number(a.size10)], [14, Number(a.size14)]]) {
  const leaves = Array.from({ length: size }, (_, i) => leafP([i % 32768, -i % 32768, 7, -7]));
  await buildTree(leaves, depth);
  const ms = [];
  for (let r = 0; r < reps; r++) {
    const t0 = performance.now();
    await buildTree(leaves, depth);
    ms.push(performance.now() - t0);
  }
  const med = [...ms].sort((x, y) => x - y)[Math.floor(reps / 2)];
  rows.push({ depth, leaves: size, hashes: 2 ** depth - 1, msMedian: med, ms, usPerHash: (med * 1000) / (2 ** depth - 1) });
}
console.log(JSON.stringify({ rows, peakRssMB: process.resourceUsage().maxRSS / 1024,
  machine: { cpu: cpus()[0]?.model, threads: cpus().length, totalRamMB: Math.round(totalmem() / 2 ** 20), os: `${platform()} ${release()}`,
    node: process.version } }));
await globalThis.curve_bn128?.terminate();
