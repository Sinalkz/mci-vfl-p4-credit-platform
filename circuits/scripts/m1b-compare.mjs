// M1b: proving time of S2b (|I| = 1000 fixed at compile time) against M1b (sizeI = 1000
// public, maxDepth = 10) under the same machine conditions. Both circuits are 773k
// constraints; proving time on this laptop drifts with background load, so the two are
// run alternately, one proof per fresh process, over several rounds.
// Usage (from circuits/):  node scripts/m1b-compare.mjs [--rounds 3]
// Needs both proving keys (build.mjs --n 64 --members --isize 1000; m1b-build.mjs --n 64 --maxdepth 10).
// The proof files the two prove scripts write are restored afterwards, so the committed
// proofs stay those of m1b-measure.mjs and the S2b spike. Writes results/m1b/compare_s2b.json.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { CIRCUITS, ROOT } from "./lib.mjs";

const { values: args } = parseArgs({ options: { rounds: { type: "string", default: "3" } } });
const kept = [resolve(ROOT, "results", "s2b", "proofs", "N64-I1000", "fused.json"),
  resolve(ROOT, "results", "m1b", "proofs", "N64-D10-I1000.json")];
const backup = kept.map((f) => readFileSync(f));

const run = (script, extra) => {
  const out = execFileSync(process.execPath, ["--max-old-space-size=14336", resolve(CIRCUITS, "scripts", script), ...extra],
    { cwd: CIRCUITS, encoding: "utf8", maxBuffer: 1 << 26, stdio: ["ignore", "pipe", "ignore"] });
  const r = JSON.parse(out.trim().split("\n").filter((l) => l.startsWith("{")).pop());
  return { proveMs: r.proveMs.all[0], witnessMs: r.witnessMs.all[0], peakRssMB: r.peakRssMB };
};
const s2b = [];
const m1b = [];
try {
  for (let i = 0; i < Number(args.rounds); i++) {
    const order = i % 2 === 0 ? ["s2b", "m1b"] : ["m1b", "s2b"];
    for (const which of order) {
      if (which === "s2b") s2b.push(run("prove-members.mjs", ["--n", "64", "--isize", "1000", "--reps", "1"]));
      else m1b.push(run("m1b-prove.mjs", ["--n", "64", "--maxdepth", "10", "--isize", "1000", "--reps", "1"]));
      console.log(which, JSON.stringify((which === "s2b" ? s2b : m1b).at(-1)));
    }
  }
} finally {
  kept.forEach((f, i) => writeFileSync(f, backup[i]));
}
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const summary = (rs) => ({ proveMsMedian: median(rs.map((r) => r.proveMs)), proveMs: rs.map((r) => r.proveMs),
  witnessMsMedian: median(rs.map((r) => r.witnessMs)), peakRssMBMax: Math.max(...rs.map((r) => r.peakRssMB)) });
const result = { rounds: Number(args.rounds), order: "alternating, fresh process per proof",
  s2bFixedI1000: summary(s2b), m1bSizeIPublicI1000: summary(m1b) };
result.ratio = result.m1bSizeIPublicI1000.proveMsMedian / result.s2bFixedI1000.proveMsMedian;
writeFileSync(resolve(ROOT, "results", "m1b", "compare_s2b.json"), JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify(result));
