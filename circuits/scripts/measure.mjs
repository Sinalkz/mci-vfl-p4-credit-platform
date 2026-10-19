// Run prove.mjs for every N in a fresh process and collect all S2 numbers.
// Usage (from circuits/): node scripts/measure.mjs [--sizes 64,128,256] [--reps 3]
// Requires: node scripts/build.mjs --n <N> for each size (and --merkle --compile-only
// for the commitment comparison). Writes results/s2/zk_measurements.json and .csv.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { CIRCUITS, ROOT, buildDir } from "./lib.mjs";

const { values: args } = parseArgs({
  options: { sizes: { type: "string", default: "64,128,256" }, reps: { type: "string", default: "3" } },
});
const sizes = args.sizes.split(",").map(Number);
const read = (p) => (existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : null);

const rows = [];
for (const n of sizes) {
  console.log(`N=${n}: proving (${args.reps} reps)...`);
  const out = execFileSync(process.execPath, ["--max-old-space-size=8192", resolve(CIRCUITS, "scripts", "prove.mjs"),
    "--n", String(n), "--reps", args.reps], { cwd: CIRCUITS, encoding: "utf8", maxBuffer: 1 << 26 });
  const m = JSON.parse(out.trim().split("\n").filter((l) => l.startsWith("{")).pop());
  const compile = read(resolve(buildDir(n), "compile.json"));
  const setup = read(resolve(buildDir(n), "setup.json"));
  const merkle = read(resolve(buildDir(n, true), "compile.json"));
  rows.push({ ...m, constraints: compile.r1csConstraints, compileSeconds: compile.compileSeconds,
    setupSeconds: setup?.setupSeconds, ptau: setup?.ptau,
    merkleConstraints: merkle?.r1csConstraints ?? null });
  console.log(JSON.stringify(rows.at(-1)));
}

const outDir = resolve(ROOT, "results", "s2");
mkdirSync(outDir, { recursive: true });
writeFileSync(resolve(outDir, "zk_measurements.json"), JSON.stringify(rows, null, 2) + "\n");
const header = "N,constraints,merkleConstraints,compileS,setupS,witnessMs,proveMs,verifyMs,peakRssMB,proofCalldataBytes,proofJsonBytes,zkeyMB,F";
const csv = rows.map((r) => [r.N, r.constraints, r.merkleConstraints, r.compileSeconds.toFixed(1), r.setupSeconds?.toFixed(1),
  r.witnessMs.median.toFixed(0), r.proveMs.median.toFixed(0), r.verifyMs.median.toFixed(1), r.peakRssMB.toFixed(0),
  r.proofCalldataBytes, r.proofJsonBytes, r.zkeyMB.toFixed(1), r.F].join(","));
writeFileSync(resolve(outDir, "zk_measurements.csv"), [header, ...csv].join("\n") + "\n");
console.log([header, ...csv].join("\n"));
