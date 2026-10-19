// M1b: collect constraints, setup, proving and verification for the sizeI-public circuit,
// with S2b (same N and depth, |I| fixed at compile time) alongside for comparison.
// Usage (from circuits/):  node scripts/m1b-measure.mjs [--n 64] [--maxdepth 10] [--sizes 1000,700] [--reps 3]
//                          node scripts/m1b-measure.mjs --collect-only   (refresh constraint rows, keep proving results)
// Needs: node scripts/m1b-build.mjs --n 64 --maxdepth 10 (compile + setup).
// Optional extra rows: compile-only builds of other maxDepth values (members-var-N<N>-D<d>).
// Writes results/m1b/zk.json.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { BUILD, CIRCUITS, ROOT } from "./lib.mjs";
import { varBuildDir } from "./m1b-lib.mjs";

const { values: args } = parseArgs({ options: {
  n: { type: "string", default: "64" }, maxdepth: { type: "string", default: "10" },
  sizes: { type: "string", default: "1000,700" }, reps: { type: "string", default: "3" },
  "collect-only": { type: "boolean", default: false },
} });
const N = Number(args.n);
const D = Number(args.maxdepth);
const read = (p) => (existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : null);
const dir = varBuildDir(N, D);
const outDir = resolve(ROOT, "results", "m1b");
mkdirSync(outDir, { recursive: true });

const previous = read(resolve(outDir, "zk.json"));
const proving = args["collect-only"] ? (previous?.proving ?? []) : [];
for (const I of args["collect-only"] ? [] : args.sizes.split(",").map(Number)) {
  console.log(`N=${N} maxDepth=${D} sizeI=${I}: proving (${args.reps} reps)...`);
  const out = execFileSync(process.execPath, ["--max-old-space-size=14336", resolve(CIRCUITS, "scripts", "m1b-prove.mjs"),
    "--n", String(N), "--maxdepth", String(D), "--isize", String(I), "--reps", args.reps],
    { cwd: CIRCUITS, encoding: "utf8", maxBuffer: 1 << 26 });
  const r = JSON.parse(out.trim().split("\n").filter((l) => l.startsWith("{")).pop());
  proving.push(r);
  console.log(JSON.stringify(r));
}

// Constraint counts of every compiled maxDepth for this N (compile-only builds included).
const depths = readdirSync(BUILD).map((d) => d.match(new RegExp(`^members-var-N${N}-D(\\d+)$`)))
  .filter(Boolean).map((m) => Number(m[1])).sort((a, b) => a - b)
  .map((d) => ({ maxDepth: d, constraints: read(resolve(varBuildDir(N, d), "compile.json"))?.r1csConstraints ?? null }));

// S2b at the same N, for the comparison: |I| = 1000 has depth 10, |I| = 10000 depth 14.
const s2b = (read(resolve(ROOT, "results", "s2b", "zk_members.json")) ?? []).filter((r) => r.N === N).map((r) => ({
  isize: r.isize, depth: r.depth, constraints: r.constraints, setupSeconds: r.setupSeconds,
  setupPeakRssMB: r.setupPeakRssMB, proveMsMedian: r.proving?.proveMs.median ?? null,
  peakRssMB: r.proving?.peakRssMB ?? null, verifyMsMedian: r.proving?.verifyMs.median ?? null,
}));

const result = { N, maxDepth: D, compile: read(resolve(dir, "compile.json")), setup: read(resolve(dir, "setup.json")),
  proving, constraintsByMaxDepth: depths, s2b };
writeFileSync(resolve(outDir, "zk.json"), JSON.stringify(result, null, 2) + "\n");
console.log("wrote results/m1b/zk.json");
