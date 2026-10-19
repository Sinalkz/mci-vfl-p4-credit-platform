// Collect S2b numbers: constraints for every (N, |I|), proving where a proving key exists.
// Usage (from circuits/): node scripts/measure-members.mjs [--configs 64:1000,64:10000] [--reps 3]
// Needs: node scripts/build.mjs --n N --members --isize I [--compile-only] for each configuration.
// Writes results/s2b/zk_members.json and results/s2b/zk_members.csv.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { CIRCUITS, ROOT, membersBuildDir } from "./lib.mjs";

const { values: args } = parseArgs({ options: {
  configs: { type: "string", default: "64:1000,64:10000,128:1000,128:10000,256:1000,256:10000" },
  reps: { type: "string", default: "3" },
  "export-all-for": { type: "string", default: "64:1000" },
} });
const read = (p) => (existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : null);
const outDir = resolve(ROOT, "results", "s2b");
mkdirSync(outDir, { recursive: true });
const prev = read(resolve(outDir, "zk_members.json")) ?? [];

const rows = [];
for (const cfg of args.configs.split(",")) {
  const [n, isize] = cfg.split(":").map(Number);
  const dir = membersBuildDir(n, isize);
  const compile = read(resolve(dir, "compile.json"));
  const setup = read(resolve(dir, "setup.json"));
  const row = { N: n, isize, depth: compile?.depth, constraints: compile?.r1csConstraints ?? null,
    compileSeconds: compile?.compileSeconds, setupSeconds: setup?.setupSeconds ?? null,
    setupPeakRssMB: setup?.setupPeakRssMB ?? null, ptau: setup?.ptau ?? null, proving: null };
  if (setup && existsSync(resolve(dir, "main_final.zkey"))) {
    console.log(`N=${n} |I|=${isize}: proving (${args.reps} reps)...`);
    const extra = args["export-all-for"].split(",").includes(cfg) ? ["--export-all"] : [];
    const out = execFileSync(process.execPath, ["--max-old-space-size=14336", resolve(CIRCUITS, "scripts", "prove-members.mjs"),
      "--n", String(n), "--isize", String(isize), "--reps", args.reps, ...extra],
      { cwd: CIRCUITS, encoding: "utf8", maxBuffer: 1 << 26 });
    row.proving = JSON.parse(out.trim().split("\n").filter((l) => l.startsWith("{")).pop());
  } else {
    // keep an earlier proving result for this configuration if there is one
    row.proving = prev.find((p) => p.N === n && p.isize === isize)?.proving ?? null;
  }
  rows.push(row);
  console.log(JSON.stringify(row));
}
writeFileSync(resolve(outDir, "zk_members.json"), JSON.stringify(rows, null, 2) + "\n");
const header = "N,isize,depth,constraints,compileS,setupS,setupPeakRssMB,witnessMs,proveMs,verifyMs,peakRssMB,distinctRows";
const csv = rows.map((r) => [r.N, r.isize, r.depth, r.constraints, r.compileSeconds?.toFixed(1), r.setupSeconds?.toFixed(1),
  r.setupPeakRssMB?.toFixed(0), r.proving?.witnessMs.median.toFixed(0), r.proving?.proveMs.median.toFixed(0),
  r.proving?.verifyMs.median.toFixed(1), r.proving?.peakRssMB.toFixed(0), r.proving?.distinctRows].join(","));
writeFileSync(resolve(outDir, "zk_members.csv"), [header, ...csv].join("\n") + "\n");
console.log([header, ...csv].join("\n"));
