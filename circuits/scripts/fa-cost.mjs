// The scoring circuit's constraint count if the power company had FA = 10, 12 or 14 inputs instead of 8 (N = 64,
// maxDepth 10), from its constraint breakdown (results/proofcost/breakdown-N64-D10.json: the parts compiled one by one
// sum to the full circuit's 773,221, residual 0). Only three parts depend on FA (circuits/src/fa_probes.circom); each
// is compiled here (--r1cs --O2, as breakdown.mjs) for FA = 8, 10, 12, 14. FA = 8 must reproduce the breakdown's
// per-instance counts; then total(FA) = full - parts(8) + parts(FA). The full circuit is not recompiled (its compile
// took free RAM from 5,556 MB to 1,673 MB) and the circuit itself is not changed.
// Usage (from circuits/):  node scripts/fa-cost.mjs   -> results/proofcost/fa-inputs-N64-D10.json
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { BUILD, CIRCUITS, ROOT } from "./lib.mjs";

const circom = process.env.CIRCOM ?? "circom";
const N = 64, D = 10;
const WIDTHS = [8, 10, 12, 14];
const breakdown = JSON.parse(readFileSync(resolve(ROOT, "results", "proofcost", `breakdown-N${N}-D${D}.json`), "utf8"));
const row = (component) => breakdown.rows.find((r) => r.component === component);
// [breakdown component, probe main for width FA, instances per proof]
const parts = [
  ["leaf range checks, tree A", (fa) => `BDLeafRanges(${fa})`, N],
  ["forward pass: bottom model f_A", (fa) => `FABottomRow(${fa})`, N],
  ["parameters + C_fA, C_model", (fa) => `FAParams(${fa})`, 1],
];

function nonLinear(main) {
  const dir = resolve(BUILD, "fa-cost", main.replace(/[^A-Za-z0-9]/g, "_"));
  mkdirSync(dir, { recursive: true });
  const file = resolve(dir, "main.circom");
  writeFileSync(file, `pragma circom 2.1.9;\ninclude "fa_probes.circom";\ncomponent main = ${main};\n`);
  const out = execFileSync(circom, [file, "--r1cs", "--O2", "-o", dir, "-l", resolve(CIRCUITS, "src"), "-l", resolve(CIRCUITS, "node_modules")],
    { encoding: "utf8" });
  return Number(out.match(/non-linear constraints:\s*(\d+)/)[1]);
}

const per = Object.fromEntries(WIDTHS.map((fa) => [fa, parts.map(([, main]) => nonLinear(main(fa)))]));
parts.forEach(([component], i) => {
  if (per[8][i] !== row(component).perInstance) throw new Error(`${component}: FA 8 gives ${per[8][i]}, the breakdown ${row(component).perInstance}`);
});
const partsTotal = (fa) => parts.reduce((s, [, , count], i) => s + per[fa][i] * count, 0);
const widths = WIDTHS.map((fa) => {
  const total = breakdown.fullConstraints - partsTotal(8) + partsTotal(fa);
  return { FA: fa, perInstance: Object.fromEntries(parts.map(([c], i) => [c, per[fa][i]])), faDependent: partsTotal(fa), total,
    vsFA8: total - breakdown.fullConstraints, vsFA8Percent: (100 * (total - breakdown.fullConstraints)) / breakdown.fullConstraints };
});
const result = { N, maxDepth: D, method: "parts that depend on FA compiled (r1cs, --O2) per width; the rest from the breakdown",
  base: { file: `results/proofcost/breakdown-N${N}-D${D}.json`, fullConstraints: breakdown.fullConstraints },
  circomVersion: execFileSync(circom, ["--version"], { encoding: "utf8" }).trim(), widths };
writeFileSync(resolve(ROOT, "results", "proofcost", `fa-inputs-N${N}-D${D}.json`), JSON.stringify(result, null, 1) + "\n");
for (const w of widths) console.log(`FA ${String(w.FA).padStart(2)}: ${w.total.toLocaleString("en")} constraints (${w.vsFA8 >= 0 ? "+" : ""}${w.vsFA8.toLocaleString("en")}, ${w.vsFA8Percent.toFixed(2)} %)`, JSON.stringify(w.perInstance));
