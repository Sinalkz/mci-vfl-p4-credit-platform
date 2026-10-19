// RESULTS-PROOFCOST item 3: attribute the scoring circuit's constraints to its parts. Compiles each probe of
// circuits/src/breakdown_probes.circom (--r1cs --O2, as m1b-build.mjs), multiplies by its count per proof, and compares
// the sum with the full circuit's r1cs count from results/proofcost/compile-N<N>-D<D>.json.
// Usage (from circuits/):  node scripts/breakdown.mjs [--n 64] [--maxdepth 10]   -> results/proofcost/breakdown-N<N>-D<D>.json
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { BUILD, CIRCUITS, ROOT } from "./lib.mjs";

const { values: a } = parseArgs({ options: { n: { type: "string", default: "64" }, maxdepth: { type: "string", default: "10" } } });
const N = Number(a.n), D = Number(a.maxdepth);
const circom = process.env.CIRCOM ?? "circom";

// [component, part, probe main, count per proof]
const parts = [
  ["index derivation", "Poseidon(seed, n), mod 2^128, mod sizeI, path bits", `BDIndex(${D})`, N],
  ["Merkle path, tree A (gammaA)", `MerkleRoot(${D})`, `BDMerklePath(${D})`, N],
  ["Merkle path, tree p1 (gammaP[0])", `MerkleRoot(${D})`, `BDMerklePath(${D})`, N],
  ["Merkle path, tree p2 (gammaP[1])", `MerkleRoot(${D})`, `BDMerklePath(${D})`, N],
  ["Merkle path, tree p3 (gammaP[2])", `MerkleRoot(${D})`, `BDMerklePath(${D})`, N],
  ["leaf range checks, tree A", "8 x SignedRange(16)", "BDLeafRanges(8)", N],
  ["leaf range checks, trees p1-p3", "3 x 4 x SignedRange(16)", "BDLeafRanges(4)", 3 * N],
  ["forward pass: bottom model f_A", "8 -> 4, ReLU, rescale", "BDBottomRow()", N],
  ["forward pass: fusion head + argmax", "16 -> 8 -> 5 (masked), argmax", "BDFusionRow()", N],
  ["F1: per-row counts", "label one-hot, TP indicators", "BDCountRow()", N],
  ["F1: macro-F1 division", "once per proof", `BDMacroF1(${N})`, 1],
  ["parameters + C_fA, C_model", "range checks, two Poseidon chains; once", "BDParams()", 1],
  ["mask + sizeI bound", "once", `BDMaskSize(${D})`, 1],
];

const cache = {};
const rows = [];
for (const [component, what, main, count] of parts) {
  if (!(main in cache)) {
    const dir = resolve(BUILD, "breakdown", main.replace(/[^A-Za-z0-9]/g, "_"));
    mkdirSync(dir, { recursive: true });
    const file = resolve(dir, "main.circom");
    writeFileSync(file, `pragma circom 2.1.9;\ninclude "breakdown_probes.circom";\ncomponent main = ${main};\n`);
    const out = execFileSync(circom, [file, "--r1cs", "--O2", "-o", dir, "-l", resolve(CIRCUITS, "src"), "-l", resolve(CIRCUITS, "node_modules")],
      { encoding: "utf8" });
    // non-linear only, as the full circuit (0 linear after --O2): a probe's output wiring can leave a linear
    // constraint that --O2 removes inside the full circuit (the fusion probe's argmax output: 1)
    cache[main] = { nonLinear: Number(out.match(/non-linear constraints:\s*(\d+)/)[1]),
      linear: Number(out.match(/(?<!non-)linear constraints:\s*(\d+)/)[1]) };
  }
  const c = cache[main];
  rows.push({ component, what, probe: main, perInstance: c.nonLinear, probeLinear: c.linear, count, total: c.nonLinear * count });
}
const full = JSON.parse(readFileSync(resolve(ROOT, "results", "proofcost", `compile-N${N}-D${D}.json`), "utf8")).result.r1csConstraints;
const sum = rows.reduce((s, r) => s + r.total, 0);
const result = { N, maxDepth: D, circuit: "circuits/src/macro_f1_members_var.circom (unchanged)", fullConstraints: full, sumOfParts: sum,
  residual: full - sum, circomVersion: execFileSync(circom, ["--version"], { encoding: "utf8" }).trim(), rows };
writeFileSync(resolve(ROOT, "results", "proofcost", `breakdown-N${N}-D${D}.json`), JSON.stringify(result, null, 1) + "\n");
for (const r of rows) console.log(`${r.component.padEnd(38)} ${String(r.perInstance).padStart(7)} x ${String(r.count).padStart(3)} = ${r.total}`);
console.log(`sum ${sum}, full ${full}, residual ${full - sum}`);
