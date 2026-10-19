// DATA1 §3, evaluation noise at N = 64: from one seed's C2 run (FULL = retrained F_r, FULL_J = joint F_j,
// LOO_1..3, BASELINE = C1a), 1,000 draws of 64 test rows with replacement, exactly as the protocol
// draws its evaluation rows (sampleIndices with seed j = 1..1000 over the test positions). No model is
// rerun: each figure's predictions on the whole test set are indexed at the drawn rows. Writes every
// draw's F per figure; the summaries are computed in py/data1_analysis.py.
//   node offchain/data1/n64.mjs --in ../results/data1/e1/c2-s1.json --out ../results/data1/e1/n64.json [--draws 1000]
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { macroF1 } from "../model.mjs";
import { n64Positions } from "./evaluate.mjs";
import { loadManifest } from "./prepared.mjs";

const { values: a } = parseArgs({ options: { in: { type: "string" }, out: { type: "string" }, draws: { type: "string", default: "1000" } } });
const run = JSON.parse(readFileSync(resolve(a.in), "utf8"));
const m = loadManifest();
const testPos = m.split.test_pos;
const at = new Map(testPos.map((k, t) => [k, t]));
const figs = Object.fromEntries(run.figures.map((f) => [f.name, f.preds]));
const names = ["FULL", "FULL_J", "LOO_1", "LOO_2", "LOO_3", "BASELINE"]; // D139: F_r and F_j (U_fed in the analysis)
const full = Object.fromEntries(names.map((n) => [n, run.figures.find((f) => f.name === n).F]));
const draws = [];
for (let j = 1; j <= Number(a.draws); j++) {
  const pos = await n64Positions(j, testPos);
  const labels = pos.map((k) => m.y[k]);
  const F = Object.fromEntries(names.map((n) => [n, macroF1(labels, pos.map((k) => figs[n][at.get(k)])).F]));
  draws.push({ j, distinct: new Set(pos).size, F });
}
writeFileSync(resolve(a.out), JSON.stringify({ source: a.in, seed: run.seed, N: 64, fullTest: full, draws }) + "\n");
console.log(JSON.stringify({ draws: draws.length, fullTest: full }));
