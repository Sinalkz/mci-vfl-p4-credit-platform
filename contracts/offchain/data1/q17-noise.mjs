// Q17 follow-up, item 2: evaluation noise against N. For N = 64, 128, 256, 512 and draw j = 1..1000, the protocol's
// draw rule (sampleIndices: Poseidon(j, k) mod |test|, with replacement) picks N test positions; each variant's l_i
// is computed on them from the full-test predictions in results/data1/q17/s{seed}.json (no model is rerun). Writes
// the raw l_i per draw; the SDs are computed in py/q17_analysis.py.
//   node offchain/data1/q17-noise.mjs [--draws 1000] [--seeds 1-5]
// --seeds (e1.mjs's parseSeeds; default 1-5): the seeds whose s{seed}.json are read. The draws do not depend on the
// seed, so each seed's l_i are the same in any seed set. The default writes noise.json; another seed set writes its
// own file (e.g. noise-s6-10.json).
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { macroF1 } from "../model.mjs";
import { parseSeeds, seedTag } from "./e1.mjs";
import { n64Positions } from "./evaluate.mjs";
import { loadManifest } from "./prepared.mjs";

export const NS = [64, 128, 256, 512];
const DIR = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "..", "results", "data1", "q17");
const { values: a } = parseArgs({ options: { draws: { type: "string", default: "1000" }, seeds: { type: "string", default: "1-5" } } });
const draws = Number(a.draws);
const seedList = parseSeeds(a.seeds);
const outFile = resolve(DIR, `noise${seedTag(seedList)}.json`);
const m = loadManifest();
const testPos = m.split.test_pos;
const at = new Map(testPos.map((k, t) => [k, t]));
const seeds = seedList.map((s) => JSON.parse(readFileSync(resolve(DIR, `s${s}.json`), "utf8")));

// l_i per variant from the figures' full-test predictions
const VARIANTS = {
  a: (f) => [f.FULL_joint, f.LOO_retrained],
  b: (f) => [f.FULL_retrained, f.LOO_retrained],
  c: (f) => [f.FULL_joint, f.LOO_masked],
  ref: (f) => [f.FULL_joint, f.LOO_central],
};
const out = { rule: "sampleIndices(j, N, |test|) over the test positions, j = 1..draws", draws, N: NS, seeds: {} };
const t0 = performance.now();
for (const run of seeds) out.seeds[run.label] = Object.fromEntries(NS.map((N) => [N, Object.fromEntries(Object.keys(VARIANTS).map((v) => [v, []]))]));
for (const N of NS) {
  for (let j = 1; j <= draws; j++) {
    const pos = await n64Positions(j, testPos, N);
    const labels = pos.map((k) => m.y[k]);
    const F = (preds) => macroF1(labels, pos.map((k) => preds[at.get(k)])).F;
    for (const run of seeds) {
      const cache = new Map();
      const Fc = (fig) => { if (!cache.has(fig)) cache.set(fig, F(fig.preds)); return cache.get(fig); };
      for (const [v, sel] of Object.entries(VARIANTS)) {
        const [full, loos] = sel(run.figures);
        out.seeds[run.label][N][v].push(loos.map((l) => Fc(full) - Fc(l)));
      }
    }
  }
  console.error(`N = ${N}: ${draws} draws, ${Math.round((performance.now() - t0) / 1000)} s`);
}
writeFileSync(outFile, JSON.stringify(out) + "\n");
console.log(JSON.stringify({ draws, N: NS, seeds: Object.keys(out.seeds), out: outFile, wallS: Math.round((performance.now() - t0) / 1000) }));
