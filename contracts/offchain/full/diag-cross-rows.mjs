// full-data diagnosis (item 4): are the five runs' 64 evaluation rows favourable to BASELINE because of what the rows
// are, or because of something that happens to a run's own evaluation rows? All five runs share one I (scenario seed
// 20265000; checked), so run j's evaluation positions are positions of the same I for every run. This script replays
// run k (as full/replay-full.mjs, equal to the chain) and scores run k's six trained models on every run j's 64
// evaluation positions (for j != k those rows are among run k's training rows), plus, as the in-sample reference,
// 1,000 draws of 64 positions from run k's own I^tr (sampleIndices(j, 64, |I^tr|), j = 1..1000).
//   VFL_MODEL_CONFIG=../results/data1/configs/e1-s<k>.json node offchain/full/diag-cross-rows.mjs \
//       --run <k> --out ../results/full/diagnosis/cross/run-<k>.json [--draws 1000]
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { sampleIndices } from "../../../circuits/scripts/members-lib.mjs";
import { assertFiWidth, CONFIG_PATH, forwardInt, macroF1, modelConfigHash } from "../model.mjs";
import { requiredFigures, scoreFigures } from "../scoring.mjs";
import { centralHq, trainCentral } from "../data1/centralized.mjs";
import { machine, peakRssMB } from "../data1/machine.mjs";
import { recsFullScenario } from "./recs-full-scenario.mjs";

const { values: opt } = parseArgs({ options: { run: { type: "string" }, out: { type: "string" }, draws: { type: "string", default: "1000" } } });
const k = Number(opt.run);
if (!opt.out || !(k >= 1 && k <= 5)) throw new Error("--run <1..5> --out <file>");
const RUNS = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "..", "results", "full", "runs");
const CONTRACTS = resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const load = (j, f) => JSON.parse(readFileSync(resolve(RUNS, `run-${j}`, f), "utf8"));
const sample = load(k, "sample.json");
if (!readFileSync(resolve(CONTRACTS, sample.modelConfig)).equals(readFileSync(CONFIG_PATH))) throw new Error(`VFL_MODEL_CONFIG is not run ${k}'s model config`);
const t0 = performance.now();
const Iraw = readFileSync(resolve(RUNS, `run-${k}`, "A", "I.json"), "utf8");
for (const j of [1, 2, 3, 4, 5]) if (readFileSync(resolve(RUNS, `run-${j}`, "A", "I.json"), "utf8") !== Iraw) throw new Error(`run ${j}'s I differs`);
const I = JSON.parse(Iraw).map((x) => BigInt(x));
const idxOf = Object.fromEntries([1, 2, 3, 4, 5].map((j) => [j, load(j, "A/summary.json").idx]));
const run = load(k, "run.json");

// ---- the replay, exactly as full/replay-full.mjs -----------------------------------------
const sc = recsFullScenario({ seed: sample.seed });
const rows = { xa: I.map((id) => sc.A.rows.get(id).xa), y: I.map((id) => sc.A.rows.get(id).y) };
const xp = sc.passives.map((p) => I.map((id) => p.rows.get(id)));
xp.forEach((x, i) => assertFiWidth(i + 1, x[0].length));
const idx = idxOf[k];
const picked = new Set(idx);
const Itr = [...I.keys()].filter((p) => !picked.has(p));
const c = trainCentral({ rows, xp, Itr, mode: "f32" });
const hq = centralHq(c.fis, xp, I.length);
const scored = scoreFigures({ fA: c.fA, fusion: c.fusion, rows, hq, Itr, idx, figures: requiredFigures(7, 3) });
for (const f of scored.figures) {
  const g = run.chainEnd.figures.find((x) => x.kind === f.kind && x.memberIndex === f.memberIndex);
  if (g.F !== f.F) throw new Error(`${f.name}: recomputed ${f.F} != on-chain ${g.F}`);
}
const predAt = (f, positions) => positions.map((p) => forwardInt(scored.fa, f.head, f.mask, rows.xa[p], hq.map((m) => m[p])));
const Fat = (positions) => Object.fromEntries(scored.figures.map((f) => [f.name, macroF1(positions.map((p) => rows.y[p]), predAt(f, positions)).F]));
const onRows = Object.fromEntries([1, 2, 3, 4, 5].map((j) => [j, { inTraining: idxOf[j].filter((p) => !picked.has(p)).length, F: Fat(idxOf[j]) }]));
const inSample = [];
for (let j = 1; j <= Number(opt.draws); j++) inSample.push(Fat((await sampleIndices(BigInt(j), 64, Itr.length)).map((t) => Itr[t])));

const out = {
  what: "full-data diagnosis item 4: run k's trained models on every run's 64 evaluation rows, and on 64-row draws from its own I^tr",
  run: k, configPath: CONFIG_PATH.replace(/.*[\\/]results[\\/]/, "results/").replaceAll("\\", "/"), configHash: modelConfigHash(),
  recomputedEqualsChain: true, onRunRows: onRows,
  inSampleDraws: { n: inSample.length, N: 64, rule: "sampleIndices(j, 64, |I^tr|) over run k's I^tr, j = 1..n", F: inSample },
  machine: machine(), wallMs: performance.now() - t0, peakRssMB: peakRssMB(),
};
mkdirSync(dirname(resolve(opt.out)), { recursive: true });
writeFileSync(resolve(opt.out), JSON.stringify(out) + "\n");
console.log(JSON.stringify({ run: k, onRunRows: Object.fromEntries(Object.entries(onRows).map(([j, v]) => [j, v.F.BASELINE])), wallS: Math.round(out.wallMs / 1000) }));
process.exit(0);
