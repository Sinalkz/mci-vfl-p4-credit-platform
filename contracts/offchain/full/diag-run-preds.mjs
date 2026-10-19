// full-data diagnosis (item 1, and the holdout part of items 3-4): a full-data run's 64-row predictions, which the run
// records do not store, recomputed by the run's own replay path (full/replay-full.mjs: the run's rows in the order of I,
// the centralized float32 training equal to split training bit for bit, scoreFigures on A's idx). The six figures must
// equal the six accepted on-chain (the proofs' public F), or the script stops. Then, with the same trained models:
//   - every figure's labels and predictions on the run's 64 evaluation positions (for the per-tier F1 and the
//     convention checks in py/full_diagnosis.py);
//   - the holdout: the 1,721 cleaned households of A that are not in I (no model of the run saw them), with their
//     prepared columns (data/recs/prepared/, the columns the passives would hold); each figure's F on all of them, and
//     on 1,000 draws of 64 of them with the protocol's index rule (sampleIndices with seed j = 1..1000, as n64.mjs);
//   - with --e1-positions <file>: E1's 1,000 N = 64 draws as test-set indices (n64.mjs's n64Positions, j = 1..1000), so
//     that py/full_diagnosis.py can recompute E1's draws with other macro-F1 implementations.
// Changes no file of the run.
//   VFL_MODEL_CONFIG=../results/data1/configs/e1-s<k>.json node offchain/full/diag-run-preds.mjs \
//       --dir ../results/full/runs/run-<k> --out ../results/full/diagnosis/preds/run-<k>.json [--draws 1000] [--e1-positions <file>]
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { sampleIndices } from "../../../circuits/scripts/members-lib.mjs";
import { assertFiWidth, CONFIG_PATH, E, forwardInt, macroF1, modelConfigHash } from "../model.mjs";
import { requiredFigures, scoreFigures } from "../scoring.mjs";
import { centralHq, trainCentral } from "../data1/centralized.mjs";
import { machine, peakRssMB } from "../data1/machine.mjs";
import { loadManifest, openPrepared, PASSIVE_NAMES } from "../data1/prepared.mjs";
import { recsFullScenario } from "./recs-full-scenario.mjs";

const { values: opt } = parseArgs({ options: { dir: { type: "string" }, out: { type: "string" }, draws: { type: "string", default: "1000" },
  "e1-positions": { type: "string" } } });
if (!opt.dir || !opt.out) throw new Error("--dir <run dir> --out <file>");
const DIR = resolve(opt.dir);
const load = (f) => JSON.parse(readFileSync(resolve(DIR, f), "utf8"));
const run = load("run.json");
const sample = load("sample.json");
if (sample.scenario !== "recs-full") throw new Error(`${DIR}: not a full-data run`);
const CONTRACTS = resolve(fileURLToPath(import.meta.url), "..", "..", "..");
if (!readFileSync(resolve(CONTRACTS, sample.modelConfig)).equals(readFileSync(CONFIG_PATH))) {
  throw new Error(`VFL_MODEL_CONFIG ${CONFIG_PATH} is not the run's model config ${sample.modelConfig}`);
}
const a = load("A/summary.json");
const I = load("A/I.json").map((x) => BigInt(x));
const idx = a.idx;
const t0 = performance.now();

// ---- the replay, exactly as full/replay-full.mjs -----------------------------------------
const sc = recsFullScenario({ seed: sample.seed });
const rows = { xa: I.map((id) => sc.A.rows.get(id).xa), y: I.map((id) => sc.A.rows.get(id).y) };
const xp = sc.passives.map((p) => I.map((id) => {
  const r = p.rows.get(id);
  if (!r) throw new Error(`id ${id} of I is missing at a passive`);
  return r;
}));
xp.forEach((rowsI, k) => assertFiWidth(k + 1, rowsI[0].length));
const picked = new Set(idx);
const Itr = [...I.keys()].filter((k) => !picked.has(k));
const c = trainCentral({ rows, xp, Itr, mode: "f32" });
const hq = centralHq(c.fis, xp, I.length);
const scored = scoreFigures({ fA: c.fA, fusion: c.fusion, rows, hq, Itr, idx, figures: requiredFigures(7, 3) });
const onChain = run.chainEnd.figures;
for (const f of scored.figures) {
  const g = onChain.find((x) => x.kind === f.kind && x.memberIndex === f.memberIndex);
  if (!g || g.F !== f.F) throw new Error(`${f.name}: recomputed ${f.F} != on-chain ${g?.F}`);
}
// the proofs' public F (public input 10 of 10: [C_model, C_fA, gammaA, gammaP[3], seed, sizeI, mask, F])
const publicF = Object.fromEntries(a.scoring.map((s) => [s.figure, { F: s.F, oracleF: s.oracleF, publicF: Number(s.publicSignals[9]),
  publicMask: Number(s.publicSignals[8]), publicSizeI: Number(s.publicSignals[7]) }]));

// ---- the holdout: A's cleaned households outside I, with their prepared columns ------------
const { m, P, W } = openPrepared();
const idsInK = [...sc.A.rows.keys()]; // recsFullScenario inserts A's rows in prepared order k
if (idsInK.length !== m.rows) throw new Error("A's rows are not every cleaned household");
const inI = new Set(I);
const holdK = idsInK.map((id, k) => (inI.has(id) ? -1 : k)).filter((k) => k >= 0);
const cols = PASSIVE_NAMES.map((n) => m.parties[n]);
const hold = { xa: holdK.map((k) => sc.A.rows.get(idsInK[k]).xa), y: holdK.map((k) => sc.A.rows.get(idsInK[k]).y) };
const xpHold = cols.map((cc) => holdK.map((k) => cc.map((j) => P[k * W + j])));
// where a passive holds a holdout household, its scenario row must be the prepared row used here
let heldChecked = 0;
sc.passives.forEach((p, i) => holdK.forEach((k, r) => {
  const row = p.rows.get(idsInK[k]);
  if (!row) return;
  if (row.length !== xpHold[i][r].length || row.some((v, j) => v !== xpHold[i][r][j])) throw new Error(`holdout row ${k} differs at passive ${i + 1}`);
  heldChecked++;
}));
if (hold.y.some((y, r) => y !== m.y[holdK[r]])) throw new Error("holdout labels differ from the manifest");
const hqHold = centralHq(c.fis, xpHold, holdK.length);
const zero = Array(E).fill(0);
const holdPreds = Object.fromEntries(scored.figures.map((f) => [f.name,
  hold.y.map((_, r) => forwardInt(scored.fa, f.head, f.mask, hold.xa[r], hqHold.map((mm) => (mm ? mm[r] : zero))))]));
const holdF = Object.fromEntries(scored.figures.map((f) => [f.name, macroF1(hold.y, holdPreds[f.name]).F]));
const holdDraws = [];
for (let j = 1; j <= Number(opt.draws); j++) {
  const pos = await sampleIndices(BigInt(j), 64, holdK.length);
  const labels = pos.map((r) => hold.y[r]);
  holdDraws.push(Object.fromEntries(scored.figures.map((f) => [f.name, macroF1(labels, pos.map((r) => holdPreds[f.name][r])).F])));
}
// E1's test positions as households: how many of them the run trained on
const testPos = loadManifest().split.test_pos;
const trainedIds = new Set(Itr.map((k) => I[k]));
const testInItr = testPos.filter((k) => trainedIds.has(idsInK[k])).length;

const out = {
  what: "full-data diagnosis item 1: a run's 64-row predictions recomputed by its replay path (equal to the chain), and its models on the 1,721-household holdout A \\ I",
  run: DIR.replace(/.*[\\/]results[\\/]/, "results/").replaceAll("\\", "/"),
  configPath: CONFIG_PATH.replace(/.*[\\/]results[\\/]/, "results/").replaceAll("\\", "/"), configHash: modelConfigHash(),
  sizeI: I.length, sizeItr: Itr.length, evalPositions: idx,
  recomputedEqualsChain: true, publicF,
  labels: idx.map((k) => rows.y[k]),
  figures: scored.figures.map((f) => ({ name: f.name, kind: f.kind, mask: f.mask, F: f.F, TP: f.TP, FP: f.FP, FN: f.FN, preds: f.preds })),
  holdout: { rows: holdK.length, heldRowsChecked: heldChecked, doeid: holdK.map((k) => m.doeid[k]), y: hold.y, F: holdF, preds: holdPreds,
    draws: { n: holdDraws.length, N: 64, rule: "sampleIndices(j, 64, rows), j = 1..n", F: holdDraws } },
  e1TestHouseholdsInRunTraining: { testRows: testPos.length, inItr: testInItr },
  machine: machine(), wallMs: performance.now() - t0, peakRssMB: peakRssMB(),
};
mkdirSync(dirname(resolve(opt.out)), { recursive: true });
writeFileSync(resolve(opt.out), JSON.stringify(out) + "\n");
if (opt["e1-positions"]) {
  const draws = [];
  for (let j = 1; j <= Number(opt.draws); j++) draws.push(await sampleIndices(BigInt(j), 64, testPos.length));
  writeFileSync(resolve(opt["e1-positions"]), JSON.stringify({ what: "E1's N = 64 draws as indices t into the manifest's test_pos (n64.mjs: n64Positions(j) = test_pos[t])",
    rule: "sampleIndices(j, 64, 3193), j = 1..n", testRows: testPos.length, draws }) + "\n");
}
console.log(JSON.stringify({ run: out.run, F: Object.fromEntries(out.figures.map((f) => [f.name, f.F])), holdoutRows: holdK.length, holdoutF: holdF,
  testInItr, wallS: Math.round(out.wallMs / 1000), peakRssMB: out.peakRssMB }));
process.exit(0);
