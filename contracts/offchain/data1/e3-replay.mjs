// DATA1 §5, E3 replay: E3 ran with the passives' noise off (Q4), so its training is deterministic. This script
// rebuilds the E3 rows in the protocol's order of I (A's I.json), trains with the centralized float32 path (equal to
// split training bit for bit, test D05), and scores on the protocol's 64 evaluation positions (A's idx). It first
// asserts that the 6 replayed figures (D139: F_r, F_j, LOO_1..3, BASELINE) equal the 6 figures accepted on-chain
// (results/data1/e3/run.json) and that U_fed = max(F_r, F_j) equals the contract's. The D118 run is kept in
// results/data1/e3-d118/ with its own replay.json.
//   VFL_MODEL_CONFIG=../results/data1/configs/e3.json node offchain/data1/e3-replay.mjs [--dir <run dir>]
// --dir: replay another E3 run, e.g. M4's (../results/m4/e3); its replay.json is written there. Each run draws its own
// seed, so its evaluation positions and figures are its own.
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { assertFiWidth, CONFIG_PATH, modelConfigHash } from "../model.mjs";
import { requiredFigures, scoreFigures } from "../scoring.mjs";
import { centralHq, trainCentral } from "./centralized.mjs";
import { machine, peakRssMB } from "./machine.mjs";
import { recsScenario } from "./recs-scenario.mjs";

const { values: opt } = parseArgs({ options: { dir: { type: "string" } } });
const DIR = opt.dir ? resolve(opt.dir) : resolve(fileURLToPath(import.meta.url), "..", "..", "..", "..", "results", "data1", "e3");
const load = (f) => JSON.parse(readFileSync(resolve(DIR, f), "utf8"));
const run = load("run.json");
const a = load("A/summary.json");
const I = load("A/I.json").map((x) => BigInt(x));
const idx = a.idx;
const t0 = performance.now();

const sc = recsScenario({ seed: load("sample.json").seed });
const rows = { xa: I.map((id) => sc.A.rows.get(id).xa), y: I.map((id) => sc.A.rows.get(id).y) };
const xp = sc.passives.map((p) => I.map((id) => {
  const r = p.rows.get(id);
  if (!r) throw new Error(`id ${id} of I is missing at a passive`);
  return r;
}));
xp.forEach((rowsI, k) => assertFiWidth(k + 1, rowsI[0].length)); // the config's widths are the data's
const picked = new Set(idx);
const Itr = [...I.keys()].filter((k) => !picked.has(k));

const c = trainCentral({ rows, xp, Itr, mode: "f32" });
const hq = centralHq(c.fis, xp, I.length);
const figs = requiredFigures(7, 3);
const scored = scoreFigures({ fA: c.fA, fusion: c.fusion, rows, hq, Itr, idx, figures: figs }).figures; // D139

const onChain = run.chainEnd.figures;
const mismatch = scored.filter((f) => onChain.find((g) => g.kind === f.kind && g.memberIndex === f.memberIndex).F !== f.F);
if (mismatch.length) throw new Error(`replay differs from the chain: ${mismatch.map((f) => `${f.name} ${f.F}`).join(", ")}`);
if (onChain.length !== scored.length) throw new Error(`${onChain.length} figures on-chain, ${scored.length} replayed`);

const F = Object.fromEntries(scored.map((f) => [f.name, f.F]));
const uFed = Math.max(F.FULL, F.FULL_J);
if (uFed !== run.chainEnd.uFed) throw new Error(`U_fed ${uFed} != the contract's ${run.chainEnd.uFed}`);
const out = {
  configPath: CONFIG_PATH, configHash: modelConfigHash(), sizeI: I.length, sizeItr: Itr.length, evalPositions: idx.length,
  replayEqualsChain: true, figures: F, uFed, jointBinds: F.FULL_J > F.FULL,
  l: [1, 2, 3].map((i) => uFed - F[`LOO_${i}`]), E: Math.max(0, uFed - F.BASELINE),
  machine: machine(), wallMs: performance.now() - t0, peakRssMB: peakRssMB(),
};
writeFileSync(resolve(DIR, "replay.json"), JSON.stringify(out, null, 1) + "\n");
console.log(JSON.stringify({ figures: F, uFed, jointBinds: out.jointBinds, l: out.l, E: out.E }));
