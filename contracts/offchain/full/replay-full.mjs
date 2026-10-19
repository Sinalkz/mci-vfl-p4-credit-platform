// full-data (Task 2, item 4): the independent recomputation of a full-data run's six figures, as data1/e3-replay.mjs does
// for E3 (imported, not copied: centralized.mjs, scoring.mjs): the run's passives' noise is off, so its training is
// deterministic. This script rebuilds the run's rows in the protocol's order of I (A's I.json) from the full-data
// scenario (full/recs-full-scenario.mjs, the run's sample.json seed), trains with the centralized float32 path (equal to
// split training bit for bit, test D05), scores on the run's 64 evaluation positions (A's idx) and asserts that the six
// replayed figures (F_r, F_j, LOO_1..3, BASELINE) equal the six accepted on-chain and U_fed = max(F_r, F_j) the
// contract's. Writes <dir>/replay.json.
//   VFL_MODEL_CONFIG=../results/data1/configs/e1-s<k>.json node offchain/full/replay-full.mjs --dir ../results/full/runs/run-<k>
// MCI loan instance: a run whose sample.json names the scenario mci-loan is rebuilt from
// scenarios/mci_loan/onchain/loan-scenario.mjs instead; everything else is the same. The rebuilt scenario must have the
// run's prepared data (sample.json prepared: the SHA-256 values), so a replay with the wrong MCI_LOAN_PREPARED stops.
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { assertFiWidth, CONFIG_PATH, modelConfigHash } from "../model.mjs";
import { requiredFigures, scoreFigures } from "../scoring.mjs";
import { centralHq, trainCentral } from "../data1/centralized.mjs";
import { machine, peakRssMB } from "../data1/machine.mjs";
import { recsFullScenario } from "./recs-full-scenario.mjs";

const { values: opt } = parseArgs({ options: { dir: { type: "string" } } });
if (!opt.dir) throw new Error("--dir <run dir>");
const DIR = resolve(opt.dir);
const load = (f) => JSON.parse(readFileSync(resolve(DIR, f), "utf8"));
const run = load("run.json");
const sample = load("sample.json");
if (!["recs-full", "mci-loan"].includes(sample.scenario)) throw new Error(`${DIR}: not a full-data run (sample.json scenario ${sample.scenario})`);
// the run's model config (runner3 records VFL_MODEL_CONFIG as given, relative to contracts/) must be this process's
const CONTRACTS = resolve(fileURLToPath(import.meta.url), "..", "..", "..");
if (!sample.modelConfig || !readFileSync(resolve(CONTRACTS, sample.modelConfig)).equals(readFileSync(CONFIG_PATH))) {
  throw new Error(`VFL_MODEL_CONFIG ${CONFIG_PATH} is not the run's model config ${sample.modelConfig}`);
}
const a = load("A/summary.json");
const I = load("A/I.json").map((x) => BigInt(x));
const idx = a.idx;
const t0 = performance.now();

const sc = sample.scenario === "mci-loan"
  ? (await import("../../../scenarios/mci_loan/onchain/loan-scenario.mjs")).loanScenario({ seed: sample.seed })
  : recsFullScenario({ seed: sample.seed });
if (sample.scenario === "mci-loan" && JSON.stringify(sc.sample.prepared) !== JSON.stringify(sample.prepared)) {
  throw new Error(`the prepared data (${JSON.stringify(sc.sample.prepared)}) is not the run's (${JSON.stringify(sample.prepared)}): set MCI_LOAN_PREPARED`);
}
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
const scored = scoreFigures({ fA: c.fA, fusion: c.fusion, rows, hq, Itr, idx, figures: requiredFigures(7, 3) }).figures;

const onChain = run.chainEnd.figures;
const mismatch = scored.filter((f) => onChain.find((g) => g.kind === f.kind && g.memberIndex === f.memberIndex).F !== f.F);
if (mismatch.length) throw new Error(`replay differs from the chain: ${mismatch.map((f) => `${f.name} ${f.F}`).join(", ")}`);
if (onChain.length !== scored.length) throw new Error(`${onChain.length} figures on-chain, ${scored.length} replayed`);
const F = Object.fromEntries(scored.map((f) => [f.name, f.F]));
const uFed = Math.max(F.FULL, F.FULL_J);
if (uFed !== run.chainEnd.uFed) throw new Error(`U_fed ${uFed} != the contract's ${run.chainEnd.uFed}`);
const out = {
  configPath: CONFIG_PATH.replace(/.*[\\/]results[\\/]/, "results/").replaceAll("\\", "/"), configHash: modelConfigHash(),
  scenario: { name: sample.scenario, seed: sample.seed }, sizeI: I.length, sizeItr: Itr.length, evalPositions: idx.length,
  replayEqualsChain: true, figures: F, uFed, jointBinds: F.FULL_J > F.FULL,
  l: [1, 2, 3].map((i) => uFed - F[`LOO_${i}`]), E: Math.max(0, uFed - F.BASELINE),
  machine: machine(), wallMs: performance.now() - t0, peakRssMB: peakRssMB(),
};
writeFileSync(resolve(DIR, "replay.json"), JSON.stringify(out, null, 1) + "\n");
console.log(JSON.stringify({ figures: F, uFed, jointBinds: out.jointBinds, l: out.l, E: out.E }));
