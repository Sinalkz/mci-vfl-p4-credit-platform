// MCI loan instance: one centralized reference run on the loan data, an adapted copy of
// contracts/offchain/data1/central-run.mjs (the thesis's centralized.mjs does the training, unchanged), evaluated on
// the whole test split through the circuit's integer path. Differences from the RECS script:
//   - the rows come from scenarios/mci_loan/offchain/prepared.mjs (MCI_LOAN_PREPARED selects the folder);
//   - the run also carries its off-chain score on the test rows (score.mjs: the margin z_1 - z_0, decision (a)) and
//     the number of rows, training and test, that its head puts in classes 2-4.
//   VFL_MODEL_CONFIG=<config> node scenarios/mci_loan/offchain/central-run.mjs --mode c4a64|c4a32|c1b --seed s --out <file>
//   c4a64  C4a: the same bottom models and fusion model trained jointly, float64, no split, no noise
//   c4a32  C4a with float32 rounding where split training transports (the bit-exact check against C2)
//   c1b    C1b: f_A and the fusion model trained from scratch on A's 8 columns only (passive slots zero)
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { centralHq, trainCentral } from "../../../contracts/offchain/data1/centralized.mjs";
import { initSeedFor } from "../../../contracts/offchain/data1/configs.mjs";
import { fOn, predictInt, saturation } from "../../../contracts/offchain/data1/evaluate.mjs";
import { machine, peakRssMB } from "../../../contracts/offchain/data1/machine.mjs";
import { CONFIG, CONFIG_PATH, modelConfigHash, quantFA, quantHead } from "../../../contracts/offchain/model.mjs";
import { PREPARED, loadPrepared } from "./prepared.mjs";
import { scoreRows } from "./score.mjs";

const { values: a } = parseArgs({ options: { mode: { type: "string" }, seed: { type: "string" }, out: { type: "string" } } });
const MODES = { c4a64: "f64", c4a32: "f32", c1b: "aOnly" };
if (!MODES[a.mode]) throw new Error("--mode c4a64|c4a32|c1b");
const seed = Number(a.seed);
if (CONFIG.initSeed !== initSeedFor(seed)) throw new Error(`config ${CONFIG_PATH} has initSeed ${CONFIG.initSeed}, not seed ${seed}'s`);
const m0 = machine();
const t0 = performance.now();
const { rows, xp, trainPos, testPos, widths, manifest } = loadPrepared({});
const testSet = new Set(testPos);
if (trainPos.some((k) => testSet.has(k))) throw new Error("training and test positions overlap");
const tTrain = performance.now();
const c = trainCentral({ rows, xp, Itr: trainPos, mode: MODES[a.mode] });
const trainingMs = performance.now() - tTrain;
const fa = quantFA(c.fA);
const head = quantHead(c.fusion);
const hq = a.mode === "c1b" ? [null, null, null] : centralHq(c.fis, xp, rows.y.length);
const mask = a.mode === "c1b" ? 0 : 7;
const preds = predictInt({ fa, head, mask, rows, hq, positions: testPos });
const test = scoreRows({ fa, head, mask, rows, hq, positions: testPos });
if (test.preds.some((p, t) => p !== preds[t])) throw new Error("score helper predictions differ from predictInt's");
const train = scoreRows({ fa, head, mask, rows, hq, positions: trainPos });
const out = {
  kind: a.mode, seed, configPath: CONFIG_PATH, configHash: modelConfigHash(), widths,
  prepared: { dir: PREPARED, aSha256: manifest.a.sha256, passiveSha256: manifest.passive.sha256 },
  machine: m0, freeRamMBAtEnd: machine().freeRamMB, wallMs: performance.now() - t0, trainingMs, peakRssMB: peakRssMB(),
  rounds: c.rounds, F: fOn(rows, testPos, preds), preds, margins: test.margins,
  unusedClass: { test: test.unusedClass, train: train.unusedClass },
  clipping: { hq: a.mode === "c1b" ? [0, 0, 0] : saturation(hq) },
};
mkdirSync(dirname(resolve(a.out)), { recursive: true });
writeFileSync(resolve(a.out), JSON.stringify(out) + "\n");
console.log(JSON.stringify({ kind: out.kind, seed, F: out.F, unusedClass: test.unusedClass + train.unusedClass,
  wallS: Math.round(out.wallMs / 1000), peakRssMB: out.peakRssMB }));
process.exit(0);
