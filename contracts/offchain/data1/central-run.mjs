// DATA1: one centralized reference run (centralized.mjs) on the prepared data, evaluated on the full
// test set through the circuit's integer path. Writes one JSON.
//   VFL_MODEL_CONFIG=<config> node offchain/data1/central-run.mjs --mode c4a64|c4a32|c1b --seed s --out <file>
//   c4a64  C4a: the same bottom models and fusion model trained jointly, float64, no split, no noise
//   c4a32  C4a with float32 rounding where split training transports (the bit-exact check against C2)
//   c1b    C1b: f_A and the fusion model trained from scratch on A's 8 columns only (passive slots zero)
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { CONFIG, CONFIG_PATH, modelConfigHash, quantFA, quantHead } from "../model.mjs";
import { centralHq, trainCentral } from "./centralized.mjs";
import { fOn, predictInt, saturation } from "./evaluate.mjs";
import { initSeedFor } from "./configs.mjs";
import { machine, peakRssMB } from "./machine.mjs";
import { loadPrepared } from "./prepared.mjs";

const { values: a } = parseArgs({ options: { mode: { type: "string" }, seed: { type: "string" }, out: { type: "string" } } });
const MODES = { c4a64: "f64", c4a32: "f32", c1b: "aOnly" };
if (!MODES[a.mode]) throw new Error("--mode c4a64|c4a32|c1b");
const seed = Number(a.seed);
if (CONFIG.initSeed !== initSeedFor(seed)) throw new Error(`config ${CONFIG_PATH} has initSeed ${CONFIG.initSeed}, not seed ${seed}'s`);
const m0 = machine();
const t0 = performance.now();
const { rows, xp, trainPos, testPos, widths } = loadPrepared({});
const testSet = new Set(testPos);
if (trainPos.some((k) => testSet.has(k))) throw new Error("training and test positions overlap");
const tTrain = performance.now();
const c = trainCentral({ rows, xp, Itr: trainPos, mode: MODES[a.mode] });
const trainingMs = performance.now() - tTrain;
const fa = quantFA(c.fA);
const head = quantHead(c.fusion);
const hq = a.mode === "c1b" ? [null, null, null] : centralHq(c.fis, xp, rows.y.length);
const preds = predictInt({ fa, head, mask: a.mode === "c1b" ? 0 : 7, rows, hq, positions: testPos });
const out = {
  kind: a.mode, seed, configPath: CONFIG_PATH, configHash: modelConfigHash(), widths, machine: m0, freeRamMBAtEnd: machine().freeRamMB,
  wallMs: performance.now() - t0, trainingMs, peakRssMB: peakRssMB(), rounds: c.rounds, F: fOn(rows, testPos, preds), preds,
  clipping: { hq: a.mode === "c1b" ? [0, 0, 0] : saturation(hq) },
};
mkdirSync(dirname(resolve(a.out)), { recursive: true });
writeFileSync(resolve(a.out), JSON.stringify(out) + "\n");
console.log(JSON.stringify({ kind: out.kind, seed, F: out.F, wallS: Math.round(out.wallMs / 1000), peakRssMB: out.peakRssMB }));
process.exit(0);
