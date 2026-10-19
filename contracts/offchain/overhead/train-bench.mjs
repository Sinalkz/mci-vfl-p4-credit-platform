// Task 4 item 1: time per training round, plain split learning vs the protocol's extra steps (train-variants.mjs),
// for one model-config seed and one row setting, in one process. Started by overhead/run.mjs, one at a time.
//   e1  DATA1 E1's rows: all 15,965 prepared households, I^tr = the 12,772 training households (200 batches a round),
//       final-round trees at maxDepth 14 (vfl-run.mjs's default)
//   e3  E3's rows (the 1,020-household subset): I = results/data1/e3/A/I.json (910), I^tr = I minus A's 64 evaluation
//       draws (848 rows, 14 batches a round), maxDepth 10
// First a warm-up (discarded): every variant and the no-transport reference once with --warm-t rounds. Then --reps
// repetitions of [local, plain, clipnoise, protocol], in that order on even repetitions and reversed on odd ones.
// local = DATA1's centralized float32 training (data1/centralized.mjs trainCentral, mode f32): the same arithmetic in
// one call stack, no messages at all; it has no per-round timer, so its round time is its total / T.
// Every repetition checks that plain's per-round losses equal local's bit for bit.
//   VFL_MODEL_CONFIG=../results/data1/configs/e1-s<seed>.json node offchain/overhead/train-bench.mjs
//       --setting e1|e3 --seed <1..5> [--reps 2] [--warm-t 1] --out <file>
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { initSeedFor } from "../data1/configs.mjs";
import { trainCentral } from "../data1/centralized.mjs";
import { machine, peakRssMB } from "../data1/machine.mjs";
import { loadPrepared } from "../data1/prepared.mjs";
import { recsScenario } from "../data1/recs-scenario.mjs";
import { CONFIG, CONFIG_PATH, fiSpec, modelConfigHash } from "../model.mjs";
import { trainOnce } from "./train-variants.mjs";

const ROOT = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "..");
export const ORDER = ["local", "plain", "clipnoise", "protocol"];

/** The rows of a setting: { rows, xp, Itr, sizeI, maxDepth, source }. */
export function settingRows(setting) {
  if (setting === "e1") {
    const d = loadPrepared({});
    return { rows: d.rows, xp: d.xp, Itr: d.trainPos, sizeI: d.rows.y.length, maxDepth: 14,
      source: "data/recs/prepared (all households; I^tr = split.train_pos)" };
  }
  if (setting === "e3") {
    const dir = resolve(ROOT, "results", "data1", "e3");
    const load = (f) => JSON.parse(readFileSync(resolve(dir, f), "utf8"));
    const I = load("A/I.json").map((x) => BigInt(x));
    const idx = load("A/summary.json").idx;
    const sc = recsScenario({ seed: load("sample.json").seed });
    const rows = { xa: I.map((id) => sc.A.rows.get(id).xa), y: I.map((id) => sc.A.rows.get(id).y) };
    const xp = sc.passives.map((p) => I.map((id) => p.rows.get(id)));
    const picked = new Set(idx);
    return { rows, xp, Itr: [...I.keys()].filter((k) => !picked.has(k)), sizeI: I.length, maxDepth: 10,
      source: "results/data1/e3: A/I.json, A/summary.json idx, sample.json seed (data1/recs-scenario.mjs)" };
  }
  throw new Error(`setting ${setting}: e1 or e3`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values: a } = parseArgs({ options: { setting: { type: "string" }, seed: { type: "string" }, reps: { type: "string", default: "2" },
    "warm-t": { type: "string", default: "1" }, out: { type: "string" } } });
  const seed = Number(a.seed);
  if (CONFIG.initSeed !== initSeedFor(seed)) throw new Error(`config ${CONFIG_PATH} has initSeed ${CONFIG.initSeed}, not seed ${seed}'s`);
  const m0 = machine();
  const t0 = performance.now();
  const d = settingRows(a.setting);
  d.xp.forEach((x, i) => { if (fiSpec(i + 1).in !== x[0].length) throw new Error(`passive ${i + 1}: width ${x[0].length} != config ${fiSpec(i + 1).in}`); });
  const loadMs = performance.now() - t0;
  const T = CONFIG.T;
  const warmT = Number(a["warm-t"]);

  const runLocal = (t) => { const s = performance.now(); const c = trainCentral({ rows: d.rows, xp: d.xp, Itr: d.Itr, mode: "f32", T: t }); return { ms: performance.now() - s, c }; };
  const warmup = {};
  for (const v of ORDER) {
    const s = performance.now();
    if (v === "local") runLocal(warmT); else await trainOnce({ variant: v, ...d, T: warmT });
    warmup[v] = performance.now() - s;
  }
  const runs = [];
  for (let rep = 0; rep < Number(a.reps); rep++) {
    const order = rep % 2 === 0 ? ORDER : [...ORDER].reverse();
    const got = {};
    for (const [pos, v] of order.entries()) {
      if (v === "local") {
        const { ms, c } = runLocal(T);
        got.local = c.rounds.map((r) => r.loss);
        runs.push({ rep, pos, variant: v, T, totalMs: ms, perRoundMs: ms / T, rounds: c.rounds.map((r) => ({ round: r.round, loss: r.loss })) });
      } else {
        const r = await trainOnce({ variant: v, ...d, T });
        if (v === "plain") got.plain = r.rounds.map((x) => x.loss);
        runs.push({ rep, pos, ...r });
      }
      console.log(JSON.stringify({ setting: a.setting, seed, rep, variant: v, ms: Math.round(runs.at(-1).totalMs) }));
    }
    if (JSON.stringify(got.plain) !== JSON.stringify(got.local)) throw new Error(`rep ${rep}: plain's losses differ from local's`);
  }
  const out = {
    item: 1, setting: a.setting, seed, configPath: CONFIG_PATH, configHash: modelConfigHash(), T, batchSize: CONFIG.batchSize,
    sizeI: d.sizeI, sizeItr: d.Itr.length, batchesPerRound: Math.ceil(d.Itr.length / CONFIG.batchSize), maxDepth: d.maxDepth, rowsSource: d.source,
    widths: d.xp.map((x) => x[0].length), order: ORDER, warmT, warmupMs: warmup, plainEqualsLocal: true,
    machine: m0, freeRamMBAtEnd: machine().freeRamMB, loadMs, wallMs: performance.now() - t0, peakRssMB: peakRssMB(), runs,
  };
  mkdirSync(dirname(resolve(a.out)), { recursive: true });
  writeFileSync(resolve(a.out), JSON.stringify(out) + "\n");
  process.exit(0);
}
