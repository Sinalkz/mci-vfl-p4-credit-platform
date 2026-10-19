// Q17 follow-up (design chat, 2026-09-29), off-chain only: the three scoring variants for one E1 seed (full test
// set) or for E3 (the protocol's 64 evaluation positions), plus a centralized leave-one-out reference.
//   (a) D118: FULL = the jointly trained fusion model; LOO_i and BASELINE retrained on the frozen embeddings
//   (b) retrained FULL: FULL retrained like the others (same config, same I_tr); LOO_i and BASELINE as in (a)
//   (c) masked LOO: FULL and LOO_i both use the joint fusion model, LOO_i with member i's slot zeroed (no
//       retraining); BASELINE retrained as in (a)
//   reference: the whole model (f_A, the other f_j, fusion) trained centrally from scratch without passive i,
//       scored with its own joint fusion model; l_i = F(joint, all) - F(joint, without i)
// The models are rebuilt, not loaded: E1's C2 runs and E3 ran without noise, so training is deterministic and the
// centralized float32 path equals split training bit for bit (test D05). The script refuses to write unless (a)
// and (b) reproduce the stored E1 figures (results/data1/e1/c2-s{seed}.json: since D139 its FULL is (b)'s F_r and
// FULL_J is (a)'s joint FULL) or the D118 on-chain E3 run (results/data1/e3-d118: its figures are (a)).
//   VFL_MODEL_CONFIG=../results/data1/configs/e1-s1.json node offchain/data1/q17-run.mjs --seed 1
//   VFL_MODEL_CONFIG=../results/data1/configs/e3.json    node offchain/data1/q17-run.mjs --e3
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { assertFiWidth, CONFIG, CONFIG_PATH, modelConfigHash } from "../model.mjs";
import { KIND, requiredFigures, scoreFigures } from "../scoring.mjs";
import { centralHq, trainCentral } from "./centralized.mjs";
import { initSeedFor } from "./configs.mjs";
import { machine, peakRssMB } from "./machine.mjs";
import { loadPrepared } from "./prepared.mjs";
import { recsScenario } from "./recs-scenario.mjs";

const RES = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "..", "results", "data1");
const OUT = resolve(RES, "q17");
const { values: a } = parseArgs({ options: { seed: { type: "string" }, e3: { type: "boolean", default: false } } });
const load = (f) => JSON.parse(readFileSync(f, "utf8"));
const t0 = performance.now();

// ---- the rows, I_tr, the evaluation positions and the figures to reproduce --------------------------
let rows, xp, Itr, idx, expected, label;
if (a.e3) {
  const d = resolve(RES, "e3-d118"); // the D118 run the design chat decided on
  const A = load(resolve(d, "A", "summary.json"));
  const I = load(resolve(d, "A", "I.json")).map((x) => BigInt(x));
  const sc = recsScenario({ seed: load(resolve(d, "sample.json")).seed });
  rows = { xa: I.map((id) => sc.A.rows.get(id).xa), y: I.map((id) => sc.A.rows.get(id).y) };
  xp = sc.passives.map((p) => I.map((id) => p.rows.get(id)));
  idx = A.idx;
  const picked = new Set(idx);
  Itr = [...I.keys()].filter((k) => !picked.has(k));
  const names = { "0,0": "FULL", "1,1": "LOO_1", "1,2": "LOO_2", "1,3": "LOO_3", "2,0": "BASELINE" };
  expected = { a: Object.fromEntries(load(resolve(d, "run.json")).chainEnd.figures.map((f) => [names[`${f.kind},${f.memberIndex}`], f.F])),
    b: load(resolve(d, "replay.json")).retrainedFull };
  label = "e3";
} else {
  const seed = Number(a.seed);
  if (CONFIG.initSeed !== initSeedFor(seed)) throw new Error(`config ${CONFIG_PATH} is not seed ${seed}'s`);
  const p = loadPrepared({});
  ({ rows, xp } = p);
  Itr = p.trainPos;
  idx = p.testPos;
  const run = load(resolve(RES, "e1", `c2-s${seed}.json`));
  const Fs = Object.fromEntries(run.figures.map((f) => [f.name, f.F])); // D139: FULL = F_r, FULL_J = F_j
  expected = { a: { FULL: Fs.FULL_J, LOO_1: Fs.LOO_1, LOO_2: Fs.LOO_2, LOO_3: Fs.LOO_3, BASELINE: Fs.BASELINE }, b: Fs.FULL };
  label = `s${seed}`;
}
xp.forEach((r, k) => assertFiWidth(k + 1, r[0].length));

const figs = requiredFigures(7, 3);
// the joint model with slot i zeroed: a FULL_JOINT figure with LOO_i's mask (analysis only)
const loo = [1, 2, 3].map((i) => ({ name: `LOO_${i}`, kind: KIND.FULL_JOINT, memberIndex: i, mask: 7 & ~(1 << (i - 1)) }));
const pick = (f) => ({ F: f.F, preds: f.preds });

// ---- the jointly trained model: variants (a), (b), (c) ---------------------------------------------
const c = trainCentral({ rows, xp, Itr, mode: "f32" });
const hq = centralHq(c.fis, xp, rows.y.length);
const all = Object.fromEntries(scoreFigures({ fA: c.fA, fusion: c.fusion, rows, hq, Itr, idx, figures: figs }).figures.map((f) => [f.name, f]));
const va = [{ ...all.FULL_J, name: "FULL" }, all.LOO_1, all.LOO_2, all.LOO_3, all.BASELINE]; // (a): the joint FULL
const vb = all.FULL; // (b): the retrained FULL
const vc = scoreFigures({ fA: c.fA, fusion: c.fusion, rows, hq, Itr, idx, figures: loo }).figures; // joint head, slot i zeroed

const Fa = Object.fromEntries(va.map((f) => [f.name, f.F]));
const bad = Object.keys(expected.a).filter((n) => expected.a[n] !== Fa[n]);
if (bad.length || vb.F !== expected.b) {
  throw new Error(`${label}: rebuilt figures differ from the stored ones: ${bad.map((n) => `${n} ${Fa[n]} != ${expected.a[n]}`).join(", ")}` +
    (vb.F !== expected.b ? ` retrained FULL ${vb.F} != ${expected.b}` : ""));
}

// ---- the centralized leave-one-out reference ---------------------------------------------------------
const ref = [1, 2, 3].map((i) => {
  const ci = trainCentral({ rows, xp, Itr, mode: "f32", exclude: [i] });
  const f = scoreFigures({ fA: ci.fA, fusion: ci.fusion, rows, hq: centralHq(ci.fis, xp, rows.y.length), Itr, idx,
    figures: [loo[i - 1]] }).figures[0];
  return pick(f);
});

const byName = Object.fromEntries(va.map((f) => [f.name, pick(f)]));
const out = {
  label, configPath: CONFIG_PATH, configHash: modelConfigHash(), sizeItr: Itr.length, evalPositions: idx.length,
  reproduced: { a: Fa, b: vb.F },
  figures: {
    FULL_joint: byName.FULL, BASELINE: byName.BASELINE,
    LOO_retrained: [1, 2, 3].map((i) => byName[`LOO_${i}`]),
    FULL_retrained: pick(vb),
    LOO_masked: vc.map(pick),
    LOO_central: ref,
  },
  machine: machine(), wallMs: performance.now() - t0, peakRssMB: peakRssMB(),
};
mkdirSync(OUT, { recursive: true });
writeFileSync(resolve(OUT, `${label}.json`), JSON.stringify(out) + "\n");
const F = (x) => x.F;
console.log(JSON.stringify({ label, a: Fa, b: vb.F, c: vc.map(F), ref: ref.map(F), wallS: Math.round(out.wallMs / 1000) }));
