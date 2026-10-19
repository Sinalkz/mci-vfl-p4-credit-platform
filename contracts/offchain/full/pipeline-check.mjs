// full-data diagnosis (item 3): the full-data runs' off-chain pipeline without the 64-row evaluation noise. The same
// functions as the runs: the full-data scenario (full/recs-full-scenario.mjs, seed 20265000) with A's rows and each
// passive's rows taken by national ID in the order of I (as party.mjs / party-m3.mjs take them), split training over
// Sessions (training.mjs activeTraining + 3 passiveTraining, in-process as data1/vfl-run.mjs), each passive's final
// round (clipAndNoise with its setting, then int16 at 2^12: the fixed-point final matrix), then scoreFigures
// (scoring.mjs: F_r retrained, F_j the jointly trained fusion model, LOO_1..3 and BASELINE retrained, frozen f_A).
// Only the rows differ from a run:
//   training rows  I^tr = the positions of I whose household is in E1's training split (manifest split.train_pos)
//   evaluation     all 3,193 households of E1's test split (manifest split.test_pos), in test_pos order
// A test household outside I (some passive misses it in the scenario) has no row in a run. Here it is appended after
// the |I| rows of I, with its prepared columns (data/recs/prepared/, the columns its passives would hold); training
// never reads it (it is not in I^tr), and the final round computes it row by row after the rows of I, so the rows of
// I are unchanged by it (with noise on, its noise draws come after those of I's rows). F is also reported on the test
// households in I and outside I separately.
// Noise: --noise off is the runs' setting (runner3.mjs passes --noise off: clip Infinity, sigma 0); --noise ldp is
// the passives' protocol setting (party-m3.mjs LOCAL_NOISE, sigma x 1) with the LDP study's seeded normals
// (data1/vfl-run.mjs: noise seed 20263000 + 10 seed + i), for comparison with results/ldp/runs/ldp-x1-s<seed>.json.
// Also each figure's F on E1's 1,000 N = 64 draws (n64.mjs's rule: the same rows as results/data1/e1/n64*.json).
//   VFL_MODEL_CONFIG=../results/data1/configs/e1-s<k>.json node offchain/full/pipeline-check.mjs --seed k --noise off|ldp \
//       --out ../results/full/diagnosis/pipeline/pc-<noise>-s<k>.json [--draws 1000]
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { Wallet } from "ethers";

import { sampleIndices } from "../../../circuits/scripts/members-lib.mjs";
import { gauss, prng } from "../data.mjs";
import { assertFiWidth, CONFIG, CONFIG_PATH, macroF1, modelConfigHash } from "../model.mjs";
import { LOCAL_NOISE } from "../party-m3.mjs";
import { requiredFigures, scoreFigures } from "../scoring.mjs";
import { activeTraining, passiveTraining } from "../training.mjs";
import { initSeedFor } from "../data1/configs.mjs";
import { saturation } from "../data1/evaluate.mjs";
import { sessionPairs } from "../data1/inproc.mjs";
import { machine, peakRssMB } from "../data1/machine.mjs";
import { openPrepared, PASSIVE_NAMES } from "../data1/prepared.mjs";
import { FULL_SEED, recsFullScenario } from "./recs-full-scenario.mjs";

const { values: a } = parseArgs({ options: { seed: { type: "string" }, noise: { type: "string" }, out: { type: "string" },
  draws: { type: "string", default: "1000" }, "wait-ms": { type: "string", default: "120000" }, "max-depth": { type: "string", default: "14" } } });
if (!["off", "ldp"].includes(a.noise)) throw new Error("--noise off|ldp");
const seed = Number(a.seed);
if (CONFIG.initSeed !== initSeedFor(seed)) throw new Error(`config ${CONFIG_PATH} has initSeed ${CONFIG.initSeed}, not seed ${seed}'s`);
const m0 = machine();
const t0 = performance.now();
const HERE = fileURLToPath(import.meta.url);
const RESULTS = resolve(HERE, "..", "..", "..", "..", "results");

// ---- rows ---------------------------------------------------------------------------------
const sc = recsFullScenario({ seed: FULL_SEED });
const I = sc.I; // sorted ascending: the order of I that A commits (gamma_I) and every party uses
const runI = resolve(RESULTS, "full", "runs", `run-${seed}`, "A", "I.json");
if (existsSync(runI) && JSON.stringify(JSON.parse(readFileSync(runI, "utf8"))) !== JSON.stringify(I.map(String))) throw new Error(`I differs from ${runI}`);
const { m, P, W } = openPrepared();
const idsInK = [...sc.A.rows.keys()]; // A's rows in prepared order k
const kOf = new Map(idsInK.map((id, k) => [id, k]));
const trainK = new Set(m.split.train_pos);
const testPos = m.split.test_pos;
const posOfK = new Map(I.map((id, p) => [kOf.get(id), p]));
const extraK = testPos.filter((k) => !posOfK.has(k)); // E1 test households outside I, appended in test_pos order
extraK.forEach((k, r) => posOfK.set(k, I.length + r));
const sizeI = I.length + extraK.length;
const cols = PASSIVE_NAMES.map((n) => m.parties[n]);
const rows = {
  xa: [...I.map((id) => sc.A.rows.get(id).xa), ...extraK.map((k) => sc.A.rows.get(idsInK[k]).xa)],
  y: [...I.map((id) => sc.A.rows.get(id).y), ...extraK.map((k) => sc.A.rows.get(idsInK[k]).y)],
};
const xp = sc.passives.map((p, i) => [
  ...I.map((id) => { const r = p.rows.get(id); if (!r) throw new Error(`id ${id} of I is missing at passive ${i + 1}`); return r; }),
  ...extraK.map((k) => cols[i].map((j) => P[k * W + j])),
]);
xp.forEach((x, i) => assertFiWidth(i + 1, x[0].length));
if (rows.y.some((y, p) => y !== m.y[p < I.length ? kOf.get(I[p]) : extraK[p - I.length]])) throw new Error("labels differ from the manifest");
const Itr = I.map((id, p) => (trainK.has(kOf.get(id)) ? p : -1)).filter((p) => p >= 0);
const idx = testPos.map((k) => posOfK.get(k)); // evaluation positions, aligned with test_pos (E1's preds order)
const trainSet = new Set(Itr);
if (idx.some((p) => trainSet.has(p))) throw new Error("training and evaluation rows overlap");
if (sizeI > 65535 || sizeI > 2 ** Number(a["max-depth"])) throw new Error(`sizeI ${sizeI} too large`);

// ---- training: the runs' functions over in-process Sessions ---------------------------------
const noises = [0, 1, 2].map((i) => (a.noise === "off" ? { clip: Infinity, sigma: 0 } : { ...LOCAL_NOISE[i] }));
const noiseSeeds = [1, 2, 3].map((i) => 20263000 + 10 * seed + i);
const normals = noiseSeeds.map((ns) => { const r = prng(BigInt(ns)); return () => gauss(r); });
const ws = [0, 1, 2, 3].map(() => Wallet.createRandom());
const chainId = 2026n;
const contract = ws[0].address;
const maxDepth = Number(a["max-depth"]);
const { pairs, close } = await sessionPairs(3, { chainId, contract, activeParty: ws[0].address, passives: ws.slice(1).map((w) => w.address) });
const tTrain = performance.now();
const [act, ...pas] = await Promise.all([
  activeTraining({ peers: pairs.map((x, i) => ({ index: i + 1, address: ws[i + 1].address, session: x.a })), rows, Itr, sizeI,
    chainId, contract, maxDepth, waitMs: Number(a["wait-ms"]) }),
  ...pairs.map((x, i) => passiveTraining({ session: x.p, index: i + 1, xp: xp[i], sizeI, chainId, contract, maxDepth,
    signer: ws[i + 1], noise: noises[i], normal: normals[i] })),
]);
const trainingMs = performance.now() - tTrain;
close();
if (act.zeroFilled.length > 0) throw new Error(`${act.zeroFilled.length} zero-filled batches`);
if (!act.finals.every((f) => f.received && f.signatureValid)) throw new Error("a final message is missing or badly signed");
const hq = act.finals.map((f) => f.hq); // the final matrices as A received them (int16 at 2^12)

// ---- scoring: the runs' scoreFigures on all test households ---------------------------------
const tScore = performance.now();
const scored = scoreFigures({ fA: act.fA, fusion: act.fusion, rows, hq, Itr, idx, figures: requiredFigures(7, 3) });
const scoringMs = performance.now() - tScore;
const inI = idx.map((p) => p < I.length);
const sub = (f, keep) => macroF1(idx.filter((_, t) => keep(t)).map((p) => rows.y[p]), f.preds.filter((_, t) => keep(t))).F;
const draws = [];
for (let j = 1; j <= Number(a.draws); j++) {
  const ts = await sampleIndices(BigInt(j), 64, testPos.length);
  const labels = ts.map((t) => rows.y[idx[t]]);
  draws.push(Object.fromEntries(scored.figures.map((f) => [f.name, macroF1(labels, ts.map((t) => f.preds[t])).F])));
}

const out = {
  what: "full-data diagnosis item 3: the runs' pipeline trained on I ∩ E1's training split, scored on all of E1's test split",
  seed, noise: a.noise === "off" ? "off (the runs' setting)" : { perPassive: noises, noiseSeeds }, configPath: CONFIG_PATH.replace(/.*[\\/]results[\\/]/, "results/").replaceAll("\\", "/"),
  configHash: modelConfigHash(), scenarioSeed: FULL_SEED, waitMs: Number(a["wait-ms"]), maxDepth,
  rows: { sizeI: I.length, appendedTestOutsideI: extraK.length, total: sizeI, train: Itr.length, test: idx.length, testInI: inI.filter(Boolean).length,
    testOutsideI: inI.filter((x) => !x).length, e1Train: m.split.train_pos.length },
  machine: m0, freeRamMBAtEnd: machine().freeRamMB, wallMs: performance.now() - t0, trainingMs, scoringMs, peakRssMB: peakRssMB(),
  training: { rounds: act.rounds.map(({ round, ms, loss, batches }) => ({ round, ms, loss, batches })), zeroFilled: act.zeroFilled.length,
    finals: act.finals.map((f) => ({ passive: f.passive, received: f.received, signatureValid: f.signatureValid })) },
  saturation: { hq: saturation(hq), entries: sizeI * 4 },
  figures: scored.figures.map((f) => ({ name: f.name, kind: f.kind, mask: f.mask, F: f.F, FinI: sub(f, (t) => inI[t]), FoutsideI: sub(f, (t) => !inI[t]),
    retrained: f.retrained, trainMs: f.trainMs, preds: f.preds })),
  testDoeid: testPos.map((k) => m.doeid[k]),
  draws: { n: draws.length, N: 64, rule: "sampleIndices(j, 64, 3193) over test_pos, j = 1..n (n64.mjs)", F: draws },
};
mkdirSync(dirname(resolve(a.out)), { recursive: true });
writeFileSync(resolve(a.out), JSON.stringify(out) + "\n");
console.log(JSON.stringify({ seed, noise: a.noise, F: Object.fromEntries(out.figures.map((f) => [f.name, f.F])), rows: out.rows,
  wallS: Math.round(out.wallMs / 1000), trainingS: Math.round(trainingMs / 1000), peakRssMB: out.peakRssMB, saturation: out.saturation.hq }));
process.exit(0);
