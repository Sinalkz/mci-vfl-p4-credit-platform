// Task 5 (LDP configurations): one LDP run of the thesis code, as vfl-run.mjs --mode ldp --sigmas does it (the same
// in-process sessions, wallets, seeded noise streams, activeTraining + 3 passiveTraining, scoreFigures on the full test
// set), with three settings vfl-run does not take; vfl-run.mjs itself is not changed:
//   --clip-scale k   each passive's clip bound is LOCAL_NOISE's x k (4/4/3 at k = 1)
//   --frac f         activeTraining and scoreFigures use a stratified subsample of the 12,772 training rows
//                    (ldp-cfg-lib.mjs stratifiedSubsample, seeded by --seed); every row of I still gets h_i
//   --cap-off-too    every passive runs through ldp-cfg-lib.mjs shadowPassive, and the figures are scored a second
//                    time on the final matrices before the int16 cap (the same 2^-12 resolution, no range limit);
//                    the protocol's own (capped) figures are unchanged
// T comes from the model config (VFL_MODEL_CONFIG, ldp-cfg-lib.mjs variantConfig); --seed must match it.
// At --clip-scale 1 --frac 1 with E1's config and the same --sigmas, the run is vfl-run.mjs --mode ldp --sigmas.
//   VFL_MODEL_CONFIG=<config> node offchain/data1/ldp-cfg-run.mjs --seed s --sigmas s1,s2,s3 [--clip-scale k]
//       [--frac f] [--cap-off-too] --out <file>
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { Wallet } from "ethers";

import { gauss, prng } from "../data.mjs";
import { CONFIG, CONFIG_PATH, E, fiSpec, modelConfigHash } from "../model.mjs";
import { LOCAL_NOISE } from "../party-m3.mjs";
import { requiredFigures, scoreFigures } from "../scoring.mjs";
import { activeTraining, passiveTraining } from "../training.mjs";
import { saturation } from "./evaluate.mjs";
import { sessionPairs } from "./inproc.mjs";
import { SUBSAMPLE_SEED, beyondInt16, shadowPassive, stratifiedSubsample } from "./ldp-cfg-lib.mjs";
import { machine, peakRssMB } from "./machine.mjs";
import { loadPrepared } from "./prepared.mjs";
import { initSeedFor } from "./configs.mjs";

const { values: a } = parseArgs({ options: {
  seed: { type: "string" }, sigmas: { type: "string" }, "clip-scale": { type: "string", default: "1" },
  frac: { type: "string", default: "1" }, "cap-off-too": { type: "boolean", default: false }, out: { type: "string" },
  "wait-ms": { type: "string", default: "120000" }, "max-depth": { type: "string", default: "14" },
} });
const sigmas = (a.sigmas ?? "").split(",").map(Number);
if (sigmas.length !== 3 || !sigmas.every((x) => Number.isFinite(x) && x >= 0)) throw new Error("--sigmas s1,s2,s3");
const clipScale = Number(a["clip-scale"]);
const frac = Number(a.frac);
if (!(clipScale > 0) || !(frac > 0 && frac <= 1)) throw new Error("bad --clip-scale or --frac");
const capOff = a["cap-off-too"];
const seed = Number(a.seed);
if (CONFIG.initSeed !== initSeedFor(seed)) throw new Error(`config ${CONFIG_PATH} has initSeed ${CONFIG.initSeed}, not seed ${seed}'s`);
const m0 = machine();
const t0 = performance.now();

const d = loadPrepared();
const { rows, xp, trainPos, testPos, widths } = d;
const sizeI = rows.y.length;
widths.forEach((w, i) => { if (fiSpec(i + 1).in !== w) throw new Error(`passive ${i + 1}: config width ${fiSpec(i + 1).in} != data width ${w}`); });
const testSet = new Set(testPos);
if (trainPos.some((k) => testSet.has(k))) throw new Error("training and test positions overlap");
if (sizeI > 65535) throw new Error("positions travel as u16");
const Itr = stratifiedSubsample(trainPos, rows.y, frac, seed);

const noises = [0, 1, 2].map((i) => ({ clip: LOCAL_NOISE[i].clip * clipScale, sigma: sigmas[i] }));
const noiseSeeds = [1, 2, 3].map((i) => 20263000 + 10 * seed + i);
const normals = noiseSeeds.map((ns) => { const r = prng(BigInt(ns)); return () => gauss(r); });
const maxDepth = Number(a["max-depth"]);
const ws = [0, 1, 2, 3].map(() => Wallet.createRandom());
const chainId = 2026n;
const contract = ws[0].address; // any address: the EIP-712 domain of this off-chain run
const { pairs, close } = await sessionPairs(3, { chainId, contract, activeParty: ws[0].address, passives: ws.slice(1).map((w) => w.address) });
const shadows = capOff ? pairs.map((x, i) => shadowPassive({ session: x.p, index: i + 1, xp: xp[i], sizeI, noise: noises[i], normal: normals[i] })) : null;

const tTrain = performance.now();
const [act, ...pas] = await Promise.all([
  activeTraining({ peers: pairs.map((x, i) => ({ index: i + 1, address: ws[i + 1].address, session: x.a })), rows, Itr, sizeI,
    chainId, contract, maxDepth, waitMs: Number(a["wait-ms"]) }),
  ...pairs.map((x, i) => passiveTraining({ session: shadows ? shadows[i].session : x.p, index: i + 1, xp: xp[i], sizeI, chainId, contract,
    maxDepth, signer: ws[i + 1], noise: noises[i], normal: shadows ? shadows[i].normal : normals[i] })),
]);
const trainingMs = performance.now() - tTrain;
close();
if (act.zeroFilled.length > 0) throw new Error(`${act.zeroFilled.length} zero-filled batches`);
if (!act.finals.every((f) => f.received && f.signatureValid)) throw new Error("a final message is missing or badly signed");

const hq = pas.map((p) => p.hq);
const trainNorms = [];
for (const m of hq) for (const k of Itr) trainNorms.push(Math.sqrt(m[k].reduce((s, v) => s + (v / 4096) ** 2, 0)));
trainNorms.sort((x, y) => x - y);
const pct = (p) => { const h = (trainNorms.length - 1) * p; const lo = Math.floor(h); return trainNorms[lo] + (trainNorms[Math.min(lo + 1, trainNorms.length - 1)] - trainNorms[lo]) * (h - lo); };

const figs = requiredFigures(7, 3);
const fig = (f) => ({ name: f.name, kind: f.kind, memberIndex: f.memberIndex, mask: f.mask, F: f.F, retrained: f.retrained, trainMs: f.trainMs, preds: f.preds });
const tScore = performance.now();
const scored = scoreFigures({ fA: act.fA, fusion: act.fusion, rows, hq, Itr, idx: testPos, figures: figs });
const scoringMs = performance.now() - tScore;
let capOffOut = null;
if (capOff) {
  const tOff = performance.now();
  const hqOff = shadows.map((s, i) => s.uncapped(hq[i]));
  const off = scoreFigures({ fA: act.fA, fusion: act.fusion, rows, hq: hqOff, Itr, idx: testPos, figures: figs });
  capOffOut = { what: "the same run scored on the final matrices before the int16 cap (2^-12 resolution, no range limit)",
    shadowCheck: "q(v, 12, 16) of every uncapped entry equals the sent int16 entry", beyondInt16: beyondInt16(hqOff),
    scoringMs: performance.now() - tOff, figures: off.figures.map(fig) };
}

const out = {
  kind: "ldp-cfg", seed, configPath: CONFIG_PATH, configHash: modelConfigHash(), widths, T: CONFIG.T, clipScale, frac,
  train: { n: Itr.length, subsampleSeed: frac === 1 ? null : SUBSAMPLE_SEED(seed),
    posSha256: createHash("sha256").update(JSON.stringify(Itr)).digest("hex") },
  noise: { sigmas, perPassive: noises, noiseSeeds },
  waitMs: Number(a["wait-ms"]), maxDepth,
  machine: m0, freeRamMBAtEnd: machine().freeRamMB, wallMs: performance.now() - t0, trainingMs, scoringMs, peakRssMB: peakRssMB(),
  rows: { total: sizeI, train: Itr.length, test: testPos.length,
    testPosSha256: createHash("sha256").update(JSON.stringify(testPos)).digest("hex") },
  training: { rounds: act.rounds.map(({ round, ms, loss, batches }) => ({ round, ms, loss, batches })), zeroFilled: act.zeroFilled,
    finals: act.finals.map((f) => ({ passive: f.passive, received: f.received, signatureValid: f.signatureValid, bytes: f.bytes })),
    passives: pas.map((p) => ({ stats: p.stats, ms: p.ms })) },
  clipping: { hq: saturation(hq), entries: sizeI * E },
  embNormTrain: { n: trainNorms.length, p50: pct(0.5), p90: pct(0.9), p95: pct(0.95), p99: pct(0.99), max: trainNorms.at(-1) },
  figures: scored.figures.map(fig),
  capOff: capOffOut,
};
mkdirSync(dirname(resolve(a.out)), { recursive: true });
writeFileSync(resolve(a.out), JSON.stringify(out) + "\n");
console.log(JSON.stringify({ kind: out.kind, seed, T: out.T, clipScale, frac, F: Object.fromEntries(out.figures.map((f) => [f.name, f.F])),
  ...(capOffOut ? { Fcapoff: Object.fromEntries(capOffOut.figures.map((f) => [f.name, f.F])) } : {}),
  wallS: Math.round(out.wallMs / 1000), trainingS: Math.round(trainingMs / 1000), peakRssMB: out.peakRssMB, clipping: out.clipping.hq }));
process.exit(0);
