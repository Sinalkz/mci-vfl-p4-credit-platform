// MCI loan instance: one off-chain VFL run on the loan data, an adapted copy of contracts/offchain/data1/vfl-run.mjs.
// The thesis's own code does the work, unchanged: activeTraining + 3 passiveTraining (training.mjs) over in-process
// sessions, then scoreFigures (scoring.mjs: FULL = F_r retrained, FULL_J = F_j the jointly trained fusion model,
// LOO_1..3 and BASELINE retrained) on the whole test split. No chain, no proofs. Differences from the RECS script:
//   - the rows come from scenarios/mci_loan/offchain/prepared.mjs (MCI_LOAN_PREPARED selects the folder);
//   - the members' accounts are the development wallets 0..3 (contracts/offchain/chain.mjs devWallet), not random
//     ones: the accounts only sign the final messages, so no figure depends on them;
//   - each figure also carries its off-chain score on the test rows (score.mjs: the margin z_1 - z_0 on the circuit's
//     integer path, decision (a)) and the number of rows, training and test, that its head puts in classes 2-4.
//   VFL_MODEL_CONFIG=<config> node scenarios/mci_loan/offchain/vfl-run.mjs --mode c2|c3|ldp --seed s [--clip C --sigma S]
//       [--sigma-scale k | --sigmas s1,s2,s3] [--partition <json>] --out <file>
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { devWallet } from "../../../contracts/offchain/chain.mjs";
import { gauss, prng } from "../../../contracts/offchain/data.mjs";
import { initSeedFor } from "../../../contracts/offchain/data1/configs.mjs";
import { saturation } from "../../../contracts/offchain/data1/evaluate.mjs";
import { sessionPairs } from "../../../contracts/offchain/data1/inproc.mjs";
import { machine, peakRssMB } from "../../../contracts/offchain/data1/machine.mjs";
import { CONFIG, CONFIG_PATH, E, fiSpec, modelConfigHash } from "../../../contracts/offchain/model.mjs";
import { LOCAL_NOISE } from "../../../contracts/offchain/party-m3.mjs";
import { requiredFigures, scoreFigures } from "../../../contracts/offchain/scoring.mjs";
import { activeTraining, passiveTraining } from "../../../contracts/offchain/training.mjs";
import { PREPARED, loadPrepared } from "./prepared.mjs";
import { scoreRows } from "./score.mjs";

const { values: a } = parseArgs({ options: {
  mode: { type: "string" }, seed: { type: "string" }, clip: { type: "string" }, sigma: { type: "string" },
  "sigma-scale": { type: "string" }, sigmas: { type: "string" },
  partition: { type: "string", default: "" }, out: { type: "string" },
  "wait-ms": { type: "string", default: "120000" }, "max-depth": { type: "string", default: "14" },
} });
if (!["c2", "c3", "ldp"].includes(a.mode)) throw new Error("--mode c2|c3|ldp");
const sigmaScale = Number(a["sigma-scale"]);
const sigmas = a.sigmas ? a.sigmas.split(",").map(Number) : null;
if (a.mode === "ldp" && (a["sigma-scale"] === undefined) === (sigmas === null)) throw new Error("--mode ldp needs --sigma-scale or --sigmas");
if (a.mode === "ldp" && !(sigmas ? sigmas.length === 3 && sigmas.every((x) => x >= 0) : sigmaScale >= 0)) throw new Error("bad --sigma-scale or --sigmas");
const seed = Number(a.seed);
if (CONFIG.initSeed !== initSeedFor(seed)) throw new Error(`config ${CONFIG_PATH} has initSeed ${CONFIG.initSeed}, not seed ${seed}'s`);
const m0 = machine();
const t0 = performance.now();

const partition = a.partition ? JSON.parse(readFileSync(resolve(a.partition), "utf8")) : null;
const d = loadPrepared({ partition });
const { rows, xp, trainPos, testPos, widths, manifest } = d;
const sizeI = rows.y.length;
widths.forEach((w, i) => { if (fiSpec(i + 1).in !== w) throw new Error(`passive ${i + 1}: config width ${fiSpec(i + 1).in} != data width ${w}`); });
const testSet = new Set(testPos);
if (trainPos.some((k) => testSet.has(k))) throw new Error("training and test positions overlap");
if (sizeI > 65535) throw new Error("positions travel as u16");

const noise = a.mode === "c2" ? { clip: Infinity, sigma: 0 } : { clip: Number(a.clip), sigma: Number(a.sigma) };
const noises = [0, 1, 2].map((i) => (a.mode !== "ldp" ? noise
  : { clip: LOCAL_NOISE[i].clip, sigma: sigmas ? sigmas[i] : sigmaScale * LOCAL_NOISE[i].sigma }));
const noiseSeeds = [1, 2, 3].map((i) => 20263000 + 10 * seed + i);
const normals = noiseSeeds.map((ns) => { const r = prng(BigInt(ns)); return () => gauss(r); });
const maxDepth = Number(a["max-depth"]);
const ws = [0, 1, 2, 3].map(devWallet);
const chainId = 2026n;
const contract = ws[0].address; // any address: the EIP-712 domain of this off-chain run
const { pairs, close } = await sessionPairs(3, { chainId, contract, activeParty: ws[0].address, passives: ws.slice(1).map((w) => w.address) });

const tTrain = performance.now();
const [act, ...pas] = await Promise.all([
  activeTraining({ peers: pairs.map((x, i) => ({ index: i + 1, address: ws[i + 1].address, session: x.a })), rows, Itr: trainPos, sizeI,
    chainId, contract, maxDepth, waitMs: Number(a["wait-ms"]) }),
  ...pairs.map((x, i) => passiveTraining({ session: x.p, index: i + 1, xp: xp[i], sizeI, chainId, contract, maxDepth,
    signer: ws[i + 1], noise: noises[i], normal: normals[i] })),
]);
const trainingMs = performance.now() - tTrain;
close();
if (act.zeroFilled.length > 0) throw new Error(`${act.zeroFilled.length} zero-filled batches`);
if (!act.finals.every((f) => f.received && f.signatureValid)) throw new Error("a final message is missing or badly signed");

const hq = pas.map((p) => p.hq);
const trainNorms = [];
for (const m of hq) for (const k of trainPos) trainNorms.push(Math.sqrt(m[k].reduce((s, v) => s + (v / 4096) ** 2, 0)));
trainNorms.sort((x, y) => x - y);
const pct = (p) => { const h = (trainNorms.length - 1) * p; const lo = Math.floor(h); return trainNorms[lo] + (trainNorms[Math.min(lo + 1, trainNorms.length - 1)] - trainNorms[lo]) * (h - lo); };

const tScore = performance.now();
const figs = requiredFigures(7, 3);
const scored = scoreFigures({ fA: act.fA, fusion: act.fusion, rows, hq, Itr: trainPos, idx: testPos, figures: figs });
const scoringMs = performance.now() - tScore;
const tHelper = performance.now();
const extra = scored.figures.map((f) => {
  const test = scoreRows({ fa: scored.fa, head: f.head, mask: f.mask, rows, hq, positions: testPos });
  if (test.preds.some((p, t) => p !== f.preds[t])) throw new Error(`${f.name}: score helper predictions differ from scoreFigures'`);
  const train = scoreRows({ fa: scored.fa, head: f.head, mask: f.mask, rows, hq, positions: trainPos });
  return { margins: test.margins, unusedClass: { test: test.unusedClass, train: train.unusedClass } };
});
const helperMs = performance.now() - tHelper;

const out = {
  kind: a.mode, seed, configPath: CONFIG_PATH, configHash: modelConfigHash(), widths, partition: a.partition || null,
  prepared: { dir: PREPARED, aSha256: manifest.a.sha256, passiveSha256: manifest.passive.sha256 },
  noise: a.mode === "c2" ? "off" : a.mode === "c3" ? { ...noise, noiseSeeds }
    : sigmas ? { sigmas, perPassive: noises, noiseSeeds } : { sigmaScale, perPassive: noises, noiseSeeds },
  waitMs: Number(a["wait-ms"]), maxDepth,
  machine: m0, freeRamMBAtEnd: machine().freeRamMB, wallMs: performance.now() - t0, trainingMs, scoringMs, helperMs, peakRssMB: peakRssMB(),
  rows: { total: sizeI, train: trainPos.length, test: testPos.length,
    testPosSha256: createHash("sha256").update(JSON.stringify(testPos)).digest("hex") },
  training: { rounds: act.rounds.map(({ round, ms, loss, batches }) => ({ round, ms, loss, batches })), zeroFilled: act.zeroFilled,
    finals: act.finals.map((f) => ({ passive: f.passive, received: f.received, signatureValid: f.signatureValid, bytes: f.bytes })),
    passives: pas.map((p) => ({ stats: p.stats, ms: p.ms })) },
  clipping: { hq: saturation(hq), entries: sizeI * E },
  embNormTrain: { n: trainNorms.length, p50: pct(0.5), p90: pct(0.9), p95: pct(0.95), p99: pct(0.99), max: trainNorms.at(-1) },
  figures: scored.figures.map((f, i) => ({ name: f.name, kind: f.kind, memberIndex: f.memberIndex, mask: f.mask, F: f.F,
    retrained: f.retrained, trainMs: f.trainMs, preds: f.preds, margins: extra[i].margins, unusedClass: extra[i].unusedClass })),
};
mkdirSync(dirname(resolve(a.out)), { recursive: true });
writeFileSync(resolve(a.out), JSON.stringify(out) + "\n");
console.log(JSON.stringify({ kind: out.kind, seed, F: Object.fromEntries(out.figures.map((f) => [f.name, f.F])),
  unusedClass: Object.fromEntries(out.figures.map((f) => [f.name, f.unusedClass.test + f.unusedClass.train])),
  wallS: Math.round(out.wallMs / 1000), trainingS: Math.round(trainingMs / 1000), peakRssMB: out.peakRssMB, clipping: out.clipping.hq }));
process.exit(0);
