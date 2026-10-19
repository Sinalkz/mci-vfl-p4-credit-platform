// RESULTS-PROOFCOST (3): the circuit inputs of E3's six scoring proofs (D139: FULL, FULL_J, LOO_1..3, BASELINE),
// rebuilt as e3-replay.mjs rebuilds the figures (deterministic training, noise off), then through circuitInputFor with
// E3's public seed, N = 64 and maxDepth = 10 -- the inputs A proved on-chain. Checks each figure's F and C_model against
// the chain; circuits/scripts/proofcost.mjs --step scoring checks the proofs' public signals against A's summary.
//   VFL_MODEL_CONFIG=../results/data1/configs/e3.json node offchain/data1/proofcost-inputs.mjs --out <dir>
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { assertFiWidth } from "../model.mjs";
import { circuitInputFor, requiredFigures, scoreFigures } from "../scoring.mjs";
import { centralHq, trainCentral } from "./centralized.mjs";
import { recsScenario } from "./recs-scenario.mjs";

const { values: args } = parseArgs({ options: { out: { type: "string" } } });
if (!args.out) throw new Error("--out <dir> required");
const DIR = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "..", "results", "data1", "e3");
const load = (f) => JSON.parse(readFileSync(resolve(DIR, f), "utf8"));
const run = load("run.json");
const a = load("A/summary.json");
const I = load("A/I.json").map((x) => BigInt(x));
const idx = a.idx;

const sc = recsScenario({ seed: load("sample.json").seed });
const rows = { xa: I.map((id) => sc.A.rows.get(id).xa), y: I.map((id) => sc.A.rows.get(id).y) };
const xp = sc.passives.map((p) => I.map((id) => p.rows.get(id)));
xp.forEach((rowsI, k) => assertFiWidth(k + 1, rowsI[0].length));
const picked = new Set(idx);
const Itr = [...I.keys()].filter((k) => !picked.has(k));
const c = trainCentral({ rows, xp, Itr, mode: "f32" });
const hq = centralHq(c.fis, xp, I.length);
const scored = scoreFigures({ fA: c.fA, fusion: c.fusion, rows, hq, Itr, idx, figures: requiredFigures(7, 3) });

mkdirSync(resolve(args.out), { recursive: true });
const out = [];
for (const fig of scored.figures) {
  const chain = run.chainEnd.figures.find((g) => g.kind === fig.kind && g.memberIndex === fig.memberIndex);
  const ci = await circuitInputFor({ fa: scored.fa, fig, rows, hq, sizeI: I.length, maxDepth: 10, seed: a.seed, N: 64 });
  if (ci.oracleF !== fig.F || fig.F !== chain.F) throw new Error(`${fig.name}: F ${fig.F}, oracle ${ci.oracleF}, chain ${chain.F}`);
  if (chain.cModel && ci.cModel.toLowerCase() !== chain.cModel.toLowerCase()) throw new Error(`${fig.name}: C_model differs from the chain`);
  writeFileSync(resolve(args.out, `${fig.name}.input.json`), JSON.stringify(ci.input));
  out.push({ name: fig.name, F: fig.F });
}
console.log(JSON.stringify(out));
