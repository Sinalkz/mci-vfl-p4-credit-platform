// DATA1: one model configuration per run. A copy of model-config.json in which only the seeds
// (initSeed = 20261000 + seed, shuffleSeed = 20262000 + seed) and the passives' input widths change
// (QUESTIONS-DATA1 Q13); its keccak256 is the run's modelConfigHash. Runs load it through
// VFL_MODEL_CONFIG (model.mjs).
//   node offchain/data1/configs.mjs --name e1-s1 --seed 1 [--widths 46,225,358] [--dir ../results/data1/configs]
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { keccak256 } from "ethers";

const HERE = resolve(fileURLToPath(import.meta.url), "..");
export const BASE_CONFIG = resolve(HERE, "..", "model-config.json");
export const CONFIG_DIR = resolve(HERE, "..", "..", "..", "results", "data1", "configs");
export const initSeedFor = (seed) => 20261000 + seed;
export const shuffleSeedFor = (seed) => 20262000 + seed;

export function writeConfig({ name, seed, widths, dir = CONFIG_DIR }) {
  const config = JSON.parse(readFileSync(BASE_CONFIG, "utf8"));
  config.initSeed = initSeedFor(seed);
  config.shuffleSeed = shuffleSeedFor(seed);
  config.architecture.fi.in = widths;
  mkdirSync(dir, { recursive: true });
  const path = resolve(dir, `${name}.json`);
  const bytes = Buffer.from(JSON.stringify(config, null, 2) + "\n", "utf8");
  writeFileSync(path, bytes);
  return { path, hash: keccak256(bytes), config };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values: a } = parseArgs({ options: { name: { type: "string" }, seed: { type: "string" }, widths: { type: "string" },
    dir: { type: "string" } } });
  let widths = a.widths?.split(",").map(Number);
  if (!widths) {
    const { loadManifest } = await import("./prepared.mjs");
    const m = loadManifest();
    widths = ["telecom", "retailer", "device"].map((p) => m.parties[p].length);
  }
  const r = writeConfig({ name: a.name, seed: Number(a.seed), widths, ...(a.dir ? { dir: resolve(a.dir) } : {}) });
  console.log(JSON.stringify({ path: r.path, hash: r.hash, widths }));
}
