// DATA1 §3, experiment E1: every run as its own child process, one at a time (brief rule 4).
//   1. for seeds 1..5 (or --seeds): C2 (vfl-run c2), C4a (central-run c4a64), C4a-f32 (c4a32), C1b (c1b)
//   2. the C3 clip bound C = p95 of seed-1 C2's per-row embedding norms on TRAINING rows (Q6), fixed
//      before any C3 run; sigma = 0.1, 0.5, 2.0 x C
//   3. for each noise level and seed: C3 (vfl-run c3)
// A run whose output file exists is skipped (resume); delete it to rerun. Refuses to start a run with
// less than 2 GB free. Writes results/data1/e1/*.json and e1.log.
//   node offchain/data1/e1.mjs [--seeds 1-5]
// --seeds: a range a-b or a list a,b,c (default 1-5, the original five); every output is named by its seed, and
// the C3 clip bound stays the one from seed 1's C2 run.
import { spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { writeConfig } from "./configs.mjs";
import { machine } from "./machine.mjs";
import { loadManifest } from "./prepared.mjs";

const HERE = resolve(fileURLToPath(import.meta.url), "..");
export const RESULTS = resolve(HERE, "..", "..", "..", "results", "data1");
const OUT = resolve(RESULTS, "e1");
const LOG = resolve(OUT, "e1.log");
export const SEEDS = [1, 2, 3, 4, 5];
export const LEVELS = [["low", 0.1], ["med", 0.5], ["high", 2.0]];
const MIN_FREE_MB = 2048;

const logTo = (file, ...x) => { const line = `${new Date().toISOString()} ${x.join(" ")}`; console.log(line); appendFileSync(file, line + "\n"); };
const log = (...x) => logTo(LOG, ...x);

/** --seeds: "a-b" (inclusive) or "a,b,c"; positive integers, no repeats; returned in increasing order. */
export function parseSeeds(text) {
  const t = String(text).trim();
  const m = /^(\d+)-(\d+)$/.exec(t);
  const seeds = m ? Array.from({ length: Number(m[2]) - Number(m[1]) + 1 }, (_, k) => Number(m[1]) + k)
    : /^\d+(,\d+)*$/.test(t) ? t.split(",").map(Number) : null;
  if (!seeds || seeds.length === 0 || seeds.some((s) => !Number.isSafeInteger(s) || s < 1) || new Set(seeds).size !== seeds.length) {
    throw new Error(`--seeds ${JSON.stringify(text)}: give a-b or a,b,c of positive integers`);
  }
  return seeds.sort((x, y) => x - y);
}

/** The file-name tag of a seed set: "" for the default 1..5 (the original names), else "-s6-10" or "-s6_8". */
export function seedTag(seeds) {
  if (JSON.stringify(seeds) === JSON.stringify(SEEDS)) return "";
  const range = seeds.length > 1 && seeds.every((s, k) => s === seeds[0] + k);
  return range ? `-s${seeds[0]}-${seeds.at(-1)}` : `-s${seeds.join("_")}`;
}

/** Runs one child process with the given model config; resolves when it exits 0, rejects otherwise. */
export function runChild(script, args, configPath, outFile, logFile = LOG) {
  const log = (...x) => logTo(logFile, ...x);
  if (existsSync(outFile)) { log("skip (exists)", outFile); return Promise.resolve("skipped"); }
  const m = machine();
  if (m.freeRamMB < MIN_FREE_MB) throw new Error(`only ${m.freeRamMB} MB free: not starting ${outFile}`);
  log("start", script, args.join(" "), "| free RAM MB", m.freeRamMB);
  const t0 = Date.now();
  return new Promise((ok, no) => {
    const p = spawn(process.execPath, [resolve(HERE, script), ...args, "--out", outFile], {
      env: { ...process.env, VFL_MODEL_CONFIG: configPath }, stdio: ["ignore", "pipe", "pipe"] });
    p.stdout.on("data", (d) => { process.stdout.write(d); appendFileSync(logFile, d); });
    p.stderr.on("data", (d) => { process.stderr.write(d); appendFileSync(logFile, d); });
    p.on("exit", (code) => {
      log("exit", code, script, args.join(" "), `${Math.round((Date.now() - t0) / 1000)} s`);
      if (code === 0) ok("done"); else no(new Error(`${script} ${args.join(" ")} exited ${code}`));
    });
  });
}

async function main() {
  const { values: opt } = parseArgs({ options: { seeds: { type: "string", default: "1-5" } } });
  const seeds = parseSeeds(opt.seeds);
  mkdirSync(OUT, { recursive: true });
  const m = loadManifest();
  const widths = ["telecom", "retailer", "device"].map((p) => m.parties[p].length);
  log("E1 start; widths", widths.join("/"), "seeds", seeds.join(","), JSON.stringify(machine()));
  const cfg = Object.fromEntries(seeds.map((s) => [s, writeConfig({ name: `e1-s${s}`, seed: s, widths })]));
  for (const s of seeds) {
    await runChild("vfl-run.mjs", ["--mode", "c2", "--seed", String(s)], cfg[s].path, resolve(OUT, `c2-s${s}.json`));
    for (const mode of ["c4a64", "c4a32", "c1b"]) {
      await runChild("central-run.mjs", ["--mode", mode, "--seed", String(s)], cfg[s].path, resolve(OUT, `${mode}-s${s}.json`));
    }
  }
  const levelsFile = resolve(OUT, "noise-levels.json");
  if (!existsSync(levelsFile)) {
    const c2 = JSON.parse(readFileSync(resolve(OUT, "c2-s1.json"), "utf8"));
    const C = c2.embNormTrain.p95;
    writeFileSync(levelsFile, JSON.stringify({ C, source: "p95 of seed-1 C2 per-row embedding L2 norms on training rows (pooled over the 3 passives)",
      levels: LEVELS.map(([name, mult]) => ({ name, mult, clip: C, sigma: mult * C })) }, null, 1) + "\n");
  }
  const levels = JSON.parse(readFileSync(levelsFile, "utf8"));
  log("C3 levels", JSON.stringify(levels.levels));
  for (const lv of levels.levels) {
    for (const s of seeds) {
      await runChild("vfl-run.mjs", ["--mode", "c3", "--seed", String(s), "--clip", String(lv.clip), "--sigma", String(lv.sigma)],
        cfg[s].path, resolve(OUT, `c3-${lv.name}-s${s}.json`));
    }
  }
  log("E1 done");
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
