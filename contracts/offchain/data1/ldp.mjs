// Local differential privacy, with vs without, in DATA1's setting (E1's rows: all 15,965 households, 12,772 training,
// 3,193 test; model configs e1-s1..e1-s5, of which e1-s1 is E3's config): the passives' protocol noise setting
// (LOCAL_NOISE, party-m3.mjs: per-row L2 clip, then Gaussian noise) with sigma x 1 (LDP on) and x 0 (LDP off: the same
// clipping, no noise), then x 0.5 and x 4 (weaker, stronger); every other setting equal, and the same seeded normal
// draws in every condition of a seed. Then each run's N = 64 draws (n64.mjs, the protocol's evaluation sampling).
//   --set scales (default):
//     0. check: C2 seed 1 again; its six figures must equal results/data1/e1/c2-s1.json's (the current code is E1's)
//     1. for each sigma scale and seed: vfl-run --mode ldp --sigma-scale k, then n64.mjs
//   --set eps: the privacy-utility curve, each passive's sigma calibrated to a per-release epsilon for its clip bound
//   (results/ldp/eps-levels.json, written by py/ldp_analysis.py --levels), against the scales set's LDP-off runs:
//     0. check: LDP off seed 1 again; its six figures must equal runs/ldp-x0-s1.json's (the reference still holds)
//     1. for each epsilon and seed: vfl-run --mode ldp --sigmas s1,s2,s3, then n64.mjs
// Every run is its own child process, one at a time (DATA1 brief rule 4); a run whose output exists is skipped
// (resume, refused if the code commit changed). Writes results/ldp/runs/*.json, the check run, the set's meta file
// (results/ldp/meta.json or meta-eps.json: the code commit, clean or not) and results/ldp/ldp.log.
//   node offchain/data1/ldp.mjs [--set scales|eps] [--seeds 1-5]
// --seeds (e1.mjs's parseSeeds; default 1-5): the E1 seeds to run (their configs results/data1/configs/e1-s*.json
// must exist: e1.mjs --seeds writes them). The runs are named by seed as before; a seed set other than 1-5 gets its
// own check run and meta file (e.g. check-c2-s1-s6-10.json, meta-s6-10.json), so the 1-5 ones stay as they are and
// the check reruns seed 1 with the current code.
import { execSync } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { parseSeeds, RESULTS, runChild, seedTag } from "./e1.mjs";
import { machine } from "./machine.mjs";

export const SCALES = [1, 0, 0.5, 4];   // LDP on, LDP off, then weaker and stronger
const { values: opt } = parseArgs({ options: { set: { type: "string", default: "scales" }, seeds: { type: "string", default: "1-5" } } });
if (!["scales", "eps"].includes(opt.set)) throw new Error("--set scales|eps");
const SEEDS = parseSeeds(opt.seeds);
const TAG = seedTag(SEEDS);   // "" for 1-5
const OUT = resolve(RESULTS, "..", "ldp");
const RUNS = resolve(OUT, "runs");
const LOG = resolve(OUT, "ldp.log");
const config = (s) => resolve(RESULTS, "configs", `e1-s${s}.json`);
const figures = (file) => Object.fromEntries(JSON.parse(readFileSync(file, "utf8")).figures.map((f) => [f.name, f.F]));

mkdirSync(RUNS, { recursive: true });
const git = (c) => execSync(`git ${c}`, { cwd: resolve(RESULTS, "..", "..") }).toString().trim();
const meta = { commit: git("rev-parse --short HEAD"), clean: git("status --porcelain --untracked-files=no -- contracts/offchain py") === "",
  started: new Date().toISOString(), ...(TAG ? { seeds: SEEDS } : {}) };
const metaFile = resolve(OUT, `${opt.set === "eps" ? "meta-eps" : "meta"}${TAG}.json`);
if (existsSync(metaFile)) {
  const before = JSON.parse(readFileSync(metaFile, "utf8"));
  if (before.commit !== meta.commit) throw new Error(`runs so far used ${before.commit}, the code is now ${meta.commit}`);
  meta.started = before.started;
}
writeFileSync(metaFile, JSON.stringify(meta, null, 1) + "\n");
appendFileSync(LOG, `${new Date().toISOString()} LDP ${opt.set} start ${JSON.stringify({ ...meta, seeds: SEEDS, machine: machine() })}\n`);

/** Reruns `args` for seed 1 into `file` and requires the six figures of `reference`. */
async function check(args, file, reference, what) {
  await runChild("vfl-run.mjs", [...args, "--seed", "1"], config(1), file, LOG);
  const [now, ref] = [figures(file), figures(reference)];
  if (JSON.stringify(now) !== JSON.stringify(ref)) throw new Error(`${what} differs: ${JSON.stringify(now)} vs ${JSON.stringify(ref)}`);
  appendFileSync(LOG, `${new Date().toISOString()} check: ${what} reproduced ${JSON.stringify(now)}\n`);
}

if (opt.set === "scales") {
  await check(["--mode", "c2"], resolve(OUT, `check-c2-s1${TAG}.json`), resolve(RESULTS, "e1", "c2-s1.json"), "E1's C2 seed 1");
  for (const k of SCALES) {
    for (const s of SEEDS) {
      const run = resolve(RUNS, `ldp-x${k}-s${s}.json`);
      await runChild("vfl-run.mjs", ["--mode", "ldp", "--seed", String(s), "--sigma-scale", String(k)], config(s), run, LOG);
      await runChild("n64.mjs", ["--in", run], config(s), resolve(RUNS, `n64-x${k}-s${s}.json`), LOG);
    }
  }
} else {
  await check(["--mode", "ldp", "--sigma-scale", "0"], resolve(OUT, `check-eps-x0-s1${TAG}.json`), resolve(RUNS, "ldp-x0-s1.json"),
    "LDP off seed 1");
  const levels = JSON.parse(readFileSync(resolve(OUT, "eps-levels.json"), "utf8")).levels;
  for (const lv of levels) {
    const sigmas = lv.passives.map((p) => String(p.sigma)).join(",");
    for (const s of SEEDS) {
      const run = resolve(RUNS, `eps-e${lv.eps}-s${s}.json`);
      await runChild("vfl-run.mjs", ["--mode", "ldp", "--seed", String(s), "--sigmas", sigmas], config(s), run, LOG);
      await runChild("n64.mjs", ["--in", run], config(s), resolve(RUNS, `n64-eps-e${lv.eps}-s${s}.json`), LOG);
    }
  }
}
appendFileSync(LOG, `${new Date().toISOString()} LDP ${opt.set} done${TAG ? ` (seeds ${SEEDS.join(",")})` : ""}\n`);
