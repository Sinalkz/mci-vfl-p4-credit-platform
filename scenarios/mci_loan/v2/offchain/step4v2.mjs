// MCI loan instance v2, Step 4: every off-chain run fixed in docs/mci/v2/02-preregistration.md (section 10). v1's
// driver (scenarios/mci_loan/offchain/step4.mjs) with v2's data, configs and output folder, and no sensitivity variant:
// the same runners (vfl-run.mjs, central-run.mjs, n64.mjs), each run its own child process, one at a time, through the
// thesis's runChild (contracts/offchain/data1/e1.mjs: a run whose output exists is skipped, so the driver resumes; no
// run starts with less than 2 GB free). The children read data/mci_loan/v2/prepared through MCI_LOAN_PREPARED.
//   --set main    for each seed s: C2 (loan2-s<s>.json), C4a-f32 and C1b (central-run.mjs), the N = 64 draws of C2
//   --set scales  check: C2 seed 1 again, its six figures and predictions equal to runs/c2-s1.json's; then for each
//                 sigma scale k (1, 0, 0.5, 4 x the protocol setting LOCAL_NOISE) and seed: ldp --sigma-scale k, then n64
//   --set eps     check: LDP off seed 1 again, equal to runs/ldp-x0-s1.json; then for each epsilon level of
//                 results/ldp/eps-levels.json (the thesis's calibration, unchanged) and seed: ldp --sigmas, then n64
// Writes results/mci_loan/v2/offchain/runs/*.json, check-*.json, meta-<set>.json (the code commit, and whether the
// code was clean; a resumed set refuses a different commit) and step4.log.
//   node scenarios/mci_loan/v2/offchain/step4v2.mjs --set main|scales|eps [--seeds 1-10]
import { execSync } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { parseSeeds, runChild } from "../../../../contracts/offchain/data1/e1.mjs";
import { machine } from "../../../../contracts/offchain/data1/machine.mjs";
import { ROOT } from "../../offchain/prepared.mjs";

const V1 = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "offchain");
const OUT = resolve(ROOT, "results", "mci_loan", "v2", "offchain");
const RUNS = resolve(OUT, "runs");
const LOG = resolve(OUT, "step4.log");
const CONFIGS = resolve(ROOT, "results", "mci_loan", "v2", "configs");
const PREPARED = resolve(ROOT, "data", "mci_loan", "v2", "prepared");
const SCALES = [1, 0, 0.5, 4]; // LDP on, LDP off, then weaker and stronger (contracts/offchain/data1/ldp.mjs)
const VFL = resolve(V1, "vfl-run.mjs");
const CENTRAL = resolve(V1, "central-run.mjs");
const N64 = resolve(V1, "n64.mjs");

const { values: opt } = parseArgs({ options: { set: { type: "string" }, seeds: { type: "string", default: "1-10" } } });
if (!["main", "scales", "eps"].includes(opt.set)) throw new Error("--set main|scales|eps");
const SEEDS = parseSeeds(opt.seeds);
const manifest = JSON.parse(readFileSync(resolve(PREPARED, "manifest.json"), "utf8"));
if (!manifest.passiveOrder) throw new Error(`${PREPARED} is not a v2 folder (no passiveOrder)`);
process.env.MCI_LOAN_PREPARED = PREPARED; // every child's loader
const config = (s) => resolve(CONFIGS, `loan2-s${s}.json`);
const log = (...x) => appendFileSync(LOG, `${new Date().toISOString()} ${x.join(" ")}\n`);

mkdirSync(RUNS, { recursive: true });
const git = (c) => execSync(`git ${c}`, { cwd: ROOT }).toString().trim();
const meta = { set: opt.set, seeds: SEEDS, commit: git("rev-parse --short HEAD"), prepared: "data/mci_loan/v2/prepared",
  clean: git("status --porcelain --untracked-files=no -- contracts circuits/scripts scenarios/mci_loan py") === "",
  started: new Date().toISOString() };
const metaFile = resolve(OUT, `meta-${opt.set}.json`);
if (existsSync(metaFile)) {
  const before = JSON.parse(readFileSync(metaFile, "utf8"));
  if (before.commit !== meta.commit) throw new Error(`runs so far used ${before.commit}, the code is now ${meta.commit}`);
  meta.started = before.started;
}
writeFileSync(metaFile, JSON.stringify(meta, null, 1) + "\n");
log(`step4v2 ${opt.set} start ${JSON.stringify({ ...meta, machine: machine() })}`);

const figures = (file) => Object.fromEntries(JSON.parse(readFileSync(file, "utf8")).figures.map((f) => [f.name, { F: f.F, preds: f.preds }]));

/** Reruns `args` for seed 1 into `file` and requires the six figures (F and test predictions) of `reference`. */
async function check(args, file, reference, what) {
  await runChild(VFL, [...args, "--seed", "1"], config(1), file, LOG);
  const [now, ref] = [figures(file), figures(reference)];
  if (JSON.stringify(now) !== JSON.stringify(ref)) throw new Error(`${what} differs`);
  log(`check: ${what} reproduced ${JSON.stringify(Object.fromEntries(Object.entries(now).map(([k, v]) => [k, v.F])))}`);
}

if (opt.set === "main") {
  for (const s of SEEDS) {
    const c2 = resolve(RUNS, `c2-s${s}.json`);
    await runChild(VFL, ["--mode", "c2", "--seed", String(s)], config(s), c2, LOG);
    for (const mode of ["c4a32", "c1b"]) {
      await runChild(CENTRAL, ["--mode", mode, "--seed", String(s)], config(s), resolve(RUNS, `${mode}-s${s}.json`), LOG);
    }
    await runChild(N64, ["--in", c2], config(s), resolve(RUNS, `n64-c2-s${s}.json`), LOG);
  }
} else if (opt.set === "scales") {
  await check(["--mode", "c2"], resolve(OUT, "check-c2-s1.json"), resolve(RUNS, "c2-s1.json"), "C2 seed 1");
  for (const k of SCALES) {
    for (const s of SEEDS) {
      const run = resolve(RUNS, `ldp-x${k}-s${s}.json`);
      await runChild(VFL, ["--mode", "ldp", "--seed", String(s), "--sigma-scale", String(k)], config(s), run, LOG);
      await runChild(N64, ["--in", run], config(s), resolve(RUNS, `n64-x${k}-s${s}.json`), LOG);
    }
  }
} else {
  await check(["--mode", "ldp", "--sigma-scale", "0"], resolve(OUT, "check-eps-x0-s1.json"), resolve(RUNS, "ldp-x0-s1.json"), "LDP off seed 1");
  const levels = JSON.parse(readFileSync(resolve(ROOT, "results", "ldp", "eps-levels.json"), "utf8")).levels;
  for (const lv of levels) {
    const sigmas = lv.passives.map((p) => String(p.sigma)).join(",");
    for (const s of SEEDS) {
      const run = resolve(RUNS, `eps-e${lv.eps}-s${s}.json`);
      await runChild(VFL, ["--mode", "ldp", "--seed", String(s), "--sigmas", sigmas], config(s), run, LOG);
      await runChild(N64, ["--in", run], config(s), resolve(RUNS, `n64-eps-e${lv.eps}-s${s}.json`), LOG);
    }
  }
}
log(`step4v2 ${opt.set} done`);
console.log(JSON.stringify({ set: opt.set, seeds: SEEDS, done: true }));
