// Task 5 (LDP configurations): runs the plan of py/ldp_configs.py (results/ldp/cfg/plan.json: per configuration and
// budget, T, the clip scale, the training fraction and each passive's sigma) for the E1 seeds, each run its own child
// process (ldp-cfg-run.mjs) with its model config: E1's (results/data1/configs/e1-s<seed>.json) at T = 20, else
// ldp-cfg-lib.mjs variantConfig written to results/ldp/cfg/model/T<T>-s<seed>.json. The base configuration also scores
// the uncapped final matrix (--cap-off-too: factor e). k = 8 is in the plan but marked blocked and is never run.
// Up to --jobs runs at a time (default 1; at most 8, the student's limit for this task); a new run starts only while the
// 1-minute load average is below --max-load (default 12 on a 20-vCPU machine), so other users' work comes first.
// A run whose output exists is skipped (resume); the code commit (contracts/offchain, py/ldp_configs.py) must be the
// one of the first runs (results/ldp/cfg/meta.json). Logs to results/ldp/cfg/ldp-cfg.log, with the load average and
// free RAM at the start and end of each batch.
//   node offchain/data1/ldp-cfg.mjs [--budgets B1,B4] [--configs base,T5,...] [--seeds 1-5] [--jobs 8] [--max-load 12]
//       [--dry-run]
import { execSync } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { loadavg } from "node:os";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { parseSeeds, RESULTS, runChild } from "./e1.mjs";
import { writeVariantConfig } from "./ldp-cfg-lib.mjs";
import { machine } from "./machine.mjs";
import { loadManifest } from "./prepared.mjs";

const MAX_JOBS = 8;
const { values: opt } = parseArgs({ options: {
  budgets: { type: "string", default: "B1,B4" }, configs: { type: "string", default: "" }, seeds: { type: "string", default: "1-5" },
  jobs: { type: "string", default: "1" }, "max-load": { type: "string", default: "12" }, "dry-run": { type: "boolean", default: false },
} });
const jobs = Number(opt.jobs);
if (!(Number.isInteger(jobs) && jobs >= 1 && jobs <= MAX_JOBS)) throw new Error(`--jobs 1..${MAX_JOBS}`);
const maxLoad = Number(opt["max-load"]);
const ROOT = resolve(RESULTS, "..", "..");
const CFG = resolve(ROOT, "results", "ldp", "cfg");
const RUNS = resolve(CFG, "runs");
const LOG = resolve(CFG, "ldp-cfg.log");
const plan = JSON.parse(readFileSync(resolve(CFG, "plan.json"), "utf8"));
const SEEDS = parseSeeds(opt.seeds);
const budgets = opt.budgets.split(",");
const wanted = opt.configs ? opt.configs.split(",") : plan.configs.filter((c) => !c.blocked && !c.name.startsWith("combined")).map((c) => c.name);
const byName = Object.fromEntries(plan.configs.map((c) => [c.name, c]));
for (const n of wanted) {
  if (!byName[n]) throw new Error(`config ${n} is not in the plan`);
  if (byName[n].blocked) throw new Error(`config ${n} is blocked: ${byName[n].blocked}`);
}
const m = loadManifest();
const widths = ["telecom", "retailer", "device"].map((p) => m.parties[p].length);

const log = (...x) => { const line = `${new Date().toISOString()} ${x.join(" ")}`; console.log(line); appendFileSync(LOG, line + "\n"); };
const load = () => loadavg().map((v) => Math.round(v * 100) / 100);
const git = (c) => execSync(`git ${c}`, { cwd: ROOT }).toString().trim();

/** The run list: every (config, budget, seed) of the selection, base first; configs without that budget are skipped. */
const runs = [];
for (const n of wanted) {
  const c = byName[n];
  for (const b of budgets) {
    if (!c.budgets[b]) continue;
    for (const s of SEEDS) {
      const configPath = c.T === 20 ? resolve(RESULTS, "configs", `e1-s${s}.json`) : resolve(CFG, "model", `T${c.T}-s${s}.json`);
      const args = ["--seed", String(s), "--sigmas", c.budgets[b].sigmas.map(String).join(","), "--clip-scale", String(c.clip_scale),
        "--frac", String(c.frac), ...(n === "base" ? ["--cap-off-too"] : [])];
      runs.push({ name: `${b}-${n}-s${s}`, c, s, configPath, args, out: resolve(RUNS, `${b}-${n}-s${s}.json`) });
    }
  }
}
if (opt["dry-run"]) {
  for (const r of runs) console.log(r.name, existsSync(r.out) ? "(exists)" : "", r.configPath, r.args.join(" "));
  process.exit(0);
}

mkdirSync(RUNS, { recursive: true });
const meta = { commit: git("rev-parse --short HEAD"), codeCommit: git("log -1 --format=%h -- contracts/offchain py/ldp_configs.py"),
  clean: git("status --porcelain --untracked-files=no -- contracts/offchain py") === "", started: new Date().toISOString() };
const metaFile = resolve(CFG, "meta.json");
if (existsSync(metaFile)) {
  const before = JSON.parse(readFileSync(metaFile, "utf8"));
  if (before.codeCommit !== meta.codeCommit) throw new Error(`runs so far used code ${before.codeCommit}, the code is now ${meta.codeCommit}`);
  meta.started = before.started;
  meta.batches = before.batches ?? [];
}
const batch = { started: new Date().toISOString(), commit: meta.commit, jobs, maxLoad, runs: runs.map((r) => r.name),
  loadAtStart: load(), freeRamMBAtStart: machine().freeRamMB };
meta.batches = [...(meta.batches ?? []), batch];
writeFileSync(metaFile, JSON.stringify(meta, null, 1) + "\n");
log(`LDP-CFG batch start ${JSON.stringify({ ...meta, batches: undefined, jobs, maxLoad, runs: runs.length, load: batch.loadAtStart, machine: machine() })}`);

for (const r of runs) if (r.c.T !== 20) writeVariantConfig(r.configPath, { seed: r.s, widths, T: r.c.T });

const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
const failed = [];
let next = 0;
async function worker(w) {
  while (next < runs.length) {
    const r = runs[next++];
    if (!existsSync(r.out)) {
      while (load()[0] >= maxLoad) { log(`worker ${w}: load ${load().join("/")} >= ${maxLoad}, waiting before ${r.name}`); await sleep(60_000); }
      log(`worker ${w}: ${r.name} | load ${load().join("/")}`);
    }
    try { await runChild("ldp-cfg-run.mjs", r.args, r.configPath, r.out, LOG); } catch (e) { failed.push(r.name); log(`FAILED ${r.name}: ${e.message}`); }
  }
}
const t0 = Date.now();
await Promise.all(Array.from({ length: jobs }, (_, w) => worker(w + 1)));
batch.ended = new Date().toISOString();
batch.wallS = Math.round((Date.now() - t0) / 1000);
batch.loadAtEnd = load();
batch.freeRamMBAtEnd = machine().freeRamMB;
batch.failed = failed;
writeFileSync(metaFile, JSON.stringify(meta, null, 1) + "\n");
log(`LDP-CFG batch done ${JSON.stringify({ wallS: batch.wallS, load: batch.loadAtEnd, failed })}`);
if (failed.length) process.exit(1);
