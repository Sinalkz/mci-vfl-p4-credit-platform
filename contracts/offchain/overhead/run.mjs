// Task 4 (off-chain overhead, Chapter 5): runs the measurements one process at a time and writes the raw files to
// results/overhead/raw/. py/overhead_report.py turns them into results/overhead/ch5-overhead.json and .md.
//   train   item 1: train-bench.mjs for the E3 subset's rows, then E1's rows, model-config seeds 1..5
//           (results/data1/configs/e1-s<seed>.json; E3's own config e3.json is byte-identical to e1-s1.json)
//   steps   item 1 attribution: steps-bench.mjs
//   merkle  item 2: merkle-bench.mjs, subset then full
//   psi     item 2: psi-bench.mjs; one subset run first (warm-up, discarded), then the subset 3 times, then 3 rounds of
//           full-a, full-b, full-c
// While each child runs, this process samples os.freemem() every 250 ms (free RAM at the start and the lowest) and
// adds it, with the wall-clock, to the child's file. A file that exists is skipped (resume); --force reruns it.
// A child starts only with at least --min-free-mb of free RAM.
//   node offchain/overhead/run.mjs [--parts train,steps,merkle,psi] [--force] [--min-free-mb 1500]
import { spawn, execSync } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { freemem } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { machine } from "../data1/machine.mjs";

const HERE = resolve(fileURLToPath(import.meta.url), "..");
const CONTRACTS = resolve(HERE, "..", "..");
const ROOT = resolve(CONTRACTS, "..");
const OUT = resolve(ROOT, "results", "overhead");
const RAW = resolve(OUT, "raw");
const LOG = resolve(OUT, "overhead.log");
export const SEEDS = [1, 2, 3, 4, 5];
export const PSI_REPS = 3;

const { values: opt } = parseArgs({ options: { parts: { type: "string", default: "train,steps,merkle,psi" },
  force: { type: "boolean", default: false }, "min-free-mb": { type: "string", default: "1500" } } });
const parts = opt.parts.split(",");
const freeMB = () => Math.round(freemem() / 2 ** 20);
const log = (s) => { const line = `${new Date().toISOString()} ${s}`; appendFileSync(LOG, line + "\n"); console.log(line); };

mkdirSync(RAW, { recursive: true });
const git = (c) => execSync(`git ${c}`, { cwd: ROOT }).toString().trim();
const meta = { commit: git("rev-parse --short HEAD"), clean: git("status --porcelain --untracked-files=no -- contracts/offchain py") === "",
  started: new Date().toISOString(), parts };
writeFileSync(resolve(OUT, `meta-${parts.join("-")}.json`), JSON.stringify(meta, null, 1) + "\n");
log(`start ${JSON.stringify({ ...meta, machine: machine() })}`);

/** Runs one child to completion (stdout and stderr to the log); returns the coordinator's measurements. */
async function runChild(script, args, out, env = {}) {
  if (existsSync(out) && !opt.force) { log(`skip (exists) ${out}`); return null; }
  const free0 = freeMB();
  if (free0 < Number(opt["min-free-mb"])) throw new Error(`only ${free0} MB free (< ${opt["min-free-mb"]} MB): not starting ${script}`);
  let low = free0;
  const timer = setInterval(() => { low = Math.min(low, freeMB()); }, 250);
  const t0 = Date.now();
  log(`run ${script} ${args.join(" ")}${env.VFL_MODEL_CONFIG ? ` (VFL_MODEL_CONFIG=${env.VFL_MODEL_CONFIG})` : ""}; free ${free0} MB`);
  const code = await new Promise((ok) => {
    const p = spawn(process.execPath, [resolve(HERE, script), ...args, "--out", out], { cwd: CONTRACTS, env: { ...process.env, ...env },
      stdio: ["ignore", "pipe", "pipe"] });
    p.stdout.on("data", (d) => appendFileSync(LOG, d));
    p.stderr.on("data", (d) => appendFileSync(LOG, d));
    p.on("exit", ok);
  });
  clearInterval(timer);
  const run = { freeRamMBAtStart: free0, freeRamMBLowest: Math.min(low, freeMB()), wallMs: Date.now() - t0, startedAt: new Date(t0).toISOString(),
    commit: meta.commit };
  if (code !== 0) throw new Error(`${script} ${args.join(" ")} exited ${code}`);
  const j = JSON.parse(readFileSync(out, "utf8"));
  j.coordinator = run;
  writeFileSync(out, JSON.stringify(j) + "\n");
  log(`done ${out} in ${(run.wallMs / 1000).toFixed(1)} s; free RAM ${run.freeRamMBAtStart} -> lowest ${run.freeRamMBLowest} MB`);
  return run;
}

const configOf = (seed) => resolve(ROOT, "results", "data1", "configs", `e1-s${seed}.json`);
if (parts.includes("train")) {
  for (const setting of ["e3", "e1"]) {
    for (const s of SEEDS) {
      await runChild("train-bench.mjs", ["--setting", setting, "--seed", String(s), "--reps", "2"], resolve(RAW, `train-${setting}-s${s}.json`),
        { VFL_MODEL_CONFIG: configOf(s) });
    }
  }
}
if (parts.includes("steps")) await runChild("steps-bench.mjs", [], resolve(RAW, "steps.json"));
if (parts.includes("merkle")) {
  for (const setting of ["subset", "full"]) await runChild("merkle-bench.mjs", ["--setting", setting, "--reps", "5"], resolve(RAW, `merkle-${setting}.json`));
}
if (parts.includes("psi")) {
  await runChild("psi-bench.mjs", ["--setting", "subset"], resolve(RAW, "psi-warmup-subset.json"));
  for (let r = 1; r <= PSI_REPS; r++) await runChild("psi-bench.mjs", ["--setting", "subset"], resolve(RAW, `psi-subset-r${r}.json`));
  for (let r = 1; r <= PSI_REPS; r++) {
    for (const setting of ["full-a", "full-b", "full-c"]) await runChild("psi-bench.mjs", ["--setting", setting], resolve(RAW, `psi-${setting}-r${r}.json`));
  }
}
log(`end ${parts.join(",")}`);
process.exit(0);
