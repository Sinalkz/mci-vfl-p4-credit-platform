// Q17 follow-up: runs q17-run.mjs for E1 seeds 1..5 and for E3, one process at a time (each with its own
// VFL_MODEL_CONFIG), then the noise-vs-N draws (q17-noise.mjs). Logs to results/data1/q17/q17.log.
//   node offchain/data1/q17.mjs [--seeds 1-5]
// --seeds (e1.mjs's parseSeeds; default 1-5): the E1 seeds (their C2 runs results/data1/e1/c2-s*.json must exist).
// With a seed set other than 1-5 the E3 run is not repeated and the draws go to their own file
// (q17-noise.mjs --seeds: e.g. results/data1/q17/noise-s6-10.json), so noise.json stays as it is.
import { spawnSync } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { parseSeeds, seedTag } from "./e1.mjs";
import { machine } from "./machine.mjs";

const HERE = resolve(fileURLToPath(import.meta.url), "..");
const RES = resolve(HERE, "..", "..", "..", "results", "data1");
const LOG = resolve(RES, "q17", "q17.log");
const { values: opt } = parseArgs({ options: { seeds: { type: "string", default: "1-5" } } });
const seeds = parseSeeds(opt.seeds);
const tag = seedTag(seeds);   // "" for 1-5
mkdirSync(resolve(RES, "q17"), { recursive: true });
const log = (s) => { const line = `${new Date().toISOString()} ${s}`; console.log(line); appendFileSync(LOG, line + "\n"); };

function child(script, args, config) {
  const free = machine().freeRamMB;
  if (free < 2000) throw new Error(`free RAM ${free} MB < 2,000 MB: not starting ${script}`);
  log(`start ${script} ${args.join(" ")} | config ${config ?? "-"} | free RAM MB ${free}`);
  const t = performance.now();
  const r = spawnSync(process.execPath, [resolve(HERE, script), ...args], {
    env: { ...process.env, ...(config ? { VFL_MODEL_CONFIG: config } : {}) }, encoding: "utf8", maxBuffer: 1 << 26 });
  log((r.stdout || "").trim());
  if (r.status !== 0) { log(r.stderr); throw new Error(`${script} ${args.join(" ")} exited ${r.status}`); }
  log(`exit 0 ${script} ${args.join(" ")} ${Math.round((performance.now() - t) / 1000)} s`);
}

log(`Q17 start seeds ${seeds.join(",")} ${JSON.stringify(machine())}`);
for (const s of seeds) child("q17-run.mjs", ["--seed", String(s)], resolve(RES, "configs", `e1-s${s}.json`));
if (!tag) child("q17-run.mjs", ["--e3"], resolve(RES, "configs", "e3.json"));
child("q17-noise.mjs", tag ? ["--seeds", opt.seeds] : [], null);
log("Q17 done");
