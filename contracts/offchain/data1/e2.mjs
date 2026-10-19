// DATA1 §4, experiment E2: C2 (with its LOO and BASELINE = C1a figures) for each of the 5 random
// partitions of the passives' original variables (py/data1_partitions.py), 5 seeds each, every run in
// its own child process, one at a time. C1b does not depend on the partition: E1's c1b-s{s}.json is
// reused by the analysis. A run whose output exists is skipped (resume).
//   node offchain/data1/e2.mjs
import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { RESULTS, SEEDS, runChild } from "./e1.mjs";
import { writeConfig } from "./configs.mjs";
import { machine } from "./machine.mjs";

const OUT = resolve(RESULTS, "e2");
const LOG = resolve(OUT, "e2.log");
mkdirSync(OUT, { recursive: true });
const parts = JSON.parse(readFileSync(resolve(OUT, "partitions.json"), "utf8"));
appendFileSync(LOG, `${new Date().toISOString()} E2 start ${JSON.stringify(machine())}\n`);
for (const p of parts) {
  for (const s of SEEDS) {
    const cfg = writeConfig({ name: `e2-p${p.k}-s${s}`, seed: s, widths: p.widths });
    await runChild("vfl-run.mjs", ["--mode", "c2", "--seed", String(s), "--partition", resolve(OUT, `partition-${p.k}.json`)],
      cfg.path, resolve(OUT, `p${p.k}-c2-s${s}.json`), LOG);
  }
}
appendFileSync(LOG, `${new Date().toISOString()} E2 done\n`);
