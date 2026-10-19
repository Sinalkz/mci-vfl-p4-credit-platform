// MCI loan instance v2: the model configurations, fixed with the v2 pre-registration. As v1's configs.mjs: every one is
// contracts/offchain/data1/configs.mjs writeConfig's copy of model-config.json (only the seeds and the passives' input
// widths change), written to results/mci_loan/v2/configs/:
//   loan2-s<s>.json         s = 1..10: the v2 split (widths from results/mci_loan/v2/prepared/manifest.json, in its
//                           passiveOrder: Bank, E-commerce company, Credit bureau)
//   loan2-s<k>-wait10s.json k = 1..5: the on-chain runs, loan2-s<k>.json with "perBatchWaitMs": 2000 -> 10000 and
//                           nothing else (as v1 and the thesis's Chapter 5 series)
//   configs.json            every file's keccak256 (the run's modelConfigHash) and SHA-256
// v2 has no sensitivity variant (docs/mci/v2/02-preregistration.md, section 5.6).
//   node scenarios/mci_loan/v2/offchain/configs2.mjs
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { keccak256 } from "../../../../contracts/node_modules/ethers/lib.esm/index.js";

import { writeConfig } from "../../../../contracts/offchain/data1/configs.mjs";
import { ROOT, passiveNames } from "../../offchain/prepared.mjs";

const DIR = resolve(ROOT, "results", "mci_loan", "v2", "configs");
const PREP = resolve(ROOT, "results", "mci_loan", "v2", "prepared");
const SEEDS = Array.from({ length: 10 }, (_, k) => k + 1);
const ONCHAIN_RUNS = [1, 2, 3, 4, 5];

const m = JSON.parse(readFileSync(resolve(PREP, "manifest.json"), "utf8"));
if (!m.passiveOrder) throw new Error("not a v2 manifest (no passiveOrder)");
const order = passiveNames(m);
const widths = order.map((p) => m.parties[p].length);

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const record = [];
const note = (name, path, extra) => {
  const bytes = readFileSync(path);
  record.push({ name, file: `results/mci_loan/v2/configs/${name}.json`, keccak256: keccak256(bytes), sha256: sha(bytes), ...extra });
};
for (const s of SEEDS) {
  const r = writeConfig({ name: `loan2-s${s}`, seed: s, widths, dir: DIR });
  note(`loan2-s${s}`, r.path, { seed: s, widths });
}
for (const k of ONCHAIN_RUNS) {
  const src = readFileSync(resolve(DIR, `loan2-s${k}.json`), "utf8");
  const parts = src.split('"perBatchWaitMs": 2000,');
  if (parts.length !== 2) throw new Error(`loan2-s${k}.json: expected one "perBatchWaitMs": 2000`);
  const path = resolve(DIR, `loan2-s${k}-wait10s.json`);
  writeFileSync(path, parts.join('"perBatchWaitMs": 10000,'));
  note(`loan2-s${k}-wait10s`, path, { seed: k, widths, onChainRun: k, change: { perBatchWaitMs: [2000, 10000] } });
}
writeFileSync(resolve(DIR, "configs.json"), JSON.stringify({
  what: "the MCI loan instance v2's model configurations (scenarios/mci_loan/v2/offchain/configs2.mjs)",
  base: "contracts/offchain/model-config.json via contracts/offchain/data1/configs.mjs writeConfig",
  passiveOrder: order, widths, configs: record,
}, null, 1) + "\n");
console.log(JSON.stringify({ order, widths, configs: record.length }));
