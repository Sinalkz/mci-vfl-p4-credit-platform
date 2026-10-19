// full-data diagnosis (item 1): E1's 1,000 N = 64 draws as indices t into the manifest's test_pos, by the rule
// data1/n64.mjs uses (evaluate.mjs n64Positions(j) = test_pos[t], t = sampleIndices(j, 64, 3193), j = 1..n), so that
// py/full_diagnosis.py can recompute every E1 draw with the other macro-F1 implementations (no model, no training).
//   node offchain/full/e1-n64-positions.mjs --out ../results/full/diagnosis/e1-n64-positions.json [--draws 1000]
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { sampleIndices } from "../../../circuits/scripts/members-lib.mjs";
import { loadManifest } from "../data1/prepared.mjs";

const { values: a } = parseArgs({ options: { out: { type: "string" }, draws: { type: "string", default: "1000" } } });
if (!a.out) throw new Error("--out <file>");
const testRows = loadManifest().split.test_pos.length;
const draws = [];
for (let j = 1; j <= Number(a.draws); j++) draws.push(await sampleIndices(BigInt(j), 64, testRows));
writeFileSync(resolve(a.out), JSON.stringify({ what: "E1's N = 64 draws as indices t into the manifest's test_pos (n64.mjs: n64Positions(j) = test_pos[t])",
  rule: `sampleIndices(j, 64, ${testRows}), j = 1..${draws.length}`, testRows, draws }) + "\n");
console.log(JSON.stringify({ draws: draws.length, testRows }));
