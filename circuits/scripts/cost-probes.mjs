// Compile the isolated pieces in circuits/src/cost_probes.circom and record their constraints.
// Usage (from circuits/): node scripts/cost-probes.mjs   -> results/s2b/cost_probes.json
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { BUILD, CIRCUITS, ROOT } from "./lib.mjs";

const probes = {
  bottomModelPerRow: "component main = ProbeBottomModelRow();",
  bottomModelFixed: "component main {public [C_fA]} = ProbeBottomModelFixed();",
  extraRootPerRow_d10: "component main {public [root]} = ProbeExtraRoot(10);",
  extraRootPerRow_d14: "component main {public [root]} = ProbeExtraRoot(14);",
  sampleIndex_I1000: "component main = ProbeSampleIndex(1000, 10);",
  sampleIndex_I10000: "component main = ProbeSampleIndex(10000, 14);",
  poseidon2: "component main = ProbePoseidon(2);",
  poseidon4: "component main = ProbePoseidon(4);",
};
const circom = process.env.CIRCOM ?? "circom";
const result = {};
for (const [name, main] of Object.entries(probes)) {
  const dir = resolve(BUILD, "probes", name);
  mkdirSync(dir, { recursive: true });
  const file = resolve(dir, "main.circom");
  writeFileSync(file, `pragma circom 2.1.9;\ninclude "cost_probes.circom";\n${main}\n`);
  const out = execFileSync(circom, [file, "--r1cs", "--O2", "-o", dir,
    "-l", resolve(CIRCUITS, "src"), "-l", resolve(CIRCUITS, "node_modules")], { encoding: "utf8" });
  result[name] = Number(out.match(/non-linear constraints:\s*(\d+)/)[1]);
  console.log(name, result[name]);
}
const dir = resolve(ROOT, "results", "s2b");
mkdirSync(dir, { recursive: true });
writeFileSync(resolve(dir, "cost_probes.json"), JSON.stringify(result, null, 2) + "\n");
