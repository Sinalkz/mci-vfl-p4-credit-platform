// Compile the macro-F1 circuit for one N and (optionally) run the Groth16 setup.
//
// Usage (from circuits/):
//   node scripts/build.mjs --n 64                 compile + setup + export verifier
//   node scripts/build.mjs --n 64 --compile-only  constraint count only
//   node scripts/build.mjs --n 64 --merkle --compile-only
//   node scripts/build.mjs --n 64 --members --isize 1000    S2b circuit (member roots)
//
// Setup: phase 1 is a public powers-of-tau ceremony file in ptau/ (Hermez or
// PSE Perpetual Powers of Tau, see fetch-ptau.mjs); phase 2 is ONE local
// contribution with fixed entropy. That is a development setup, not a secure
// ceremony.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import * as snarkjs from "snarkjs";

import { CIRCUITS, ROOT, buildDir, membersBuildDir } from "./lib.mjs";

const { values: args } = parseArgs({
  options: {
    n: { type: "string" },
    merkle: { type: "boolean", default: false },
    members: { type: "boolean", default: false },
    isize: { type: "string" },
    "compile-only": { type: "boolean", default: false },
    "setup-only": { type: "boolean", default: false }, // reuse main.r1cs and compile.json
    ptau: { type: "string" },
  },
});
const N = Number(args.n);
if (!Number.isInteger(N) || N < 1) throw new Error("--n required");
const isize = Number(args.isize);
if (args.members && !(isize >= 2)) throw new Error("--isize required with --members");
const depth = Math.max(1, Math.ceil(Math.log2(isize)));
const dir = args.members ? membersBuildDir(N, isize) : buildDir(N, args.merkle);
mkdirSync(dir, { recursive: true });

const mainPath = resolve(dir, "main.circom");
writeFileSync(mainPath, args.members
  ? `pragma circom 2.1.9;\ninclude "macro_f1_members.circom";\n` +
    `component main {public [C_model, C_fA, gammaA, gammaP, seed, mask, F]} = MacroF1Members(${N}, ${isize}, ${depth});\n`
  : `pragma circom 2.1.9;\ninclude "macro_f1.circom";\n` +
    `component main {public [C_model, C_eval, F]} = MacroF1(${N}, 16, 8, 5, ${args.merkle ? 1 : 0});\n`);

// ---- compile ------------------------------------------------------------------
const circom = process.env.CIRCOM ?? "circom";
let t0 = performance.now();
let compile;
if (args["setup-only"] && existsSync(resolve(dir, "compile.json"))) {
  compile = JSON.parse(readFileSync(resolve(dir, "compile.json"), "utf8"));
} else {
  const out = execFileSync(circom, [mainPath, "--r1cs", "--wasm", "--sym", "--O2", "-o", dir,
    "-l", resolve(CIRCUITS, "src"), "-l", resolve(CIRCUITS, "node_modules")], { encoding: "utf8", maxBuffer: 1 << 26 });
  const compileSeconds = (performance.now() - t0) / 1000;
  const grab = (label) => Number((out.match(new RegExp(`(?<!non-)${label}:\\s*(\\d+)`)) ?? [])[1]);
  compile = {
    N, merkle: args.merkle, members: args.members, isize: args.members ? isize : undefined,
    depth: args.members ? depth : undefined, compileSeconds,
    nonLinearConstraints: grab("non-linear constraints"),
    linearConstraints: grab("linear constraints"),
    publicInputs: grab("public inputs"),
    privateInputs: grab("private inputs"),
    wires: grab("wires"),
  };
  const info = await snarkjs.r1cs.info(resolve(dir, "main.r1cs"));
  compile.r1csConstraints = info.nConstraints;
  compile.circomVersion = execFileSync(circom, ["--version"], { encoding: "utf8" }).trim();
  writeFileSync(resolve(dir, "compile.json"), JSON.stringify(compile, null, 2) + "\n");
}
console.log(JSON.stringify(compile));
if (args["compile-only"]) process.exit(0);

// ---- setup ----------------------------------------------------------------------
const need = Math.ceil(Math.log2(compile.r1csConstraints + compile.publicInputs + 1));
// Any phase-1 file with at least 2^need powers works; take the first present.
const candidates = [];
for (let k = Math.max(need, 8); k <= 28; k++) {
  candidates.push(`powersOfTau28_hez_final_${k}.ptau`, `ppot_0080_${String(k).padStart(2, "0")}.ptau`);
}
const ptau = args.ptau ?? candidates.map((f) => resolve(CIRCUITS, "ptau", f)).find(existsSync);
if (!ptau) throw new Error(`no ptau with >= 2^${need} powers; run: node scripts/fetch-ptau.mjs ${need}`);

t0 = performance.now();
await snarkjs.zKey.newZKey(resolve(dir, "main.r1cs"), ptau, resolve(dir, "main_0000.zkey"));
await snarkjs.zKey.contribute(resolve(dir, "main_0000.zkey"), resolve(dir, "main_final.zkey"),
  "spike single contribution", "thesis-poc spike fixed entropy -- NOT a secure ceremony");
const setupSeconds = (performance.now() - t0) / 1000;
const vkey = await snarkjs.zKey.exportVerificationKey(resolve(dir, "main_final.zkey"));
writeFileSync(resolve(dir, "verification_key.json"), JSON.stringify(vkey, null, 1));

// Solidity verifier, renamed per N so all sizes can coexist in one Hardhat project.
const templates = { groth16: readFileSync(resolve(CIRCUITS, "node_modules", "snarkjs", "templates",
  "verifier_groth16.sol.ejs"), "utf8") };
const sol = await snarkjs.zKey.exportSolidityVerifier(resolve(dir, "main_final.zkey"), templates);
const name = args.members ? `MembersVerifierN${N}I${isize}` : `MacroF1VerifierN${N}`;
const verifiersDir = resolve(ROOT, "contracts", "contracts", "verifiers");
mkdirSync(verifiersDir, { recursive: true });
writeFileSync(resolve(verifiersDir, `${name}.sol`), sol.replace(/contract Groth16Verifier/, `contract ${name}`));

writeFileSync(resolve(dir, "setup.json"), JSON.stringify({ ptau: ptau.replace(ROOT, "."), ptauPower: need,
  setupSeconds, setupPeakRssMB: process.resourceUsage().maxRSS / 1024, verifierContract: name }, null, 2) + "\n");
console.log(`setup done in ${setupSeconds.toFixed(1)} s; verifier -> contracts/contracts/verifiers/${name}.sol`);
process.exit(0);
