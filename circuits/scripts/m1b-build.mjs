// M1b: compile circuits/src/macro_f1_members_var.circom for (N, maxDepth) and, unless
// --compile-only, run the Groth16 setup and export the Solidity verifier.
// Usage (from circuits/):
//   node scripts/m1b-build.mjs --n 64 --maxdepth 10
//   node scripts/m1b-build.mjs --n 8 --maxdepth 5 --compile-only     (circuit tests)
//   node scripts/m1b-build.mjs --n 64 --maxdepth 14 --setup-only --no-verifier   (proof-cost measurements)
// Same setup procedure as build.mjs (S2): public phase-1 ptau, ONE local phase-2
// contribution with fixed entropy. A development setup, not a secure ceremony.
import { execFileSync } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import * as snarkjs from "snarkjs";

import { CIRCUITS, ROOT } from "./lib.mjs";
import { fileSha256, varBuildDir, varVerifierName } from "./m1b-lib.mjs";

const { values: args } = parseArgs({ options: {
  n: { type: "string" }, maxdepth: { type: "string" },
  "compile-only": { type: "boolean", default: false },
  "setup-only": { type: "boolean", default: false },
  "no-verifier": { type: "boolean", default: false }, // measurements only (RESULTS-PROOFCOST): leave contracts/ untouched
  "r1cs-only": { type: "boolean", default: false },   // RESULTS-PROOFCOST: constraint count only (no wasm/sym); implies --compile-only
  cpp: { type: "boolean", default: false },           // RESULTS-PROOFCOST: also emit the C++ witness generator (--c --no_asm)
} });
if (args["r1cs-only"]) args["compile-only"] = true;

/** nConstraints from the r1cs header section (type 1), without loading the file (iden3 r1cs binary format). */
function r1csHeaderConstraints(path) {
  const fd = openSync(path, "r");
  const read = (pos, len) => { const b = Buffer.alloc(len); readSync(fd, b, 0, len, pos); return b; };
  try {
    const nSections = read(8, 4).readUInt32LE(0);
    let pos = 12;
    for (let s = 0; s < nSections; s++) {
      const h = read(pos, 12);
      const type = h.readUInt32LE(0);
      const size = Number(h.readBigUInt64LE(4));
      if (type === 1) {
        const fieldSize = read(pos + 12, 4).readUInt32LE(0);
        return read(pos + 12 + 4 + fieldSize + 4 * 4 + 8, 4).readUInt32LE(0); // after nWires, nPubOut, nPubIn, nPrvIn, nLabels
      }
      pos += 12 + size;
    }
    throw new Error("no r1cs header section");
  } finally { closeSync(fd); }
}
const N = Number(args.n);
const D = Number(args.maxdepth);
if (!(N >= 1) || !(D >= 1 && D <= 20)) throw new Error("--n and --maxdepth (1..20) required");
const dir = varBuildDir(N, D);
mkdirSync(dir, { recursive: true });
const mainPath = resolve(dir, "main.circom");
writeFileSync(mainPath, `pragma circom 2.1.9;\ninclude "macro_f1_members_var.circom";\n` +
  `component main {public [C_model, C_fA, gammaA, gammaP, seed, sizeI, mask, F]} = MacroF1MembersVar(${N}, ${D});\n`);

const circom = process.env.CIRCOM ?? "circom";
const sha256 = fileSha256; // chunked: an r1cs can pass readFileSync's 2 GiB limit (N 512: 2.64 GB)
// A build that already has a proving key (e.g. the protocol's N64/maxDepth10) is never altered by this script:
// a recompile goes to <dir>/recompile and must reproduce the r1cs byte for byte; a setup must not replace the key.
const keyed = existsSync(resolve(dir, "main_final.zkey"));
let compile;
if (args["setup-only"] && existsSync(resolve(dir, "compile.json"))) {
  compile = JSON.parse(readFileSync(resolve(dir, "compile.json"), "utf8"));
} else {
  const outDir = keyed ? resolve(dir, "recompile") : dir;
  rmSync(resolve(dir, "recompile"), { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });
  const t0 = performance.now();
  const outputs = args["r1cs-only"] ? ["--r1cs"] : ["--r1cs", "--wasm", "--sym"];
  const out = execFileSync(circom, [mainPath, ...outputs, "--O2", "-o", outDir,
    "-l", resolve(CIRCUITS, "src"), "-l", resolve(CIRCUITS, "node_modules")], { encoding: "utf8", maxBuffer: 1 << 26 });
  const compileSeconds = (performance.now() - t0) / 1000;
  let cppSeconds;
  if (args.cpp) { // with --c circom names every output main_c.*: emit the C++ generator on its own, into <dir>/cpp/main_c_cpp
    mkdirSync(resolve(dir, "cpp"), { recursive: true });
    const t1 = performance.now();
    execFileSync(circom, [mainPath, "--c", "--no_asm", "--O2", "-o", resolve(dir, "cpp"),
      "-l", resolve(CIRCUITS, "src"), "-l", resolve(CIRCUITS, "node_modules")], { encoding: "utf8", maxBuffer: 1 << 26 });
    cppSeconds = (performance.now() - t1) / 1000;
    outputs.push("+ separate --c --no_asm run -> cpp/main_c_cpp");
  }
  const grab = (label) => Number((out.match(new RegExp(`(?<!non-)${label}:\\s*(\\d+)`)) ?? [])[1]);
  compile = {
    N, maxDepth: D, compileSeconds, ...(cppSeconds !== undefined ? { cppSeconds } : {}),
    nonLinearConstraints: grab("non-linear constraints"), linearConstraints: grab("linear constraints"),
    publicInputs: grab("public inputs"), privateInputs: grab("private inputs"), wires: grab("wires"),
  };
  const r1csPath = resolve(outDir, "main.r1cs");
  compile.r1csConstraints = r1csHeaderConstraints(r1csPath);
  if (!args["r1cs-only"] && (await snarkjs.r1cs.info(r1csPath)).nConstraints !== compile.r1csConstraints) {
    throw new Error("r1cs header reader disagrees with snarkjs r1cs.info");
  }
  compile.r1csSha256 = sha256(r1csPath);
  compile.outputs = outputs;
  compile.circomVersion = execFileSync(circom, ["--version"], { encoding: "utf8" }).trim();
  if (keyed) {
    const existing = existsSync(resolve(dir, "main.r1cs")) ? sha256(resolve(dir, "main.r1cs")) : null;
    rmSync(outDir, { recursive: true, force: true });
    if (existing !== compile.r1csSha256) {
      throw new Error(`this recompile would change the r1cs of the existing proving key in ${dir} ` +
        `(${existing} -> ${compile.r1csSha256}); the build was left untouched`);
    }
    compile.r1csIdenticalToKeyedBuild = true; // the keyed build's files and compile.json are kept as they are
  } else {
    writeFileSync(resolve(dir, "compile.json"), JSON.stringify(compile, null, 2) + "\n");
  }
}
console.log(JSON.stringify(compile));
if (args["compile-only"]) { await globalThis.curve_bn128?.terminate(); process.exit(0); }

if (keyed && args["no-verifier"]) {
  // a new contribution draws fresh randomness: the key would change while the exported verifier stays the old one
  throw new Error(`--no-verifier setup refuses to replace the existing proving key in ${dir}`);
}
const need = Math.ceil(Math.log2(compile.r1csConstraints + compile.publicInputs + 1));
const candidates = [];
for (let k = Math.max(need, 8); k <= 28; k++) {
  candidates.push(`powersOfTau28_hez_final_${k}.ptau`, `ppot_0080_${String(k).padStart(2, "0")}.ptau`);
}
const ptau = candidates.map((f) => resolve(CIRCUITS, "ptau", f)).find(existsSync);
if (!ptau) throw new Error(`no ptau with >= 2^${need} powers; run: node scripts/fetch-ptau.mjs ${need}`);

const t0 = performance.now();
await snarkjs.zKey.newZKey(resolve(dir, "main.r1cs"), ptau, resolve(dir, "main_0000.zkey"));
await snarkjs.zKey.contribute(resolve(dir, "main_0000.zkey"), resolve(dir, "main_final.zkey"),
  "spike single contribution", "thesis-poc spike fixed entropy -- NOT a secure ceremony");
const setupSeconds = (performance.now() - t0) / 1000;
const vkey = await snarkjs.zKey.exportVerificationKey(resolve(dir, "main_final.zkey"));
writeFileSync(resolve(dir, "verification_key.json"), JSON.stringify(vkey, null, 1));

if (args["no-verifier"]) {
  writeFileSync(resolve(dir, "setup.json"), JSON.stringify({ ptau: ptau.replace(ROOT, "."), ptauPower: need,
    setupSeconds, setupPeakRssMB: process.resourceUsage().maxRSS / 1024, verifierContract: null }, null, 2) + "\n");
  console.log(`setup done in ${setupSeconds.toFixed(1)} s (no verifier exported)`);
  await globalThis.curve_bn128?.terminate();
  process.exit(0);
}
const templates = { groth16: readFileSync(resolve(CIRCUITS, "node_modules", "snarkjs", "templates",
  "verifier_groth16.sol.ejs"), "utf8") };
const sol = await snarkjs.zKey.exportSolidityVerifier(resolve(dir, "main_final.zkey"), templates);
const name = varVerifierName(N, D);
writeFileSync(resolve(ROOT, "contracts", "contracts", "verifiers", `${name}.sol`),
  sol.replace(/contract Groth16Verifier/, `contract ${name}`));
writeFileSync(resolve(dir, "setup.json"), JSON.stringify({ ptau: ptau.replace(ROOT, "."), ptauPower: need,
  setupSeconds, setupPeakRssMB: process.resourceUsage().maxRSS / 1024, verifierContract: name }, null, 2) + "\n");
console.log(`setup done in ${setupSeconds.toFixed(1)} s; verifier -> contracts/contracts/verifiers/${name}.sol`);
await globalThis.curve_bn128?.terminate();
process.exit(0);
