// Multi-party phase 2 of the Groth16 setup for a scoring circuit (n = 3, N = 64; maxDepth 10 is the deployed one, 14
// the full-data one), on the public multi-party phase-1 file (PSE Perpetual Powers of Tau, 80 contributions, 2^20).
// Steps, one process each (mpc-step.mjs), in order: groth16 setup -> zkey contribute by power, mci, digikala, maker (each
// with its own entropy) -> zkey verify against the r1cs and the ptau -> export of the verification key and the Solidity
// verifier (MembersVarVerifierN<N>D<maxDepth>Mpc). Every step records its time, peak memory (the process's maximum
// resident set), available memory before it and its lowest value during the step (sampled every 250 ms) and the size of
// what it wrote. A step whose record exists is skipped (resume); the development key in build/members-var-N<N>-D<maxDepth>
// is never touched. Before setup, the circuit's fit in the phase-1 file is checked by snarkjs's own rule (groth16Fit).
// Outputs: build/members-var-N<N>-D<maxDepth>-mpc/main_0000..0004.zkey, main_final.zkey (a hard link to main_0004.zkey),
// verification_key.json; <out>/steps/*.json, ceremony.json, verification_key.json;
// contracts/contracts/verifiers/MembersVarVerifierN<N>D<maxDepth>Mpc.sol.
// Usage (from circuits/):
//   node scripts/mpc-ceremony.mjs                                            (maxDepth 10 -> results/mpc/setup)
//   node scripts/mpc-ceremony.mjs --maxdepth 14 --out ../results/full/setup [--verify-guard 1.31:512]
// --verify-guard <ratio>:<MB>: before verify, stop (exit 3, record steps/6-verify-guard.json) unless available memory is at
// least <ratio> x the setup step's peak + <MB> (the maxDepth 10 run's verify/setup peak ratio was 1.31).
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { closeSync, copyFileSync, createReadStream, existsSync, linkSync, mkdirSync, openSync, readFileSync, readSync, statSync,
  writeFileSync } from "node:fs";
import { cpus, freemem, platform, release, totalmem } from "node:os";
import { resolve } from "node:path";
import { pipeline } from "node:stream/promises";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

import { BUILD, CIRCUITS, ROOT } from "./lib.mjs";
import { varBuildDir } from "./m1b-lib.mjs";

export const MEMBERS = ["power", "mci", "digikala", "maker"];
const PTAU = { file: resolve(CIRCUITS, "ptau", "ppot_0080_20.ptau"), power: 20,
  source: "PSE Perpetual Powers of Tau, 80 contributions, truncated to 2^20 (pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_20.ptau); fetched by scripts/fetch-ptau.mjs (the Hermez files answer HTTP 403 from this network)" };

/**
 * The ceremony's circuit, folders and contract name from the command line. Defaults: the deployed circuit (N 64,
 * maxDepth 10) into results/mpc/setup, as the first run. Any other circuit needs --out, so that its step records can never
 * be mistaken for (and resumed from) the maxDepth 10 run's.
 */
export function ceremonyConfig(argv = []) {
  const { values: a } = parseArgs({ args: argv, options: { n: { type: "string", default: "64" }, maxdepth: { type: "string", default: "10" },
    out: { type: "string" }, "keys-dir": { type: "string" }, contract: { type: "string" }, "verify-guard": { type: "string" } } });
  const N = Number(a.n), D = Number(a.maxdepth);
  if (!(N >= 1) || !(D >= 1 && D <= 20)) throw new Error("--n and --maxdepth (1..20)");
  if (!a.out && !(N === 64 && D === 10)) throw new Error(`--out is required for N ${N}, maxDepth ${D} (the default results/mpc/setup holds the maxDepth 10 run)`);
  let guard = null;
  if (a["verify-guard"]) {
    const [ratio, marginMB] = a["verify-guard"].split(":").map(Number);
    if (!(ratio > 0) || !(marginMB >= 0)) throw new Error("--verify-guard <ratio>:<MB>, e.g. 1.31:512");
    guard = { ratio, marginMB };
  }
  return { N, D, devDir: varBuildDir(N, D), r1cs: resolve(varBuildDir(N, D), "main.r1cs"),
    keys: a["keys-dir"] ? resolve(a["keys-dir"]) : resolve(BUILD, `members-var-N${N}-D${D}-mpc`),
    contract: a.contract ?? `MembersVarVerifierN${N}D${D}Mpc`,
    out: a.out ? resolve(a.out) : resolve(ROOT, "results", "mpc", "setup"), guard };
}

/** The r1cs header (iden3 r1cs binary format, section type 1), read without loading the constraints. */
export function r1csHeader(path) {
  const fd = openSync(path, "r");
  const read = (pos, len) => { const b = Buffer.alloc(len); readSync(fd, b, 0, len, pos); return b; };
  try {
    const nSections = read(8, 4).readUInt32LE(0);
    let pos = 12;
    for (let s = 0; s < nSections; s++) {
      const h = read(pos, 12);
      if (h.readUInt32LE(0) === 1) {
        const fieldSize = read(pos + 12, 4).readUInt32LE(0);
        const b = read(pos + 16 + fieldSize, 28);
        return { nWires: b.readUInt32LE(0), nOutputs: b.readUInt32LE(4), nPubInputs: b.readUInt32LE(8), nPrvInputs: b.readUInt32LE(12),
          nLabels: Number(b.readBigUInt64LE(16)), nConstraints: b.readUInt32LE(24) };
      }
      pos += 12 + Number(h.readBigUInt64LE(4));
    }
    throw new Error("no r1cs header section");
  } finally { closeSync(fd); }
}

/**
 * snarkjs 0.7.6's own rule for a Groth16 setup (src/zkey_new.js): cirPower = log2(nConstraints + nPubInputs + nOutputs) + 1,
 * with misc.js's integer log2 (the floor, so cirPower is the bit length of the sum); refused if cirPower > the ptau's power;
 * the domain is 2^cirPower. Equivalently nConstraints + nPublic + 1 <= 2^power (the setup adds one row per public signal
 * and one for the constant 1).
 */
export function groth16Fit({ nConstraints, nPubInputs, nOutputs }, ptauPower) {
  const v = nConstraints + nPubInputs + nOutputs;
  const cirPower = v.toString(2).length;
  const rows = v + 1;
  return { nConstraints, nPublic: nPubInputs + nOutputs, rows, cirPower, domain: 2 ** cirPower, ptauPower, fits: cirPower <= ptauPower,
    marginRows: 2 ** ptauPower - rows, rule: "snarkjs 0.7.6 src/zkey_new.js: log2(nConstraints + nPubInputs + nOutputs) + 1 <= ptau power" };
}

const rel = (p) => p.replace(ROOT, ".").replaceAll("\\", "/");
async function digest(algo, file) {
  const h = createHash(algo);
  await pipeline(createReadStream(file), h);
  return h.digest("hex");
}
/** Available memory in MB: macOS (free + inactive + speculative) pages, elsewhere os.freemem(). */
function availMB() {
  if (platform() !== "darwin") return Math.round(freemem() / 2 ** 20);
  const v = spawnSync("vm_stat", { encoding: "utf8" }).stdout;
  const pages = (k) => Number((v.match(new RegExp(`Pages ${k}:\\s+(\\d+)`)) ?? [])[1] ?? 0);
  const size = Number((v.match(/page size of (\d+) bytes/) ?? [])[1] ?? 16384);
  return Math.round(((pages("free") + pages("inactive") + pages("speculative")) * size) / 2 ** 20);
}
/** Runs one mpc-step.mjs process; resolves with its exit status and stdout, sampling available memory meanwhile. */
function runStep(args) {
  return new Promise((ok) => {
    let lowest = availMB();
    const timer = setInterval(() => { try { lowest = Math.min(lowest, availMB()); } catch { /* keep sampling */ } }, 250);
    const p = spawn(process.execPath, [resolve(CIRCUITS, "scripts", "mpc-step.mjs"), ...args], { cwd: CIRCUITS, stdio: ["ignore", "pipe", "inherit"] });
    let stdout = "";
    p.stdout.on("data", (d) => { stdout += d; });
    p.on("exit", (status) => { clearInterval(timer); ok({ status, stdout, availMBLowest: Math.min(lowest, availMB()) }); });
  });
}

async function main() {
  const C = ceremonyConfig(process.argv.slice(2));
  const STEPS = resolve(C.out, "steps");
  mkdirSync(C.keys, { recursive: true });
  mkdirSync(STEPS, { recursive: true });
  const zkey = (k) => resolve(C.keys, `main_${String(k).padStart(4, "0")}.zkey`);
  const fit = groth16Fit(r1csHeader(C.r1cs), PTAU.power);
  console.log(`fit: ${fit.rows} rows (${fit.nConstraints} constraints + ${fit.nPublic} public + 1), domain 2^${fit.cirPower}, ` +
    `phase 1 2^${PTAU.power}: ${fit.fits ? `fits, margin ${fit.marginRows}` : "DOES NOT FIT"}`);
  if (!fit.fits) process.exit(1);

  const plan = [
    { id: "1-setup", args: ["setup", C.r1cs, PTAU.file, zkey(0)], out: zkey(0) },
    ...MEMBERS.map((m, i) => ({ id: `${i + 2}-contribute-${m}`, args: ["contribute", zkey(i), zkey(i + 1), m], out: zkey(i + 1) })),
    { id: "6-verify", args: ["verify", C.r1cs, PTAU.file, zkey(MEMBERS.length)] },
    { id: "7-export", args: ["export", zkey(MEMBERS.length), resolve(C.keys, "verification_key.json"),
      resolve(ROOT, "contracts", "contracts", "verifiers", `${C.contract}.sol`), C.contract] },
  ];
  const records = [];
  for (const s of plan) {
    const recFile = resolve(STEPS, `${s.id}.json`);
    if (existsSync(recFile) && (!s.out || existsSync(s.out))) {
      console.log(`skip ${s.id} (recorded)`);
      records.push(JSON.parse(readFileSync(recFile, "utf8")));
      continue;
    }
    const availMBBefore = availMB();
    if (s.id === "6-verify" && C.guard) {
      const setupPeak = records.find((r) => r.step === "setup").peakRssMB;
      const needMB = Math.round(setupPeak * C.guard.ratio + C.guard.marginMB);
      const g = { id: "6-verify-guard", setupPeakRssMB: setupPeak, ...C.guard, needMB, availMB: availMBBefore, at: new Date().toISOString() };
      writeFileSync(resolve(STEPS, "6-verify-guard.json"), JSON.stringify({ ...g, stopped: availMBBefore < needMB }, null, 1) + "\n");
      if (availMBBefore < needMB) {
        console.log(`STOP before verify: available ${availMBBefore} MB < ${needMB} MB (setup peak ${Math.round(setupPeak)} MB x ${C.guard.ratio} + ${C.guard.marginMB})`);
        process.exit(3);
      }
      console.log(`verify guard: available ${availMBBefore} MB >= ${needMB} MB`);
    }
    console.log(`${new Date().toISOString().slice(11, 19)} ${s.id}: start, available memory ${availMBBefore} MB`);
    const t0 = performance.now();
    const r = await runStep(s.args);
    const wallS = (performance.now() - t0) / 1000;
    if (r.status !== 0) throw new Error(`${s.id} exited ${r.status}\n${r.stdout.slice(-2000)}`);
    const res = JSON.parse(r.stdout.trim().split("\n").filter((l) => l.startsWith("{")).pop());
    const rec = { id: s.id, ...res, output: res.output && rel(res.output), verificationKey: res.verificationKey && rel(res.verificationKey),
      verifier: res.verifier && rel(res.verifier), wallS, availMBBefore, availMBLowest: r.availMBLowest,
      startedAt: new Date(Date.now() - wallS * 1000).toISOString() };
    writeFileSync(recFile, JSON.stringify(rec, null, 1) + "\n");
    console.log(`${s.id}: ${wallS.toFixed(1)} s, peak ${Math.round(res.peakRssMB)} MB, available lowest ${r.availMBLowest} MB` +
      (res.outputBytes ? `, ${(res.outputBytes / 2 ** 20).toFixed(1)} MB` : ""));
    records.push(rec);
  }

  const final = zkey(MEMBERS.length);
  if (!existsSync(resolve(C.keys, "main_final.zkey"))) linkSync(final, resolve(C.keys, "main_final.zkey")); // the prover's name
  copyFileSync(resolve(C.keys, "verification_key.json"), resolve(C.out, "verification_key.json"));
  const verify = records.find((r) => r.step === "verify");
  const ceremony = {
    what: `multi-party Groth16 phase 2 for the scoring circuit (n = 3, N = ${C.N}, maxDepth ${C.D})`,
    phase1: { ...PTAU, file: rel(PTAU.file), blake2b512: await digest("blake2b512", PTAU.file), bytes: statSync(PTAU.file).size },
    circuit: { r1cs: rel(C.r1cs), sha256: await digest("sha256", C.r1cs), ...JSON.parse(readFileSync(resolve(C.devDir, "compile.json"), "utf8")) },
    fit,
    members: MEMBERS,
    steps: records.map((r) => ({ id: r.id, step: r.step, member: r.member ?? null, wallS: r.wallS, peakRssMB: r.peakRssMB,
      availMBBefore: r.availMBBefore, availMBLowest: r.availMBLowest ?? null, outputMB: r.outputBytes ? r.outputBytes / 2 ** 20 : null,
      contributionHash: r.contributionHash ?? null })),
    verified: verify?.verified === true, verifyTranscript: verify?.transcript ?? [],
    finalZkey: { file: rel(final), sha256: await digest("sha256", final), bytes: statSync(final).size },
    verificationKey: { file: rel(resolve(C.out, "verification_key.json")), sha256: await digest("sha256", resolve(C.out, "verification_key.json")) },
    verifierContract: C.contract,
    machine: { platform: `${platform()} ${release()}`, cpu: cpus()[0]?.model, cores: cpus().length, totalRamMB: Math.round(totalmem() / 2 ** 20) },
    tools: { node: process.version, snarkjs: JSON.parse(readFileSync(resolve(CIRCUITS, "node_modules", "snarkjs", "package.json"), "utf8")).version },
    note: "all four contributions ran on one machine, one after another, each in its own process with its own entropy; in a real deployment each member contributes on its own machine and destroys its entropy",
  };
  writeFileSync(resolve(C.out, "ceremony.json"), JSON.stringify(ceremony, null, 1) + "\n");
  console.log(`ceremony: verified ${ceremony.verified}; final zkey sha256 ${ceremony.finalZkey.sha256.slice(0, 16)}...; vkey sha256 ${ceremony.verificationKey.sha256.slice(0, 16)}...`);
  if (!ceremony.verified) process.exit(1);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
