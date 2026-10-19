// RESULTS-PROOFCOST: one measurement step per call, each in child processes, one at a time. Records the system's free
// RAM at the start and its minimum during the step (sampled every 250 ms) next to the step's own output.
// Free RAM: os.freemem() (Windows: available physical memory); on macOS, where os.freemem() counts only never-used
// pages, free + inactive + speculative + purgeable pages from vm_stat (the memory the OS can hand out without swapping).
// Usage (from circuits/):
//   node scripts/proofcost.mjs --step compile --n 32 --maxdepth 10 [--r1cs-only] [--cpp]   (m1b-build --compile-only)
//   node scripts/proofcost.mjs --step setup   --n 64 --maxdepth 14        (m1b-build --setup-only --no-verifier)
//   node scripts/proofcost.mjs --step prove   --n 64 --maxdepth 14 [--isize 1000] [--reps 3]   (m1b-prove)
//   node scripts/proofcost.mjs --step scoring --conc 2 [--threads 3] --inputs <dir>   (E3's six scoring proofs, <= conc at a time)
//   node scripts/proofcost.mjs --step threads --inputs <dir> [--threads-list 1,2,3,6,12] [--reps 3]   (E3's FULL proof, N64 D10)
//   node scripts/proofcost.mjs --step compare --n 128 --maxdepth 10 [--input <file>] [--cpp] [--reps 3] [--warmup 1] [--threads 10]
//        (wasm vs C++ witness, snarkjs vs rapidsnark; macOS/Linux; RAPIDSNARK = path of rapidsnark's prover binary)
//   node scripts/proofcost.mjs --step cpp-make --n 64 --maxdepth 10 --cpath <gmp/include:dir with nlohmann/> --libpath <dir with libgmp.a>
//        (macOS: builds the C++ witness generator emitted by compile --cpp)
// <dir> holds <FIGURE>.input.json from contracts/offchain/data1/proofcost-inputs.mjs. Writes results/proofcost/[<label>/]<step>-*.json
// (--machine-label <label> for a machine other than the laptop the first tables were measured on).
// --threads / --threads-list set the prover thread count through scripts/threads-preload.mjs: a MEASUREMENT setting only.
// Server options (results/proofcost/server): --threads k also applies to the compile, setup, prove and compare steps
// (the snarkjs worker pool of each node child; for a run pinned to k CPUs with taskset); --warmup w (compare) runs each
// tool w times untimed before the timed reps and records those runs apart; --max-old <MB> replaces node's heap cap of
// 14336 MB for the children (m1b-build's snarkjs.r1cs.info holds every constraint in the heap: about 4.6 GB at N 64).
// On Linux /usr/bin/time is GNU time (-v, kbytes) and os.freemem() is /proc/meminfo MemAvailable.
import { spawn } from "node:child_process";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { availableParallelism, cpus, freemem, platform, release, totalmem } from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import * as snarkjs from "snarkjs";

import { CIRCUITS, ROOT } from "./lib.mjs";
import { membersVarInput, varBuildDir } from "./m1b-lib.mjs";
import { loadS2b } from "./members-lib.mjs";
import { peakRssMBFromTime, timeFlag } from "./native-prover.mjs";

const { values: a } = parseArgs({ options: {
  step: { type: "string" }, n: { type: "string" }, maxdepth: { type: "string" }, isize: { type: "string", default: "1000" },
  reps: { type: "string", default: "3" }, conc: { type: "string", default: "1" }, inputs: { type: "string" },
  input: { type: "string" }, threads: { type: "string" }, "threads-list": { type: "string", default: "1,2,3,6,12" },
  "r1cs-only": { type: "boolean", default: false }, cpp: { type: "boolean", default: false },
  "machine-label": { type: "string" }, cpath: { type: "string" }, libpath: { type: "string" },
  "min-free-mb-per-proof": { type: "string", default: "6100" }, // E3's largest single-proof peak RSS was 6,027 MB
  warmup: { type: "string", default: "0" }, "max-old": { type: "string", default: "14336" },
} });
const OUT = resolve(ROOT, "results", "proofcost", ...(a["machine-label"] ? [a["machine-label"]] : []));
mkdirSync(OUT, { recursive: true });
const MB = 2 ** 20;
const MAX_OLD = Number(a["max-old"]);
const THREADS = a.threads ? Number(a.threads) : undefined; // compile/setup/prove/compare; scoring and threads read a.threads
const FIGS = ["FULL", "FULL_J", "LOO_1", "LOO_2", "LOO_3", "BASELINE"];
const PRELOAD = resolve(CIRCUITS, "scripts", "threads-preload.mjs");

function availMB() {
  if (platform() !== "darwin") return Math.round(freemem() / MB);
  const vm = execFileSync("vm_stat", { encoding: "utf8" });
  const page = Number(vm.match(/page size of (\d+) bytes/)[1]);
  const pages = (k) => Number((vm.match(new RegExp(`Pages ${k}:\\s+(\\d+)`)) ?? [0, 0])[1]);
  return Math.round(((pages("free") + pages("inactive") + pages("speculative") + pages("purgeable")) * page) / MB);
}
const machine = () => ({ cpu: cpus()[0].model, threads: cpus().length, totalRamMB: Math.round(totalmem() / MB), node: process.version,
  os: `${platform()} ${release()}`, freeRamMethod: platform() === "darwin" ? "vm_stat free+inactive+speculative+purgeable" :
    platform() === "linux" ? "os.freemem() (/proc/meminfo MemAvailable)" : "os.freemem()",
  ...(platform() === "linux" ? { availableParallelism: availableParallelism() } : {}) }); // the CPUs this process may run on (taskset)

/**
 * Spawns node (or `cmd` when given); resolves with stdout. `time`: wrap in /usr/bin/time (-l macOS, -v GNU/Linux) and
 * return peak RSS.
 * `keys: "dev"`: m3-prove-figure.mjs with the development key and snarkjs, as the scoring tables were measured (its own
 * defaults are the multi-party key and rapidsnark).
 */
function run(args, { maxOld, threads, cmd, time, keys } = {}) {
  return new Promise((ok, no) => {
    const nodeArgs = [...(maxOld ? [`--max-old-space-size=${maxOld}`] : []), ...(threads ? ["--import", pathToFileURL(PRELOAD).href] : [])];
    let exe = cmd ?? process.execPath;
    let argv = cmd ? args : [...nodeArgs, ...args];
    if (time) { argv = [timeFlag(), exe, ...argv]; exe = "/usr/bin/time"; }
    const env = { ...process.env, ...(threads ? { PROVER_THREADS: String(threads) } : {}),
      ...(keys === "dev" ? { VFL_KEY_DIR: varBuildDir(64, 10), VFL_PROVER: "snarkjs" } : {}) };
    const t0 = performance.now();
    const p = spawn(exe, argv, { cwd: CIRCUITS, env, stdio: ["ignore", "pipe", "pipe"] });
    let out = "", err = "";
    p.stdout.on("data", (d) => { out += d; if (!time) process.stdout.write(d); });
    p.stderr.on("data", (d) => { err += d; if (!time) process.stderr.write(d); });
    p.on("exit", (c) => {
      const ms = performance.now() - t0;
      if (c !== 0) return no(new Error(`${exe} ${argv.slice(0, 3).join(" ")} exited ${c}\n${err.slice(-2000)}`));
      const rss = time ? peakRssMBFromTime(err) : undefined;
      ok(time ? { out, ms, peakRssMB: rss } : out);
    });
  });
}
const lastJson = (s) => JSON.parse(s.trim().split("\n").filter((l) => l.startsWith("{")).pop());
const median = (xs) => [...xs].sort((x, y) => x - y)[Math.floor(xs.length / 2)];

async function measured(fn) {
  const freeStartMB = availMB();
  let minFree = freeStartMB;
  const timer = setInterval(() => { try { minFree = Math.min(minFree, availMB()); } catch { /* keep sampling */ } }, 250);
  const t0 = performance.now();
  try {
    const result = await fn();
    return { freeStartMB, minFreeMB: minFree, wallS: (performance.now() - t0) / 1000, result };
  } finally { clearInterval(timer); }
}

const script = (f) => resolve(CIRCUITS, "scripts", f);
const e3Expected = () => JSON.parse(readFileSync(resolve(ROOT, "results", "data1", "e3", "A", "summary.json"), "utf8")).scoring;
const vkeyOf = (dir) => JSON.parse(readFileSync(resolve(dir, "verification_key.json"), "utf8"));
const N = a.n, D = a.maxdepth;
let name, rec;

if (a.step === "compile") {
  name = `compile-N${N}-D${D}${a["r1cs-only"] ? "-r1cs" : ""}`;
  rec = await measured(async () => lastJson(await run([script("m1b-build.mjs"), "--n", N, "--maxdepth", D, "--compile-only",
    ...(a["r1cs-only"] ? ["--r1cs-only"] : []), ...(a.cpp ? ["--cpp"] : [])], { maxOld: MAX_OLD, threads: THREADS }))); // r1cs.info needs > node's default heap from ~1M constraints
} else if (a.step === "setup") {
  name = `setup-N${N}-D${D}`;
  rec = await measured(async () => {
    await run([script("m1b-build.mjs"), "--n", N, "--maxdepth", D, "--setup-only", "--no-verifier"], { maxOld: MAX_OLD, threads: THREADS });
    return JSON.parse(readFileSync(resolve(varBuildDir(Number(N), Number(D)), "setup.json"), "utf8"));
  });
} else if (a.step === "prove") {
  name = `prove-N${N}-D${D}`;
  rec = await measured(async () => lastJson(await run([script("m1b-prove.mjs"), "--n", N, "--maxdepth", D, "--isize", a.isize,
    "--reps", a.reps, "--proof-dir", resolve(OUT, "proofs")], { maxOld: MAX_OLD, threads: THREADS })));
} else if (a.step === "threads") {
  // E3's FULL statement at N 64, maxDepth 10; one witness, then `reps` proofs per thread count, each in its own process.
  name = "threads-N64-D10";
  const dir = varBuildDir(64, 10);
  const work = resolve(OUT, "work-threads");
  mkdirSync(work, { recursive: true });
  const want = e3Expected().find((s) => s.figure === "FULL").publicSignals;
  rec = await measured(async () => {
    const wtns = resolve(work, "FULL.wtns");
    lastJson(await run([script("prove-one.mjs"), "witness", resolve(a.inputs, "FULL.input.json"), resolve(dir, "main_js", "main.wasm"), wtns]));
    const rows = [];
    for (const k of a["threads-list"].split(",").map(Number)) {
      const reps = [];
      for (let r = 0; r < Number(a.reps); r++) {
        const pf = resolve(work, `t${k}-r${r}.proof.json`);
        const o = lastJson(await run([script("prove-one.mjs"), "prove", resolve(dir, "main_final.zkey"), wtns, pf], { maxOld: 14336, threads: k }));
        const { proof, publicSignals } = JSON.parse(readFileSync(pf, "utf8"));
        if (!(await snarkjs.groth16.verify(vkeyOf(dir), publicSignals, proof))) throw new Error(`threads ${k}: proof does not verify`);
        if (JSON.stringify(publicSignals) !== JSON.stringify(want)) throw new Error(`threads ${k}: public signals differ from E3's FULL`);
        reps.push({ ms: o.ms, peakRssMB: o.peakRssMB, workers: o.workers, free: availMB() });
      }
      if (reps.some((x) => x.workers !== k)) throw new Error(`threads ${k}: snarkjs used ${reps.map((x) => x.workers)} workers`);
      rows.push({ threads: k, proveMs: reps.map((x) => x.ms), proveMsMedian: median(reps.map((x) => x.ms)),
        peakRssMB: Math.max(...reps.map((x) => x.peakRssMB)), workers: k, verified: true, publicSignalsEqualE3: true });
      console.log(JSON.stringify(rows.at(-1)));
    }
    return { input: "E3 FULL (N 64, maxDepth 10, sizeI 910)", reps: Number(a.reps), rows };
  });
  rmSync(work, { recursive: true, force: true });
} else if (a.step === "scoring") {
  const conc = Number(a.conc);
  const threads = a.threads ? Number(a.threads) : undefined;
  name = `scoring-conc${conc}${threads ? `-t${threads}` : ""}`;
  const need = conc * Number(a["min-free-mb-per-proof"]);
  const free = availMB();
  if (free < need) {
    rec = { notRun: `needs ${need} MB free for ${conc} proofs at once (${a["min-free-mb-per-proof"]} MB each), ${free} MB free`, freeStartMB: free, threads };
  } else {
    const expected = e3Expected();
    const work = resolve(OUT, "work");
    mkdirSync(work, { recursive: true });
    rec = await measured(async () => {
      const t0 = performance.now();
      const queue = [...FIGS];
      const proofs = {};
      const worker = async () => {
        for (let f = queue.shift(); f; f = queue.shift()) {
          const start = (performance.now() - t0) / 1000;
          const outFile = resolve(work, `${f}.proof.json`);
          await run([script("m3-prove-figure.mjs"), resolve(a.inputs, `${f}.input.json`), outFile], { threads, keys: "dev" });
          const r = JSON.parse(readFileSync(outFile, "utf8"));
          const want = expected.find((s) => s.figure === f).publicSignals;
          if (JSON.stringify(r.publicSignals) !== JSON.stringify(want)) throw new Error(`${f}: public signals differ from E3's`);
          proofs[f] = { startS: start, endS: (performance.now() - t0) / 1000, witnessMs: r.witnessMs, proveMs: r.proveMs,
            verifyMs: r.verifyMs, peakRssMB: r.peakRssMB, publicSignalsEqualE3: true };
        }
      };
      await Promise.all(Array.from({ length: conc }, worker));
      return { conc, threads: threads ?? cpus().length, proofs: FIGS.map((f) => ({ figure: f, ...proofs[f] })) };
    });
    rmSync(work, { recursive: true, force: true });
  }
} else if (a.step === "cpp-make") {
  // Builds <build>/cpp/main_c_cpp/main_c (from compile --cpp) with circom's Makefile; CPATH / LIBRARY_PATH (--cpath,
  // --libpath) point the compiler at GMP's header, nlohmann/json.hpp and a directory holding only libgmp.a (static link).
  name = `cpp-make-N${N}-D${D}`;
  const src = resolve(varBuildDir(Number(N), Number(D)), "cpp", "main_c_cpp");
  const env = { ...process.env, CPATH: a.cpath ?? "", LIBRARY_PATH: a.libpath ?? "" };
  rec = await measured(async () => {
    const t0 = performance.now();
    // -D_LONG_LONG_LIMB: circom's --no_asm field code passes uint64_t* (unsigned long long) where GMP's arm64 macOS limb
    // is unsigned long; both are 64-bit with one calling convention (checked by the byte-identical witness in compare)
    const cflags = `-std=c++11 -O3 -I.${platform() === "darwin" ? " -D_LONG_LONG_LIMB" : ""}`;
    execFileSync("make", ["-C", src, "-j4", `CFLAGS=${cflags}`], { env, encoding: "utf8", maxBuffer: 1 << 26 });
    return { makeSeconds: (performance.now() - t0) / 1000, cflags, compiler: execFileSync("c++", ["--version"], { encoding: "utf8" }).split("\n")[0],
      binary: "cpp/main_c_cpp/main_c", linkedLibs: execFileSync("otool", ["-L", resolve(src, "main_c")], { encoding: "utf8" }).trim().split("\n").slice(1).map((l) => l.trim()) };
  });
} else if (a.step === "compare") {
  // wasm vs C++ witness (C++ timings only if its .wtns is byte-identical to the wasm one), snarkjs vs rapidsnark proving.
  name = `compare-N${N}-D${D}`;
  const dir = varBuildDir(Number(N), Number(D));
  const work = resolve(OUT, `work-compare-N${N}-D${D}`);
  mkdirSync(work, { recursive: true });
  const reps = Number(a.reps);
  const W = Number(a.warmup);
  // `W` untimed warm-up runs, then `reps` timed ones; each run does the same work (and the same checks)
  const repeat = async (fn) => { const warm = [], runs = []; for (let r = -W; r < reps; r++) (r < 0 ? warm : runs).push(await fn()); return { warm, runs }; };
  const warmRec = (warm) => (W ? { warmup: warm.map((x) => ({ ms: x.ms, peakRssMB: x.peakRssMB })) } : {});
  rec = await measured(async () => {
    let input = a.input;
    let F;
    if (!input) {
      const built = await membersVarInput(Number(N), Number(a.isize), Number(D), { data: loadS2b() });
      input = resolve(work, "input.json");
      writeFileSync(input, JSON.stringify(built.input));
      F = Number(built.input.F);
    }
    const wW = resolve(work, "w_wasm.wtns");
    const { warm: wasmWarm, runs: wasm } = await repeat(async () =>
      lastJson(await run([script("prove-one.mjs"), "witness", input, resolve(dir, "main_js", "main.wasm"), wW])));
    let cpp = null;
    if (a.cpp) {
      // circom's C++ JSON reader rejects negative numbers (snarkjs reduces them mod p): same input, every value as its
      // field element (decimal string of v mod p); the byte-identical witness check below confirms the equivalence
      const P = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;
      const fe = (v) => (Array.isArray(v) ? v.map(fe) : String(((BigInt(v) % P) + P) % P));
      const inCpp = resolve(work, "input_field.json");
      writeFileSync(inCpp, JSON.stringify(Object.fromEntries(Object.entries(JSON.parse(readFileSync(input, "utf8"))).map(([k, v]) => [k, fe(v)]))));
      const wC = resolve(work, "w_cpp.wtns");
      const { warm, runs } = await repeat(() => run([inCpp, wC], { cmd: resolve(dir, "cpp", "main_c_cpp", "main_c"), time: true }));
      const identical = readFileSync(wW).equals(readFileSync(wC));
      cpp = identical ? { identicalToWasm: true, ms: runs.map((x) => x.ms), msMedian: median(runs.map((x) => x.ms)),
        peakRssMB: Math.max(...runs.map((x) => x.peakRssMB)), ...warmRec(warm) } : { identicalToWasm: false };
    }
    const vk = vkeyOf(dir);
    let ps;
    const { warm: snarkWarm, runs: snark } = await repeat(async () => {
      const pf = resolve(work, "snarkjs.proof.json");
      const o = lastJson(await run([script("prove-one.mjs"), "prove", resolve(dir, "main_final.zkey"), wW, pf], { maxOld: MAX_OLD, threads: THREADS }));
      const { proof, publicSignals } = JSON.parse(readFileSync(pf, "utf8"));
      if (!(await snarkjs.groth16.verify(vk, publicSignals, proof))) throw new Error("snarkjs proof does not verify");
      ps = publicSignals;
      return o;
    });
    let rapid = null;
    if (process.env.RAPIDSNARK) {
      const { warm, runs } = await repeat(async () => {
        const pf = resolve(work, "rapid.proof.json"), pub = resolve(work, "rapid.public.json");
        const o = await run([resolve(dir, "main_final.zkey"), wW, pf, pub], { cmd: process.env.RAPIDSNARK, time: true });
        const proof = JSON.parse(readFileSync(pf, "utf8")), publicSignals = JSON.parse(readFileSync(pub, "utf8"));
        if (!(await snarkjs.groth16.verify(vk, publicSignals, proof))) throw new Error("rapidsnark proof does not verify");
        if (JSON.stringify(publicSignals) !== JSON.stringify(ps)) throw new Error("rapidsnark public signals differ from snarkjs's");
        return o;
      });
      rapid = { ms: runs.map((x) => x.ms), msMedian: median(runs.map((x) => x.ms)), peakRssMB: Math.max(...runs.map((x) => x.peakRssMB)),
        verifiedBySnarkjs: true, publicSignalsEqualSnarkjs: true, ...warmRec(warm) };
    }
    const zkeyMB = statSync(resolve(dir, "main_final.zkey")).size / MB;
    return { N: Number(N), maxDepth: Number(D), input: a.input ? a.input.replace(/.*[\\/]/, "") : `S2b data, sizeI ${a.isize}`, F, reps,
      ...(W ? { warmup: W } : {}), zkeyMB, publicSignals: ps,
      wasmWitness: { ms: wasm.map((x) => x.ms), msMedian: median(wasm.map((x) => x.ms)), peakRssMB: Math.max(...wasm.map((x) => x.peakRssMB)),
        ...warmRec(wasmWarm) },
      cppWitness: cpp,
      snarkjs: { ms: snark.map((x) => x.ms), msMedian: median(snark.map((x) => x.ms)), peakRssMB: Math.max(...snark.map((x) => x.peakRssMB)),
        workers: snark[0].workers, verified: true, ...warmRec(snarkWarm) },
      rapidsnark: rapid };
  });
  rmSync(work, { recursive: true, force: true });
} else throw new Error("--step compile|setup|prove|threads|scoring|cpp-make|compare");

const file = resolve(OUT, `${name}.json`);
// the measurement settings (the defaults unless a server option was given); OMP_NUM_THREADS reaches rapidsnark via env
const settings = { maxOldMB: MAX_OLD, threads: THREADS ?? null, warmup: Number(a.warmup), ompNumThreads: process.env.OMP_NUM_THREADS ?? null,
  rapidsnark: process.env.RAPIDSNARK ?? null, circom: process.env.CIRCOM ?? null };
writeFileSync(file, JSON.stringify({ step: a.step, N: N ? Number(N) : undefined, maxDepth: D ? Number(D) : undefined,
  machine: machine(), at: new Date().toISOString(), settings, ...rec }, null, 1) + "\n");
console.log(`wrote ${file.replace(ROOT, ".")}`);
await globalThis.curve_bn128?.terminate();
process.exit(0);
