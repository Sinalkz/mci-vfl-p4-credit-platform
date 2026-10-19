// full-data (Task 2, item 4, the student's decision 4): the alignment the full-data runs need, measured before the runs,
// so that the proposal's alignment period can be declared as at least twice the measured time. Exactly as in a run
// (party.mjs): A in one process runs the PSI with the 3 passives in parallel (alignment.mjs activePair, one TCP link
// each), every passive in its own process (passiveSide); then A builds I, computes gamma_A at depth 14 and sends the
// signed I to each passive. The sets are the full-data scenario's (full/recs-full-scenario.mjs, scenario seed
// 20265000), not synthetic. No chain: the contract address and the keys are made up for the benchmark (the PSI and the
// channel do not depend on them). Timed from A's first incoming connection (each process builds its scenario first) to
// the last signed I sent; also each part. Every process's peak memory is its own.
// The declared alignment period: ceil(2 x the slowest repetition's alignment time / 2 s per block), rounded up to the
// next multiple of 10 blocks (QBFT, chain/qbftConfigFile.json: 2 s per block), and never below party.mjs's default 60.
// Usage (from contracts/):  node offchain/full/bench-psi-full.mjs [--reps 2] [--out ../results/full/psi-bench.json]
//                           [--limit n]  (the first n of each set only: the test's small run)
// MCI loan instance: --scenario mci-loan measures the loan scenario's sets (scenarios/mci_loan/onchain/loan-scenario.mjs,
// scenario seed 20267005) instead; the default --scenario recs-full is the thesis's measurement, unchanged.
import { spawn } from "node:child_process";
import { cpus, freemem, platform, release, totalmem } from "node:os";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { Wallet } from "ethers";

import { activePair, buildI, gammaA, passiveSide, sendI } from "../alignment.mjs";
import { bytesToHex, channelKeyPair, hexToBytes } from "../channel.mjs";
import { gammaI } from "../encoding.mjs";
import { connect, listen } from "../link.mjs";
import { FULL_SEED, recsFullScenario } from "./recs-full-scenario.mjs";

const HERE = fileURLToPath(import.meta.url);
const { values: args } = parseArgs({ options: {
  reps: { type: "string", default: "2" }, out: { type: "string" }, limit: { type: "string" }, "max-depth": { type: "string", default: "14" },
  side: { type: "string" }, index: { type: "string" }, port: { type: "string" }, keys: { type: "string" },
  "block-seconds": { type: "string", default: "2" }, scenario: { type: "string", default: "recs-full" },
} });
if (!["recs-full", "mci-loan"].includes(args.scenario)) throw new Error("--scenario recs-full|mci-loan");
const LOAN = args.scenario === "mci-loan" ? await import("../../../scenarios/mci_loan/onchain/loan-scenario.mjs") : null;
const SCENARIO_SEED = LOAN ? LOAN.LOAN_SEED : FULL_SEED;
const CONTRACT = "0x" + "c0".repeat(20);
const PASSIVE_ADDRS = ["0x" + "b1".repeat(20), "0x" + "b2".repeat(20), "0x" + "b3".repeat(20)];

/** The scenario, optionally cut to the first `limit` IDs of each set (the test's small run). */
function sets() {
  const sc = LOAN ? LOAN.loanScenario({ seed: SCENARIO_SEED }) : recsFullScenario({ seed: FULL_SEED });
  if (!args.limit) return sc;
  const n = Number(args.limit);
  const a = sc.A.ids.slice(0, n);
  const aSet = new Set(a);
  const passives = sc.passives.map((p) => ({ ...p, ids: [...p.ids.filter((x) => aSet.has(x)).slice(0, Math.floor(0.9 * n)), ...p.ids.filter((x) => !sc.A.rows.has(x)).slice(0, Math.floor(0.1 * n))] }));
  const pSets = passives.map((p) => new Set(p.ids));
  return { ...sc, A: { ...sc.A, ids: a }, passives, I: a.filter((x) => pSets.every((s) => s.has(x))).sort((x, y) => (x < y ? -1 : 1)) };
}

if (args.side) {
  // ---- one side (child process) ----
  const k = JSON.parse(args.keys);
  const signer = new Wallet(k.signerA);
  const chain = { contract: CONTRACT, chainId: 2026n, activeParty: signer.address, activeKey: k.pkA, passives: PASSIVE_ADDRS,
    passiveKey: async (addr) => k.pkP[PASSIVE_ADDRS.findIndex((x) => x.toLowerCase() === addr.toLowerCase())] };
  const tb = performance.now();
  const sc = sets();
  const buildScenarioMs = performance.now() - tb;
  let res;
  if (args.side === "A") {
    const pairsP = [];
    const arrivals = [];
    let t0 = null;
    const server = await listen(0, (link) => {
      t0 ??= performance.now();
      arrivals.push(performance.now() - t0);
      pairsP.push(activePair({ link, chain, sk: hexToBytes(k.skA), ids: sc.A.ids, timeoutMs: 3_600_000 }));
    });
    console.log(`PORT ${server.address().port}`);
    while (pairsP.length < 3) await new Promise((ok) => setTimeout(ok, 5));
    const pairs = await Promise.all(pairsP);
    const tPsi = performance.now();
    const I = buildI(pairs.map((x) => x.intersection));
    const tI = performance.now();
    const gA = await gammaA(I, sc.A.rows, Number(args["max-depth"]));
    const tG = performance.now();
    for (const pr of pairs) await sendI(pr, I, undefined, signer);
    const tS = performance.now();
    const expected = sc.I.map(String);
    res = { side: "A", buildScenarioMs, arrivalsMs: arrivals, aSetSize: sc.A.ids.length,
      psiMs: tPsi - t0, buildIMs: tI - tPsi, gammaAMs: tG - tI, sendIMs: tS - tG, alignmentMs: tS - t0,
      sizeI: I.length, IEqualsScenario: JSON.stringify(I.map(String)) === JSON.stringify(expected), gammaI: gammaI(I), gammaA: gA,
      pairs: pairs.map((x) => ({ passive: x.passive, pairwise: x.intersection.length, pSetSize: x.pSetSize, timings: x.timings,
        bytesOut: x.link.bytesOut, bytesIn: x.link.bytesIn, session: { ...x.session.stats } })) };
    await new Promise((ok) => setTimeout(ok, 300));
    for (const pr of pairs) pr.link.destroy();
    server.close();
  } else {
    const i = Number(args.index);
    const link = await connect(Number(args.port));
    const t0 = performance.now();
    const r = await passiveSide({ link, chain, sk: hexToBytes(k.skP[i]), self: PASSIVE_ADDRS[i], ids: sc.passives[i].ids, timeoutMs: 3_600_000 });
    res = { side: `p${i + 1}`, buildScenarioMs, pSetSize: sc.passives[i].ids.length, received: r.I.length, ms: performance.now() - t0,
      timings: r.timings, bytesOut: link.bytesOut, bytesIn: link.bytesIn, session: { ...r.session.stats } };
    link.destroy();
  }
  res.peakRssMB = process.resourceUsage().maxRSS / 1024;
  console.log("RESULT " + JSON.stringify(res));
  process.exit(0);
}

// ---- coordinator ----
const run = (argv) => {
  const p = spawn(process.execPath, [HERE, ...argv], { stdio: ["ignore", "pipe", "inherit"] });
  let out = "";
  const port = new Promise((ok) => p.stdout.on("data", (d) => { out += d; const m = out.match(/PORT (\d+)/); if (m) ok(Number(m[1])); }));
  const done = new Promise((ok, no) => p.on("exit", (c) => (c === 0 ? ok(JSON.parse(out.match(/RESULT (.*)/)[1])) : no(new Error(`side exited ${c}`)))));
  return { port, done };
};
const reps = [];
for (let r = 0; r < Number(args.reps); r++) {
  const a = channelKeyPair();
  const ps = [0, 1, 2].map(() => channelKeyPair());
  const keys = JSON.stringify({ skA: bytesToHex(a.sk), pkA: bytesToHex(a.pk), skP: ps.map((p) => bytesToHex(p.sk)),
    pkP: ps.map((p) => bytesToHex(p.pk)), signerA: Wallet.createRandom().privateKey });
  const freeRamMBAtStart = Math.round(freemem() / 2 ** 20);
  const t0 = Date.now();
  const A = run(["--side", "A", "--keys", keys, ...(args.limit ? ["--limit", args.limit] : []), "--max-depth", args["max-depth"],
    "--scenario", args.scenario]);
  const port = String(await A.port);
  const P = [0, 1, 2].map((i) => run(["--side", "p", "--index", String(i), "--port", port, "--keys", keys, ...(args.limit ? ["--limit", args.limit] : []),
    "--scenario", args.scenario]));
  const [ra, ...rp] = await Promise.all([A.done, ...P.map((p) => p.done)]);
  const row = { rep: r + 1, freeRamMBAtStart, wallMs: Date.now() - t0, A: ra, passives: rp,
    correct: ra.IEqualsScenario && rp.every((x) => x.received === ra.sizeI) };
  reps.push(row);
  console.log(JSON.stringify({ rep: r + 1, alignmentS: +(ra.alignmentMs / 1000).toFixed(1), psiS: +(ra.psiMs / 1000).toFixed(1),
    gammaAS: +(ra.gammaAMs / 1000).toFixed(2), sizeI: ra.sizeI, A_MB: Math.round(ra.peakRssMB), p_MB: rp.map((x) => Math.round(x.peakRssMB)),
    resent: ra.pairs.reduce((s, x) => s + x.session.retransmitted, 0) + rp.reduce((s, x) => s + x.session.retransmitted, 0), correct: row.correct }));
}
const blockS = Number(args["block-seconds"]);
const slowest = Math.max(...reps.map((x) => x.A.alignmentMs)) / 1000;
const declared = Math.max(60, Math.ceil(Math.ceil((2 * slowest) / blockS) / 10) * 10); // never below party.mjs's default
const result = {
  what: "the full-data alignment (PSI with 3 passives in parallel, build I, gamma_A at depth 14, signed I to each), measured "
        + "to declare the alignment period (contracts/offchain/full/bench-psi-full.mjs)",
  scenario: { name: args.scenario, seed: SCENARIO_SEED, limit: args.limit ? Number(args.limit) : null },
  machine: { cpu: cpus()[0]?.model, threads: cpus().length, totalRamMB: Math.round(totalmem() / 2 ** 20), os: `${platform()} ${release()}`,
    node: process.version },
  reps, slowestAlignmentS: slowest, blockSeconds: blockS,
  alignmentPeriod: { blocks: declared, seconds: declared * blockS, rule: "ceil(2 x slowest alignment / block time), rounded up to a multiple of 10 blocks, at least the default 60",
    default: 60, ratioToSlowest: (declared * blockS) / slowest },
  note: "no chain and no Besu during the measurement; the factor 2 is the margin for the run's concurrent load and the "
        + "transactions (commitAlignment) that also fall in the window",
};
const out = args.out ? resolve(args.out) : resolve(dirname(HERE), "..", "..", "..", "results", "full", "psi-bench.json");
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(result, null, 1) + "\n");
console.log(JSON.stringify({ slowestAlignmentS: slowest, alignmentPeriodBlocks: declared, all_correct: reps.every((x) => x.correct) }));
if (!reps.every((x) => x.correct)) process.exit(1);
