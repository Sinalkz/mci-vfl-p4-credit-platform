// M3 end-to-end runner on the S1 Besu network (4 QBFT validators, full mesh). Five processes:
// A (power node), the 3 passives (their own nodes) and the arbiter (a node picked at random).
//   node offchain/runner3.mjs --run i6   full path: setup -> alignment -> reveal -> training ->
//                                        reconciliation -> scoring (5 proofs) -> Scored
//   node offchain/runner3.mjs --run i7   equivocation: p2 registers another root than it signed
//   node offchain/runner3.mjs --run i8   non-receipt: p3 withholds its final message from A; resend
//                                        request, objection (Dc), arbiter relay, certifyDelivery
//   node offchain/runner3.mjs --run i9   scoring timeout: A never scores; the arbiter resolves (D115)
// M4: a Scored run continues through settlement (acknowledge / objectAlignment / settleBackup) and everyone withdraws.
//   --out-dir <dir>: write there instead of the run's default directory (M4's E3 run: ../results/m4/e3).
//   --machine-label <text>, --besu-launch <text>: recorded in run.json (defaults: the laptop, docker-compose). The chain
//   record also holds each node's Besu version and genesis block hash and the SHA-256 of chain/data/genesis.json.
//   --setup mpc|dev (default mpc): the Groth16 keys. mpc = the multi-party key of circuits/scripts/mpc-ceremony.mjs
//   (build/members-var-N64-D10-mpc, MembersVarVerifierN64D10Mpc, results/mpc/setup/verification_key.json); dev = the
//   development key (build/members-var-N64-D10, verifier MembersVarVerifierN64D10, results/m1b/verification_key.json).
//   --prover rapidsnark|snarkjs (default rapidsnark): A's scoring prover (rapidsnark needs RAPIDSNARK, its prover binary).
//   Both reach the child processes as VFL_KEY_DIR, VFL_VK_FILE and VFL_PROVER and are recorded in run.json. The defaults
//   are circuits/scripts/prover-config.mjs's; a missing rapidsnark binary or key file fails here, before anything runs.
// full-data (Task 2, items 3-4): --run full = the e3 path on the full-data scenario (full/recs-full-scenario.mjs: A holds
//   all 15,965 cleaned households; --scenario-seed, default 20265000), with the maxDepth 14 circuit's multi-party key
//   (--setup mpc-d14, the default for --run full: MembersVarVerifierN64D14Mpc, build/members-var-N64-D14-mpc,
//   results/full/setup/verification_key.json; A's proposal gets maxDepth 14) and --out-dir required. Also, for any run:
//   --run-seed k (every member's seed-commitment secret from k, full/run-seed.mjs) and --alignment-period n (blocks,
//   A's proposal; party.mjs's default 60 otherwise). All recorded in run.json.
// MCI loan instance: --run mci-loan = --run full on the loan scenario (scenarios/mci_loan/onchain/loan-scenario.mjs: the
//   Bank holds all 15,965 cohort applicants; --scenario-seed, default 20267005), with the same key, rules and records.
// The runner only deploys the shared contracts from an operator account (token, M1b verifier,
// factory). Writes results/m3/<run>/: run.json, each process's summary and log.
import { spawn } from "node:child_process";
import { cpus, freemem, totalmem } from "node:os";
import { createWriteStream, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { keccak256 } from "ethers";

import { KEY_DIRS, PROVERS, checkKeyDir, checkRapidsnark } from "../../circuits/scripts/prover-config.mjs";
import { ChainClient, chainIdentity, devWallet } from "./chain.mjs";

const { values: args } = parseArgs({ options: { run: { type: "string", default: "i6" }, "out-dir": { type: "string" },
  "machine-label": { type: "string", default: "Windows 11 laptop, i7-9750H, 16 GB" },
  "besu-launch": { type: "string", default: "docker-compose (chain/docker-compose.yml)" },
  setup: { type: "string" }, prover: { type: "string", default: "rapidsnark" }, // --setup: mpc (mpc-d14 for --run full)
  "run-seed": { type: "string" }, "scenario-seed": { type: "string" }, "alignment-period": { type: "string" } } });
const RUN = args.run;
if (!["i6", "i7", "i8", "i9", "e3", "full", "mci-loan"].includes(RUN)) throw new Error("--run i6|i7|i8|i9|e3|full|mci-loan");
const LOAN = RUN === "mci-loan"; // the MCI loan instance: as full, on the loan data
const SETUP_NAME = args.setup ?? (RUN === "full" || LOAN ? "mpc-d14" : "mpc");
const RECS = RUN === "e3" || RUN === "full"; // the RECS data, a run-specific model config (VFL_MODEL_CONFIG), noise off
for (const [flag, v] of [["--run-seed", args["run-seed"]], ["--scenario-seed", args["scenario-seed"]], ["--alignment-period", args["alignment-period"]]]) {
  if (v !== undefined && !/^\d+$/.test(v)) throw new Error(`${flag} ${v}: a non-negative integer`);
}
if ((RUN === "full" || LOAN) && !args["out-dir"]) throw new Error(`--run ${RUN} needs --out-dir`);
const HERE = resolve(fileURLToPath(import.meta.url), "..");
const ROOT = resolve(HERE, "..", "..");
const SETUPS = {
  dev: { verifier: "MembersVarVerifierN64D10", keyDir: KEY_DIRS.dev, vkFile: resolve(ROOT, "results", "m1b", "verification_key.json"), maxDepth: 10 },
  mpc: { verifier: "MembersVarVerifierN64D10Mpc", keyDir: KEY_DIRS.mpc,
    vkFile: resolve(ROOT, "results", "mpc", "setup", "verification_key.json"), maxDepth: 10 },
  "mpc-d14": { verifier: "MembersVarVerifierN64D14Mpc", keyDir: KEY_DIRS["mpc-d14"],
    vkFile: resolve(ROOT, "results", "full", "setup", "verification_key.json"), maxDepth: 14 },
};
const SETUP = SETUPS[SETUP_NAME];
if (!SETUP) throw new Error("--setup dev|mpc|mpc-d14");
if ((RUN === "full" || LOAN) && SETUP.maxDepth !== 14) throw new Error(`--run ${RUN} needs --setup mpc-d14 (|X_A| = 15,965 > 2^${SETUP.maxDepth})`);
if (!PROVERS.includes(args.prover)) throw new Error("--prover snarkjs|rapidsnark");
if (args.prover === "rapidsnark") checkRapidsnark(process.env.RAPIDSNARK);
checkKeyDir(SETUP.keyDir);
process.env.VFL_KEY_DIR = SETUP.keyDir;
process.env.VFL_VK_FILE = SETUP.vkFile;
process.env.VFL_PROVER = args.prover;
// e3 (DATA1 §5): the I6 path on real RECS rows (data1/recs-scenario.mjs), passives' noise off (Q4); the model
// config comes from VFL_MODEL_CONFIG, which every child process inherits. Checked before anything is deleted.
{
  const { checkRunConfig, fiSpec } = await import("./model.mjs");
  let dataWidths = null;
  if (RECS || LOAN) {
    const { loadManifest, PASSIVE_NAMES, passiveNames } = await import(LOAN ? "../../scenarios/mci_loan/offchain/prepared.mjs" : "./data1/prepared.mjs");
    const m = loadManifest();
    dataWidths = (LOAN ? passiveNames(m) : PASSIVE_NAMES).map((n) => m.parties[n].length); // the loan's v2: the manifest's order
  }
  // full and mci-loan: e3's rule (a run-specific model config is required and must fit the data's widths)
  checkRunConfig({ run: RECS || LOAN ? "e3" : RUN, configEnv: process.env.VFL_MODEL_CONFIG, fiIn: [1, 2, 3].map((i) => fiSpec(i).in), dataWidths });
}
const out = args["out-dir"] ? resolve(args["out-dir"]) : RUN === "e3" ? resolve(ROOT, "results", "data1", "e3") : resolve(ROOT, "results", "m3", RUN);
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
const NODES = { power: 8545, mci: 8546, digikala: 8547, maker: 8548 };
const rpc = (n) => `http://127.0.0.1:${NODES[n]}`;
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const t0 = Date.now();

const machine = { host: args["machine-label"], cpus: cpus().length, totalRamMB: Math.round(totalmem() / 2 ** 20),
  freeRamMBAtStart: Math.round(freemem() / 2 ** 20) };
const operator = new ChainClient(rpc("power"), 5, "operator");
const arbiterNode = Object.keys(NODES)[Math.floor(Math.random() * 4)];
const probes = Object.keys(NODES).map((n) => new ChainClient(rpc(n), 5, "probe").provider);
const peers = await Promise.all(probes.map((p) => p.send("admin_peers", []).then((x) => x.length)));
const validators = await operator.provider.send("qbft_getValidatorsByBlockNumber", ["latest"]);
const identity = await chainIdentity(probes, resolve(ROOT, "chain", "data", "genesis.json")); // M4: Besu version, genesis hash
const { clientVersion } = identity;
if (peers.some((x) => x !== 3) || validators.length < 4) throw new Error("not a 4-validator full mesh");
log(clientVersion, "genesis", identity.genesisBlockHash, "validators", validators.length, "arbiter via", arbiterNode, "free RAM MB", machine.freeRamMBAtStart);

const token = await operator.deploy("deploy MockERC20", "MockERC20");
const verifier = await operator.deploy(`deploy ${SETUP.verifier}`, SETUP.verifier);
const factory = await operator.deploy("deploy FederationFactory", "FederationFactory");
const fromBlock = await operator.provider.getBlockNumber();
log("infrastructure deployed; factory", factory.target);

const passives = [1, 2, 3].map((i) => devWallet(i).address);
const arbiter = devWallet(4).address;
const common = ["--run-dir", out, "--factory", factory.target, "--token", token.target, "--arbiter", arbiter, "--m3",
  ...(RECS || LOAN ? ["--scenario", LOAN ? "mci-loan" : RUN === "full" ? "recs-full" : "recs", "--noise", "off"] : []),
  ...(args["scenario-seed"] !== undefined ? ["--scenario-seed", args["scenario-seed"]] : []),
  ...(args["run-seed"] !== undefined ? ["--run-seed", args["run-seed"]] : [])];
if (RECS || LOAN) {
  const scenarioSeed = args["scenario-seed"] !== undefined ? Number(args["scenario-seed"]) : undefined;
  const sc = LOAN ? (await import("../../scenarios/mci_loan/onchain/loan-scenario.mjs")).loanScenario({ seed: scenarioSeed })
    : RUN === "full"
      ? (await import("./full/recs-full-scenario.mjs")).recsFullScenario({ seed: scenarioSeed })
      : (await import("./data1/recs-scenario.mjs")).recsScenario({ seed: scenarioSeed });
  writeFileSync(resolve(out, "sample.json"), JSON.stringify({ ...sc.sample, sizeI: sc.I.length, pairwise: sc.pairwise.map((p) => p.length),
    classBoundaries: sc.classBoundaries.map(String), modelConfig: process.env.VFL_MODEL_CONFIG ?? null }, null, 1) + "\n");
}
const aFlags = ["--scoring-period", RUN === "i9" ? "20" : "400", "--delta-r", "30", ...(RUN === "i9" ? ["--no-score"] : []),
  "--max-depth", String(SETUP.maxDepth), ...(args["alignment-period"] !== undefined ? ["--alignment-period", args["alignment-period"]] : [])];
const pFlags = (i) => [...(RUN === "i7" && i === 1 ? ["--equivocate"] : []), ...(RUN === "i8" && i === 2 ? ["--withhold-final"] : [])];
const specs = [
  { name: "A", script: "party.mjs", argv: ["--role", "A", "--rpc", rpc("power"), "--verifier", verifier.target, "--passives", passives.join(","), ...aFlags] },
  ...["mci", "digikala", "maker"].map((node, i) => ({ name: `p${i + 1}`, script: "party.mjs",
    argv: ["--role", "p", "--index", String(i), "--rpc", rpc(node), "--from-block", String(fromBlock), ...pFlags(i)] })),
  { name: "arbiter", script: "arbiter-party.mjs", argv: ["--rpc", rpc(arbiterNode), ...(RUN === "i9" ? ["--resolve-scoring-timeout"] : [])] },
];
const procs = specs.map((s) => {
  const argv = s.script === "party.mjs" ? [...s.argv, ...common] : [...s.argv, "--run-dir", out];
  const p = spawn(process.execPath, [resolve(HERE, s.script), ...argv], { cwd: resolve(HERE, ".."), stdio: ["ignore", "pipe", "pipe"] });
  const f = createWriteStream(resolve(out, `${s.name}.log`));
  p.stdout.pipe(f); p.stderr.pipe(f);
  p.stdout.on("data", (d) => process.stdout.write(d));
  p.stderr.on("data", (d) => process.stderr.write(`[${s.name} stderr] ${d}`));
  return { ...s, proc: p, done: new Promise((ok) => p.on("exit", (code) => ok(code))) };
});
const codes = await Promise.all(procs.map((p) => p.done));
const read = (n) => { try { return JSON.parse(readFileSync(resolve(out, n, "summary.json"), "utf8")); } catch { return null; } };
const parties = Object.fromEntries(specs.map((s) => [s.name, read(s.name)]));
const fed = operator.contract("Federation", JSON.parse(readFileSync(resolve(out, "federation.json"), "utf8")).federation);
const STATES = ["Setup", "Alignment", "Revealing", "Training", "SetupAborted", "RevealFailed", "AlignmentFailed", "Reconciliation", "Scoring", "Scored", "ScoringFailed", "Settled"];
// M4: the settlement as the chain recorded it (events from the run's creation on)
const ev = async (name) => (await fed.queryFilter(fed.filters[name](), fromBlock)).map((e) => ({ block: e.blockNumber, args: [...e.args].map(String) }));
const credited = {};
for (const e of await ev("Credited")) credited[e.args[0]] = String(BigInt(credited[e.args[0]] ?? 0) + BigInt(e.args[1]));
const names = Object.fromEntries([["A", devWallet(0).address], ...passives.map((a, i) => [`p${i + 1}`, a]), ["arbiter", arbiter]].map(([k, a]) => [a, k]));
const settledEv = (await ev("Settled"))[0] ?? null;
const P = await fed.proposal(); // M4: the run's terms as the contract holds them (for py/m4_e3.py)
const result = {
  run: RUN, machine, chain: { ...identity, launch: args["besu-launch"], validators, peersPerNode: peers, chainId: 2026 },
  runSeed: args["run-seed"] ?? null, scenarioSeed: args["scenario-seed"] ?? null, alignmentPeriodFlag: args["alignment-period"] ?? null,
  keys: { setup: SETUP_NAME, verifierContract: SETUP.verifier, verifier: verifier.target, vkFile: SETUP.vkFile.replace(ROOT, ".").replaceAll("\\", "/"),
    vkHash: keccak256(readFileSync(SETUP.vkFile)), proposalVkHash: P.vkHash },
  prover: args.prover,
  federation: fed.target, arbiterNode, exitCodes: Object.fromEntries(specs.map((s, i) => [s.name, codes[i]])),
  finalState: STATES[Number(await fed.state())], wallClockMs: Date.now() - t0, operatorTxs: operator.txs, parties,
  chainEnd: {
    proposal: { B: String(P.B), rho: String(P.rho), Dp: String(P.Dp), Dc: String(P.Dc), K: Number(P.K), deltaA: Number(P.deltaA),
      deltaR: Number(P.deltaR), scoringPeriod: Number(P.scoringPeriod), finalWait: Number(P.finalWait), T: Number(P.T), N: Number(P.N),
      setupPeriod: Number(P.setupPeriod), alignmentPeriod: Number(P.alignmentPeriod), revealWindow: Number(P.revealWindow),
      maxDepth: Number(P.maxDepth), passives: [...P.passives] },
    trainingEndBlock: Number(await fed.trainingEndBlock()), reconciliationEndBlock: Number(await fed.reconciliationEndBlock()),
    forfeitedToPool: String(await fed.forfeitedToPool()), dcEscrow: String(await fed.dcEscrow()), leftCount: Number(await fed.leftCount()),
    acceptedFigures: Number(await fed.acceptedFigures()), requiredFigures: Number(await fed.requiredFigures()),
    members: await Promise.all(passives.map(async (a) => { const s = await fed.finalStatus(a); return { address: a, gamma: s.gamma, left: s.left, equivocated: s.equivocated, dpForfeited: s.dpForfeited, objection: Number(s.objection) }; })),
    owed: Object.fromEntries(await Promise.all([["A", devWallet(0).address], ...passives.map((a, i) => [`p${i + 1}`, a]), ["arbiter", arbiter]]
      .map(async ([k, a]) => [k, String(await fed.owed(a))]))),
    gammaFA: await fed.gammaFA(), gammaG: await fed.gammaG(), uFed: Number(await fed.uFed()), // D139: U_fed = max(F_r, F_j)
    figures: await Promise.all([[0, 0], [3, 0], [1, 1], [1, 2], [1, 3], [2, 0]].map(async ([k, j]) => { const f = await fed.figure(k, j); return { kind: k, memberIndex: j, accepted: f.accepted, F: Number(f.F), cModel: f.cModel }; })),
    settlement: { // M4
      scoredBlock: Number(await fed.scoredBlock()), settlementDeadline: Number(await fed.settlementDeadline()),
      path: ["none", "normal", "objection", "backup"][Number(await fed.settlementPath())],
      settled: settledEv && { block: settledEv.block, path: Number(settledEv.args[0]), E: Number(settledEv.args[1]), estate: Number(settledEv.args[2]),
        awards: settledEv.args[3].split(",").filter((x) => x !== "").map(Number) },
      acknowledged: (await ev("Acknowledged")).map((e) => ({ block: e.block, passive: names[e.args[0]] ?? e.args[0] })),
      objections: (await ev("AlignmentObjectionUpheld")).map((e) => ({ block: e.block, passive: names[e.args[0]] ?? e.args[0], listHash: e.args[1] })),
      credited: Object.fromEntries(Object.entries(credited).map(([a, v]) => [names[a] ?? a, v])),
      withdrawn: (await ev("Withdrawn")).map((e) => ({ block: e.block, to: names[e.args[0]] ?? e.args[0], amount: e.args[1] })),
      lockedForever: String(await fed.lockedForever()),
      contractBalance: String(await token.balanceOf(fed.target)),
    },
  },
};
writeFileSync(resolve(out, "run.json"), JSON.stringify(result, (_, v) => (typeof v === "bigint" ? v.toString() : v), 2) + "\n");
log("final state", result.finalState, "exit codes", JSON.stringify(result.exitCodes));
process.exit(codes.every((c) => c === 0) ? 0 : 1);
