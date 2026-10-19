// M2 end-to-end runner on the S1 Besu network (4 QBFT validators, full mesh).
//   node offchain/runner.mjs --run i4    full run: setup -> channels -> PSI -> commitAlignment ->
//                                         passive checks -> reveals -> Training (s fixed) -> I^ev, I^tr
//   node offchain/runner.mjs --run i5    alignment-timeout path (A never aligns; small alignmentPeriod)
// Four party processes (offchain/party.mjs): A on the power node (8545) and each passive on its
// own node (8546-8548). The runner itself only deploys the shared contracts (token, M1b
// verifier, factory) from an operator account and plays the arbiter (not a member, no node:
// it sends through a node picked at random), and, in I5, calls resolveAlignmentTimeout.
// Writes results/m2/<run>/: run.json, the parties' summaries, logs and output files.
import { spawn } from "node:child_process";
import { cpus, freemem, totalmem } from "node:os";
import { mkdirSync, readFileSync, rmSync, existsSync, createWriteStream, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { ChainClient, devWallet } from "./chain.mjs";
import { bytesToHex, channelKeyPair } from "./channel.mjs";

const { values: args } = parseArgs({ options: { run: { type: "string", default: "i4" } } });
const HERE = resolve(fileURLToPath(import.meta.url), "..");
const ROOT = resolve(HERE, "..", "..");
const out = resolve(ROOT, "results", "m2", args.run);
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
const NODES = { power: 8545, mci: 8546, digikala: 8547, maker: 8548 };
const rpc = (n) => `http://127.0.0.1:${NODES[n]}`;
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const t0 = Date.now();

// ---- machine and chain ------------------------------------------------------------------
const machine = { host: "Windows 11 laptop, i7-9750H, 16 GB", cpus: cpus().length, totalRamMB: Math.round(totalmem() / 2 ** 20),
  freeRamMBAtStart: Math.round(freemem() / 2 ** 20) };
const operator = new ChainClient(rpc("power"), 5, "operator");
const arbiterNode = Object.keys(NODES)[Math.floor(Math.random() * 4)];
const arbiter = new ChainClient(rpc(arbiterNode), 4, "arbiter");
const peers = await Promise.all(Object.keys(NODES).map((n) => new ChainClient(rpc(n), 5, "probe").provider.send("admin_peers", []).then((x) => x.length)));
const validators = await operator.provider.send("qbft_getValidatorsByBlockNumber", ["latest"]);
const clientVersion = await operator.provider.send("web3_clientVersion", []);
if (peers.some((x) => x !== 3) || validators.length < 4) throw new Error("not a 4-validator full mesh");
log(clientVersion, "validators", validators.length, "peers", peers.join(","), "arbiter via", arbiterNode, "free RAM MB", machine.freeRamMBAtStart);

// ---- shared contracts ------------------------------------------------------------------------
const token = await operator.deploy("deploy MockERC20", "MockERC20");
const verifier = await operator.deploy("deploy MembersVarVerifierN64D10", "MembersVarVerifierN64D10");
const factory = await operator.deploy("deploy FederationFactory", "FederationFactory");
const fromBlock = await operator.provider.getBlockNumber();
log("infrastructure deployed; token", token.target, "factory", factory.target);

// ---- the four party processes ----------------------------------------------------------------
const passives = [1, 2, 3].map((i) => devWallet(i).address);
const common = ["--run-dir", out, "--factory", factory.target, "--token", token.target, "--arbiter", arbiter.address];
const alignmentPeriod = args.run === "i5" ? "8" : "60";
const specs = [
  { name: "A", node: "power", argv: ["--role", "A", "--rpc", rpc("power"), "--verifier", verifier.target, "--passives", passives.join(","),
    "--alignment-period", alignmentPeriod, ...(args.run === "i5" ? ["--no-align"] : [])] },
  ...["mci", "digikala", "maker"].map((node, i) => ({ name: `p${i + 1}`, node,
    argv: ["--role", "p", "--index", String(i), "--rpc", rpc(node), "--from-block", String(fromBlock)] })),
];
const procs = specs.map((s) => {
  const p = spawn(process.execPath, [resolve(HERE, "party.mjs"), ...s.argv, ...common], { cwd: resolve(HERE, ".."), stdio: ["ignore", "pipe", "pipe"] });
  const f = createWriteStream(resolve(out, `${s.name}.log`));
  p.stdout.pipe(f); p.stderr.pipe(f);
  p.stdout.on("data", (d) => process.stdout.write(d));
  p.stderr.on("data", (d) => process.stderr.write(`[${s.name} stderr] ${d}`));
  return { ...s, proc: p, done: new Promise((ok) => p.on("exit", (code) => ok(code))) };
});

// ---- the arbiter ---------------------------------------------------------------------------------
const fedFile = resolve(out, "federation.json");
await arbiter.until(async () => existsSync(fedFile), "A's run", 600_000);
const fed = arbiter.contract("Federation", JSON.parse(readFileSync(fedFile, "utf8")).federation);
await arbiter.call("acceptArbiter", fed, "acceptArbiter", [bytesToHex(channelKeyPair().pk)]);
log("arbiter accepted through", arbiterNode);
if (args.run === "i5") {
  await arbiter.until(async () => Number(await fed.state()) !== 0, "start");
  const deadline = await fed.alignmentDeadline();
  log("started; alignmentDeadline", deadline);
  await arbiter.until(async () => BigInt(await arbiter.provider.getBlockNumber()) > deadline, "deadline", 600_000);
  await arbiter.call("resolveAlignmentTimeout", fed, "resolveAlignmentTimeout");
  log("alignment timeout resolved");
}

const codes = await Promise.all(procs.map((p) => p.done));
const read = (n) => JSON.parse(readFileSync(resolve(out, n, "summary.json"), "utf8"));
const parties = Object.fromEntries(specs.map((s) => [s.name, read(s.name)]));
const state = ["Setup", "Alignment", "Revealing", "Training", "SetupAborted", "RevealFailed", "AlignmentFailed", "Reconciliation", "Scoring", "Scored", "ScoringFailed"][Number(await fed.state())];
const result = {
  run: args.run, machine, chain: { clientVersion, validators, peersPerNode: peers, chainId: 2026 },
  federation: fed.target, arbiterNode, exitCodes: Object.fromEntries(specs.map((s, i) => [s.name, codes[i]])), finalState: state,
  wallClockMs: Date.now() - t0, operatorTxs: operator.txs, arbiterTxs: arbiter.txs, parties,
};
if (args.run === "i4") {
  const iev = (n) => readFileSync(resolve(out, n, "I_ev.json"), "utf8");
  result.checks = {
    allGammaIPass: ["p1", "p2", "p3"].every((p) => parties[p].gammaICheck.pass),
    sameIevEverywhere: ["p1", "p2", "p3"].every((p) => iev(p) === iev("A")),
    sizeI: parties.A.sizeI,
  };
} else {
  result.owed = Object.fromEntries(Object.entries(parties).map(([k, v]) => [k, v.owed]));
}
writeFileSync(resolve(out, "run.json"), JSON.stringify(result, null, 2) + "\n");
log("final state", state, "exit codes", JSON.stringify(result.exitCodes), JSON.stringify(result.checks ?? result.owed));
process.exit(codes.every((c) => c === 0) ? 0 : 1);
