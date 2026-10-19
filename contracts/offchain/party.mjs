// M2 party process: plays A or one passive from setup to the evaluation rows, as its own OS
// process with its own account, node, channel key pair and PSI scalars (kept in memory).
// Started by offchain/runner.mjs; usable by hand:
//   node offchain/party.mjs --role A --rpc http://127.0.0.1:8545 --run-dir <dir> --factory <addr> --token <addr>
//     --verifier <addr> --arbiter <addr> --passives <a1,a2,a3> [--alignment-period 60] [--no-align]
//   node offchain/party.mjs --role p --index 0 --rpc http://127.0.0.1:8546 --run-dir <dir> --factory <addr>
//     --token <addr> --arbiter <addr> --from-block <n>
// Off-chain coordination is limited to A's TCP endpoint (run-dir/a-endpoint.json); everything
// else (the run's address, keys, sizes, the seed) is read from the chain.
// Writes run-dir/<A|p1..p3>/summary.json and, once s is fixed, I.json, I_ev.json, I_tr.json.
// M3: with --m3 the process continues with training, reconciliation and scoring (party-m3.mjs).
//   A: [--no-score] (I9) [--delta-r n] [--scoring-period n];  passive: [--equivocate] (I7) [--withhold-final] (I8)
// full-data (Task 2): --scenario recs-full (full/recs-full-scenario.mjs, every cleaned household at A); A: --max-depth n
// (the proposal's maxDepth, default 10; 14 for the full data); every member: --run-seed k (full/run-seed.mjs: the
// seed-commitment secret r derived from k instead of drawn at random, so run k's on-chain seed is reproducible).
// MCI loan instance: --scenario mci-loan (scenarios/mci_loan/onchain/loan-scenario.mjs), as recs-full otherwise.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { activePair, buildI, checkGammaI, evaluationRows, gammaA, passiveSide, readChain, sendI } from "./alignment.mjs";
import { ChainClient } from "./chain.mjs";
import { bytesToHex, channelKeyPair } from "./channel.mjs";
import { genScenario } from "./data.mjs";
import { gammaI } from "./encoding.mjs";
import { connect, listen } from "./link.mjs";

const { values: args } = parseArgs({ options: {
  role: { type: "string" }, index: { type: "string", default: "0" }, rpc: { type: "string" }, "run-dir": { type: "string" },
  factory: { type: "string" }, token: { type: "string" }, verifier: { type: "string" }, arbiter: { type: "string" },
  passives: { type: "string" }, "from-block": { type: "string", default: "0" },
  "setup-period": { type: "string", default: "150" }, "alignment-period": { type: "string", default: "60" },
  "reveal-window": { type: "string", default: "30" }, "no-align": { type: "boolean", default: false },
  m3: { type: "boolean", default: false }, "no-score": { type: "boolean", default: false },
  equivocate: { type: "boolean", default: false }, "withhold-final": { type: "boolean", default: false },
  "delta-r": { type: "string", default: "30" }, "scoring-period": { type: "string", default: "400" },
  // DATA1 E3: the RECS scenario (data1/recs-scenario.mjs) and the passives' noise switch (QUESTIONS-DATA1 Q4)
  scenario: { type: "string", default: "" }, "scenario-seed": { type: "string", default: "20265000" },
  noise: { type: "string", default: "local" },
  "max-depth": { type: "string", default: "10" }, "run-seed": { type: "string", default: "" }, // full-data
} });
const isA = args.role === "A";
const idx = Number(args.index);
const name = isA ? "A" : `p${idx + 1}`;
const outDir = resolve(args["run-dir"], name);
mkdirSync(outDir, { recursive: true });
const client = new ChainClient(args.rpc, isA ? 0 : idx + 1, name);
const State = ["Setup", "Alignment", "Revealing", "Training", "SetupAborted", "RevealFailed", "AlignmentFailed", "Reconciliation", "Scoring", "Scored", "ScoringFailed", "Settled"];

const t0 = Date.now();
const steps = [];
async function step(label, fn) {
  const s = Date.now();
  const r = await fn();
  steps.push({ step: label, startMs: s - t0, ms: Date.now() - s });
  console.log(`[${name}] ${label} ${Date.now() - s} ms`);
  return r;
}
const summary = { party: name, address: client.address, rpc: args.rpc, steps, txs: client.txs };
const finish = (extra = {}) => {
  writeFileSync(resolve(outDir, "summary.json"), JSON.stringify({ ...summary, ...extra, rpcErrorsRetried: client.rpcErrors ?? 0, totalMs: Date.now() - t0 },
    (_, v) => (typeof v === "bigint" ? v.toString() : v), 2) + "\n");
};
const writeJson = (f, v) => writeFileSync(resolve(outDir, f), JSON.stringify(v, (_, x) => (typeof x === "bigint" ? x.toString() : x)) + "\n");

const scenario = args.scenario === "recs" // each party uses only its own part
  ? (await import("./data1/recs-scenario.mjs")).recsScenario({ seed: Number(args["scenario-seed"]) })
  : args.scenario === "recs-full"
    ? (await import("./full/recs-full-scenario.mjs")).recsFullScenario({ seed: Number(args["scenario-seed"]) })
    : args.scenario === "mci-loan" // the MCI loan instance (scenarios/mci_loan/onchain/loan-scenario.mjs)
      ? (await import("../../scenarios/mci_loan/onchain/loan-scenario.mjs")).loanScenario({ seed: Number(args["scenario-seed"]) })
      : genScenario();
/** The seed-commitment secret: from the run seed (full-data runs) or a fresh 256-bit secret. */
const seedSecret = async () => (args["run-seed"]
  ? (await import("./full/run-seed.mjs")).runSeedSecret(args["run-seed"], name)
  : BigInt(bytesToHex(channelKeyPair().sk)));
const keys = channelKeyPair();   // X25519; the public key goes on-chain, the secret stays here
const token = client.contract("MockERC20", args.token);
const factory = client.contract("FederationFactory", args.factory);
const stateOf = async (fed) => State[Number(await fed.state())];

if (isA) {
  const passives = args.passives.split(",");
  const { defaultProposal } = await import("./proposal.mjs");
  // Q5 check in its normal refusing mode (QUESTIONS-M2 Q6 answer: |X_A| <= 1,024 since M3).
  const p = defaultProposal({ activeParty: client.address, passives, arbiter: args.arbiter, verifier: args.verifier, token: args.token,
    activeChannelKey: bytesToHex(keys.pk), setupPeriod: Number(args["setup-period"]), alignmentPeriod: Number(args["alignment-period"]),
    revealWindow: Number(args["reveal-window"]), activeCustomers: scenario.A.ids.length, maxDepth: Number(args["max-depth"]),
    ...(scenario.classBoundaries ? { classBoundaries: scenario.classBoundaries } : {}),
    deltaR: Number(args["delta-r"]), scoringPeriod: Number(args["scoring-period"]) });
  summary.capacityCheck = { activeCustomers: scenario.A.ids.length, cap: 2 ** Number(p.maxDepth), mode: "refuse" };
  const rA = await seedSecret(); // a fresh 256-bit secret for the seed commitment (or the run seed's, full-data)
  const fed = await step("create (mint B, approve, factory.create)", async () => {
    await client.call("mint B", token, "mint", [client.address, p.B]);
    await client.call("approve B", token, "approve", [args.factory, p.B]);
    const rc = await client.call("create", factory, "create", [p]);
    const ev = rc.logs.map((l) => { try { return factory.interface.parseLog(l); } catch { return null; } }).find((e) => e?.name === "FederationCreated");
    return client.contract("Federation", ev.args.federation);
  });
  summary.federation = fed.target;
  writeFileSync(resolve(args["run-dir"], "federation.json"), JSON.stringify({ federation: fed.target }) + "\n");
  await step("commitSeed", async () => client.call("commitSeed", fed, "commitSeed", [await fed.seedCommitment(rA, client.address)]));
  await step("wait for the start (all members ready)", () => client.until(async () => (await stateOf(fed)) !== "Setup", "start"));
  if (args["no-align"]) {
    // I5: A never aligns. Wait for the timeout to be resolved, then collect what is owed.
    await step("wait (no alignment) until AlignmentFailed", () => client.until(async () => (await stateOf(fed)) === "AlignmentFailed", "timeout"));
    summary.owed = String(await fed.owed(client.address));
    await step("withdraw", () => client.call("withdraw", fed, "withdraw"));
    finish({ state: await stateOf(fed) });
    process.exit(0);
  }
  const chain = await readChain(fed);
  const pairsP = [];
  const server = await listen(0, (link) => pairsP.push(activePair({ link, chain, sk: keys.sk, ids: scenario.A.ids, timeoutMs: 600_000 })));
  writeFileSync(resolve(args["run-dir"], "a-endpoint.json"), JSON.stringify({ host: "127.0.0.1", port: server.address().port }) + "\n");
  const pairs = await step("PSI with 3 passives (in parallel)", async () => {
    await client.until(async () => pairsP.length === chain.passives.length, "3 connections");
    return Promise.all(pairsP);
  });
  const I = await step("build I (intersection of pairwise results, sorted)", async () => buildI(pairs.map((x) => x.intersection)));
  const gI = gammaI(I);
  const gA = await step("gamma_A (Poseidon Merkle, depth maxDepth)", () => gammaA(I, scenario.A.rows, chain.maxDepth));
  await step("send I and its AlignedSet signature to each passive", async () => { for (const pr of pairs) await sendI(pr, I, undefined, client.wallet); });
  await step("commitAlignment", () => client.call("commitAlignment", fed, "commitAlignment", [gI, I.length, gA]));
  await step("revealSeed", () => client.call("revealSeed A", fed, "revealSeed", [rA]));
  await step("wait for s (state Training)", () => client.until(async () => (await stateOf(fed)) === "Training", "Training (s fixed)"));
  const ev = await step("evaluation rows from on-chain s and sizeI", () => evaluationRows(fed, I));
  writeJson("I.json", I); writeJson("I_ev.json", ev.Iev); writeJson("I_tr.json", ev.Itr);
  summary.psi = pairs.map((x) => ({ passive: x.passive, pairwise: x.intersection.length, pSetSize: x.pSetSize,
    bytesOut: x.link.bytesOut, bytesIn: x.link.bytesIn, session: { ...x.session.stats }, timings: x.timings }));
  if (args.m3) {
    const { activeM3, relayInboxHandler } = await import("./party-m3.mjs");
    const relayInbox = new Map();
    const relayServer = await listen(0, relayInboxHandler({ fed, keys, contract: fed.target, chainId: (await client.provider.getNetwork()).chainId,
      activeParty: client.address, inbox: relayInbox }));
    writeFileSync(resolve(args["run-dir"], "a-relay-endpoint.json"), JSON.stringify({ host: "127.0.0.1", port: relayServer.address().port }) + "\n");
    await activeM3({ client, fed, pairs, I, ev, scenario, keys, runDir: args["run-dir"], outDir, args, step, summary, relayInbox });
    relayServer.close();
  }
  for (const pr of pairs) pr.link.destroy();
  server.close();
  finish({ state: await stateOf(fed), sizeI: I.length, gammaI: gI, gammaA: gA, seed: ev.s, idx: ev.idx });
  process.exit(0);
} else {
  const ids = scenario.passives[idx].ids;
  const fed = await step("discover the run on-chain (FederationCreated listing this passive)", async () => {
    let found = null;
    await client.until(async () => {
      const logs = await factory.queryFilter(factory.filters.FederationCreated(), Number(args["from-block"]));
      for (const l of logs) {
        const f = client.contract("Federation", l.args.federation);
        if ((await f.proposal()).passives.some((a) => a.toLowerCase() === client.address.toLowerCase())) found = f;
      }
      return found !== null;
    }, "FederationCreated");
    return found;
  });
  summary.federation = fed.target;
  const Dp = (await fed.proposal()).Dp;
  const r = await seedSecret();
  await step("register (mint Dp, approve, register, commitSeed)", async () => {
    await client.call("mint Dp", token, "mint", [client.address, Dp]);
    await client.call("approve Dp", token, "approve", [fed.target, Dp]);
    await client.call("register", fed, "register", [bytesToHex(keys.pk), args.arbiter]);
    await client.call("commitSeed", fed, "commitSeed", [await fed.seedCommitment(r, client.address)]);
  });
  await step("wait for the start", () => client.until(async () => (await stateOf(fed)) !== "Setup", "start"));
  const chain = await readChain(fed);
  const endpoint = resolve(args["run-dir"], "a-endpoint.json");
  const link = await step("connect to A", async () => {
    let l = null;
    await client.until(async () => {
      if ((await stateOf(fed)) !== "Alignment") return true; // e.g. AlignmentFailed (I5)
      if (!existsSync(endpoint)) return false;
      try { l = await connect(JSON.parse(readFileSync(endpoint, "utf8")).port); return true; } catch { return false; }
    }, "A's endpoint", 900_000);
    return l;
  });
  if (!link) {
    summary.owed = String(await fed.owed(client.address));
    await step("withdraw", () => client.call("withdraw", fed, "withdraw"));
    finish({ state: await stateOf(fed) });
    process.exit(0);
  }
  const res = await step("PSI with A, then receive I", () => passiveSide({ link, chain, sk: keys.sk, self: client.address, ids, timeoutMs: 600_000 }));
  await step("wait for commitAlignment", () => client.until(async () => (await stateOf(fed)) !== "Alignment", "commitAlignment"));
  const check = await step("gamma_I check against the chain", () => checkGammaI(fed, res.I, name));
  await step("wait for A's reveal", () => client.until(async () => (await fed.member(chain.activeParty)).revealed, "A's reveal"));
  await step("revealSeed", () => client.call("revealSeed", fed, "revealSeed", [r]));
  await step("wait for s (state Training)", () => client.until(async () => (await stateOf(fed)) === "Training", "Training (s fixed)"));
  const ev = await step("evaluation rows from on-chain s and sizeI", () => evaluationRows(fed, res.I));
  writeJson("I.json", res.I); writeJson("I_ev.json", ev.Iev); writeJson("I_tr.json", ev.Itr);
  summary.psi = { aSetSize: res.aSetSize, bytesOut: link.bytesOut, bytesIn: link.bytesIn, session: { ...res.session.stats }, timings: res.timings };
  summary.alignment = { alignSig: res.alignSig, listHash: res.listHash, gammaIOnChain: check.onChain, matches: check.pass }; // M4 objection evidence
  if (args.m3) {
    const { passiveM3 } = await import("./party-m3.mjs");
    await passiveM3({ client, fed, session: res.session, I: res.I, idx, scenario, keys, runDir: args["run-dir"], args, step, summary,
      alignment: { alignSig: res.alignSig, listHash: res.listHash, matches: check.pass } });
  }
  link.destroy();
  finish({ state: await stateOf(fed), gammaICheck: check, sizeI: res.I.length, idx: ev.idx });
  process.exit(0);
}
