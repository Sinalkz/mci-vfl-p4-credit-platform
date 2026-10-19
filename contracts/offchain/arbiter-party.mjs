// M3: the arbiter as its own process (account 4, a node picked by the runner).
//   node offchain/arbiter-party.mjs --rpc <url> --run-dir <dir> [--resolve-scoring-timeout]
// Accepts the role with a fresh X25519 key, then serves the relay (M3 §4): a passive hands it a
// sealed final message; it checks the signature and the root against the member's gamma,
// reseals it to A's registered key, delivers it to A and certifies the delivery on-chain.
// With --resolve-scoring-timeout (I9) it ends a run whose scoring deadline passed.
// Writes run-dir/arbiter/summary.json.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { arbiterRelay, RMSG } from "./arbiter.mjs";
import { ChainClient } from "./chain.mjs";
import { bytesToHex, channelKeyPair, deriveChannelKeys, hexToBytes } from "./channel.mjs";
import { listen } from "./link.mjs";
import { acceptRelay, dialRelay, memberOf, relayPayload } from "./party-m3.mjs";

const { values: args } = parseArgs({ options: {
  rpc: { type: "string" }, "run-dir": { type: "string" }, "resolve-scoring-timeout": { type: "boolean", default: false },
} });
const outDir = resolve(args["run-dir"], "arbiter");
mkdirSync(outDir, { recursive: true });
const client = new ChainClient(args.rpc, 4, "arbiter");
const keys = channelKeyPair();
const t0 = Date.now();
const steps = [];
const summary = { party: "arbiter", address: client.address, rpc: args.rpc, steps, txs: client.txs, relays: [] };
const step = async (label, fn) => {
  const s = Date.now();
  const r = await fn();
  steps.push({ step: label, startMs: s - t0, ms: Date.now() - s });
  console.log(`[arbiter] ${label} ${Date.now() - s} ms`);
  return r;
};
const finish = (extra = {}) => writeFileSync(resolve(outDir, "summary.json"),
  JSON.stringify({ ...summary, ...extra, rpcErrorsRetried: client.rpcErrors ?? 0, totalMs: Date.now() - t0 }, (_, v) => (typeof v === "bigint" ? v.toString() : v), 2) + "\n");

const fedFile = resolve(args["run-dir"], "federation.json");
await step("wait for A's run", () => client.until(async () => existsSync(fedFile), "A's run", 600_000));
const fed = client.contract("Federation", JSON.parse(readFileSync(fedFile, "utf8")).federation);
await step("acceptArbiter", () => client.call("acceptArbiter", fed, "acceptArbiter", [bytesToHex(keys.pk)]));
const chainId = (await client.provider.getNetwork()).chainId;
const contract = fed.target;
const p = await fed.proposal();

// The relay endpoint (the only off-chain coordination: its port, in the run folder).
const pending = [];
const server = await listen(0, (link) => pending.push((async () => {
  const { from, session } = await acceptRelay(link, async (member) => deriveChannelKeys({ selfSk: keys.sk,
    peerPk: hexToBytes((await fed.member(member)).channelKey), contract, chainId, activeParty: client.address, passive: member }), "A");
  const m = await session.recv(600_000);
  session.close();
  const { member, box } = memberOf(m.payload);
  const rec = { member, from, receivedAtBlock: await client.provider.getBlockNumber() };
  if (member.toLowerCase() !== from.toLowerCase()) { summary.relays.push({ ...rec, ok: false, reason: "sender is not the member" }); return; }
  const res = await arbiterRelay({ fed, box, arbiterSk: keys.sk, member, chainId });
  if (!res.ok) { summary.relays.push({ ...rec, ok: false, reason: res.reason }); return; }
  const aPort = JSON.parse(readFileSync(resolve(args["run-dir"], "a-relay-endpoint.json"), "utf8")).port;
  const toA = await dialRelay({ port: aPort, self: client.address, sk: keys.sk, peerPk: p.activeChannelKey, contract, chainId,
    activeParty: p.activeParty, passive: client.address, role: "p" });
  await toA.send(RMSG.box, relayPayload(member, res.boxForA));
  toA.close();
  await step(`certifyDelivery ${member.slice(0, 8)}..`, () => client.call("certifyDelivery", fed, "certifyDelivery", [member, res.root]));
  summary.relays.push({ ...rec, ok: true, root: res.root, deliveredAtBlock: await client.provider.getBlockNumber() });
})().catch((e) => summary.relays.push({ ok: false, error: String(e) }))));
writeFileSync(resolve(args["run-dir"], "arb-endpoint.json"), JSON.stringify({ host: "127.0.0.1", port: server.address().port }) + "\n");

const TERMINAL = [4, 5, 6, 10, 11]; // SetupAborted, RevealFailed, AlignmentFailed, ScoringFailed, Settled (M4)
await step("serve the relay until the run ends", () => client.until(async () => {
  const st = Number(await fed.state());
  if (args["resolve-scoring-timeout"] && st === 8) {
    const deadline = await fed.scoringDeadline();
    if (BigInt(await client.provider.getBlockNumber()) > deadline) {
      await client.call("resolveScoringTimeout", fed, "resolveScoringTimeout");
      return true;
    }
  }
  return TERMINAL.includes(st);
}, "the end of the run", 3_600_000));
await Promise.all(pending);
server.close();
// M4 (D130): D_c was credited to the arbiter when each relay objection closed
const owedArb = await fed.owed(client.address);
summary.owed = String(owedArb);
if (owedArb > 0n) await step("withdraw (D_c)", () => client.call("withdraw", fed, "withdraw"));
finish({ state: Number(await fed.state()) });
process.exit(0);
