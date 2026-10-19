// M3 party phases after the seed: training, reconciliation, scoring (A) and training, final
// commitment, resend/relay handling, settlement (passive). Called by party.mjs with --m3.
// Every off-chain message is carried by a Session (M2): A <-> passive on the PSI sessions;
// passive -> arbiter and arbiter -> A on sessions keyed from the registered X25519 keys.
// Relay messages start with a clear hello naming the sender, then one sealed "relay/box"
// whose payload is member address (20 bytes) || sealed box (arbiter.mjs).
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { activeOpenRelayed, RMSG, seal } from "./arbiter.mjs";
import { Channel, clearBody, deriveChannelKeys, hexToBytes, parseBody } from "./channel.mjs";
import { addressBytes, concatBytes, utf8 } from "./encoding.mjs";
import { connect } from "./link.mjs";
import { assertFiWidth } from "./model.mjs";
import { circuitInputFor, gammaFAHex, gammaGHex, KIND, proveInChild, requiredFigures, scoreFigures } from "./scoring.mjs";
import { Session } from "./session.mjs";
import { TMSG, activeTraining, decFinal, embeddingRootHex, finalSigner, passiveTraining, waitForFinalCommits } from "./training.mjs";

/** Each passive's own noise setting (local, not in the proposal; recorded in its summary). */
export const LOCAL_NOISE = [{ clip: 4, sigma: 0.05 }, { clip: 4, sigma: 0.1 }, { clip: 3, sigma: 0.05 }];
const RESEND_WAIT_BLOCKS = 3;        // blocks A waits for a direct resend before filing (Q10)

const hex = (b) => "0x" + Buffer.from(b).toString("hex");
const memberOf = (payload) => ({ member: hex(payload.subarray(0, 20)), box: payload.subarray(20) });
export const relayPayload = (member, box) => concatBytes(addressBytes(member), box);

/** Opens a Session to `port` with a clear hello naming `self`; keys from our secret and the peer's registered key. */
export async function dialRelay({ port, self, sk, peerPk, contract, chainId, activeParty, passive, role }) {
  const link = await connect(port);
  await link.send(clearBody("hello", utf8(JSON.stringify({ address: self }))));
  return new Session(link, new Channel(role, deriveChannelKeys({ selfSk: sk, peerPk: hexToBytes(peerPk), contract, chainId, activeParty, passive })));
}

/** Server side of a relay hop: reads the hello, derives the keys through keysFor(address), returns { from, session }. */
export async function acceptRelay(link, keysFor, role) {
  const hello = parseBody(await link.recv(60_000));
  const from = JSON.parse(new TextDecoder().decode(hello.rest)).address;
  return { from, session: new Session(link, new Channel(role, await keysFor(from))) };
}

// ---- A ----------------------------------------------------------------------------------

export async function activeM3({ client, fed, pairs, I, ev, scenario, keys, runDir, outDir, args, step, summary, relayInbox }) {
  const chainId = (await client.provider.getNetwork()).chainId;
  const contract = fed.target;
  const p = await fed.proposal();
  const maxDepth = Number(p.maxDepth);
  const addrs = [...p.passives];
  const peers = addrs.map((a, i) => ({ index: i + 1, address: a, session: pairs.find((x) => x.passive.toLowerCase() === a.toLowerCase()).session }));
  const rows = { xa: I.map((id) => scenario.A.rows.get(id).xa), y: I.map((id) => scenario.A.rows.get(id).y) };
  const picked = new Set(ev.idx);
  const Itr = [...I.keys()].filter((k) => !picked.has(k));

  const tr = await step("training: T rounds, then the final messages", () =>
    activeTraining({ peers, rows, Itr, sizeI: I.length, chainId, contract, maxDepth }));
  summary.training = { rounds: tr.rounds, zeroFilled: tr.zeroFilled, finals: tr.finals.map(({ hq, ...f }) => f),
    finalBytes: tr.finals.map((f) => f.bytes ?? null) };
  // M4 (D129): an off-chain rule of A's process; the contract does not enforce it
  summary.finalWait = await step(`wait for every passive's commitFinal (at most finalWait = ${p.finalWait} blocks)`, () =>
    waitForFinalCommits({ provider: client.provider, fed, addrs, finalWait: Number(p.finalWait) }));
  const gFA = await gammaFAHex(tr.fA);
  const gG = await gammaGHex(tr.fusion); // D139: the jointly trained fusion model, F_j is proven for it
  summary.gammaG = gG;
  await step("commitFA (C_fA and gammaG)", () => client.call("commitFA", fed, "commitFA", [gFA, gG]));

  // Reconciliation: equivocation proofs, resend requests, objections.
  const binding = addrs.map(() => null);
  for (const [i, f] of tr.finals.entries()) {
    const st = await fed.finalStatus(addrs[i]);
    if (st.left || !f.received || !f.signatureValid) continue;
    if (st.gamma.toLowerCase() === f.root.toLowerCase()) { binding[i] = f.hq; continue; }
    await step(`proveEquivocation p${i + 1} (registered ${st.gamma.slice(0, 10)}.., signed ${f.root.slice(0, 10)}..)`, () =>
      client.call(`proveEquivocation p${i + 1}`, fed, "proveEquivocation", [addrs[i], f.root, f.sig]));
    binding[i] = f.hq;
  }
  for (const [i, a] of addrs.entries()) {
    const st = await fed.finalStatus(a);
    if (st.left || binding[i]) continue;
    await step(`requestResend p${i + 1}`, () => client.call(`requestResend p${i + 1}`, fed, "requestResend", [a]));
    const direct = await step(`wait ${RESEND_WAIT_BLOCKS} blocks for a direct resend from p${i + 1}`, async () => {
      const until = (await client.provider.getBlockNumber()) + RESEND_WAIT_BLOCKS;
      let got = null;
      while (!got && (await client.provider.getBlockNumber()) < until) {
        try { const m = await peers[i].session.recv(2000); if (m.type === TMSG.final) got = decFinal(m.payload); } catch {
          await new Promise((ok) => setTimeout(ok, 500)); // nothing yet, or the session is gone
        }
      }
      return got;
    });
    if (direct) {
      const root = await embeddingRootHex(direct.hq, maxDepth);
      if (root.toLowerCase() === st.gamma.toLowerCase()) { binding[i] = direct.hq; continue; }
    }
    await step(`fileNonReceipt p${i + 1} (mint and approve Dc)`, async () => {
      await client.call("mint Dc", client.contract("MockERC20", p.token), "mint", [client.address, p.Dc]);
      await client.call("approve Dc", client.contract("MockERC20", p.token), "approve", [contract, p.Dc]);
      await client.call(`fileNonReceipt p${i + 1}`, fed, "fileNonReceipt", [a]);
    });
    const relayed = await step(`receive p${i + 1}'s final message through the arbiter`, () =>
      client.until(async () => relayInbox.has(a.toLowerCase()) || (await fed.finalStatus(a)).objection !== 1n, "relay", 900_000)
        .then(() => relayInbox.get(a.toLowerCase())));
    if (relayed) {
      const got = await activeOpenRelayed({ box: relayed, activeSk: keys.sk, contract, chainId, member: a, maxDepth });
      summary.relayed = { member: a, root: got.root, signatureValid: got.signatureValid, rootIsGamma: got.root === st.gamma };
      if (got.signatureValid && got.root.toLowerCase() === st.gamma.toLowerCase()) binding[i] = got.hq;
    }
  }
  await step("wait for the end of reconciliation (window passed, no objection open)", () => client.until(async () =>
    BigInt(await client.provider.getBlockNumber()) > (await fed.reconciliationDeadline()) && (await fed.openObjections()) === 0n, "reconciliation", 900_000));
  await step("finalizeReconciliation", () => client.call("finalizeReconciliation", fed, "finalizeReconciliation"));

  if (args["no-score"]) {
    await step("no scoring (I9): wait for ScoringFailed", () => client.until(async () => Number(await fed.state()) === 10, "ScoringFailed", 3_600_000));
    summary.owed = String(await fed.owed(client.address));
    return;
  }
  // Scoring: retrain the fusion model per figure, prove one figure at a time, submit.
  const mask = Number(await fed.remainingMask());
  const hq = binding.map((m, i) => ((mask >> i) & 1 ? m : null));
  const figs = requiredFigures(mask, addrs.length);
  const scored = await step("retrain the fusion model for each figure", async () =>
    scoreFigures({ fA: tr.fA, fusion: tr.fusion, rows, hq, Itr, idx: ev.idx, figures: figs })); // D139: F_r, F_j, LOO, BASELINE
  const proofsDir = resolve(tmpdir(), `m3-proofs-${fed.target}`); // witness and input files (large); proofs go into the summary
  summary.scoring = [];
  for (const fig of scored.figures) {
    const ci = await circuitInputFor({ fa: scored.fa, fig, rows, hq, sizeI: I.length, maxDepth, seed: ev.s, N: Number(p.N) });
    if (ci.oracleF !== fig.F) throw new Error(`${fig.name}: JS F ${fig.F} != oracle F ${ci.oracleF}`);
    const pr = await step(`prove ${fig.name} (own process)`, () => proveInChild(ci.input, proofsDir, fig.name));
    if (fig.kind === KIND.FULL_JOINT) {
      if (ci.cModel.toLowerCase() !== gG.toLowerCase()) throw new Error("D139: F_j's C_model differs from the committed gammaG");
      await step("submitJointScore FULL_J", () => client.call("submitJointScore FULL_J", fed, "submitJointScore", [fig.F, pr.proof8]));
    } else {
      await step(`submitScore ${fig.name}`, () => client.call(`submitScore ${fig.name}`, fed, "submitScore", [fig.kind, fig.memberIndex, fig.F, ci.cModel, pr.proof8]));
    }
    summary.scoring.push({ figure: fig.name, mask: fig.mask, F: fig.F, oracleF: ci.oracleF, cModel: ci.cModel, retrained: fig.retrained, trainMs: fig.trainMs,
      witnessMs: pr.witnessMs, proveMs: pr.proveMs, verifyMs: pr.verifyMs, peakRssMB: pr.peakRssMB, freeRamMBAtStart: pr.freeRamMBAtStart,
      prover: pr.prover ?? "snarkjs", proverPeakRssMB: pr.proverPeakRssMB ?? null,
      proof: pr.proof8, publicSignals: pr.publicSignals });
  }
  await step("wait for Scored", () => client.until(async () => settlementStarted(Number(await fed.state())), "Scored or Settled"));
  await settlement({ client, fed, step, summary, response: null });
}

// ---- settlement (M4) --------------------------------------------------------------------

const SCORED = 9;
/**
 * Gas limit floor for acknowledge. Whether an acknowledge settles the run depends on the order in which the passives'
 * acknowledgements land, which its estimate cannot know: an early one is estimated at ~78k, the settling one uses ~241k
 * (results/m4/gas.csv). Test M14 checks every normal-path vector stays below this floor (the circuit fixes n = 3).
 */
export const ACK_GAS_FLOOR = 600_000;
const SETTLED = 11;
const PATHS = ["none", "normal", "objection", "backup"];

/**
 * The wait after scoring ends at Scored, or at Settled (an upheld objection settles in the block after Scored);
 * orFailed: also at ScoringFailed (10, a passive's wait).
 */
export const settlementStarted = (state, { orFailed = false } = {}) => state === SCORED || state === SETTLED || (orFailed && state === 10);

/** A passive's response after Scored: its alignment evidence, or null if it left or the run has already settled. */
export const settlementResponse = ({ state, left, alignment }) => (state !== SCORED || left ? null : alignment ?? { matches: true });

/**
 * M4, after Scored. A passive responds: acknowledge() when its gamma_I check passed, else objectAlignment() with A's
 * AlignedSet signature over the hash of the list it received (response: { alignSig, listHash, matches }); A responds
 * nothing (response null). Then every party waits for Settled; if the response window passes first, it calls
 * settleBackup (anyone may). Finally it withdraws what it is owed.
 */
export async function settlement({ client, fed, step, summary, response }) {
  const s = (summary.settlement = {});
  if (response) {
    if (response.matches) {
      s.response = "acknowledge";
      await step("acknowledge", () => client.call("acknowledge", fed, "acknowledge", [], { minGas: ACK_GAS_FLOOR }));
    } else {
      s.response = "objectAlignment";
      await step("objectAlignment (A's AlignedSet signature)", () =>
        client.call("objectAlignment", fed, "objectAlignment", [response.alignSig, response.listHash]));
    }
  }
  const deadline = Number(await fed.settlementDeadline());
  s.settlementDeadline = deadline;
  await step("wait for Settled (settleBackup after the response window)", () => client.until(async () => {
    const st = Number(await fed.state());
    if (st === SETTLED) return true;
    if (st === SCORED && (await client.provider.getBlockNumber()) > deadline) {
      await client.call("settleBackup", fed, "settleBackup").catch(() => {}); // another party may have been first
      return Number(await fed.state()) === SETTLED;
    }
    return false;
  }, "Settled", 3_600_000));
  s.path = PATHS[Number(await fed.settlementPath())];
  s.owed = String(await fed.owed(client.address));
  if (BigInt(s.owed) > 0n) await step("withdraw", () => client.call("withdraw", fed, "withdraw"));
}

// ---- passive ------------------------------------------------------------------------------

export async function passiveM3({ client, fed, session, I, idx, scenario, keys, runDir, args, step, summary, alignment }) {
  const chainId = (await client.provider.getNetwork()).chainId;
  const contract = fed.target;
  const p = await fed.proposal();
  const maxDepth = Number(p.maxDepth);
  const self = client.address;
  const xp = I.map((id) => scenario.passives[idx].rows.get(id));
  assertFiWidth(idx + 1, xp[0].length); // a row of another width would train silently (Dense.forward does not check)
  const noise = args.noise === "off" ? { clip: Infinity, sigma: 0 } : LOCAL_NOISE[idx]; // DATA1 E3: --noise off (Q4)
  summary.noise = args.noise === "off" ? "off" : noise;
  // M4 (D129): commitFinal before the final message leaves (beforeSend), so A cannot hold the message before the root.
  let registered;
  const t = await step("training, commitFinal, then the final message", () => passiveTraining({ session, index: idx + 1, xp, sizeI: I.length,
    chainId, contract, maxDepth, signer: client.wallet, noise, hooks: { withholdFinal: !!args["withhold-final"],
      beforeSend: async ({ root, hq }) => {
        registered = root;
        if (args.equivocate) {
          // I7: register the root of a slightly different matrix than the one signed and sent.
          const other = hq.map((r) => [...r]);
          other[0][0] += 1;
          registered = await embeddingRootHex(other, maxDepth);
        }
        await client.call("commitFinal", fed, "commitFinal", [registered]);
      } } }));
  summary.final = { signedRoot: t.root, registeredRoot: registered, messageBytes: t.finalPayload.length, withheld: !!args["withhold-final"],
    training: t.stats, trainingMs: t.ms };
  // Until the run ends: answer a resend request (directly, unless the path to A is broken: I8),
  // and hand the final message to the arbiter when A files a non-receipt objection.
  let resent = false;
  let relayed = false;
  await step("reconciliation and scoring: wait for the end, answering requests", () => client.until(async () => {
    const st = await fed.finalStatus(self);
    if (st.resendRequestBlock !== 0n && !resent) {
      resent = true;
      if (!args["withhold-final"]) await session.send(TMSG.final, t.finalPayload).catch(() => {});
      summary.resendRequest = { atBlock: Number(st.resendRequestBlock), answeredDirectly: !args["withhold-final"] };
    }
    if (st.objection === 1n && !relayed) {
      relayed = true;
      const arbKey = await fed.arbiterEncKey();
      const port = JSON.parse(readFileSync(resolve(runDir, "arb-endpoint.json"), "utf8")).port;
      const s = await dialRelay({ port, self, sk: keys.sk, peerPk: arbKey, contract, chainId, activeParty: p.arbiter, passive: self, role: "p" });
      const box = seal({ message: t.finalPayload, recipientPk: arbKey, contract, chainId, member: self });
      await s.send(RMSG.box, relayPayload(self, box));
      s.close();
      summary.relay = { handedToArbiterAtBlock: await client.provider.getBlockNumber(), boxBytes: box.length };
    }
    return settlementStarted(Number(await fed.state()), { orFailed: true });
  }, "Scored, Settled or ScoringFailed", 3_600_000));
  const st = Number(await fed.state());
  if (st === SCORED || st === SETTLED) {
    const left = (await fed.finalStatus(self)).left;
    await settlement({ client, fed, step, summary, response: settlementResponse({ state: st, left, alignment }) });
    return;
  }
  summary.owed = String(await fed.owed(self));
  if (Number(await fed.state()) === 10) await step("withdraw", () => client.call("withdraw", fed, "withdraw"));
}

/** A's listener for boxes delivered by the arbiter (member -> box), keyed with the arbiter's registered key. */
export function relayInboxHandler({ fed, keys, contract, chainId, activeParty, inbox }) {
  return async (link) => {
    const arbKey = await fed.arbiterEncKey();
    const arbiter = (await fed.proposal()).arbiter;
    const { session } = await acceptRelay(link, async () =>
      deriveChannelKeys({ selfSk: keys.sk, peerPk: hexToBytes(arbKey), contract, chainId, activeParty, passive: arbiter }), "A");
    const m = await session.recv(600_000);
    const { member, box } = memberOf(m.payload);
    inbox.set(member.toLowerCase(), box);
  };
}

export { memberOf, finalSigner };
