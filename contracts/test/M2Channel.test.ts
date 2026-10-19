// M2 §5.1: secure channel tests C01..C06 (claude-code-M2-alignment.md §7), and C07..C08
// for the retransmission layer (offchain/session.mjs, answer to QUESTIONS-M2 Q2).
// The channel keys are read from the contract; messages pass through the relay process
// (offchain/relay.mjs), which drops, modifies, replays, reorders or substitutes them.
// C01..C06 test the Channel itself (seal/open); C07..C08 test a Session on each side.
import { expect } from "chai";
import { randomBytes } from "node:crypto";
import { network } from "hardhat";

import { startedRunWithKeys, startRelay } from "./helpers/m2.js";
// @ts-ignore -- plain ESM modules
import { Channel, ChannelError, bytesToHex, channelKeyPair, clearBody, deriveChannelKeys, hexToBytes, parseBody } from "../offchain/channel.mjs";
// @ts-ignore
import { connect, listen } from "../offchain/link.mjs";
// @ts-ignore
import { Session } from "../offchain/session.mjs";

const { ethers } = await network.create();
const signers = await ethers.getSigners();
const A = signers[0];
const P = signers.slice(1, 4);
const ARB = signers[11];

let run: Awaited<ReturnType<typeof startedRunWithKeys>>;
const procs: any[] = [];

/** Channel keys exactly as each side derives them: its own secret key, the peer's key from the chain. */
async function keysFromChain(side: "A" | "p", i: number, peerPkOverride?: Uint8Array) {
  const peerPk = peerPkOverride ?? hexToBytes(side === "A"
    ? (await run.fed.member(P[i].address)).channelKey
    : (await run.fed.proposal()).activeChannelKey);
  return deriveChannelKeys({
    selfSk: side === "A" ? run.keysA.sk : run.keysP[i].sk, peerPk,
    contract: run.addr, chainId: run.chainId, activeParty: A.address, passive: P[i].address,
  });
}

/**
 * A listens; passive i connects through a relay with `policy`. The passive sends a clear
 * hello offering a public key (ignored by A), then both sides build their channels.
 */
async function pair(i: number, policy: object = {}, opts: { passivePeerPk?: Uint8Array } = {}) {
  let resolveA: (l: any) => void;
  const aLinkP = new Promise<any>((r) => { resolveA = r; });
  const server = await listen(0, (l: any) => resolveA(l));
  const relay = await startRelay(server.address().port, policy);
  procs.push(relay.proc);
  const pLink = await connect(relay.port);
  await pLink.send(clearBody("hello", new TextEncoder().encode(JSON.stringify({ address: P[i].address, pk: bytesToHex(run.keysP[i].pk) }))));
  const aLink = await aLinkP;
  const hello = parseBody(await aLink.recv());
  const helloMsg = JSON.parse(new TextDecoder().decode(hello.rest));
  const aCh = new Channel("A", await keysFromChain("A", P.findIndex((s: any) => s.address === helloMsg.address)));
  const pCh = new Channel("p", await keysFromChain("p", i, opts.passivePeerPk));
  return { aLink, pLink, aCh, pCh, helloMsg, close: () => { pLink.destroy(); aLink.destroy(); server.close(); relay.proc.kill(); } };
}

/** A tries to open the next frame; returns "ok" or the ChannelError code. */
async function aOpens(pr: any) {
  try { pr.aCh.open(await pr.aLink.recv(5000)); return "ok"; } catch (e: any) {
    if (e instanceof ChannelError) return e.code;
    throw e;
  }
}

describe("M2 secure channel (A <-> passive, keys from the chain, through the relay process)", function () {
  before(async function () {
    run = await startedRunWithKeys(ethers, { A, P, ARB });
  });
  after(function () { for (const p of procs) p.kill(); });

  it("C01 round trip in both directions, 100 messages each: all decrypt, in order", async function () {
    const pr = await pair(0);
    const sent: string[] = [];
    for (let k = 0; k < 100; k++) {
      const m = randomBytes(1 + (k * 37) % 2000);
      sent.push(m.toString("hex"));
      await pr.pLink.send(pr.pCh.seal("c01/p2a", m));
      await pr.aLink.send(pr.aCh.seal("c01/a2p", m));
    }
    for (let k = 0; k < 100; k++) {
      const a = pr.aCh.open(await pr.aLink.recv());
      const p = pr.pCh.open(await pr.pLink.recv());
      expect(Buffer.from(a.payload).toString("hex")).to.equal(sent[k]);
      expect(Buffer.from(p.payload).toString("hex")).to.equal(sent[k]);
      expect(a.type).to.equal("c01/p2a");
    }
    expect(pr.aCh.stats.rejected + pr.pCh.stats.rejected).to.equal(0);
    pr.close();
  });

  it("C02 the relay offers its own public key in the hello: no effect; a side forced to use it fails the first tag check", async function () {
    const relayKeys = channelKeyPair();
    const pr = await pair(1, { p2a: [{ op: "substituteHelloPk", at: 0, pk: bytesToHex(relayKeys.pk) }] });
    expect(pr.helloMsg.pk).to.equal(bytesToHex(relayKeys.pk)); // A saw the relay's key...
    await pr.pLink.send(pr.pCh.seal("c02", new Uint8Array([1, 2, 3])));
    expect(await aOpens(pr)).to.equal("ok");                   // ...and ignored it: keys come from the chain
    pr.close();
    // The passive is forced to derive with the relay's key instead of A's on-chain key.
    const forced = await pair(1, {}, { passivePeerPk: relayKeys.pk });
    await forced.pLink.send(forced.pCh.seal("c02", new Uint8Array([1, 2, 3])));
    expect(await aOpens(forced)).to.equal("tag");
    forced.close();
  });

  it("C03 the relay flips one bit in a ciphertext: rejected", async function () {
    // p2a frame 0 is the hello; frame 1 is the first sealed message.
    const pr = await pair(0, { p2a: [{ op: "flip", at: 1, byte: -1, bit: 0 }] });
    await pr.pLink.send(pr.pCh.seal("c03", randomBytes(64)));
    expect(await aOpens(pr)).to.equal("tag");
    pr.close();
  });

  it("C04 replay, reorder and drop are rejected; a message is accepted only when its counter is next", async function () {
    const send4 = async (pr: any) => { for (let k = 0; k < 4; k++) await pr.pLink.send(pr.pCh.seal("c04", Uint8Array.of(k))); };
    // Replay: after m2 (frame 3) the relay sends m1 (frame 2) again.  A sees m0 m1 m2 m1' m3.
    const rp = await pair(0, { p2a: [{ op: "replay", at: 3, of: 2 }] });
    await send4(rp);
    expect([await aOpens(rp), await aOpens(rp), await aOpens(rp), await aOpens(rp), await aOpens(rp)])
      .to.deep.equal(["ok", "ok", "ok", "counter", "ok"]);
    rp.close();
    // Reorder: m1 and m2 swapped.  A sees m0 m2 m1 m3: m2 skips a counter; m1 is then next; m3 skips m2.
    const ro = await pair(0, { p2a: [{ op: "swap", at: 2 }] });
    await send4(ro);
    expect([await aOpens(ro), await aOpens(ro), await aOpens(ro), await aOpens(ro)])
      .to.deep.equal(["ok", "counter", "ok", "counter"]);
    ro.close();
    // Drop: m1 never arrives.  A sees m0 m2 m3: both later messages skip counter 1.
    const dr = await pair(0, { p2a: [{ op: "drop", at: 2 }] });
    await send4(dr);
    expect([await aOpens(dr), await aOpens(dr), await aOpens(dr)]).to.deep.equal(["ok", "counter", "counter"]);
    dr.close();
  });

  it("C05 the same members and keys under two different contract instances: different channel keys", async function () {
    const other = (await ethers.deployContract("MockERC20")).target as string; // any second contract address
    const base = { selfSk: run.keysP[0].sk, peerPk: run.keysA.pk, chainId: run.chainId, activeParty: A.address, passive: P[0].address };
    const k1 = deriveChannelKeys({ ...base, contract: run.addr });
    const k2 = deriveChannelKeys({ ...base, contract: other });
    expect(bytesToHex(k1.kAp)).to.not.equal(bytesToHex(k2.kAp));
    expect(bytesToHex(k1.kpA)).to.not.equal(bytesToHex(k2.kpA));
    // And a second real run with the same members and the same key pairs:
    const run2 = await startedRunWithKeys(ethers, { A, P, ARB });
    const k3 = deriveChannelKeys({ ...base, contract: run2.addr });
    expect(bytesToHex(k3.kAp)).to.not.equal(bytesToHex(k1.kAp));
  });

  it("C06 the two directions use different keys, and both sides derive the same pair", async function () {
    for (let i = 0; i < 3; i++) {
      const ka = await keysFromChain("A", i);
      const kp = await keysFromChain("p", i);
      expect(bytesToHex(ka.kAp)).to.not.equal(bytesToHex(ka.kpA));
      expect(bytesToHex(ka.kAp)).to.equal(bytesToHex(kp.kAp));
      expect(bytesToHex(ka.kpA)).to.equal(bytesToHex(kp.kpA));
    }
  });

  // Retransmission timing for C07/C08: first resend after 150 ms, doubling. On loopback an ack
  // takes about a millisecond, so each lost frame costs exactly one resend round.
  const FAST = { rtoMs: 150, maxRtoMs: 2_000, maxRetries: 5 };
  const sessions = (pr: any, opts: object = FAST) => ({ a: new Session(pr.aLink, pr.aCh, opts), p: new Session(pr.pLink, pr.pCh, opts) });
  const msgs = (k: number) => Array.from({ length: k }, (_, j) => Uint8Array.from(randomBytes(40 + j)));
  const hex = (b: Uint8Array) => Buffer.from(b).toString("hex");
  /** p sends all messages at once (counters 0..k-1); A receives k; returns what A received. */
  async function exchange(s: any, sent: Uint8Array[]) {
    const acked = Promise.all(sent.map((m) => s.p.send("c07", m)));
    const got: string[] = [];
    for (let k = 0; k < sent.length; k++) got.push(hex((await s.a.recv(5000)).payload));
    await acked; // every message acknowledged
    return got;
  }
  /** True if A has nothing more to deliver within ms (an extra delivery would be a replay accepted). */
  async function nothingMore(s: any, ms = 400) {
    try { await s.a.recv(ms); return false; } catch (e: any) { return e.code === "timeout"; }
  }

  it("C07 a dropped frame is recovered by retransmission; so is a dropped ack; the sender gives up after its retries", async function () {
    // (a) The relay drops m1 (p2a frame 2). A drops m2 and m3 (counter ahead), acks m0; p resends
    // m1..m3 byte for byte after its timeout; A accepts them in order.
    const pr = await pair(0, { p2a: [{ op: "drop", at: 2 }] });
    const s = sessions(pr);
    const sent = msgs(4);
    expect(await exchange(s, sent)).to.deep.equal(sent.map(hex));
    expect(s.a.stats.delivered).to.equal(4);
    expect(s.a.stats.droppedFuture).to.equal(2);
    expect(s.p.stats.retransmitted).to.equal(3);
    expect(await nothingMore(s)).to.equal(true);
    pr.close();

    // (b) The relay drops A's ack of m0 (a2p frame 0). p resends m0; A sees a duplicate of an
    // accepted counter, acks it again and does not deliver it again.
    const pa = await pair(0, { a2p: [{ op: "drop", at: 0 }] });
    const sa = sessions(pa);
    const one = msgs(1);
    expect(await exchange(sa, one)).to.deep.equal(one.map(hex));
    expect(sa.p.stats.retransmitted).to.equal(1);
    expect(sa.a.stats.duplicates).to.equal(1);
    expect(sa.a.stats.acksSent).to.equal(2);
    expect(await nothingMore(sa)).to.equal(true);
    pa.close();

    // (c) The relay drops everything p sends after the hello. p resends maxRetries = 3 times
    // (after 150, 300 and 600 ms), waits 1,200 ms more and fails with "retries"; A delivers nothing.
    const pg = await pair(0, { p2a: [{ op: "dropFrom", at: 1 }] });
    const sg = sessions(pg, { ...FAST, maxRetries: 3 });
    const t0 = performance.now();
    const err = await sg.p.send("c07", Uint8Array.of(1)).then(() => null, (e: any) => e);
    expect(err).to.be.instanceOf(ChannelError);
    expect(err.code).to.equal("retries");
    expect(sg.p.stats.retransmitted).to.equal(3);
    expect(performance.now() - t0).to.be.at.least(150 + 300 + 600 + 1200 - 50);
    expect(sg.a.stats.delivered).to.equal(0);
    // Once failed, the session refuses further sends with the same error.
    expect((await sg.p.send("c07", Uint8Array.of(2)).then(() => null, (e: any) => e)).code).to.equal("retries");
    pg.close();
  });

  it("C08 a tampered frame is recovered by retransmission; a tampered ack too; a replayed old frame is still rejected", async function () {
    // (a) The relay flips a bit in m0's tag (p2a frame 1). A drops it silently (no error, no
    // ack) and drops m1, m2 (counter ahead); p resends m0..m2; A delivers the original m0..m2.
    const pr = await pair(0, { p2a: [{ op: "flip", at: 1, byte: -1, bit: 0 }] });
    const s = sessions(pr);
    const sent = msgs(3);
    expect(await exchange(s, sent)).to.deep.equal(sent.map(hex));
    expect(s.a.stats.droppedTag).to.equal(1);
    expect(s.a.stats.droppedFuture).to.equal(2);
    expect(s.p.stats.retransmitted).to.equal(3);
    pr.close();

    // (b) The relay flips a bit in A's ack of m0 (a2p frame 0). p drops the ack, resends m0,
    // and A acknowledges the duplicate again.
    const pa = await pair(0, { a2p: [{ op: "flip", at: 0, byte: -1, bit: 3 }] });
    const sa = sessions(pa);
    const one = msgs(1);
    expect(await exchange(sa, one)).to.deep.equal(one.map(hex));
    expect(sa.p.stats.droppedAck).to.equal(1);
    expect(sa.p.stats.retransmitted).to.equal(1);
    expect(sa.a.stats.duplicates).to.equal(1);
    pa.close();

    // (c) After forwarding m2 (p2a frame 3), the relay sends m0 (frame 1) again. A sees
    // m0 m1 m2 m0' m3: m0' is a valid frame of an already-accepted counter, so it is acked
    // and dropped; A delivers m0..m3 exactly once, and nothing afterwards.
    const rp = await pair(0, { p2a: [{ op: "replay", at: 3, of: 1 }] });
    const sr = sessions(rp);
    const four = msgs(4);
    expect(await exchange(sr, four)).to.deep.equal(four.map(hex));
    expect(sr.a.stats.delivered).to.equal(4);
    expect(sr.a.stats.duplicates).to.equal(1);
    expect(sr.p.stats.retransmitted).to.equal(0);
    expect(await nothingMore(sr)).to.equal(true);
    // The strict Channel rule underneath is unchanged: the same old frame offered to open()
    // (which the session no longer calls) is rejected with "counter".
    const old = sr.p.ch.seal("c08", Uint8Array.of(9)); // counter 4, accepted below
    await sr.p.link.send(old);
    expect(hex((await sr.a.recv(5000)).payload)).to.equal("09");
    expect(() => sr.a.ch.open(old)).to.throw(ChannelError).with.property("code", "counter");
    rp.close();
  });
});
