// M2 §5.2: PSI tests P01..P06 (M2 alignment spec §7). A and three passives run the
// real protocol over TCP in this process; keys come from a run on the in-process chain.
import { expect } from "chai";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { keccak256 } from "ethers";
import { network } from "hardhat";

import { runAlignmentOffChain, startedRunWithKeys } from "./helpers/m2.js";
// @ts-ignore -- plain ESM modules
import { Channel, deriveChannelKeys, hexToBytes } from "../offchain/channel.mjs";
// @ts-ignore
import { genScenario, prng } from "../offchain/data.mjs";
// @ts-ignore
import { idBytes, sortIds } from "../offchain/encoding.mjs";

const { ethers } = await network.create();
const signers = await ethers.getSigners();
const A = signers[0];
const P = signers.slice(1, 4);
const ARB = signers[11];

let run: Awaited<ReturnType<typeof startedRunWithKeys>>;
const scenario = genScenario();
const asSet = (xs: bigint[]) => new Set(xs.map(String));
const sameSet = (x: bigint[], y: bigint[]) => x.length === y.length && [...asSet(x)].every((v) => asSet(y).has(v));

/** Records every plaintext sealed by any Channel while `fn` runs (the messages before encryption). */
async function withPlaintexts<T>(fn: () => Promise<T>) {
  const seen: { type: string; plaintext: Uint8Array }[] = [];
  const orig = Channel.prototype.seal;
  Channel.prototype.seal = function (type: string, plaintext: Uint8Array) { seen.push({ type, plaintext }); return orig.call(this, type, plaintext); };
  try { return { result: await fn(), seen }; } finally { Channel.prototype.seal = orig; }
}

describe("M2 PSI (ristretto255, RFC 9380 hash-to-group; 3 passives over TCP)", function () {
  before(async function () {
    run = await startedRunWithKeys(ethers, { A, P, ARB });
  });

  it("P01 random sets incl. empty, full and partial overlaps: the PSI result equals the plaintext intersection", async function () {
    const rng = prng(7n);
    const id = () => 1_000_000_000n + (rng.u64() % 9_000_000_000n);
    const pool = Array.from({ length: 400 }, id);
    const rowsFor = (ids: bigint[]) => new Map(ids.map((x) => [x, { xa: Array(8).fill(0), y: 0 }]));
    const pick = (k: number) => [...pool].sort(() => rng.float() - 0.5).slice(0, k);
    const cases: [string, bigint[], bigint[][]][] = [
      ["A empty", [], [pick(50), pick(50), pick(50)]],
      ["one passive empty", pick(60), [pick(80), [], pick(80)]],
      ["disjoint", pool.slice(0, 100), [pool.slice(100, 200), pool.slice(200, 300), pool.slice(300, 400)]],
      ["identical (full overlap)", pool.slice(0, 120), [pool.slice(0, 120), pool.slice(0, 120), pool.slice(0, 120)]],
      ["partial, random", pick(150), [pick(200), pick(200), pick(200)]],
      ["partial, A a subset of every passive", pool.slice(0, 30), [pool.slice(0, 90), pool.slice(0, 60), pool.slice(0, 40)]],
    ];
    for (const [name, aIds, pIds] of cases) {
      const sc = { A: { ids: aIds, rows: rowsFor(aIds) }, passives: pIds.map((ids) => ({ ids })) };
      const r = await runAlignmentOffChain({ fed: run.fed, run, A, P, scenario: sc });
      const aSet = asSet(aIds);
      pIds.forEach((ids, i) => expect(sameSet(r.pairs[i].intersection, ids.filter((x) => aSet.has(String(x)))), `${name}: pair ${i + 1}`).to.equal(true));
      const truth = sortIds(aIds.filter((x) => pIds.every((ids) => asSet(ids).has(String(x)))));
      expect(r.I.map(String), name).to.deep.equal(truth.map(String));
      r.passives.forEach((p: any) => expect(p.ok, name).to.equal(true));
    }
  });

  it("P02 no raw ID, decimal string, keccak256 or sha256 of an ID appears on the wire (nor in any PSI message)", async function () {
    const dir = mkdtempSync(join(tmpdir(), "m2-wire-"));
    const { result, seen } = await withPlaintexts(() =>
      runAlignmentOffChain({ fed: run.fed, run, A, P, scenario, relay: true, captureDir: dir }));
    expect(result.I.length).to.equal(1000);
    const allIds = [...new Set([...scenario.A.ids, ...scenario.passives.flatMap((p: any) => p.ids)].map(String))].map(BigInt);
    const pat = { 8: new Set<string>(), 10: new Set<string>(), 32: new Set<string>() } as Record<number, Set<string>>;
    for (const x of allIds) {
      const raw = Buffer.from(idBytes(x));
      const dec = Buffer.from(String(x), "utf8");
      pat[8].add(raw.toString("hex"));
      pat[10].add(dec.toString("hex"));
      for (const src of [raw, dec]) {
        pat[32].add(keccak256(src).slice(2));
        pat[32].add(createHash("sha256").update(src).digest("hex"));
      }
    }
    const scan = (bytes: Uint8Array) => {
      const hex = Buffer.from(bytes).toString("hex");
      let hits = 0;
      for (const w of [8, 10, 32]) for (let o = 0; o + w <= bytes.length; o++) if (pat[w].has(hex.slice(2 * o, 2 * (o + w)))) hits++;
      return hits;
    };
    let frames = 0;
    let wireHits = 0;
    for (const f of readdirSync(dir)) {
      for (const line of readFileSync(join(dir, f), "utf8").trim().split("\n")) {
        frames++;
        wireHits += scan(Buffer.from(JSON.parse(line).hex, "hex"));
      }
    }
    // Per passive: hello, a-blinded, p-on-a, p-blinded, I, I-sig (M4: A's AlignedSet signature), and one ack (33 bytes)
    // for each of the 5 sealed messages; nothing is lost here, so nothing is resent.
    expect(frames).to.equal(3 * (6 + 5));
    expect(result.pairs.every((pr: any) => pr.session.stats.retransmitted === 0)).to.equal(true);
    expect(wireHits).to.equal(0);
    const psiMsgs = seen.filter((m) => m.type.startsWith("psi/"));
    expect(psiMsgs.length).to.equal(9);
    expect(psiMsgs.reduce((n, m) => n + scan(m.plaintext), 0)).to.equal(0);
    // The one message whose plaintext does hold IDs is align/I (by design, §5.3): it is
    // encrypted, and the wire scan above found none of them.
    expect(seen.filter((m) => m.type === "align/I").every((m) => scan(m.plaintext) > 0)).to.equal(true);
    // M4: the signature message holds a 65-byte signature over the list's hash, never an ID or an ID's hash
    const sigs = seen.filter((m) => m.type === "align/I-sig");
    expect(sigs.length).to.equal(3);
    expect(sigs.every((m) => m.plaintext.length === 65 && scan(m.plaintext) === 0)).to.equal(true);
  });

  it("P03 outputs: A learns each pairwise intersection; each passive learns only I", async function () {
    const r = await runAlignmentOffChain({ fed: run.fed, run, A, P, scenario });
    for (let i = 0; i < 3; i++) {
      expect(sameSet(r.pairs[i].intersection, scenario.pairwise[i])).to.equal(true);
      expect(r.pairs[i].intersection.length).to.equal([1010, 1005, 1005][i]); // larger than |I|
      const p = r.passives[i];
      expect(p.ok).to.equal(true);
      expect(p.I.map(String)).to.deep.equal(sortIds(scenario.I).map(String));
      // Apart from I, a passive only ever receives blinded points (a·H(x)); what it can count is A's set size.
      expect(p.aSetSize).to.equal(1020);
    }
  });

  it("P04 two runs with the same inputs: the blinded values are unrelated", async function () {
    const blinded = async () => {
      const { seen } = await withPlaintexts(() => runAlignmentOffChain({ fed: run.fed, run, A, P, scenario }));
      const vals = new Set<string>();
      // PSI values only: align/I holds IDs, and align/I-sig is A's deterministic (RFC 6979) signature over the same list
      for (const m of seen.filter((x) => x.type !== "align/I" && x.type !== "align/I-sig")) {
        for (let o = 0; o < m.plaintext.length; o += 32) vals.add(Buffer.from(m.plaintext.subarray(o, o + 32)).toString("hex"));
      }
      return vals;
    };
    const v1 = await blinded();
    const v2 = await blinded();
    expect(v1.size).to.be.greaterThan(10_000);
    expect([...v1].filter((x) => v2.has(x)).length).to.equal(0);
  });

  it("P05 the lists A and each passive send are in neither input order nor sorted order", async function () {
    const orders: { who: string; input: bigint[]; sent: bigint[] }[] = [];
    await runAlignmentOffChain({ fed: run.fed, run, A, P, scenario,
      hooksA: { onOrder: (p: string, order: bigint[]) => orders.push({ who: `A->${p}`, input: scenario.A.ids, sent: order }) },
      hooksP: [0, 1, 2].map((i) => ({ onOrder: (order: bigint[]) => orders.push({ who: `p${i + 1}`, input: scenario.passives[i].ids, sent: order }) })),
    });
    expect(orders.length).to.equal(6);
    for (const o of orders) {
      expect(o.sent.map(String), o.who).to.not.deep.equal(o.input.map(String));
      expect(o.sent.map(String), o.who).to.not.deep.equal(sortIds(o.input).map(String));
      expect(sameSet(o.sent, o.input), o.who).to.equal(true);
    }
  });

  it("P06 the PSI scalars are not the channel keys (nor each other, across pairs and runs)", async function () {
    const scalars: bigint[] = [];
    const collect = () => ({ hooksA: { onScalar: (_: string, a: bigint) => scalars.push(a) },
      hooksP: [0, 1, 2].map(() => ({ onScalar: (b: bigint) => scalars.push(b) })) });
    await runAlignmentOffChain({ fed: run.fed, run, A, P, scenario, ...collect() });
    await runAlignmentOffChain({ fed: run.fed, run, A, P, scenario, ...collect() });
    expect(scalars.length).to.equal(12);
    const asInts = (b: Uint8Array) => [BigInt("0x" + Buffer.from(b).toString("hex")), BigInt("0x" + Buffer.from(b).reverse().toString("hex"))];
    const keyInts = new Set<string>();
    const chainId = run.chainId;
    for (let i = 0; i < 3; i++) {
      const k = deriveChannelKeys({ selfSk: run.keysA.sk, peerPk: hexToBytes((await run.fed.member(P[i].address)).channelKey),
        contract: run.addr, chainId, activeParty: A.address, passive: P[i].address });
      for (const b of [run.keysA.sk, run.keysP[i].sk, k.kAp, k.kpA]) for (const v of asInts(Uint8Array.from(b))) keyInts.add(String(v));
    }
    for (const s of scalars) expect(keyInts.has(String(s))).to.equal(false);
    expect(new Set(scalars.map(String)).size).to.equal(12);
  });
});
