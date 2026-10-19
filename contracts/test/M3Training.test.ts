// M3 §3: training tests T01..T05 (claude-code-M3-training-scoring.md §6).
// T01..T03 run the whole off-chain protocol in this process over TCP (PSI, then training on the
// same sessions); T04, T05 need only the contract.
import { expect } from "chai";
import { keccak256, recoverAddress, toBeHex, toUtf8Bytes, TypedDataEncoder } from "ethers";
import { network } from "hardhat";

import { BN254_R, State } from "./helpers/federation.js";
import { startedRunWithKeys } from "./helpers/m2.js";
import { fieldRoot, quickTraining as qt, signFinal, trainAll, trainingRun } from "./helpers/m3.js";
// @ts-ignore -- plain ESM modules
import { genScenario } from "../offchain/data.mjs";
// @ts-ignore
import { FINAL_MARKER, FINAL_TYPES, finalDomain, waitForFinalCommits } from "../offchain/training.mjs";
// @ts-ignore
import { CONFIG } from "../offchain/model.mjs";

const { ethers, networkHelpers } = await network.create();
const signers = await ethers.getSigners();
const A = signers[0];
const P = signers.slice(1, 4);
const ARB = signers[11];
const OUT = signers[12];
const scenario = genScenario();
const S = { A, P, ARB };

const quickTraining = () => qt(ethers, S);
const root = fieldRoot;
const sign = signFinal;

describe("M3 training (split model over the M2 sessions; commitFinal, commitFA)", function () {
  it("T01 a full training run of T rounds completes; each passive's gamma_i equals the root A recomputes from h_i", async function () {
    const ctx = await trainingRun(ethers, S, scenario);
    try {
      const t = await trainAll(ctx, P);
      expect(t.active.rounds.length).to.equal(CONFIG.T);
      expect(t.active.zeroFilled).to.deep.equal([]);
      expect(t.active.rounds.at(-1).loss).to.be.lessThan(t.active.rounds[0].loss);
      for (let i = 0; i < 3; i++) {
        await (await ctx.run.fed.connect(P[i]).commitFinal(t.passives[i].root)).wait();
        const f = t.active.finals[i];
        expect(f.received).to.equal(true);
        expect(f.signatureValid).to.equal(true);
        expect(f.root).to.equal(t.passives[i].root);
        expect((await ctx.run.fed.finalStatus(P[i].address)).gamma).to.equal(f.root);
      }
    } finally { ctx.close(); }
  });

  it("T02 the relay drops one passive's embedding for one batch: A zero-fills that batch and training completes", async function () {
    // Passive 2's 5th embedding (round 1, batch 4) is dropped; its retransmission arrives after
    // the 3 s timeout, later than the 2 s per-batch wait.
    const ctx = await trainingRun(ethers, S, scenario, { relayPolicies: [{}, { p2a: [{ op: "dropType", type: "train/emb", nth: 4 }] }, {}] });
    try {
      const t = await trainAll(ctx, P, { T: 2 });
      const z = t.active.zeroFilled;
      expect(z.some((x: any) => x.passive === 2 && x.round === 1 && x.batch === 4)).to.equal(true);
      expect(z.every((x: any) => x.passive === 2)).to.equal(true); // the others were never zero-filled
      expect(t.active.rounds.length).to.equal(2);
      expect(t.passives[1].stats.missedGrads).to.be.greaterThan(0); // no gradient for the zero-filled batch
      expect(t.active.finals.every((f: any) => f.received && f.signatureValid)).to.equal(true);
    } finally { ctx.close(); }
  });

  it("T03 a passive stops after round 1 and never calls commitFinal: counted as left at commitFA, Dp in forfeitedToPool", async function () {
    const ctx = await trainingRun(ethers, S, scenario);
    try {
      const t = await trainAll(ctx, P, { T: 3, hooksP: [{}, {}, { stopAfterRound: 1 }] });
      expect(t.passives[2].stopped).to.equal(true);
      expect(t.active.finals[2].received).to.equal(false);
      expect(t.active.zeroFilled.filter((x: any) => x.passive === 3).length).to.equal(2 * t.active.rounds[1].batches);
      for (const i of [0, 1]) await (await ctx.run.fed.connect(P[i]).commitFinal(t.passives[i].root)).wait();
      const fed = ctx.run.fed;
      await expect(fed.connect(A).commitFA(root("C_fA"), root("gamma_g"))).to.emit(fed, "Left").withArgs(P[2].address, ctx.run.p.Dp);
      expect(await fed.state()).to.equal(BigInt(State.Reconciliation));
      expect((await fed.finalStatus(P[2].address)).left).to.equal(true);
      expect((await fed.finalStatus(P[0].address)).left).to.equal(false);
      expect(await fed.forfeitedToPool()).to.equal(ctx.run.p.Dp);
      expect(await fed.leftCount()).to.equal(1n);
      expect(await fed.trainingEndBlock()).to.equal(BigInt(await ethers.provider.getBlockNumber()));
    } finally { ctx.close(); }
  });

  it("T04 the EIP-712 signature verifies on-chain; it fails under another contract instance, another chainId, another marker", async function () {
    const run = await quickTraining();
    const other = await quickTraining();
    const chainId = run.chainId;
    const r = root("h_1");
    const sig = await sign(P[0], chainId, run.addr, r);
    // On-chain digest of this run recovers the signer.
    expect(recoverAddress(await run.fed.finalMessageHash(P[0].address, r), sig)).to.equal(P[0].address);
    expect(await run.fed.finalMessageHash(P[0].address, r))
      .to.equal(TypedDataEncoder.hash(finalDomain(chainId, run.addr), FINAL_TYPES, { sender: P[0].address, root: r, marker: FINAL_MARKER }));
    // Another instance: its digest does not recover the signer.
    expect(recoverAddress(await other.fed.finalMessageHash(P[0].address, r), sig)).to.not.equal(P[0].address);
    // Another chainId, another marker: signatures over those do not recover to the signer under this run's digest.
    const sigChain = await sign(P[0], 999n, run.addr, r);
    const sigMarker = await sign(P[0], chainId, run.addr, r, "round-19");
    for (const s of [sigChain, sigMarker]) expect(recoverAddress(await run.fed.finalMessageHash(P[0].address, r), s)).to.not.equal(P[0].address);
    // And proveEquivocation rejects all three foreign signatures (gamma registered differs from r).
    await (await run.fed.connect(P[0]).commitFinal(root("registered"))).wait();
    await (await other.fed.connect(P[0]).commitFinal(root("registered"))).wait();
    await (await run.fed.connect(A).commitFA(root("C_fA"), root("gamma_g"))).wait();
    const sigOther = await sign(P[0], chainId, other.addr, r);
    for (const s of [sigOther, sigChain, sigMarker]) {
      await expect(run.fed.proveEquivocation(P[0].address, r, s)).to.be.revertedWithCustomError(run.fed, "SignerMismatch");
    }
    await expect(run.fed.proveEquivocation(P[0].address, r, sig)).to.emit(run.fed, "EquivocationProven");
  });

  it("T05 commitFinal after commitFA, commitFA by a non-A account, commitFA twice: revert; commitFinal twice, by A, or a zero / non-field root: revert", async function () {
    const run = await quickTraining();
    const fed = run.fed;
    await expect(fed.connect(A).commitFinal(root("x"))).to.be.revertedWithCustomError(fed, "NotPassive");
    await expect(fed.connect(P[0]).commitFinal(ethers.ZeroHash)).to.be.revertedWithCustomError(fed, "ZeroCommitment");
    await expect(fed.connect(P[0]).commitFinal(toBeHex(BN254_R, 32))).to.be.revertedWithCustomError(fed, "NotAFieldElement");
    await (await fed.connect(P[0]).commitFinal(root("p1"))).wait();
    await expect(fed.connect(P[0]).commitFinal(root("p1b"))).to.be.revertedWithCustomError(fed, "AlreadyCommitted");
    for (const s of [P[1], ARB, OUT]) await expect(fed.connect(s).commitFA(root("C_fA"), root("gamma_g"))).to.be.revertedWithCustomError(fed, "NotActiveParty");
    await (await fed.connect(A).commitFA(root("C_fA"), root("gamma_g"))).wait();
    await expect(fed.connect(A).commitFA(root("C_fA"), root("gamma_g"))).to.be.revertedWithCustomError(fed, "WrongState");
    await expect(fed.connect(P[1]).commitFinal(root("p2"))).to.be.revertedWithCustomError(fed, "WrongState");
    // passives 2 and 3 had not committed: both left
    expect(await fed.leftCount()).to.equal(2n);
    expect(await fed.forfeitedToPool()).to.equal(2n * run.p.Dp);
    void networkHelpers;
  });

  // ---- M4 (D129): a passive registers gamma_i before its final message; A waits finalWait blocks ----------

  it("T06 M4 (D129): with commitFinal before the final message, every gamma is on-chain when A holds all final messages; commitFA at once leaves no one out", async function () {
    this.timeout(1_800_000);
    const ctx = await trainingRun(ethers, S, scenario);
    try {
      const t = await trainAll(ctx, P, { commitFirst: true });
      const fed = ctx.run.fed;
      for (let i = 0; i < 3; i++) {
        expect(t.active.finals[i].received).to.equal(true);
        expect((await fed.finalStatus(P[i].address)).finalCommitted, `p${i + 1}`).to.equal(true);
        expect((await fed.finalStatus(P[i].address)).gamma).to.equal(t.passives[i].root);
      }
      const rc = await (await fed.connect(A).commitFA(fieldRoot("C_fA"), fieldRoot("gamma_g"))).wait();
      const ev = rc.logs.map((l: any) => fed.interface.parseLog(l)).find((e: any) => e?.name === "TrainingEnded");
      expect(ev.args.left).to.equal(0n);
    } finally { ctx.close(); }
  });

  it("T07 M4 (D129): waitForFinalCommits returns at once when every passive committed, else after the proposal's finalWait blocks", async function () {
    const run = await quickTraining();
    const fed = run.fed;
    const addrs = P.map((s) => s.address);
    const finalWait = Number((await fed.proposal()).finalWait);
    expect(finalWait).to.equal(15);
    for (let i = 0; i < 2; i++) await (await fed.connect(P[i]).commitFinal(root(`r${i}`))).wait();
    const start = await ethers.provider.getBlockNumber();
    const waiting = waitForFinalCommits({ provider: ethers.provider, fed, addrs, finalWait, pollMs: 10 });
    for (let k = 0; k < finalWait; k++) await networkHelpers.mine(1);
    const r = await waiting;
    expect(r.allCommitted).to.equal(false);
    expect(r.atBlock).to.be.at.least(start + finalWait);
    await (await fed.connect(P[2]).commitFinal(root("r2"))).wait();
    const r2 = await waitForFinalCommits({ provider: ethers.provider, fed, addrs, finalWait, pollMs: 10 });
    expect(r2.allCommitted).to.equal(true);
  });
});
