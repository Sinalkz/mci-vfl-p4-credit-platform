// M3 §4: reconciliation tests R01..R06 (M3 training and scoring spec §6).
import { expect } from "chai";
import { network } from "hardhat";

import { State } from "./helpers/federation.js";
import { fieldRoot, reconciliationRun, signFinal } from "./helpers/m3.js";
// @ts-ignore -- plain ESM modules
import { activeOpenRelayed, arbiterRelay, RMSG, seal } from "../offchain/arbiter.mjs";
// @ts-ignore
import { Channel, deriveChannelKeys } from "../offchain/channel.mjs";
// @ts-ignore
import { connect, listen } from "../offchain/link.mjs";
// @ts-ignore
import { Session } from "../offchain/session.mjs";
// @ts-ignore
import { encFinal } from "../offchain/training.mjs";

const { ethers, networkHelpers } = await network.create();
const signers = await ethers.getSigners();
const A = signers[0];
const P = signers.slice(1, 4);
const ARB = signers[11];
const OUT = signers[12];
const S = { A, P, ARB };

const deadlineOf = async (fed: any) => Number(await fed.reconciliationDeadline());
const mineTo = async (block: number) => {
  const now = await ethers.provider.getBlockNumber();
  if (block > now) await networkHelpers.mine(block - now);
};
/** A holds Dc for objections and approves the run. */
async function fundDc(run: any, times = 3) {
  await (await run.token.mint(A.address, BigInt(times) * run.p.Dc)).wait();
  await (await run.token.connect(A).approve(run.addr, BigInt(times) * run.p.Dc)).wait();
}

/** A Session pair over TCP with M2 channel keys from two X25519 key pairs (roles "A" and "p"). */
async function sessionPair(keysA: any, keysP: any, ctx: { contract: string; chainId: bigint; activeParty: string; passive: string }) {
  let got: (l: any) => void;
  const lp = new Promise<any>((r) => { got = r; });
  const server = await listen(0, (l: any) => got(l));
  const pLink = await connect(server.address().port);
  const aLink = await lp;
  const kA = deriveChannelKeys({ selfSk: keysA.sk, peerPk: keysP.pk, ...ctx });
  const kP = deriveChannelKeys({ selfSk: keysP.sk, peerPk: keysA.pk, ...ctx });
  const a = new Session(aLink, new Channel("A", kA));
  const p = new Session(pLink, new Channel("p", kP));
  return { a, p, close: () => { a.close(); p.close(); server.close(); } };
}

describe("M3 reconciliation (equivocation, resend requests, objections, arbiter relay)", function () {
  it("R01 a passive signs root X and registers root Y: proveEquivocation succeeds, Dp is forfeited, gamma becomes X", async function () {
    const Y = fieldRoot("registered Y");
    const { run, finals } = await reconciliationRun(ethers, S, { registered: [null, Y, null] });
    const X = finals[1].root;
    expect((await run.fed.finalStatus(P[1].address)).gamma).to.equal(Y);
    await expect(run.fed.connect(OUT).proveEquivocation(P[1].address, X, finals[1].sig))
      .to.emit(run.fed, "EquivocationProven").withArgs(P[1].address, Y, X, run.p.Dp);
    const st = await run.fed.finalStatus(P[1].address);
    expect(st.gamma).to.equal(X);
    expect(st.equivocated).to.equal(true);
    expect(st.dpForfeited).to.equal(true);
    expect(st.left).to.equal(false); // still scored
    expect(await run.fed.forfeitedToPool()).to.equal(run.p.Dp);
    expect(await run.fed.remainingMask()).to.equal(7n);
  });

  it("R02 proveEquivocation with equal roots, with a forged signature, after the window: revert", async function () {
    const Y = fieldRoot("registered Y");
    const { run, finals } = await reconciliationRun(ethers, S, { registered: [null, Y, null] });
    const fed = run.fed;
    // equal roots: passive 1 registered exactly what it signed
    await expect(fed.proveEquivocation(P[0].address, finals[0].root, finals[0].sig)).to.be.revertedWithCustomError(fed, "NoEquivocation");
    // forged: X signed by someone else (A), or passive 3's signature presented for passive 2
    const forged = await signFinal(A, run.chainId, run.addr, finals[1].root);
    await expect(fed.proveEquivocation(P[1].address, finals[1].root, forged)).to.be.revertedWithCustomError(fed, "SignerMismatch");
    await expect(fed.proveEquivocation(P[1].address, finals[2].root, finals[2].sig)).to.be.revertedWithCustomError(fed, "SignerMismatch");
    // after the window
    await mineTo((await deadlineOf(fed)) + 1);
    await expect(fed.proveEquivocation(P[1].address, finals[1].root, finals[1].sig)).to.be.revertedWithCustomError(fed, "WindowClosed");
  });

  it("R03 fileNonReceipt without a prior request, not from A, after the window: revert; with a request, Dc is escrowed", async function () {
    const { run } = await reconciliationRun(ethers, S);
    const fed = run.fed;
    await fundDc(run);
    await expect(fed.connect(A).fileNonReceipt(P[0].address)).to.be.revertedWithCustomError(fed, "NoResendRequest");
    await expect(fed.connect(P[1]).requestResend(P[0].address)).to.be.revertedWithCustomError(fed, "NotActiveParty");
    await expect(fed.connect(A).requestResend(P[0].address)).to.emit(fed, "ResendRequested");
    await expect(fed.connect(A).requestResend(P[0].address)).to.be.revertedWithCustomError(fed, "AlreadyRequested");
    await expect(fed.connect(ARB).fileNonReceipt(P[0].address)).to.be.revertedWithCustomError(fed, "NotActiveParty");
    const before = await run.token.balanceOf(run.addr);
    await expect(fed.connect(A).fileNonReceipt(P[0].address)).to.emit(fed, "NonReceiptFiled");
    expect(await fed.dcEscrow()).to.equal(run.p.Dc);
    expect((await run.token.balanceOf(run.addr)) - before).to.equal(run.p.Dc);
    expect(await fed.openObjections()).to.equal(1n);
    await expect(fed.connect(A).fileNonReceipt(P[0].address)).to.be.revertedWithCustomError(fed, "AlreadyFiled");
    // after the window (a request made inside it does not extend it)
    await (await fed.connect(A).requestResend(P[1].address)).wait();
    await mineTo((await deadlineOf(fed)) + 1);
    await expect(fed.connect(A).fileNonReceipt(P[1].address)).to.be.revertedWithCustomError(fed, "WindowClosed");
    await expect(fed.connect(A).requestResend(P[2].address)).to.be.revertedWithCustomError(fed, "WindowClosed");
  });

  it("R04 the relay path: sealed box to the arbiter, checked against gamma, resealed to A; certifyDelivery with gamma; another root or a non-arbiter reverts", async function () {
    const { run, finals } = await reconciliationRun(ethers, S);
    const fed = run.fed;
    await fundDc(run);
    for (const i of [0, 1]) {
      await (await fed.connect(A).requestResend(P[i].address)).wait();
      await (await fed.connect(A).fileNonReceipt(P[i].address)).wait();
    }
    const contract = run.addr;
    const chainId = run.chainId;
    const member = P[0].address;
    // 1-2: passive 1 seals its final message to the arbiter's registered key and hands it over.
    const hop1 = await sessionPair(run.keysArb, run.keysP[0], { contract, chainId, activeParty: ARB.address, passive: member });
    const box = seal({ message: encFinal(finals[0].sig, finals[0].hq), recipientPk: await fed.arbiterEncKey(), contract, chainId, member });
    await hop1.p.send(RMSG.box, box);
    const atArb = await hop1.a.recv(5000);
    hop1.close();
    // 3: the arbiter checks the signature and the root against the member's gamma; 4: reseals to A.
    const res = await arbiterRelay({ fed, box: atArb.payload, arbiterSk: run.keysArb.sk, member, chainId });
    expect(res.ok).to.equal(true);
    expect(res.root).to.equal(finals[0].root);
    const hop2 = await sessionPair(run.keysA, run.keysArb, { contract, chainId, activeParty: A.address, passive: ARB.address });
    await hop2.p.send(RMSG.box, res.boxForA);
    const atA = await hop2.a.recv(5000);
    hop2.close();
    const got = await activeOpenRelayed({ box: atA.payload, activeSk: run.keysA.sk, contract, chainId, member, maxDepth: 10 });
    expect(got.signatureValid).to.equal(true);
    expect(got.root).to.equal(finals[0].root);
    expect(got.hq).to.deep.equal(finals[0].hq);
    // A box for the arbiter does not open with A's key, and vice versa.
    const wrongKey = await activeOpenRelayed({ box, activeSk: run.keysA.sk, contract, chainId, member, maxDepth: 10 }).then(() => "opened", () => "rejected");
    expect(wrongKey).to.equal("rejected");
    // On-chain: another root and a non-arbiter revert; the arbiter with gamma closes the objection.
    await expect(fed.connect(ARB).certifyDelivery(member, finals[1].root)).to.be.revertedWithCustomError(fed, "RootMismatch");
    for (const s of [A, P[0], OUT]) await expect(fed.connect(s).certifyDelivery(member, res.root)).to.be.revertedWithCustomError(fed, "NotArbiter");
    await expect(fed.connect(ARB).certifyDelivery(member, res.root)).to.emit(fed, "DeliveryCertified").withArgs(member, res.root);
    expect((await fed.finalStatus(member)).objection).to.equal(2n); // Certified
    expect(await fed.openObjections()).to.equal(1n); // passive 2's is still open
    // The arbiter refuses a box whose root is not the member's gamma (passive 2 sends passive 3's matrix).
    const wrong = seal({ message: encFinal(finals[1].sig, finals[2].hq), recipientPk: await fed.arbiterEncKey(), contract, chainId, member: P[1].address });
    expect((await arbiterRelay({ fed, box: wrong, arbiterSk: run.keysArb.sk, member: P[1].address, chainId })).ok).to.equal(false);
  });

  it("R05 relay timeout: the member is counted as left and its Dp forfeited", async function () {
    const { run } = await reconciliationRun(ethers, S);
    const fed = run.fed;
    await fundDc(run);
    await (await fed.connect(A).requestResend(P[2].address)).wait();
    await (await fed.connect(A).fileNonReceipt(P[2].address)).wait();
    const relayDeadline = Number((await fed.finalStatus(P[2].address)).relayDeadline);
    await mineTo(relayDeadline - 1); // the next transaction lands in block relayDeadline, the window's last
    await expect(fed.resolveRelayTimeout(P[2].address)).to.be.revertedWithCustomError(fed, "WindowOpen");
    await expect(fed.connect(ARB).certifyDelivery(P[2].address, fieldRoot("x"))).to.be.revertedWithCustomError(fed, "RootMismatch"); // still inside
    await networkHelpers.mine(1);
    await expect(fed.connect(ARB).certifyDelivery(P[2].address, (await fed.finalStatus(P[2].address)).gamma)).to.be.revertedWithCustomError(fed, "WindowClosed");
    await expect(fed.connect(OUT).resolveRelayTimeout(P[2].address)).to.emit(fed, "Left").withArgs(P[2].address, run.p.Dp);
    const st = await fed.finalStatus(P[2].address);
    expect(st.left).to.equal(true);
    expect(st.objection).to.equal(3n); // TimedOut
    expect(await fed.forfeitedToPool()).to.equal(run.p.Dp);
    expect(await fed.remainingMask()).to.equal(3n); // passives 1, 2
    expect(await fed.openObjections()).to.equal(0n);
    await expect(fed.resolveRelayTimeout(P[2].address)).to.be.revertedWithCustomError(fed, "NoOpenObjection");
  });

  it("R06 finalizeReconciliation while an objection is open, or before the window ends: revert", async function () {
    const { run } = await reconciliationRun(ethers, S);
    const fed = run.fed;
    await fundDc(run);
    await expect(fed.finalizeReconciliation()).to.be.revertedWithCustomError(fed, "WindowOpen");
    await (await fed.connect(A).requestResend(P[0].address)).wait();
    await (await fed.connect(A).fileNonReceipt(P[0].address)).wait();
    await mineTo((await deadlineOf(fed)) + 1);
    await expect(fed.finalizeReconciliation()).to.be.revertedWithCustomError(fed, "ObjectionsOpen").withArgs(1n);
    await (await fed.connect(ARB).certifyDelivery(P[0].address, (await fed.finalStatus(P[0].address)).gamma)).wait();
    await expect(fed.connect(OUT).finalizeReconciliation()).to.emit(fed, "ReconciliationFinalized");
    expect(await fed.state()).to.equal(BigInt(State.Scoring));
    expect(await fed.requiredFigures()).to.equal(6n); // D139: F_r, F_j, LOO_1..3, BASELINE
    expect(await fed.scoringDeadline()).to.equal(BigInt(await ethers.provider.getBlockNumber()) + BigInt(run.p.scoringPeriod));
  });

  it("R07 M4 (D130): certifyDelivery credits the arbiter D_c; the escrow is back to 0; A gets nothing back", async function () {
    const { run } = await reconciliationRun(ethers, S);
    const fed = run.fed;
    await fundDc(run);
    await (await fed.connect(A).requestResend(P[0].address)).wait();
    await (await fed.connect(A).fileNonReceipt(P[0].address)).wait();
    await expect(fed.connect(ARB).certifyDelivery(P[0].address, (await fed.finalStatus(P[0].address)).gamma))
      .to.emit(fed, "Credited").withArgs(ARB.address, run.p.Dc);
    expect(await fed.owed(ARB.address)).to.equal(run.p.Dc);
    expect(await fed.dcEscrow()).to.equal(0n);
    expect(await fed.owed(A.address)).to.equal(0n);
  });

  it("R08 M4 (D130, Q7 a): a relay timeout credits the arbiter D_c too; the escrow is back to 0", async function () {
    const { run } = await reconciliationRun(ethers, S);
    const fed = run.fed;
    await fundDc(run);
    await (await fed.connect(A).requestResend(P[1].address)).wait();
    await (await fed.connect(A).fileNonReceipt(P[1].address)).wait();
    await mineTo(Number((await fed.finalStatus(P[1].address)).relayDeadline));
    await expect(fed.connect(OUT).resolveRelayTimeout(P[1].address)).to.emit(fed, "Credited").withArgs(ARB.address, run.p.Dc);
    expect(await fed.owed(ARB.address)).to.equal(run.p.Dc);
    expect(await fed.dcEscrow()).to.equal(0n);
    expect(await fed.owed(A.address)).to.equal(0n);
  });
});
