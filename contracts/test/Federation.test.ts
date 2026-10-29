// M1/M2: federation contract, setup, alignment and seed reveal. Test IDs U01..U24 are
// the ones in the M1 setup spec, section 9, U25..U27 (and the extended U24) those in
// the M2 alignment spec, section 7; the thesis cites them.
import { expect } from "chai";
import { TypedDataEncoder } from "ethers";
// @ts-ignore -- plain ESM module
import { FINAL_MARKER, FINAL_TYPES, finalDomain } from "../offchain/training.mjs";
// @ts-ignore -- plain ESM module
import { emptyRoots } from "../../circuits/scripts/empty-roots.mjs";
import { ZeroAddress, ZeroHash, keccak256, toBeHex, toUtf8Bytes } from "ethers";
import { network } from "hardhat";

import {
  BN254_R, State, defaultProposal, randomKey, randomR, seedCommitment, seedFromReveals, type Proposal,
} from "./helpers/federation.js";
// Section 7: the S2b index derivation is reused, not reimplemented.
// @ts-ignore -- plain ESM module from the circuits package
import { sampleIndices } from "../../circuits/scripts/members-lib.mjs";

const { ethers, networkHelpers } = await network.create();
const signers = await ethers.getSigners();
const A = signers[0];                 // active party (power company)
const P = signers.slice(1, 11);       // up to 10 passives, member index 1..10 in this order
const ARB = signers[11];              // arbiter
const OUT = signers[12];              // an account with no role
const EXTRA = signers[13];            // an 11th passive for the n > 10 case

const SIZE_I = 1000n;
const GAMMA_I = toBeHex(BigInt(keccak256(toUtf8Bytes("gamma_I"))) % BN254_R, 32);
const GAMMA_A = toBeHex(BigInt(keccak256(toUtf8Bytes("gamma_A"))) % BN254_R, 32);

let token: any, verifier: any, factory: any, fedIface: any;

type Run = Awaited<ReturnType<typeof createRun>>;

/** A creates a run through the factory (B approved first); passives get Dp and approve the run. */
async function createRun(n = 3, overrides: Partial<Proposal> = {}) {
  const passives = P.slice(0, n);
  const p = defaultProposal({
    activeParty: A.address, passives: passives.map((s) => s.address), arbiter: ARB.address,
    verifier: await verifier.getAddress(), token: await token.getAddress(),
  }, overrides);
  await token.mint(A.address, p.B);
  await token.connect(A).approve(await factory.getAddress(), p.B);
  const tx = await factory.connect(A).create(p);
  const rc = await tx.wait();
  const ev = rc.logs.map((l: any) => factory.interface.parseLog(l)).find((e: any) => e?.name === "FederationCreated");
  const fed = await ethers.getContractAt("Federation", ev.args.federation);
  const addr = await fed.getAddress();
  for (const s of passives) {
    await token.mint(s.address, p.Dp);
    await token.connect(s).approve(addr, p.Dp);
  }
  return { fed, addr, p, passives, tx, createBlock: rc.blockNumber as number, rA: randomR(), rP: passives.map(() => randomR()) };
}

const register = (run: Run, s: any) => run.fed.connect(s).register(randomKey(), ARB.address);
const accept = (run: Run) => run.fed.connect(ARB).acceptArbiter(randomKey());
const commitA = (run: Run) => run.fed.connect(A).commitSeed(seedCommitment(run.rA, A.address, run.addr));
const commitP = (run: Run, i: number) =>
  run.fed.connect(run.passives[i]).commitSeed(seedCommitment(run.rP[i], run.passives[i].address, run.addr));

/** Every setup condition met; the arbiter's acceptance is the last action. */
async function startRun(n = 3, overrides: Partial<Proposal> = {}) {
  const run = await createRun(n, overrides);
  await commitA(run);
  for (let i = 0; i < n; i++) { await register(run, run.passives[i]); await commitP(run, i); }
  await accept(run);
  return run;
}

async function alignedRun(n = 3, overrides: Partial<Proposal> = {}) {
  const run = await startRun(n, overrides);
  await run.fed.connect(A).commitAlignment(GAMMA_I, SIZE_I, GAMMA_A);
  return run;
}

const blockNow = () => ethers.provider.getBlockNumber();
/** Mine so that the next transaction is included in block `target` (automine). */
async function nextTxAt(target: number | bigint) {
  const cur = await blockNow();
  const gap = Number(target) - 1 - cur;
  if (gap < 0) throw new Error(`already past block ${target}`);
  if (gap > 0) await networkHelpers.mine(gap);
}
const receiptBlock = async (txp: Promise<any>) => (await (await txp).wait()).blockNumber as number;
const revealP = (run: Run, i: number) => run.fed.connect(run.passives[i]).revealSeed(run.rP[i]);

describe("M1 Federation: setup phase and seed reveal", function () {
  before(async function () {
    token = await ethers.deployContract("MockERC20");
    verifier = await ethers.deployContract("MembersVerifierN64I1000");
    factory = await ethers.deployContract("FederationFactory");
    fedIface = await ethers.getContractAt("Federation", ZeroAddress); // error ABI for the matchers
  });

  // ---- Section 4: proposal and factory -----------------------------------------

  it("U01 factory happy path: run holds B, every proposal field readable, state Setup", async function () {
    const run = await createRun(3);
    await expect(run.tx).to.emit(factory, "FederationCreated").withArgs(run.addr, A.address, run.p.B);
    await expect(run.tx).to.emit(run.fed, "PoolHeld").withArgs(run.p.B);
    expect(await token.balanceOf(run.addr)).to.equal(run.p.B);
    expect(await run.fed.poolHeld()).to.equal(true);
    expect(await run.fed.state()).to.equal(State.Setup);
    expect(await run.fed.factory()).to.equal(await factory.getAddress());
    expect(await run.fed.creationBlock()).to.equal(BigInt(run.createBlock));
    expect(await run.fed.setupDeadline()).to.equal(BigInt(run.createBlock + run.p.setupPeriod));
    expect(await run.fed.passiveCount()).to.equal(3n);
    const got: any = await run.fed.proposal();
    for (const [k, v] of Object.entries(run.p)) {
      if (Array.isArray(v)) expect([...got[k]].map(String), k).to.deep.equal(v.map(String));
      else if (typeof v === "string") expect(String(got[k]).toLowerCase(), k).to.equal(v.toLowerCase());
      else expect(BigInt(got[k]), k).to.equal(BigInt(v));
    }
    for (let i = 0; i < 3; i++) expect((await run.fed.member(run.passives[i].address)).index).to.equal(BigInt(i + 1));
  });

  it("U28 M4 (D129): finalWait is stored and returned by proposal(); 0 is accepted, the contract never enforces it", async function () {
    for (const finalWait of [15, 0]) {
      const run = await createRun(3, { finalWait });
      expect((await run.fed.proposal()).finalWait).to.equal(BigInt(finalWait));
    }
  });

  it("U29 M4 (D128): emptyRoot() is the all-zero-leaf Poseidon root at the run's maxDepth, for every maxDepth 1..20", async function () {
    const roots: bigint[] = await emptyRoots();
    for (let d = 1; d <= 20; d++) {
      const run = await createRun(3, { maxDepth: d });
      expect(BigInt(await run.fed.emptyRoot()), `maxDepth ${d}`).to.equal(roots[d - 1]);
    }
  });

  it("U30 M4: the EIP-712 digest of a final-round message equals ethers' TypedDataEncoder (domain VFLFederation / 1 / chain / run)", async function () {
    const run = await createRun(3);
    const chainId = (await ethers.provider.getNetwork()).chainId;
    const root = "0x" + "11".repeat(32);
    const want = TypedDataEncoder.hash(finalDomain(chainId, run.addr), FINAL_TYPES, { sender: P[0].address, root, marker: FINAL_MARKER });
    expect(await run.fed.finalMessageHash(P[0].address, root)).to.equal(want);
  });

  describe("U02 each invalid proposal rule makes create revert", function () {
    const base = () => defaultProposal({
      activeParty: A.address, passives: P.slice(0, 3).map((s) => s.address), arbiter: ARB.address,
      verifier: ZeroAddress, token: ZeroAddress, // replaced below, once deployed
    });
    const cases: [string, (p: Proposal) => Partial<Proposal>, string][] = [
      ["n = 2 (< 3)", (p) => ({ passives: p.passives.slice(0, 2) }), "PassiveCount"],
      ["n = 11 (> 10)", () => ({ passives: [...P.map((s) => s.address), EXTRA.address] }), "PassiveCount"],
      ["a passive is the zero address", (p) => ({ passives: [p.passives[0], ZeroAddress, p.passives[2]] }), "InvalidPassive"],
      ["a passive is listed twice", (p) => ({ passives: [p.passives[0], p.passives[1], p.passives[0]] }), "InvalidPassive"],
      ["a passive is the active party", (p) => ({ passives: [p.passives[0], A.address, p.passives[2]] }), "InvalidPassive"],
      ["a passive is the arbiter", (p) => ({ passives: [p.passives[0], p.passives[1], ARB.address] }), "InvalidPassive"],
      ["arbiter is the zero address", () => ({ arbiter: ZeroAddress }), "InvalidArbiter"],
      ["arbiter is the active party", () => ({ arbiter: A.address }), "InvalidArbiter"],
      ["activeChannelKey is empty", () => ({ activeChannelKey: "0x" }), "BadKeyLength"],
      ["activeChannelKey has 31 bytes", () => ({ activeChannelKey: "0x" + "11".repeat(31) }), "BadKeyLength"],
      ["activeChannelKey has 33 bytes", () => ({ activeChannelKey: "0x" + "11".repeat(33) }), "BadKeyLength"],
      ["modelConfigHash is zero", () => ({ modelConfigHash: ZeroHash }), "ZeroHash"],
      ["vkHash is zero", () => ({ vkHash: ZeroHash }), "ZeroHash"],
      ["maxDepth = 0", () => ({ maxDepth: 0 }), "InvalidMaxDepth"],
      ["maxDepth = 21", () => ({ maxDepth: 21 }), "InvalidMaxDepth"],
      ["metricId = 2", () => ({ metricId: 2 }), "UnsupportedMetric"],
      ["metricId = 0", () => ({ metricId: 0 }), "UnsupportedMetric"],
      ["3 class boundaries", () => ({ classBoundaries: [1n, 2n, 3n] }), "InvalidClassBoundaries"],
      ["5 class boundaries", () => ({ classBoundaries: [1n, 2n, 3n, 4n, 5n] }), "InvalidClassBoundaries"],
      ["class boundaries with a tie", () => ({ classBoundaries: [1n, 2n, 2n, 3n] }), "InvalidClassBoundaries"],
      ["class boundaries decreasing", () => ({ classBoundaries: [4n, 3n, 2n, 1n] }), "InvalidClassBoundaries"],
      ["T = 0", () => ({ T: 0 }), "ZeroRoundsOrDraws"],
      ["N = 0", () => ({ N: 0 }), "ZeroRoundsOrDraws"],
      ["verifier has no code", () => ({ verifier: OUT.address }), "NoCode"],
      ["token has no code", () => ({ token: OUT.address }), "NoCode"],
      ["setupPeriod = 0", () => ({ setupPeriod: 0 }), "ZeroPeriod"],
      ["revealWindow = 0", () => ({ revealWindow: 0 }), "ZeroPeriod"],
      ["deltaR = 0", () => ({ deltaR: 0 }), "ZeroPeriod"],
      ["deltaA = 0", () => ({ deltaA: 0 }), "ZeroPeriod"],
      ["scoringPeriod = 0 (M3)", () => ({ scoringPeriod: 0 }), "ZeroPeriod"],
      ["K = 0", () => ({ K: 0 }), "ZeroK"],
      ["B = 0", () => ({ B: 0n }), "ZeroAmount"],
      ["rho = 0", () => ({ rho: 0n }), "ZeroAmount"],
      ["Dp = 0", () => ({ Dp: 0n }), "ZeroAmount"],
      ["Dc = 0", () => ({ Dc: 0n }), "ZeroAmount"],
    ];
    for (const [name, change, error] of cases) {
      it(`U02 ${name} -> ${error}`, async function () {
        const p0 = { ...base(), verifier: await verifier.getAddress(), token: await token.getAddress() };
        const bad = { ...p0, ...change(p0) };
        await token.mint(A.address, p0.B);
        await token.connect(A).approve(await factory.getAddress(), p0.B);
        await expect(factory.connect(A).create(bad)).to.be.revertedWithCustomError(fedIface, error);
        await expect(factory.connect(A).create(p0)).to.emit(factory, "FederationCreated"); // control: valid
      });
    }
    it("U02 factory: activeParty is not the caller -> NotActiveParty", async function () {
      const p0 = { ...base(), verifier: await verifier.getAddress(), token: await token.getAddress() };
      await expect(factory.connect(P[0]).create(p0)).to.be.revertedWithCustomError(factory, "NotActiveParty");
    });
  });

  it("U03 create reverts when A has not approved B (none, or B - 1)", async function () {
    const p = defaultProposal({
      activeParty: A.address, passives: P.slice(0, 3).map((s) => s.address), arbiter: ARB.address,
      verifier: await verifier.getAddress(), token: await token.getAddress(),
    });
    await token.mint(A.address, p.B);
    await token.connect(A).approve(await factory.getAddress(), 0n);
    await expect(factory.connect(A).create(p)).to.be.revertedWithCustomError(token, "ERC20InsufficientAllowance");
    await token.connect(A).approve(await factory.getAddress(), p.B - 1n);
    await expect(factory.connect(A).create(p)).to.be.revertedWithCustomError(token, "ERC20InsufficientAllowance");
  });

  // ---- Section 6: functions ------------------------------------------------------

  it("U04 ABI: no non-view function beyond section 6 (plus the factory-only activate of section 4); no setter", async function () {
    const nonView = (c: any) => c.interface.fragments
      .filter((f: any) => f.type === "function" && f.stateMutability !== "view" && f.stateMutability !== "pure")
      .map((f: any) => f.name).sort();
    expect(nonView(fedIface)).to.deep.equal([
      "abortSetup", "acceptArbiter", "acknowledge", "activate", "certifyDelivery", "commitAlignment", "commitFA", "commitFinal", "commitSeed",
      "fileNonReceipt", "finalizeReconciliation", "objectAlignment", "proveEquivocation", "register", "requestResend", "resolveAlignmentTimeout",
      "resolveRelayTimeout", "resolveRevealFailure", "resolveScoringTimeout", "revealSeed", "settleBackup", "submitJointScore", "submitScore",
      "withdraw",
    ]); // M4 added acknowledge, objectAlignment, settleBackup // DATA1 D139 added submitJointScore // M3 added commitFinal .. resolveScoringTimeout (M3 training and scoring spec §3.4, §4, §5.2)
    expect(nonView(factory)).to.deep.equal(["create"]);
    const all = fedIface.interface.fragments.filter((f: any) => f.type === "function").map((f: any) => f.name);
    const adminLike = /^(set[A-Z_]|update|change|pause|unpause|upgrade|transferOwnership|renounceOwnership|owner$)/;
    expect(all.filter((n: string) => adminLike.test(n))).to.deep.equal([]);
  });

  it("U05 register rejects: unlisted account, A, wrong arbiterEcho, a key that is not 32 bytes, Dp not approved, a second registration, after the deadline", async function () {
    const run = await createRun(3);
    const [p1, p2, p3] = run.passives;
    await expect(run.fed.connect(OUT).register(randomKey(), ARB.address)).to.be.revertedWithCustomError(run.fed, "NotPassive");
    await expect(run.fed.connect(A).register(randomKey(), ARB.address)).to.be.revertedWithCustomError(run.fed, "NotPassive");
    await expect(run.fed.connect(p1).register(randomKey(), OUT.address)).to.be.revertedWithCustomError(run.fed, "ArbiterMismatch");
    for (const bad of ["0x", "0x" + "11".repeat(31), "0x" + "11".repeat(33)]) {
      await expect(run.fed.connect(p1).register(bad, ARB.address))
        .to.be.revertedWithCustomError(run.fed, "BadKeyLength").withArgs((bad.length - 2) / 2);
    }
    await token.connect(p2).approve(run.addr, 0n);
    await expect(register(run, p2)).to.be.revertedWithCustomError(token, "ERC20InsufficientAllowance");
    await expect(register(run, p1)).to.emit(run.fed, "Registered");
    expect(await token.balanceOf(run.addr)).to.equal(run.p.B + run.p.Dp);
    await expect(register(run, p1)).to.be.revertedWithCustomError(run.fed, "AlreadyRegistered");
    await nextTxAt(run.createBlock + run.p.setupPeriod + 1);
    await expect(register(run, p3)).to.be.revertedWithCustomError(run.fed, "SetupClosed");
  });

  it("U06 every member-only function called from an unlisted account reverts", async function () {
    const run = await createRun(3);
    await expect(run.fed.connect(OUT).register(randomKey(), ARB.address)).to.be.revertedWithCustomError(run.fed, "NotPassive");
    await expect(run.fed.connect(OUT).acceptArbiter(randomKey())).to.be.revertedWithCustomError(run.fed, "NotArbiter");
    await expect(run.fed.connect(OUT).commitSeed(seedCommitment(1n, OUT.address, run.addr))).to.be.revertedWithCustomError(run.fed, "NotMember");
    await expect(run.fed.connect(OUT).activate()).to.be.revertedWithCustomError(run.fed, "NotFactory");
    const started = await startRun(3);
    await expect(started.fed.connect(OUT).commitAlignment(GAMMA_I, SIZE_I, GAMMA_A)).to.be.revertedWithCustomError(started.fed, "NotActiveParty");
    await started.fed.connect(A).commitAlignment(GAMMA_I, SIZE_I, GAMMA_A);
    await expect(started.fed.connect(OUT).revealSeed(1n)).to.be.revertedWithCustomError(started.fed, "NotMember");
    await started.fed.connect(A).revealSeed(started.rA);
    await expect(started.fed.connect(OUT).revealSeed(1n)).to.be.revertedWithCustomError(started.fed, "NotMember");
  });

  it("U07 acceptArbiter rejects: a non-arbiter, a key that is not 32 bytes, a second call, a call after the deadline", async function () {
    const run = await createRun(3);
    await expect(run.fed.connect(P[0]).acceptArbiter(randomKey())).to.be.revertedWithCustomError(run.fed, "NotArbiter");
    await expect(run.fed.connect(A).acceptArbiter(randomKey())).to.be.revertedWithCustomError(run.fed, "NotArbiter");
    for (const bad of ["0x", "0x" + "11".repeat(31), "0x" + "11".repeat(33)]) {
      await expect(run.fed.connect(ARB).acceptArbiter(bad))
        .to.be.revertedWithCustomError(run.fed, "BadKeyLength").withArgs((bad.length - 2) / 2);
    }
    const key = randomKey();
    await expect(run.fed.connect(ARB).acceptArbiter(key)).to.emit(run.fed, "ArbiterAccepted").withArgs(ARB.address, key);
    expect(await run.fed.arbiterEncKey()).to.equal(key);
    await expect(accept(run)).to.be.revertedWithCustomError(run.fed, "AlreadyAccepted");
    const late = await createRun(3);
    await nextTxAt(late.createBlock + late.p.setupPeriod + 1);
    await expect(accept(late)).to.be.revertedWithCustomError(late.fed, "SetupClosed");
  });

  it("U08 commitSeed rejects: an unregistered passive, non-members, a second commit, zero, after the deadline", async function () {
    const run = await createRun(3);
    await expect(commitP(run, 0)).to.be.revertedWithCustomError(run.fed, "NotRegistered");
    await expect(run.fed.connect(OUT).commitSeed(seedCommitment(1n, OUT.address, run.addr))).to.be.revertedWithCustomError(run.fed, "NotMember");
    await expect(run.fed.connect(ARB).commitSeed(seedCommitment(1n, ARB.address, run.addr))).to.be.revertedWithCustomError(run.fed, "NotMember");
    await register(run, run.passives[0]);
    await expect(run.fed.connect(run.passives[0]).commitSeed(ethers.ZeroHash)).to.be.revertedWithCustomError(run.fed, "ZeroCommitment");
    await expect(commitP(run, 0)).to.emit(run.fed, "SeedCommitted");
    await expect(commitP(run, 0)).to.be.revertedWithCustomError(run.fed, "AlreadyCommitted");
    await expect(commitA(run)).to.emit(run.fed, "SeedCommitted");
    await expect(commitA(run)).to.be.revertedWithCustomError(run.fed, "AlreadyCommitted");
    await register(run, run.passives[1]);
    await nextTxAt(run.createBlock + run.p.setupPeriod + 1);
    await expect(commitP(run, 1)).to.be.revertedWithCustomError(run.fed, "SetupClosed");
  });

  describe("U09 the run starts exactly on the last setup action", function () {
    async function expectStartsOn(run: Run, last: () => Promise<any>) {
      expect(await run.fed.state()).to.equal(State.Setup);
      const tx = await last();
      const rc = await tx.wait();
      await expect(tx).to.emit(run.fed, "Started").withArgs(rc.blockNumber);
      expect(await run.fed.state()).to.equal(State.Alignment);
      expect(await run.fed.startBlock()).to.equal(BigInt(rc.blockNumber));
    }

    it("U09 last action is acceptArbiter", async function () {
      const run = await createRun(3);
      await commitA(run);
      for (let i = 0; i < 3; i++) { await register(run, run.passives[i]); await commitP(run, i); }
      await expectStartsOn(run, () => accept(run));
    });

    it("U09 last action is commitSeed (by a passive, and by A)", async function () {
      const run = await createRun(3);
      await accept(run);
      await commitA(run);
      for (let i = 0; i < 3; i++) await register(run, run.passives[i]);
      await commitP(run, 0);
      await commitP(run, 2);
      await expectStartsOn(run, () => commitP(run, 1));

      const run2 = await createRun(3);
      for (let i = 0; i < 3; i++) { await register(run2, run2.passives[i]); await commitP(run2, i); }
      await accept(run2);
      await expectStartsOn(run2, () => commitA(run2));
    });
  });

  it("U10 after the start: register, acceptArbiter, commitSeed and abortSetup revert", async function () {
    const run = await startRun(3);
    await expect(register(run, run.passives[0])).to.be.revertedWithCustomError(run.fed, "WrongState");
    await expect(accept(run)).to.be.revertedWithCustomError(run.fed, "WrongState");
    await expect(commitA(run)).to.be.revertedWithCustomError(run.fed, "WrongState");
    await nextTxAt(run.createBlock + run.p.setupPeriod + 1);
    await expect(run.fed.connect(OUT).abortSetup()).to.be.revertedWithCustomError(run.fed, "WrongState");
  });

  it("U11 deadline passes with one passive unregistered: abortSetup refunds B and Dp; withdraw pays once", async function () {
    const run = await createRun(3);
    const [p1, p2, p3] = run.passives;
    await commitA(run);
    await accept(run);
    await register(run, p1); await commitP(run, 0);
    await register(run, p2);
    await nextTxAt(run.createBlock + run.p.setupPeriod + 1);
    await expect(run.fed.connect(OUT).abortSetup()).to.emit(run.fed, "SetupAborted");
    expect(await run.fed.state()).to.equal(State.SetupAborted);
    expect(await run.fed.owed(A.address)).to.equal(run.p.B);
    expect(await run.fed.owed(p1.address)).to.equal(run.p.Dp);
    expect(await run.fed.owed(p2.address)).to.equal(run.p.Dp);
    expect(await run.fed.owed(p3.address)).to.equal(0n);

    await expect(run.fed.connect(A).withdraw()).to.changeTokenBalances(ethers, token, [A, run.fed], [run.p.B, -run.p.B]);
    await expect(run.fed.connect(A).withdraw()).to.changeTokenBalances(ethers, token, [A, run.fed], [0n, 0n]);
    await expect(run.fed.connect(p1).withdraw()).to.changeTokenBalances(ethers, token, [p1, run.fed], [run.p.Dp, -run.p.Dp]);
    await expect(run.fed.connect(p1).withdraw()).to.changeTokenBalances(ethers, token, [p1], [0n]);
    await expect(run.fed.connect(p3).withdraw()).to.changeTokenBalances(ethers, token, [p3], [0n]);
    await run.fed.connect(p2).withdraw();
    expect(await token.balanceOf(run.addr)).to.equal(0n);
  });

  it("U12 abortSetup reverts before (and at) the deadline, and after the start", async function () {
    const run = await createRun(3);
    await expect(run.fed.connect(OUT).abortSetup()).to.be.revertedWithCustomError(run.fed, "SetupOpen");
    await nextTxAt(run.createBlock + run.p.setupPeriod); // the deadline block itself is still setup
    await expect(run.fed.connect(OUT).abortSetup()).to.be.revertedWithCustomError(run.fed, "SetupOpen");
    const started = await startRun(3);
    await expect(started.fed.connect(OUT).abortSetup()).to.be.revertedWithCustomError(started.fed, "WrongState");
  });

  it("U13 commitAlignment rejects: not A, wrong state, gammaA >= BN254_R, sizeI = 0, a second call", async function () {
    const fresh = await createRun(3);
    await expect(fresh.fed.connect(A).commitAlignment(GAMMA_I, SIZE_I, GAMMA_A)).to.be.revertedWithCustomError(fresh.fed, "WrongState");
    const run = await startRun(3);
    const tooBig = toBeHex(BN254_R, 32);
    await expect(run.fed.connect(P[0]).commitAlignment(GAMMA_I, SIZE_I, GAMMA_A)).to.be.revertedWithCustomError(run.fed, "NotActiveParty");
    await expect(run.fed.connect(A).commitAlignment(GAMMA_I, SIZE_I, tooBig)).to.be.revertedWithCustomError(run.fed, "NotAFieldElement");
    await expect(run.fed.connect(A).commitAlignment(GAMMA_I, 0n, GAMMA_A)).to.be.revertedWithCustomError(run.fed, "ZeroSize");
    const largest = toBeHex(BN254_R - 1n, 32); // the largest field element is accepted
    const tx = run.fed.connect(A).commitAlignment(largest, SIZE_I, GAMMA_A);
    const blk = await receiptBlock(tx);
    await expect(tx).to.emit(run.fed, "AlignmentCommitted").withArgs(largest, SIZE_I, GAMMA_A, blk);
    expect(await run.fed.state()).to.equal(State.Revealing);
    expect(await run.fed.alignmentBlock()).to.equal(BigInt(blk));
    await expect(run.fed.connect(A).commitAlignment(GAMMA_I, SIZE_I, GAMMA_A)).to.be.revertedWithCustomError(run.fed, "WrongState");
  });

  it("U13 sizeI cap (D106): sizeI = 2^maxDepth is accepted, 2^maxDepth + 1 reverts", async function () {
    for (const maxDepth of [1, 10, 20]) {
      const run = await startRun(3, { maxDepth });
      const cap = 1n << BigInt(maxDepth);
      await expect(run.fed.connect(A).commitAlignment(GAMMA_I, cap + 1n, GAMMA_A))
        .to.be.revertedWithCustomError(run.fed, "SizeAboveCap").withArgs(cap + 1n, cap);
      await expect(run.fed.connect(A).commitAlignment(GAMMA_I, cap, GAMMA_A)).to.emit(run.fed, "AlignmentCommitted");
      expect(await run.fed.sizeI()).to.equal(cap);
    }
  });

  it("U25 commitAlignment reverts after alignmentDeadline and with gammaI = 0; alignmentPeriod = 0 is rejected at creation", async function () {
    await expect(createRun(3, { alignmentPeriod: 0 })).to.be.revertedWithCustomError(fedIface, "ZeroPeriod");
    const run = await startRun(3);
    const deadline = (await run.fed.startBlock()) + BigInt(run.p.alignmentPeriod);
    expect(await run.fed.alignmentDeadline()).to.equal(deadline);
    await expect(run.fed.connect(A).commitAlignment(ethers.ZeroHash, SIZE_I, GAMMA_A))
      .to.be.revertedWithCustomError(run.fed, "ZeroCommitment");
    // gammaI is a plain keccak256 hash: values >= BN254_R are accepted (M2 drops that check).
    await nextTxAt(deadline); // the deadline block itself is still in time
    const big = toBeHex((1n << 256n) - 1n, 32);
    await expect(run.fed.connect(A).commitAlignment(big, SIZE_I, GAMMA_A)).to.emit(run.fed, "AlignmentCommitted");
    expect(await run.fed.gammaI()).to.equal(big);
    const late = await startRun(3);
    const lateDeadline = await late.fed.alignmentDeadline();
    await nextTxAt(lateDeadline + 1n);
    await expect(late.fed.connect(A).commitAlignment(GAMMA_I, SIZE_I, GAMMA_A))
      .to.be.revertedWithCustomError(late.fed, "WindowClosed").withArgs(lateDeadline);
  });

  it("U26 resolveAlignmentTimeout reverts before (and at) the deadline, and in the wrong state", async function () {
    const setup = await createRun(3);
    await expect(setup.fed.connect(OUT).resolveAlignmentTimeout()).to.be.revertedWithCustomError(setup.fed, "WrongState");
    const run = await startRun(3);
    await expect(run.fed.connect(OUT).resolveAlignmentTimeout()).to.be.revertedWithCustomError(run.fed, "WindowOpen");
    await nextTxAt(await run.fed.alignmentDeadline());
    await expect(run.fed.connect(OUT).resolveAlignmentTimeout()).to.be.revertedWithCustomError(run.fed, "WindowOpen");
    const aligned = await alignedRun(3);
    await nextTxAt((await aligned.fed.alignmentDeadline()) + 1n);
    await expect(aligned.fed.connect(OUT).resolveAlignmentTimeout()).to.be.revertedWithCustomError(aligned.fed, "WrongState");
  });

  describe("U27 resolveAlignmentTimeout after the deadline credits exactly as D105; state AlignmentFailed", function () {
    async function timeout(B: bigint) {
      const run = await startRun(3, { B });
      await nextTxAt((await run.fed.alignmentDeadline()) + 1n);
      await expect(run.fed.connect(OUT).resolveAlignmentTimeout()).to.emit(run.fed, "AlignmentTimedOut");
      expect(await run.fed.state()).to.equal(State.AlignmentFailed);
      await expect(run.fed.connect(A).commitAlignment(GAMMA_I, SIZE_I, GAMMA_A)).to.be.revertedWithCustomError(run.fed, "WrongState");
      return run;
    }
    it("U27 B >= n*Dp: each passive owed 2*Dp; A owed B - n*Dp", async function () {
      const run = await timeout(1_000_003n);
      for (const s of run.passives) expect(await run.fed.owed(s.address)).to.equal(2n * run.p.Dp);
      expect(await run.fed.owed(A.address)).to.equal(1_000_003n - 3n * run.p.Dp);
    });
    it("U27 B < n*Dp: c = floor(B/n), remainder to passive index 1; A owed 0", async function () {
      const B = 1_003n;
      const run = await timeout(B);
      const Dp = run.p.Dp;
      expect(await run.fed.owed(run.passives[0].address)).to.equal(Dp + B / 3n + (B % 3n));
      expect(await run.fed.owed(run.passives[1].address)).to.equal(Dp + B / 3n);
      expect(await run.fed.owed(run.passives[2].address)).to.equal(Dp + B / 3n);
      expect(await run.fed.owed(A.address)).to.equal(0n);
    });
  });

  it("U14 happy reveal, A first then passives in random order: s matches off-chain, s < BN254_R, Training (was Seeded)", async function () {
    for (const n of [3, 10]) {
      const run = await alignedRun(n);
      await run.fed.connect(A).revealSeed(run.rA);
      const order = [...Array(n).keys()].sort(() => Math.random() - 0.5);
      for (const i of order.slice(0, -1)) {
        await revealP(run, i);
        expect(await run.fed.state()).to.equal(State.Revealing);
      }
      const expected = seedFromReveals(run.rA, run.rP);
      await expect(revealP(run, order.at(-1)!)).to.emit(run.fed, "SeedFixed").withArgs(expected, SIZE_I);
      expect(await run.fed.seed()).to.equal(expected);
      expect(expected < BN254_R).to.equal(true);
      expect(await run.fed.state()).to.equal(State.Training);
      expect(await token.balanceOf(run.addr)).to.equal(run.p.B + BigInt(n) * run.p.Dp); // deposits stay locked
    }
  });

  it("S7 the S2b index derivation applied to the contract's s gives N indices in [0, sizeI)", async function () {
    const run = await alignedRun(3);
    await run.fed.connect(A).revealSeed(run.rA);
    for (let i = 0; i < 3; i++) await revealP(run, i);
    const s = await run.fed.seed();
    const sizeI = await run.fed.sizeI();
    const idx: number[] = await sampleIndices(s, run.p.N, sizeI);
    expect(idx.length).to.equal(run.p.N);
    for (const k of idx) expect(k >= 0 && BigInt(k) < sizeI, `index ${k}`).to.equal(true);
  });

  it("U15 a passive reveals before A; A reveals before commitAlignment", async function () {
    const started = await startRun(3);
    await expect(started.fed.connect(A).revealSeed(started.rA)).to.be.revertedWithCustomError(started.fed, "WrongState");
    await started.fed.connect(A).commitAlignment(GAMMA_I, SIZE_I, GAMMA_A);
    await expect(revealP(started, 0)).to.be.revertedWithCustomError(started.fed, "ActiveNotRevealed");
  });

  it("U16 a wrong r reverts (A and a passive)", async function () {
    const run = await alignedRun(3);
    await expect(run.fed.connect(A).revealSeed(run.rA + 1n)).to.be.revertedWithCustomError(run.fed, "BadOpening");
    await run.fed.connect(A).revealSeed(run.rA);
    await expect(run.fed.connect(A).revealSeed(run.rA)).to.be.revertedWithCustomError(run.fed, "AlreadyRevealed");
    await expect(run.fed.connect(run.passives[0]).revealSeed(run.rP[1])).to.be.revertedWithCustomError(run.fed, "BadOpening");
    await revealP(run, 0);
    await expect(revealP(run, 0)).to.be.revertedWithCustomError(run.fed, "AlreadyRevealed");
  });

  it("U17 a copied commitment and one made for another instance can never be opened: both end as non-revealers", async function () {
    const other = await createRun(3);
    const run = await createRun(3);
    const [p1, p2, p3] = run.passives;
    const c1 = seedCommitment(run.rP[0], p1.address, run.addr);
    const c3other = seedCommitment(run.rP[2], p3.address, other.addr);
    expect(await other.fed.seedCommitment(run.rP[2], p3.address)).to.equal(c3other); // valid there
    await commitA(run);
    await register(run, p1); await run.fed.connect(p1).commitSeed(c1);
    await register(run, p2); await run.fed.connect(p2).commitSeed(c1);      // copies p1's commitment
    await register(run, p3); await run.fed.connect(p3).commitSeed(c3other); // made for `other`
    await accept(run);
    await run.fed.connect(A).commitAlignment(GAMMA_I, SIZE_I, GAMMA_A);
    const aBlock = await receiptBlock(run.fed.connect(A).revealSeed(run.rA));
    await revealP(run, 0);
    await expect(run.fed.connect(p2).revealSeed(run.rP[0])).to.be.revertedWithCustomError(run.fed, "BadOpening");
    await expect(run.fed.connect(p3).revealSeed(run.rP[2])).to.be.revertedWithCustomError(run.fed, "BadOpening");
    await nextTxAt(aBlock + run.p.revealWindow + 1);
    await expect(run.fed.connect(OUT).resolveRevealFailure()).to.emit(run.fed, "RevealFailed").withArgs(false, 1n, 0n);
    expect(await run.fed.owed(p2.address)).to.equal(0n);
    expect(await run.fed.owed(p3.address)).to.equal(0n);
    expect(await run.fed.owed(p1.address)).to.equal(3n * run.p.Dp);
  });

  it("U18 A reveals at alignmentBlock + revealWindow; passives can still reveal up to aRevealBlock + revealWindow", async function () {
    const run = await alignedRun(3);
    const last = (await run.fed.alignmentBlock()) + BigInt(run.p.revealWindow);
    await nextTxAt(last);
    expect(await receiptBlock(run.fed.connect(A).revealSeed(run.rA))).to.equal(Number(last));
    expect(await run.fed.aRevealBlock()).to.equal(last);
    await revealP(run, 0);
    await revealP(run, 1);
    const lastP = last + BigInt(run.p.revealWindow);
    await nextTxAt(lastP);
    expect(await receiptBlock(revealP(run, 2))).to.equal(Number(lastP));
    expect(await run.fed.state()).to.equal(State.Training);
  });

  it("U19 a passive reveal at aRevealBlock + revealWindow + 1 reverts (and A's one block late)", async function () {
    const late = await alignedRun(3);
    await nextTxAt((await late.fed.alignmentBlock()) + BigInt(late.p.revealWindow) + 1n);
    await expect(late.fed.connect(A).revealSeed(late.rA)).to.be.revertedWithCustomError(late.fed, "WindowClosed");

    const run = await alignedRun(3);
    const aBlock = await receiptBlock(run.fed.connect(A).revealSeed(run.rA));
    await revealP(run, 0);
    await nextTxAt(aBlock + run.p.revealWindow + 1);
    await expect(revealP(run, 1)).to.be.revertedWithCustomError(run.fed, "WindowClosed")
      .withArgs(BigInt(aBlock + run.p.revealWindow));
  });

  it("U20 n = 3, one passive misses: A owed B; revealers owed Dp + Dp/2, remainder to the first in proposal order", async function () {
    const run = await alignedRun(3); // Dp = 1001 (odd): the forfeited 1001 splits 500 + 500, remainder 1
    const [p1, p2, p3] = run.passives;
    const aBlock = await receiptBlock(run.fed.connect(A).revealSeed(run.rA));
    await revealP(run, 2); // p3 before p2: the remainder follows proposal order, not reveal order
    await revealP(run, 1);
    await nextTxAt(aBlock + run.p.revealWindow + 1);
    await expect(run.fed.connect(OUT).resolveRevealFailure()).to.emit(run.fed, "RevealFailed").withArgs(false, 2n, 0n);
    const Dp = run.p.Dp;
    expect(await run.fed.state()).to.equal(State.RevealFailed);
    expect(await run.fed.owed(A.address)).to.equal(run.p.B);
    expect(await run.fed.owed(p1.address)).to.equal(0n);
    expect(await run.fed.owed(p2.address)).to.equal(Dp + Dp / 2n + (Dp % 2n)); // 1502
    expect(await run.fed.owed(p3.address)).to.equal(Dp + Dp / 2n);             // 1501
  });

  it("U21 every passive misses its window: A owed B; nobody owed Dp; n*Dp stays locked", async function () {
    const run = await alignedRun(4);
    const aBlock = await receiptBlock(run.fed.connect(A).revealSeed(run.rA));
    await nextTxAt(aBlock + run.p.revealWindow + 1);
    await expect(run.fed.connect(OUT).resolveRevealFailure()).to.emit(run.fed, "RevealFailed").withArgs(false, 0n, 4n * run.p.Dp);
    expect(await run.fed.owed(A.address)).to.equal(run.p.B);
    for (const s of run.passives) expect(await run.fed.owed(s.address)).to.equal(0n);
    expect(await run.fed.lockedForever()).to.equal(4n * run.p.Dp);
    await run.fed.connect(A).withdraw();
    for (const s of run.passives) await run.fed.connect(s).withdraw();
    expect(await token.balanceOf(run.addr)).to.equal(4n * run.p.Dp);
  });

  describe("U22 A misses its window [D105, updated]: each passive owed Dp + c; A owed the rest of B", function () {
    async function missA(n: number, B: bigint) {
      const run = await alignedRun(n, { B });
      await nextTxAt((await run.fed.alignmentBlock()) + BigInt(run.p.revealWindow) + 1n);
      await expect(run.fed.connect(OUT).resolveRevealFailure()).to.emit(run.fed, "RevealFailed").withArgs(true, 0n, 0n);
      expect(await run.fed.state()).to.equal(State.RevealFailed);
      return run;
    }

    it("U22 B >= n*Dp: c = Dp; A owed B - n*Dp", async function () {
      const run = await missA(3, 1_000_003n);
      const Dp = run.p.Dp;
      for (const s of run.passives) expect(await run.fed.owed(s.address)).to.equal(2n * Dp);
      expect(await run.fed.owed(A.address)).to.equal(1_000_003n - 3n * Dp);
    });

    it("U22 B = n*Dp (boundary): c = Dp; A owed 0", async function () {
      const run = await missA(3, 3n * 1_001n);
      for (const s of run.passives) expect(await run.fed.owed(s.address)).to.equal(2n * run.p.Dp);
      expect(await run.fed.owed(A.address)).to.equal(0n);
    });

    it("U22 B < n*Dp: c = floor(B/n), remainder to passive index 1; A owed 0", async function () {
      const B = 1_003n; // n*Dp = 3003; c = 334, remainder 1
      const run = await missA(3, B);
      const Dp = run.p.Dp;
      const [p1, p2, p3] = run.passives;
      expect(await run.fed.owed(p1.address)).to.equal(Dp + B / 3n + (B % 3n)); // 1336
      expect(await run.fed.owed(p2.address)).to.equal(Dp + B / 3n);            // 1335
      expect(await run.fed.owed(p3.address)).to.equal(Dp + B / 3n);
      expect(await run.fed.owed(A.address)).to.equal(0n);
    });
  });

  it("U23 resolveRevealFailure reverts before any window expires, and after the seed is fixed", async function () {
    const setup = await createRun(3);
    await expect(setup.fed.connect(OUT).resolveRevealFailure()).to.be.revertedWithCustomError(setup.fed, "WrongState");
    const run = await alignedRun(3);
    const lastA = (await run.fed.alignmentBlock()) + BigInt(run.p.revealWindow);
    await expect(run.fed.connect(OUT).resolveRevealFailure()).to.be.revertedWithCustomError(run.fed, "WindowOpen");
    await nextTxAt(lastA); // last block of A's window
    await expect(run.fed.connect(OUT).resolveRevealFailure()).to.be.revertedWithCustomError(run.fed, "WindowOpen");
    const run2 = await alignedRun(3);
    const aBlock = await receiptBlock(run2.fed.connect(A).revealSeed(run2.rA));
    await revealP(run2, 0);
    await nextTxAt(aBlock + run2.p.revealWindow); // last block of the passives' window
    await expect(run2.fed.connect(OUT).resolveRevealFailure()).to.be.revertedWithCustomError(run2.fed, "WindowOpen");
    const seeded = await alignedRun(3);
    await seeded.fed.connect(A).revealSeed(seeded.rA);
    for (let i = 0; i < 3; i++) await revealP(seeded, i);
    await nextTxAt((await blockNow()) + 50);
    await expect(seeded.fed.connect(OUT).resolveRevealFailure()).to.be.revertedWithCustomError(seeded.fed, "WrongState");
  });

  it("U24 conservation (extended in M2 with AlignmentFailed): on every terminal path, owed plus locked equals what the run collected (B + n*Dp once started)", async function () {
    // Deterministic PRNG so a failure can be replayed.
    let st = 0x2026_0924;
    const rnd = () => { st = (st * 1103515245 + 12345) >>> 0; return st / 2 ** 32; };
    const pick = (lo: number, hi: number) => lo + Math.floor(rnd() * (hi - lo + 1));
    const paths = ["seeded", "passivesMissed", "activeMissed", "aborted", "alignmentFailed"] as const;
    const TRIALS = 50;
    const tally: Record<string, number> = {};
    for (let t = 0; t < TRIALS; t++) {
      const path = paths[t % paths.length];
      const n = pick(3, 10);
      const Dp = BigInt(pick(1, 5_000));
      const B = rnd() < 0.5 ? BigInt(pick(1, n * Number(Dp))) : BigInt(pick(n * Number(Dp), 10_000_000));
      const run = await createRun(n, { B, Dp });
      let registered = n;
      let locked = 0n;
      if (path === "aborted") {
        // A random subset registers; the arbiter never accepts, so the run cannot start.
        registered = 0;
        if (rnd() < 0.5) await commitA(run);
        for (let i = 0; i < n; i++) {
          if (rnd() < 0.5) continue;
          await register(run, run.passives[i]); registered++;
          if (rnd() < 0.5) await commitP(run, i);
        }
        await nextTxAt(run.createBlock + run.p.setupPeriod + 1);
        await run.fed.connect(OUT).abortSetup();
      } else {
        await commitA(run);
        for (let i = 0; i < n; i++) { await register(run, run.passives[i]); await commitP(run, i); }
        await accept(run);
        if (path === "alignmentFailed") {
          await nextTxAt((await run.fed.alignmentDeadline()) + 1n);
          await run.fed.connect(OUT).resolveAlignmentTimeout();
        } else await run.fed.connect(A).commitAlignment(GAMMA_I, SIZE_I, GAMMA_A);
        if (path === "alignmentFailed") {
          // settled above (D107 = D105 rule)
        } else if (path === "activeMissed") {
          await nextTxAt((await run.fed.alignmentBlock()) + BigInt(run.p.revealWindow) + 1n);
          await run.fed.connect(OUT).resolveRevealFailure();
        } else {
          const aBlock = await receiptBlock(run.fed.connect(A).revealSeed(run.rA));
          const revealers = path === "seeded" ? [...Array(n).keys()]
            : [...Array(n).keys()].filter(() => rnd() < 0.5).slice(0, n - 1); // possibly none, never all
          for (const i of revealers) await revealP(run, i);
          if (path === "seeded") {
            locked = B + BigInt(n) * Dp; // kept for later phases
          } else {
            await nextTxAt(aBlock + run.p.revealWindow + 1);
            await run.fed.connect(OUT).resolveRevealFailure();
            if (revealers.length === 0) locked = BigInt(n) * Dp;
            expect(await run.fed.lockedForever()).to.equal(locked);
          }
        }
      }
      const collected = B + BigInt(registered) * Dp;
      const everyone = [A, ...run.passives, ARB, OUT];
      const owed = await Promise.all(everyone.map((s) => run.fed.owed(s.address)));
      const sumOwed = owed.reduce((a: bigint, b: bigint) => a + b, 0n);
      const ctx = `trial ${t} path ${path} n ${n} B ${B} Dp ${Dp}`;
      expect(await token.balanceOf(run.addr), ctx).to.equal(collected);
      expect(sumOwed + locked, ctx).to.equal(collected);
      for (let i = 0; i < everyone.length; i++) {
        await expect(run.fed.connect(everyone[i]).withdraw(), ctx).to.changeTokenBalances(ethers, token, [everyone[i]], [owed[i]]);
      }
      expect(await token.balanceOf(run.addr), ctx).to.equal(locked);
      tally[path] = (tally[path] ?? 0) + 1;
    }
    expect(Object.values(tally).reduce((a, b) => a + b, 0)).to.equal(TRIALS);
  });
});
