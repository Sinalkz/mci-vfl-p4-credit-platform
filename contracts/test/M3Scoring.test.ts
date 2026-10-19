// M3 §5: scoring tests S01..S07 (claude-code-M3-training-scoring.md §6), updated for D139 (DATA1; D137 + D139,
// which supersede D118): FULL is scored twice, each with its own proof. F_r = the retrained FULL (submitScore, kind
// FULL, its own C_model); F_j = the fusion model trained jointly during split training, committed at commitFA as
// gammaG (submitJointScore, kind FULL_JOINT, no C_model). The contract sets U_fed = max(F_r, F_j). LOO and BASELINE
// stay retrained. S08..S11 are new.
// S01, S03, S04, S08 use one real run: PSI, T rounds of split training, final commitments, reconciliation, then the
// 6 figures proved with the M1b circuit (one proof at a time, each in its own process). S05..S07, S09..S11 use
// MockScoreVerifier or no verifier for the bookkeeping.
import { expect } from "chai";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { network } from "hardhat";

import { ZeroHash, toBeHex } from "ethers";
import { BN254_R, State } from "./helpers/federation.js";
import { fieldRoot, quickTraining, reconciliationRun, signFinal, trainAll, trainingRun } from "./helpers/m3.js";
// @ts-ignore -- plain ESM modules
import { genScenario } from "../offchain/data.mjs";
// @ts-ignore
import { circuitInputFor, gammaFAHex, gammaGHex, KIND, proveInChild, requiredFigures, scoreFigures } from "../offchain/scoring.mjs";
// @ts-ignore
import { embeddingRootHex } from "../offchain/training.mjs";
// @ts-ignore
import { KEY_DIRS } from "../../circuits/scripts/prover-config.mjs";

// The real-verifier tests deploy the development key's verifier (helpers/m2.ts), so their proofs (proveInChild ->
// m3-prove-figure.mjs, whose defaults are the multi-party key and rapidsnark) are made with the development key, by
// snarkjs unless VFL_PROVER says otherwise.
process.env.VFL_KEY_DIR = KEY_DIRS.dev;
process.env.VFL_PROVER ??= "snarkjs";

// M4 (D128): the all-zero-leaf root at depth 10, built the circuit's way from one row of -32768 (packs to leaf 0)
const EMPTY_ROOT_10 = await embeddingRootHex([[-32768, -32768, -32768, -32768]], 10);

const { ethers, networkHelpers } = await network.create();
const signers = await ethers.getSigners();
const A = signers[0];
const P = signers.slice(1, 4);
const ARB = signers[11];
const OUT = signers[12];
const S = { A, P, ARB };

const mineTo = async (block: number) => {
  const now = await ethers.provider.getBlockNumber();
  if (block > now) await networkHelpers.mine(block - now);
};
const FAKE_PROOF = ["1", "0", "0", "0", "0", "0", "0", "0"]; // accepted by MockScoreVerifier
const BAD_PROOF = ["0", "0", "0", "0", "0", "0", "0", "0"]; // rejected by it

/** A run in Scoring with MockScoreVerifier. */
async function mockScoringRun(opts: { skipFinal?: boolean[]; registered?: (string | null)[]; equivocate?: number | number[] } = {}) {
  const verifier = await ethers.deployContract("MockScoreVerifier");
  const { run, finals } = await reconciliationRun(ethers, S, { ...opts, overrides: { verifier: await verifier.getAddress() } });
  for (const i of opts.equivocate === undefined ? [] : [opts.equivocate].flat()) {
    await (await run.fed.proveEquivocation(P[i].address, finals[i].root, finals[i].sig)).wait();
  }
  await mineTo(Number(await run.fed.reconciliationDeadline()));
  await (await run.fed.finalizeReconciliation()).wait();
  return { run, finals };
}

/** One mock figure: FULL_JOINT through submitJointScore, every other kind through submitScore. */
const mockSubmit = (fed: any, kind: number, j: number, F: number | bigint = 1, proof = FAKE_PROOF) =>
  kind === KIND.FULL_JOINT ? fed.connect(A).submitJointScore(F, proof)
    : fed.connect(A).submitScore(kind, j, F, fieldRoot(`m${kind}${j}`), proof);
const ALL = [[KIND.FULL, 0], [KIND.FULL_JOINT, 0], [KIND.LOO, 1], [KIND.LOO, 2], [KIND.LOO, 3], [KIND.BASELINE, 0]];

describe("M3 scoring (F_r, F_j, LOO_j, BASELINE with the M1b verifier; U_fed = max(F_r, F_j); D115 timeout)", function () {
  // ---- the real run -----------------------------------------------------------------
  let real: any;
  let snap: any;
  let preCommit: any;
  let trained: any;
  before(async function () {
    this.timeout(3_600_000);
    const scenario = genScenario();
    const ctx = await trainingRun(ethers, S, scenario);
    const t = await trainAll(ctx, P);
    ctx.close();
    const fed = ctx.run.fed;
    preCommit = await networkHelpers.takeSnapshot(); // S13 replays the run from here with passive 3 left
    trained = { ctx, t };
    for (let i = 0; i < 3; i++) await (await fed.connect(P[i]).commitFinal(t.passives[i].root)).wait();
    await (await fed.connect(A).commitFA(await gammaFAHex(t.active.fA), await gammaGHex(t.active.fusion))).wait();
    await mineTo(Number(await fed.reconciliationDeadline()));
    await (await fed.finalizeReconciliation()).wait();
    const hq = t.active.finals.map((f: any) => f.hq);
    const figs = requiredFigures(Number(await fed.remainingMask()));
    const scored = scoreFigures({ fA: t.active.fA, fusion: t.active.fusion, rows: ctx.rowsA, hq, Itr: ctx.Itr, idx: ctx.ev.idx, figures: figs });
    const seed = await fed.seed();
    const workDir = resolve(tmpdir(), `m3-s01-${Date.now()}`);
    const proofs: any = {};
    for (const fig of scored.figures) {
      const ci = await circuitInputFor({ fa: scored.fa, fig, rows: ctx.rowsA, hq, sizeI: ctx.r.I.length, maxDepth: 10, seed, N: 64 });
      const pr = await proveInChild(ci.input, workDir, fig.name);
      proofs[fig.name] = { ...fig, ...ci, ...pr };
    }
    real = { fed, proofs, scored };
    snap = await networkHelpers.takeSnapshot();
  });

  const submit = (f: any, kind = f.kind, memberIndex = f.memberIndex, F = f.F, proof = f.proof8, from = A) =>
    kind === KIND.FULL_JOINT ? real.fed.connect(from).submitJointScore(F, proof)
      : real.fed.connect(from).submitScore(kind, memberIndex, F, f.cModel, proof);

  it("S01 n = 3, all 6 figures: each accepted, each F equals the Python oracle, and U_fed = max(F_r, F_j)", async function () {
    await snap.restore();
    const names = ["FULL", "FULL_J", "LOO_1", "LOO_2", "LOO_3", "BASELINE"];
    expect(Object.keys(real.proofs)).to.deep.equal(names);
    expect(real.proofs.FULL_J.cModel).to.equal(await real.fed.gammaG());    // F_j: the committed joint model
    expect(real.proofs.FULL.cModel).to.not.equal(await real.fed.gammaG());  // F_r: a retrained model, its own C_model
    expect(real.proofs.FULL.mask).to.equal(real.proofs.FULL_J.mask);
    for (const name of names) {
      const f = real.proofs[name];
      expect(f.F).to.equal(f.oracleF); // JS exact F == py/s2b_oracle.py on the drawn rows
      expect(f.publicSignals[9]).to.equal(String(f.F));
      await expect(submit(f)).to.emit(real.fed, "ScoreAccepted").withArgs(f.kind, f.memberIndex, f.F, f.cModel, f.mask);
      const stored = await real.fed.figure(f.kind, f.memberIndex);
      expect(stored.accepted).to.equal(true);
      expect(stored.F).to.equal(BigInt(f.F));
    }
    expect(await real.fed.state()).to.equal(BigInt(State.Scored));
    expect(await real.fed.acceptedFigures()).to.equal(6n);
    expect(await real.fed.uFed()).to.equal(BigInt(Math.max(real.proofs.FULL.F, real.proofs.FULL_J.F)));
    console.log("      F:", names.map((n) => `${n} ${real.proofs[n].F}`).join(", "));
  });

  it("S02 the submitScore ABI has no parameter for gamma, seed, sizeI or mask; submitJointScore has no C_model either", async function () {
    const frag = real.fed.interface.getFunction("submitScore");
    expect(frag.inputs.map((i: any) => `${i.type} ${i.name}`)).to.deep.equal(
      ["uint8 kind", "uint8 memberIndex", "uint256 F", "bytes32 cModel", "uint256[8] proof"]);
    for (const i of frag.inputs) expect(i.name).to.not.match(/gamma|seed|size|mask/i);
    const joint = real.fed.interface.getFunction("submitJointScore");
    expect(joint.inputs.map((i: any) => `${i.type} ${i.name}`)).to.deep.equal(["uint256 F", "uint256[8] proof"]);
  });

  it("S03 LOO_1's proof submitted as LOO_2, a wrong F, FULL's proof submitted as BASELINE: each reverts", async function () {
    await snap.restore();
    const { FULL, LOO_1 } = real.proofs;
    await expect(submit(LOO_1, KIND.LOO, 2)).to.be.revertedWithCustomError(real.fed, "InvalidProof");
    await expect(submit(FULL, KIND.FULL, 0, FULL.F + 1)).to.be.revertedWithCustomError(real.fed, "InvalidProof");
    await expect(submit(FULL, KIND.BASELINE, 0)).to.be.revertedWithCustomError(real.fed, "InvalidProof"); // with FULL's C_model
    // also: not A; after the deadline
    await expect(submit(FULL, KIND.FULL, 0, FULL.F, FULL.proof8, OUT)).to.be.revertedWithCustomError(real.fed, "NotActiveParty");
    expect(await real.fed.acceptedFigures()).to.equal(0n);
    await mineTo(Number(await real.fed.scoringDeadline()));
    await expect(submit(FULL)).to.be.revertedWithCustomError(real.fed, "WindowClosed");
    await expect(submit(real.proofs.FULL_J)).to.be.revertedWithCustomError(real.fed, "WindowClosed");
  });

  it("S04 resubmitting after an invalid proof works; replacing an accepted figure reverts", async function () {
    await snap.restore();
    const { LOO_3, FULL_J } = real.proofs;
    const broken = [...LOO_3.proof8];
    broken[0] = (BigInt(broken[0]) ^ 1n).toString();
    await expect(submit(LOO_3, LOO_3.kind, 3, LOO_3.F, broken)).to.be.revertedWithCustomError(real.fed, "InvalidProof");
    await expect(submit(LOO_3)).to.emit(real.fed, "ScoreAccepted");
    await expect(submit(LOO_3)).to.be.revertedWithCustomError(real.fed, "FigureAccepted");
    await expect(submit(LOO_3, LOO_3.kind, 3, LOO_3.F + 1)).to.be.revertedWithCustomError(real.fed, "FigureAccepted");
    await expect(submit(FULL_J)).to.emit(real.fed, "ScoreAccepted");
    await expect(submit(FULL_J)).to.be.revertedWithCustomError(real.fed, "FigureAccepted");
  });

  // ---- bookkeeping with the test verifier ---------------------------------------------

  it("S05 one passive left in training: exactly 5 figures are required, and its mask bit is 0 everywhere", async function () {
    const { run } = await mockScoringRun({ skipFinal: [false, false, true] });
    const fed = run.fed;
    expect(await fed.requiredFigures()).to.equal(5n);
    expect(await fed.remainingMask()).to.equal(3n);
    await expect(fed.connect(A).submitScore(KIND.LOO, 3, 1, fieldRoot("m"), FAKE_PROOF)).to.be.revertedWithCustomError(fed, "MemberLeft");
    const masks: bigint[] = [];
    for (const [kind, j] of [[KIND.FULL, 0], [KIND.FULL_JOINT, 0], [KIND.LOO, 1], [KIND.LOO, 2], [KIND.BASELINE, 0]]) {
      const rc = await (await mockSubmit(fed, kind, j, 1000 + j)).wait();
      masks.push(fed.interface.parseLog(rc.logs.find((l: any) => fed.interface.parseLog(l)?.name === "ScoreAccepted")).args.mask);
    }
    expect(masks).to.deep.equal([3n, 3n, 2n, 1n, 0n]);
    expect(masks.every((m) => (m & 4n) === 0n)).to.equal(true);
    expect(await fed.state()).to.equal(BigInt(State.Scored));
    // the public inputs: the left passive's gamma slot holds the empty-tree root at depth maxDepth (M4, D128)
    const pub = await fed.publicInputs(3n, 1n, fieldRoot("m"));
    expect(pub[5]).to.equal(BigInt(EMPTY_ROOT_10));
    expect(pub[8]).to.equal(3n);
  });

  it("S06 scoring timeout: remaining passives get Dp + an equal share of B (remainder to the first), A nothing", async function () {
    // (a) all 3 remain, one figure accepted before the deadline
    const a = await mockScoringRun();
    const { B, Dp } = a.run.p;
    await (await mockSubmit(a.run.fed, KIND.FULL, 0)).wait();
    await expect(a.run.fed.resolveScoringTimeout()).to.be.revertedWithCustomError(a.run.fed, "WindowOpen");
    await mineTo(Number(await a.run.fed.scoringDeadline()));
    await expect(a.run.fed.connect(OUT).resolveScoringTimeout()).to.emit(a.run.fed, "ScoringTimedOut");
    expect(await a.run.fed.state()).to.equal(BigInt(State.ScoringFailed));
    expect(await a.run.fed.owed(P[0].address)).to.equal(Dp + B / 3n + (B % 3n));
    expect(await a.run.fed.owed(P[1].address)).to.equal(Dp + B / 3n);
    expect(await a.run.fed.owed(P[2].address)).to.equal(Dp + B / 3n);
    expect(await a.run.fed.owed(A.address)).to.equal(0n);
    // (b) passive 1 left (no commitFinal), passive 2 equivocated (Dp forfeited, Q3): the B remainder goes to passive 2;
    // the forfeited deposits (passive 1's and 2's Dp) go to the remaining non-equivocator, passive 3 (M4 Q3 rule)
    const b = await mockScoringRun({ skipFinal: [true, false, false], registered: [null, fieldRoot("Y"), null], equivocate: 1 });
    await mineTo(Number(await b.run.fed.scoringDeadline()));
    await (await b.run.fed.resolveScoringTimeout()).wait();
    expect(await b.run.fed.owed(P[0].address)).to.equal(0n);
    expect(await b.run.fed.owed(P[1].address)).to.equal(B / 2n + (B % 2n)); // no Dp back, no share of the forfeits
    expect(await b.run.fed.owed(P[2].address)).to.equal(Dp + B / 2n + 2n * Dp);
    expect(await b.run.fed.forfeitedToPool()).to.equal(0n);
    expect(await b.run.fed.lockedForever()).to.equal(0n);
    // (c) after Scored, the timeout is not available
    const c = await mockScoringRun();
    for (const [kind, j] of ALL) await (await mockSubmit(c.run.fed, kind, j)).wait();
    await mineTo(Number(await c.run.fed.scoringDeadline()) + 1);
    await expect(c.run.fed.resolveScoringTimeout()).to.be.revertedWithCustomError(c.run.fed, "WrongState");
    // (d) five of six figures (F_j missing) is not Scored: the timeout applies
    const e = await mockScoringRun();
    for (const [kind, j] of ALL.filter(([k]) => k !== KIND.FULL_JOINT)) await (await mockSubmit(e.run.fed, kind, j)).wait();
    expect(await e.run.fed.state()).to.equal(BigInt(State.Scoring));
    await mineTo(Number(await e.run.fed.scoringDeadline()) + 1);
    await expect(e.run.fed.resolveScoringTimeout()).to.emit(e.run.fed, "ScoringTimedOut");
    // an invalid proof reverts and A may resubmit (bookkeeping side of S04)
    const d = await mockScoringRun();
    await expect(mockSubmit(d.run.fed, KIND.FULL, 0, 1, BAD_PROOF)).to.be.revertedWithCustomError(d.run.fed, "InvalidProof");
    await expect(mockSubmit(d.run.fed, KIND.FULL_JOINT, 0, 1, BAD_PROOF)).to.be.revertedWithCustomError(d.run.fed, "InvalidProof");
  });

  it("S14 D141: scoring timeout with every passive left: B is credited to A, the three forfeited D_p are locked", async function () {
    const r = await mockScoringRun({ skipFinal: [true, true, true] });
    const { B, Dp } = r.run.p;
    expect(await r.run.fed.leftCount()).to.equal(3n);
    await mineTo(Number(await r.run.fed.scoringDeadline()));
    await expect(r.run.fed.connect(OUT).resolveScoringTimeout()).to.emit(r.run.fed, "ScoringTimedOut");
    expect(await r.run.fed.state()).to.equal(BigInt(State.ScoringFailed));
    expect(await r.run.fed.owed(A.address)).to.equal(B);
    for (const s of P) expect(await r.run.fed.owed(s.address)).to.equal(0n);
    expect(await r.run.fed.lockedForever()).to.equal(3n * Dp);
    expect(await r.run.fed.forfeitedToPool()).to.equal(0n);
    expect(await r.run.token.balanceOf(r.run.addr)).to.equal(B + 3n * Dp);
  });

  it("S15 scoring timeout: the forfeited deposits are split equally among the remaining non-equivocators, remainder to the first", async function () {
    // passive 1 left; passives 2 and 3 remain and did not equivocate: passive 1's Dp (odd) is split 2 ways
    const r = await mockScoringRun({ skipFinal: [true, false, false] });
    const { B, Dp } = r.run.p;
    await mineTo(Number(await r.run.fed.scoringDeadline()));
    await (await r.run.fed.resolveScoringTimeout()).wait();
    expect(await r.run.fed.owed(P[0].address)).to.equal(0n);
    expect(await r.run.fed.owed(P[1].address)).to.equal(Dp + B / 2n + (B % 2n) + Dp / 2n + (Dp % 2n));
    expect(await r.run.fed.owed(P[2].address)).to.equal(Dp + B / 2n + Dp / 2n);
    expect(await r.run.fed.forfeitedToPool()).to.equal(0n);
    expect(await r.run.fed.lockedForever()).to.equal(0n);
    expect(await r.run.token.balanceOf(r.run.addr)).to.equal(B + 2n * Dp + Dp); // B + the 2 refunds + the forfeit, all owed
  });

  it("S16 scoring timeout: every remaining passive equivocated: each gets its share of B, no Dp; the forfeited deposits stay locked", async function () {
    const r = await mockScoringRun({ registered: [fieldRoot("Y0"), fieldRoot("Y1"), fieldRoot("Y2")], equivocate: [0, 1, 2] });
    const { B, Dp } = r.run.p;
    expect(await r.run.fed.forfeitedToPool()).to.equal(3n * Dp);
    await mineTo(Number(await r.run.fed.scoringDeadline()));
    await (await r.run.fed.resolveScoringTimeout()).wait();
    expect(await r.run.fed.owed(P[0].address)).to.equal(B / 3n + (B % 3n));
    expect(await r.run.fed.owed(P[1].address)).to.equal(B / 3n);
    expect(await r.run.fed.owed(P[2].address)).to.equal(B / 3n);
    expect(await r.run.fed.owed(A.address)).to.equal(0n);
    expect(await r.run.fed.forfeitedToPool()).to.equal(0n);
    expect(await r.run.fed.lockedForever()).to.equal(3n * Dp);
  });

  it("S07 conservation on every M3 terminal path: owed + locked + forfeitedToPool + Dc escrow (+ the run's pending B and Dp when Scored) = deposits = balance", async function () {
    this.timeout(1_800_000);
    const paths: string[] = [];
    for (let trial = 0; trial < 24; trial++) {
      const skip = [0, 1, 2].map((i) => trial % 4 === 1 && i === trial % 3);           // one passive left in training
      const eqv = trial % 3 === 2 ? (trial % 2) : undefined;                             // an equivocation
      const registered = [0, 1, 2].map((i) => (i === eqv ? fieldRoot(`Y${trial}`) : null));
      const verifier = await ethers.deployContract("MockScoreVerifier");
      const { run, finals } = await reconciliationRun(ethers, S, { skipFinal: skip, registered, overrides: { verifier: await verifier.getAddress() } });
      const fed = run.fed;
      if (eqv !== undefined && !skip[eqv]) await (await fed.proveEquivocation(P[eqv].address, finals[eqv].root, finals[eqv].sig)).wait();
      // an objection on a remaining passive: certified, timed out, or none
      let filed = 0n;
      const objTarget = [0, 1, 2].find((i) => !skip[i]);
      const obj = ["none", "certified", "timeout"][trial % 3];
      if (obj !== "none" && objTarget !== undefined) {
        await (await run.token.mint(A.address, run.p.Dc)).wait();
        await (await run.token.connect(A).approve(run.addr, run.p.Dc)).wait();
        await (await fed.connect(A).requestResend(P[objTarget].address)).wait();
        await (await fed.connect(A).fileNonReceipt(P[objTarget].address)).wait();
        filed = run.p.Dc;
        if (obj === "certified") await (await fed.connect(ARB).certifyDelivery(P[objTarget].address, (await fed.finalStatus(P[objTarget].address)).gamma)).wait();
        else { await mineTo(Number((await fed.finalStatus(P[objTarget].address)).relayDeadline)); await (await fed.resolveRelayTimeout(P[objTarget].address)).wait(); }
      }
      await mineTo(Number(await fed.reconciliationDeadline()));
      await (await fed.finalizeReconciliation()).wait();
      const scoreAll = trial % 2 === 0;
      if (scoreAll) {
        const mask = Number(await fed.remainingMask());
        for (const [kind, j] of ALL) {
          if (kind === KIND.LOO && !(mask & (1 << (j - 1)))) continue;
          await (await mockSubmit(fed, kind, j)).wait();
        }
        expect(await fed.state()).to.equal(BigInt(State.Scored));
      } else {
        await mineTo(Number(await fed.scoringDeadline()));
        await (await fed.resolveScoringTimeout()).wait();
        expect(await fed.state()).to.equal(BigInt(State.ScoringFailed));
      }
      const who = [A, ...P, ARB];
      let owed = 0n;
      for (const s of who) owed += await fed.owed(s.address);
      let pending = 0n;
      if (scoreAll) {
        pending = run.p.B;
        for (const s of P) if (!(await fed.finalStatus(s.address)).dpForfeited) pending += run.p.Dp;
      }
      expect(await fed.dcEscrow(), `trial ${trial}`).to.equal(0n); // M4 (D130): D_c always went to the arbiter
      const buckets = owed + (await fed.lockedForever()) + (await fed.forfeitedToPool()) + (await fed.dcEscrow()) + pending;
      const deposits = run.p.B + 3n * run.p.Dp + filed;
      expect(buckets, `trial ${trial}`).to.equal(deposits);
      expect(await run.token.balanceOf(run.addr), `trial ${trial}`).to.equal(deposits);
      // every withdrawal pays exactly what is owed
      for (const s of who) {
        const due = await fed.owed(s.address);
        const before = await run.token.balanceOf(s.address);
        await (await fed.connect(s).withdraw()).wait();
        expect((await run.token.balanceOf(s.address)) - before).to.equal(due);
      }
      expect(await run.token.balanceOf(run.addr)).to.equal(deposits - owed);
      paths.push(`${scoreAll ? "Scored" : "ScoringFailed"}/${skip.some(Boolean) ? "left" : "all"}/${eqv !== undefined ? "eqv" : "-"}/${obj}`);
    }
    expect(new Set(paths.map((p) => p.split("/")[0]))).to.deep.equal(new Set(["Scored", "ScoringFailed"]));
    expect(new Set(paths.map((p) => p.split("/")[3]))).to.deep.equal(new Set(["none", "certified", "timeout"]));
  });

  // ---- D139 (DATA1) --------------------------------------------------------------------

  it("S08 D139: F_j must be proven for the committed gammaG; the retrained FULL's valid proof is rejected as F_j", async function () {
    await snap.restore();
    const { FULL, FULL_J } = real.proofs;
    const committed = await real.fed.gammaG();
    expect(FULL.cModel).to.not.equal(committed);                             // a different fusion model ...
    expect(FULL.publicSignals[0]).to.equal(BigInt(FULL.cModel).toString());  // ... whose proof is valid for its own C_model
    await expect(real.fed.connect(A).submitJointScore(FULL.F, FULL.proof8)).to.be.revertedWithCustomError(real.fed, "InvalidProof");
    await expect(submit(FULL_J)).to.emit(real.fed, "ScoreAccepted").withArgs(KIND.FULL_JOINT, 0, FULL_J.F, committed, FULL_J.mask);
  });

  it("S09 D139: submitScore with kind FULL_JOINT reverts BadFigure (F_j takes no C_model); submitJointScore uses gammaG", async function () {
    const { run } = await mockScoringRun();
    await expect(run.fed.connect(A).submitScore(KIND.FULL_JOINT, 0, 1, fieldRoot("m"), FAKE_PROOF)).to.be.revertedWithCustomError(run.fed, "BadFigure");
    await expect(run.fed.connect(A).submitJointScore(1, FAKE_PROOF)).to.emit(run.fed, "ScoreAccepted")
      .withArgs(KIND.FULL_JOINT, 0, 1, await run.fed.gammaG(), 7);
  });

  it("S10 D139: commitFA with a zero or non-field gammaG reverts; a valid one is stored", async function () {
    const run = await quickTraining(ethers, S);
    const fed = run.fed;
    for (let i = 0; i < 3; i++) await (await fed.connect(P[i]).commitFinal(fieldRoot(`p${i}`))).wait();
    await expect(fed.connect(A).commitFA(fieldRoot("C_fA"), ZeroHash)).to.be.revertedWithCustomError(fed, "ZeroCommitment");
    await expect(fed.connect(A).commitFA(fieldRoot("C_fA"), toBeHex(BN254_R, 32))).to.be.revertedWithCustomError(fed, "NotAFieldElement");
    const rc = await (await fed.connect(A).commitFA(fieldRoot("C_fA"), fieldRoot("gamma_g"))).wait();
    const ev = rc.logs.map((l: any) => fed.interface.parseLog(l)).find((e: any) => e?.name === "TrainingEnded");
    expect([ev.args.gammaFA, ev.args.gammaG, ev.args.left]).to.deep.equal([fieldRoot("C_fA"), fieldRoot("gamma_g"), 0n]);
    expect(await fed.gammaG()).to.equal(fieldRoot("gamma_g"));
  });

  it("S11 D139: U_fed = max(F_r, F_j), set once both are accepted, in either order, ties included", async function () {
    for (const [first, Fr, Fj] of [[KIND.FULL, 4000, 4300], [KIND.FULL_JOINT, 4000, 4300], [KIND.FULL, 4500, 4300], [KIND.FULL_JOINT, 4200, 4200]]) {
      const { run } = await mockScoringRun();
      const fed = run.fed;
      const second = first === KIND.FULL ? KIND.FULL_JOINT : KIND.FULL;
      const Fof = (k: number) => (k === KIND.FULL ? Fr : Fj);
      await (await mockSubmit(fed, first, 0, Fof(first))).wait();
      expect(await fed.uFed()).to.equal(0n); // not set until both FULL figures are accepted
      await expect(mockSubmit(fed, second, 0, Fof(second))).to.emit(fed, "UFedSet").withArgs(Fr, Fj, Math.max(Fr, Fj));
      expect(await fed.uFed()).to.equal(BigInt(Math.max(Fr, Fj)));
    }
  });

  // ---- M4 (D128): a passive who left is scored against the empty-tree root ----------------

  it("S12 M4 (D128): a left passive's public-input slot is the empty-tree root, whether it never committed or timed out", async function () {
    const { run } = await mockScoringRun({ skipFinal: [false, true, false] });          // passive 2 never committed
    expect(await run.fed.emptyRoot()).to.equal(EMPTY_ROOT_10);
    expect((await run.fed.publicInputs(5n, 1n, fieldRoot("m")))[4]).to.equal(BigInt(EMPTY_ROOT_10));
    expect((await run.fed.publicInputs(5n, 1n, fieldRoot("m")))[3]).to.equal(BigInt((await run.fed.finalStatus(P[0].address)).gamma));
    // passive 1 committed, then left by a relay timeout: its stored gamma is not used either
    const verifier = await ethers.deployContract("MockScoreVerifier");
    const { run: r2 } = await reconciliationRun(ethers, S, { overrides: { verifier: await verifier.getAddress() } });
    await (await r2.token.mint(A.address, r2.p.Dc)).wait();
    await (await r2.token.connect(A).approve(r2.addr, r2.p.Dc)).wait();
    await (await r2.fed.connect(A).requestResend(P[0].address)).wait();
    await (await r2.fed.connect(A).fileNonReceipt(P[0].address)).wait();
    await mineTo(Number((await r2.fed.finalStatus(P[0].address)).relayDeadline));
    await (await r2.fed.resolveRelayTimeout(P[0].address)).wait();
    expect((await r2.fed.finalStatus(P[0].address)).gamma).to.not.equal(EMPTY_ROOT_10);
    expect((await r2.fed.publicInputs(6n, 1n, fieldRoot("m")))[3]).to.equal(BigInt(EMPTY_ROOT_10));
  });

  it("S13 M4 (D128), real proof: with passive 3 left, BASELINE proven with -32768 rows for its slot is accepted", async function () {
    this.timeout(1_800_000);
    await preCommit.restore(); // invalidates later snapshots: this test runs last
    const { ctx, t } = trained;
    const fed = ctx.run.fed;
    for (let i = 0; i < 2; i++) await (await fed.connect(P[i]).commitFinal(t.passives[i].root)).wait();
    await (await fed.connect(A).commitFA(await gammaFAHex(t.active.fA), await gammaGHex(t.active.fusion))).wait();
    expect((await fed.finalStatus(P[2].address)).left).to.equal(true);
    await mineTo(Number(await fed.reconciliationDeadline()));
    await (await fed.finalizeReconciliation()).wait();
    expect(await fed.remainingMask()).to.equal(3n);
    const hq = [t.active.finals[0].hq, t.active.finals[1].hq, null];
    const base = requiredFigures(3).filter((f: any) => f.kind === KIND.BASELINE);
    const [fig] = scoreFigures({ fA: t.active.fA, fusion: t.active.fusion, rows: ctx.rowsA, hq, Itr: ctx.Itr, idx: ctx.ev.idx, figures: base }).figures;
    const fa = scoreFigures({ fA: t.active.fA, fusion: t.active.fusion, rows: ctx.rowsA, hq, Itr: ctx.Itr, idx: ctx.ev.idx, figures: [] }).fa;
    const ci = await circuitInputFor({ fa, fig, rows: ctx.rowsA, hq, sizeI: ctx.r.I.length, maxDepth: 10, seed: await fed.seed(), N: 64 });
    expect(ci.input.gammaP[2]).to.equal(BigInt(EMPTY_ROOT_10).toString());
    const pr = await proveInChild(ci.input, resolve(tmpdir(), `m4-s13-${Date.now()}`), "BASELINE");
    await expect(fed.connect(A).submitScore(KIND.BASELINE, 0, fig.F, ci.cModel, pr.proof8)).to.emit(fed, "ScoreAccepted");
  });
});
