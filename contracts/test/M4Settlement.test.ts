// M4 settlement (claude-code-M4-settlement.md; plan docs/superpowers/plans/2026-09-30-m4-settlement.md).
// Expected values come from py/settlement_reference.py through test/vectors/m4_settlement.json (py/m4_vectors.py).
import { expect } from "chai";
import { readFileSync } from "node:fs";
import { network } from "hardhat";

import { State } from "./helpers/federation.js";
import { reconciliationRun } from "./helpers/m3.js";
import { alignedSig, Case, FAKE_PROOF, scoredRun } from "./helpers/m4.js";
import { keccak256, toUtf8Bytes } from "ethers";
// @ts-ignore -- plain ESM module
import { ACK_GAS_FLOOR } from "../offchain/party-m3.mjs";

const { ethers, networkHelpers } = await network.create();
const signers = await ethers.getSigners();
const A = signers[0];
const P = signers.slice(1, 4);
const ARB = signers[11];
const OUT = signers[12];
const S = { A, P, ARB };
const VECTORS: Case[] = JSON.parse(readFileSync(new URL("./vectors/m4_settlement.json", import.meta.url), "utf8")).cases;
const big = (xs: number[]) => xs.map((x) => BigInt(x));

describe("M4 settlement", function () {
  // ---- Task 5: the division ------------------------------------------------------------
  it(`M01 settlementDivision equals the Python reference on all ${VECTORS.length} vector cases`, async function () {
    this.timeout(1_800_000);
    for (const c of VECTORS) {
      const { fed } = await scoredRun(ethers, networkHelpers, S, c);
      expect(await fed.state(), c.name).to.equal(BigInt(State.Scored));
      const [E, Eprime, l, a] = await fed.settlementDivision();
      expect([E, Eprime], c.name).to.deep.equal([BigInt(c.expected.E), BigInt(c.expected.Eprime)]);
      expect([...l], c.name).to.deep.equal(big(c.expected.l));
      expect([...a], c.name).to.deep.equal(big(c.expected.a));
    }
  });

  it("M02 settlementDivision reverts before Scored", async function () {
    const { run } = await reconciliationRun(ethers, S);
    await expect(run.fed.settlementDivision()).to.be.revertedWithCustomError(run.fed, "WrongState");
  });

  // ---- Task 6: acknowledge and the normal path -------------------------------------------
  async function expectOwed(fed: any, c: Case) {
    expect(await fed.owed(A.address), `${c.name}: A`).to.equal(BigInt(c.expected.owedA));
    for (let i = 0; i < 3; i++) expect(await fed.owed(P[i].address), `${c.name}: p${i + 1}`).to.equal(BigInt(c.expected.owedP[i]));
    expect(await fed.lockedForever(), `${c.name}: locked`).to.equal(BigInt(c.expected.locked));
    expect(await fed.forfeitedToPool(), `${c.name}: forfeits distributed`).to.equal(0n);
  }

  it("M03 normal path: every remaining passive acknowledges; the last one settles; payouts equal the reference", async function () {
    this.timeout(1_800_000);
    const cases = VECTORS.filter((c) => c.path === "normal");
    for (const c of cases) {
      const { fed } = await scoredRun(ethers, networkHelpers, S, c);
      const remaining = [0, 1, 2].filter((i) => !c.left[i]);
      for (const [k, i] of remaining.entries()) {
        const tx = fed.connect(P[i]).acknowledge();
        await expect(tx, c.name).to.emit(fed, "Acknowledged").withArgs(P[i].address);
        if (k < remaining.length - 1) expect(await fed.state()).to.equal(BigInt(State.Scored));
        else await expect(tx).to.emit(fed, "Settled").withArgs(1, c.expected.E, c.expected.Eprime, big(c.expected.a));
      }
      expect(await fed.state()).to.equal(BigInt(State.Settled));
      expect(await fed.settlementPath()).to.equal(1n); // Normal
      await expectOwed(fed, c);
    }
  });

  it("M04 acknowledge reverts: outsider, A, arbiter, a left passive, twice, before Scored, after the window, after Settled", async function () {
    const c = VECTORS.find((x) => x.name === "p3 left, p1 equivocated: normal")!;
    const { run } = await reconciliationRun(ethers, S);
    await expect(run.fed.connect(P[0]).acknowledge()).to.be.revertedWithCustomError(run.fed, "WrongState");
    const { fed, mineTo } = await scoredRun(ethers, networkHelpers, S, c);
    for (const s of [OUT, A, ARB]) await expect(fed.connect(s).acknowledge()).to.be.revertedWithCustomError(fed, "NotPassive");
    await expect(fed.connect(P[2]).acknowledge()).to.be.revertedWithCustomError(fed, "MemberLeft");
    await (await fed.connect(P[0]).acknowledge()).wait();
    await expect(fed.connect(P[0]).acknowledge()).to.be.revertedWithCustomError(fed, "AlreadyAcknowledged");
    const deadline = Number(await fed.settlementDeadline());
    expect(deadline).to.equal(Number(await fed.scoredBlock()) + 3 * 30); // K * deltaA
    await mineTo(deadline - 1); // the next transaction lands in the window's last block
    await (await fed.connect(P[1]).acknowledge()).wait(); // last remaining passive: settles
    expect(await fed.state()).to.equal(BigInt(State.Settled));
    await expect(fed.connect(P[1]).acknowledge()).to.be.revertedWithCustomError(fed, "WrongState");
    // a separate run: one block past the window
    const d = await scoredRun(ethers, networkHelpers, S, c);
    await d.mineTo(Number(await d.fed.settlementDeadline()));
    await expect(d.fed.connect(P[0]).acknowledge()).to.be.revertedWithCustomError(d.fed, "WindowClosed");
  });

  it("M05 an equivocator gets its award but no D_p and no share of the forfeits; a passive who left gets nothing", async function () {
    const c = VECTORS.find((x) => x.name === "p3 left, p1 equivocated: normal")!;
    const { fed } = await scoredRun(ethers, networkHelpers, S, c);
    await (await fed.connect(P[0]).acknowledge()).wait();
    await (await fed.connect(P[1]).acknowledge()).wait();
    const [, , , a] = [0, 0, 0, c.expected.a];
    expect(await fed.owed(P[0].address)).to.equal(BigInt(c.rho * a[0]));                      // award only
    expect(await fed.owed(P[1].address)).to.equal(BigInt(c.rho * a[1] + c.Dp + 2 * c.Dp));   // + both forfeited deposits
    expect(await fed.owed(P[2].address)).to.equal(0n);
    await expectOwed(fed, c);
  });

  // ---- Task 7: objectAlignment and the objection path -------------------------------------
  it("M06 an upheld objection settles at once: rewards, deposits and forfeits as normal, B - rho*sum(a) to the remaining passives, A nothing", async function () {
    this.timeout(1_800_000);
    for (const c of VECTORS.filter((x) => x.path === "objection")) {
      const { run, fed } = await scoredRun(ethers, networkHelpers, S, c);
      for (let i = 0; i < 3; i++) if (c.acks[i]) await (await fed.connect(P[i]).acknowledge()).wait();
      const who = P[c.objector!];
      const h = keccak256(toUtf8Bytes(`list sent to ${who.address}`));
      const tx = fed.connect(who).objectAlignment(await alignedSig(A, run, who.address, h), h);
      await expect(tx, c.name).to.emit(fed, "AlignmentObjectionUpheld").withArgs(who.address, h);
      await expect(tx, c.name).to.emit(fed, "Settled").withArgs(2, c.expected.E, c.expected.Eprime, big(c.expected.a));
      expect(await fed.state()).to.equal(BigInt(State.Settled));
      expect(await fed.settlementPath()).to.equal(2n);
      await expectOwed(fed, c);
      const other = [0, 1, 2].find((i) => !c.left[i] && !c.acks[i] && i !== c.objector);
      if (other !== undefined) await expect(fed.connect(P[other]).acknowledge()).to.be.revertedWithCustomError(fed, "WrongState");
      await expect(fed.connect(who).objectAlignment(await alignedSig(A, run, who.address, h), h)).to.be.revertedWithCustomError(fed, "WrongState");
    }
  });

  it("M07 a baseless objection reverts: h = gamma_I, another signer, a signature for another passive or another run, outside the window, a left passive, an outsider, after acknowledging", async function () {
    const c = VECTORS.find((x) => x.name === "p3 left, p1 equivocated: normal")!;
    const { run, fed, mineTo } = await scoredRun(ethers, networkHelpers, S, c);
    const gI = await fed.gammaI();
    const h = keccak256(toUtf8Bytes("another list"));
    await expect(fed.connect(P[1]).objectAlignment(await alignedSig(A, run, P[1].address, gI), gI)).to.be.revertedWithCustomError(fed, "NoMisalignment");
    await expect(fed.connect(P[1]).objectAlignment(await alignedSig(ARB, run, P[1].address, h), h)).to.be.revertedWithCustomError(fed, "SignerMismatch");
    await expect(fed.connect(P[1]).objectAlignment(await alignedSig(A, run, P[0].address, h), h)).to.be.revertedWithCustomError(fed, "SignerMismatch");
    const otherRun = await scoredRun(ethers, networkHelpers, S, c);
    await expect(fed.connect(P[1]).objectAlignment(await alignedSig(A, otherRun.run, P[1].address, h), h)).to.be.revertedWithCustomError(fed, "SignerMismatch");
    await expect(fed.connect(P[2]).objectAlignment(await alignedSig(A, run, P[2].address, h), h)).to.be.revertedWithCustomError(fed, "MemberLeft");
    await expect(fed.connect(OUT).objectAlignment(await alignedSig(A, run, OUT.address, h), h)).to.be.revertedWithCustomError(fed, "NotPassive");
    await (await fed.connect(P[0]).acknowledge()).wait();
    await expect(fed.connect(P[0]).objectAlignment(await alignedSig(A, run, P[0].address, h), h)).to.be.revertedWithCustomError(fed, "AlreadyAcknowledged");
    expect(await fed.state()).to.equal(BigInt(State.Scored)); // nothing settled
    await mineTo(Number(await fed.settlementDeadline()));
    await expect(fed.connect(P[1]).objectAlignment(await alignedSig(A, run, P[1].address, h), h)).to.be.revertedWithCustomError(fed, "WindowClosed");
  });

  // ---- Task 8: the backup path ------------------------------------------------------------
  // D142 (supersedes M4 Q6): after the window, settleBackup settles exactly as the normal path; silence counts as acceptance.
  it("M08 D142: settleBackup after K*deltaA pays exactly the normal path (rho*a_i, D_p unless forfeited, forfeits to non-equivocators, the rest of B to A); path 3 recorded", async function () {
    this.timeout(1_800_000);
    for (const c of VECTORS.filter((x) => x.path === "backup")) {
      const { fed, mineTo } = await scoredRun(ethers, networkHelpers, S, c);
      for (let i = 0; i < 3; i++) if (c.acks[i]) await (await fed.connect(P[i]).acknowledge()).wait();
      if ([0, 1, 2].every((i) => c.left[i] || c.acks[i])) throw new Error(`${c.name}: every remaining passive acknowledged (normal path)`);
      const deadline = Number(await fed.settlementDeadline());
      await mineTo(deadline - 1); // the next transaction lands in the window's last block
      await expect(fed.connect(OUT).settleBackup(), c.name).to.be.revertedWithCustomError(fed, "WindowOpen");
      await networkHelpers.mine(1); // a reverted call mines no block: the next transaction lands one past the window
      await expect(fed.connect(OUT).settleBackup(), c.name).to.emit(fed, "Settled").withArgs(3, c.expected.E, c.expected.Eprime, big(c.expected.a));
      expect(await fed.state()).to.equal(BigInt(State.Settled));
      expect(await fed.settlementPath()).to.equal(3n);
      expect(await fed.owed(OUT.address)).to.equal(0n);
      await expectOwed(fed, c);
      await expect(fed.connect(OUT).settleBackup()).to.be.revertedWithCustomError(fed, "WrongState");
    }
  });

  // ---- Task 9: conservation on every M4 terminal path, and the size limit ---------------------
  it("M12 conservation on every M4 terminal path (normal, objection, backup with and without responders) x left / equivocation / relay objection: owed + locked = deposits = balance; withdrawals pay exactly", async function () {
    this.timeout(3_600_000);
    const paths: string[] = [];
    const figures = { Fr: 4400, Fj: 4300, loo: [4100, 3900, 3600], base: 3000 };
    for (let trial = 0; trial < 24; trial++) {
      const path = ["normal", "objection", "backup", "backup-none"][trial % 4];
      const variant = Math.floor(trial / 4) % 6; // 0 none, 1 left, 2 eqv, 3 relay certified, 4 relay timeout, 5 left + eqv
      const skip = [false, false, variant === 1 || variant === 5];
      const eqv = variant === 2 || variant === 5 ? 0 : undefined;
      const verifier = await ethers.deployContract("MockScoreVerifier");
      const { run, finals } = await reconciliationRun(ethers, S, {
        skipFinal: skip, registered: [0, 1, 2].map((i) => (i === eqv ? `0x${"0".repeat(63)}${i + 1}` : null)),
        overrides: { verifier: await verifier.getAddress(), B: 1_000_003n + BigInt(trial), rho: 997n, Dp: 1_001n + BigInt(trial) },
      });
      const fed = run.fed;
      if (eqv !== undefined) await (await fed.proveEquivocation(P[eqv].address, finals[eqv].root, finals[eqv].sig)).wait();
      let filed = 0n;
      if (variant === 3 || variant === 4) {
        await (await run.token.mint(A.address, run.p.Dc)).wait();
        await (await run.token.connect(A).approve(run.addr, run.p.Dc)).wait();
        await (await fed.connect(A).requestResend(P[1].address)).wait();
        await (await fed.connect(A).fileNonReceipt(P[1].address)).wait();
        filed = run.p.Dc;
        if (variant === 3) await (await fed.connect(ARB).certifyDelivery(P[1].address, (await fed.finalStatus(P[1].address)).gamma)).wait();
        else {
          const rd = Number((await fed.finalStatus(P[1].address)).relayDeadline);
          const now = await ethers.provider.getBlockNumber();
          if (rd > now) await networkHelpers.mine(rd - now);
          await (await fed.resolveRelayTimeout(P[1].address)).wait();
        }
      }
      const rec = Number(await fed.reconciliationDeadline());
      const now = await ethers.provider.getBlockNumber();
      if (rec > now) await networkHelpers.mine(rec - now);
      await (await fed.finalizeReconciliation()).wait();
      const left = await Promise.all([0, 1, 2].map(async (i) => (await fed.finalStatus(P[i].address)).left));
      await (await fed.connect(A).submitScore(0, 0, figures.Fr, `0x${"0".repeat(62)}aa`, FAKE_PROOF)).wait();
      await (await fed.connect(A).submitJointScore(figures.Fj, FAKE_PROOF)).wait();
      for (let j = 1; j <= 3; j++) if (!left[j - 1]) await (await fed.connect(A).submitScore(1, j, figures.loo[j - 1], `0x${"0".repeat(63)}${j}`, FAKE_PROOF)).wait();
      await (await fed.connect(A).submitScore(2, 0, figures.base, `0x${"0".repeat(62)}bb`, FAKE_PROOF)).wait();
      expect(await fed.state(), `trial ${trial}`).to.equal(BigInt(State.Scored));
      const remaining = [0, 1, 2].filter((i) => !left[i]);
      if (path === "normal") for (const i of remaining) await (await fed.connect(P[i]).acknowledge()).wait();
      else if (path === "objection") {
        const who = P[remaining[remaining.length - 1]];
        const h = keccak256(toUtf8Bytes(`other ${trial}`));
        await (await fed.connect(who).objectAlignment(await alignedSig(A, run, who.address, h), h)).wait();
      } else {
        if (path === "backup" && remaining.length > 1) await (await fed.connect(P[remaining[0]]).acknowledge()).wait();
        const dl = Number(await fed.settlementDeadline());
        const b = await ethers.provider.getBlockNumber();
        if (dl + 1 > b) await networkHelpers.mine(dl + 1 - b);
        await (await fed.connect(OUT).settleBackup()).wait();
      }
      expect(await fed.state(), `trial ${trial}`).to.equal(BigInt(State.Settled));
      const who = [A, ...P, ARB, OUT];
      let owed = 0n;
      for (const s of who) owed += await fed.owed(s.address);
      const locked = await fed.lockedForever();
      expect(await fed.forfeitedToPool(), `trial ${trial}`).to.equal(0n);
      expect(await fed.dcEscrow(), `trial ${trial}`).to.equal(0n);
      const deposits = run.p.B + 3n * run.p.Dp + filed;
      expect(owed + locked, `trial ${trial}`).to.equal(deposits);
      expect(await run.token.balanceOf(run.addr), `trial ${trial}`).to.equal(deposits);
      for (const s of who) {
        const due = await fed.owed(s.address);
        const before = await run.token.balanceOf(s.address);
        await (await fed.connect(s).withdraw()).wait();
        expect((await run.token.balanceOf(s.address)) - before, `trial ${trial}`).to.equal(due);
      }
      expect(await run.token.balanceOf(run.addr), `trial ${trial}`).to.equal(locked);
      paths.push(`${path}/${variant}`);
    }
    expect(new Set(paths.map((p) => p.split("/")[0]))).to.deep.equal(new Set(["normal", "objection", "backup", "backup-none"]));
    expect(new Set(paths.map((p) => p.split("/")[1])).size).to.equal(6);
  });

  it("M13 Federation's runtime code fits EIP-170 (24,576 bytes)", async function () {
    const art = JSON.parse(readFileSync(new URL("../artifacts/contracts/Federation.sol/Federation.json", import.meta.url), "utf8"));
    const size = (art.deployedBytecode.length - 2) / 2;
    console.log(`      Federation runtime: ${size} bytes`);
    expect(size).to.be.at.most(24_576);
  });

  it("M09 settleBackup reverts before Scored and after a normal settlement", async function () {
    const { run } = await reconciliationRun(ethers, S);
    await expect(run.fed.settleBackup()).to.be.revertedWithCustomError(run.fed, "WrongState");
    const c = VECTORS.find((x) => x.name === "E3 figures, normal")!;
    const { fed, mineTo } = await scoredRun(ethers, networkHelpers, S, c);
    for (let i = 0; i < 3; i++) await (await fed.connect(P[i]).acknowledge()).wait();
    await mineTo(Number(await fed.settlementDeadline()) + 1);
    await expect(fed.settleBackup()).to.be.revertedWithCustomError(fed, "WrongState");
  });

  // Final review: rho is only required to be nonzero; rho * sum(l) must not overflow, or settleBackup reverts for good.
  it("M15 the backup path settles with a huge rho (no overflow); payouts equal the reference (D142: as normal)", async function () {
    const math = await ethers.deployContract("SettlementMath");
    const rho = 2n ** 255n, B = 2n ** 255n + 7n;
    const o = await math.settle({ path: 3, B, rho, Dp: 1_001n, uFed: 4000n, uA: 3000n, forfeited: 0n,
      loo: [3900n, 3800n, 3700n], left: 0n, eqv: 0n });
    // py/settlement_reference.py settle("backup", ...) (D142: as normal): E' = floor(B / rho) = 1, awards 0, 0, 0
    // (Talmud(1; 100, 200, 300) floors to 0 each), every passive its D_p, A the whole of B, nothing locked
    expect(o.estate).to.equal(1n);
    expect([...o.awards]).to.deep.equal([0n, 0n, 0n]);
    expect([...o.credits]).to.deep.equal([1_001n, 1_001n, 1_001n]);
    expect(o.creditA).to.equal(B);
    expect(o.locked).to.equal(0n);
  });

  // D141 (QUESTIONS-M4 Q1): if no passive remains, B returns to A and the forfeited deposits stay locked. No one can
  // acknowledge, so settleBackup is the only call; it pays as the normal path (D142): A gets B, the forfeits are locked.
  it("M16 D141: every passive left: settleBackup after K*deltaA pays as the normal path, A gets B, the three D_p are locked", async function () {
    const base = VECTORS.find((x) => x.name === "E3 figures, normal")!;
    const c: Case = { ...base, name: "every passive left", path: "backup", left: [true, true, true], acks: [false, false, false] };
    const { fed, mineTo } = await scoredRun(ethers, networkHelpers, S, c);
    expect(await fed.state()).to.equal(BigInt(State.Scored));
    for (let i = 0; i < 3; i++) await expect(fed.connect(P[i]).acknowledge()).to.be.revertedWithCustomError(fed, "MemberLeft");
    await expect(fed.settleBackup()).to.be.revertedWithCustomError(fed, "WindowOpen");
    await mineTo(Number(await fed.settlementDeadline()) + 1);
    await expect(fed.connect(OUT).settleBackup()).to.emit(fed, "Settled").withArgs(3, c.expected.E, c.expected.Eprime, [0n, 0n, 0n]);
    expect(await fed.settlementPath()).to.equal(3n);
    expect(await fed.owed(A.address)).to.equal(BigInt(c.B));
    for (let i = 0; i < 3; i++) expect(await fed.owed(P[i].address)).to.equal(0n);
    expect(await fed.lockedForever()).to.equal(BigInt(3 * c.Dp));
    expect(await fed.forfeitedToPool()).to.equal(0n);
  });

  /** Every party's owed amount and the locked amount, for comparing two runs. */
  async function outcome(fed: any) {
    return { A: await fed.owed(A.address), P: await Promise.all(P.map((s) => fed.owed(s.address))), locked: await fed.lockedForever() };
  }

  // D142: with fewer than a third of validators colluding, a response sent to several nodes is included within
  // (colluders + 1) blocks, so a missing response at the window's end is silence. Keeping a passive silent must not pay.
  it("M17 D142: a coalition (A, p1, p2) that keeps p3 silent gains nothing: settleBackup pays every party exactly what the normal path pays", async function () {
    const c = VECTORS.find((x) => x.name === "E3 figures, normal")!;     // p3 holds the only positive claim
    const normal = await scoredRun(ethers, networkHelpers, S, c);
    for (let i = 0; i < 3; i++) await (await normal.fed.connect(P[i]).acknowledge()).wait();
    expect(await normal.fed.settlementPath()).to.equal(1n);
    const honest = await outcome(normal.fed);
    const kept = await scoredRun(ethers, networkHelpers, S, c);
    for (const i of [0, 1]) await (await kept.fed.connect(P[i]).acknowledge()).wait();   // p3's response never lands
    await kept.mineTo(Number(await kept.fed.settlementDeadline()) + 1);
    await (await kept.fed.connect(A).settleBackup()).wait();
    expect(await kept.fed.settlementPath()).to.equal(3n);
    const silenced = await outcome(kept.fed);
    expect(silenced).to.deep.equal(honest);
    expect(silenced.P[2]).to.equal(BigInt(c.expected.owedP[2]));          // p3 still gets rho*a_3 + D_p
  });

  it("M18 D142: a passive that goes offline: after the window A is paid exactly the normal amount, the others theirs, nothing locked", async function () {
    const c = VECTORS.find((x) => x.name === "large B: E' = E")!;          // every claim paid in full, A keeps the rest of B
    const { fed, mineTo } = await scoredRun(ethers, networkHelpers, S, c);
    for (const i of [0, 2]) await (await fed.connect(P[i]).acknowledge()).wait();        // p2 is offline
    await mineTo(Number(await fed.settlementDeadline()) + 1);
    await (await fed.connect(OUT).settleBackup()).wait();
    expect(await fed.owed(A.address)).to.equal(BigInt(c.expected.owedA));
    for (let i = 0; i < 3; i++) expect(await fed.owed(P[i].address), `p${i + 1}`).to.equal(BigInt(c.expected.owedP[i]));
    expect(await fed.lockedForever()).to.equal(BigInt(c.expected.locked));
    expect(c.expected.locked).to.equal(0);
  });

  // Found by the E3 run on Besu: the three passives acknowledged in one block; p1's gas limit (its estimate as an early
  // acknowledge + 100k) did not cover the settlement its acknowledge ran as the last one, and it ran out of gas.
  it("M14 a settling acknowledge costs more than an early acknowledge's estimate + 100k, and less than party-m3's ACK_GAS_FLOOR on every normal-path vector", async function () {
    this.timeout(1_800_000);
    for (const c of VECTORS.filter((x) => x.path === "normal")) {
      const { fed } = await scoredRun(ethers, networkHelpers, S, c);
      const remaining = [0, 1, 2].filter((i) => !c.left[i]);
      const early = await fed.connect(P[remaining[0]]).acknowledge.estimateGas();
      for (const i of remaining.slice(1)) await (await fed.connect(P[i]).acknowledge()).wait();
      const rc = await (await fed.connect(P[remaining[0]]).acknowledge()).wait();
      expect(await fed.state(), c.name).to.equal(BigInt(State.Settled));
      if (c.name === "E3 figures, normal") expect(rc.gasUsed, `${c.name}: settling vs early estimate`).to.be.greaterThan(early + 100_000n);
      expect(rc.gasUsed, c.name).to.be.lessThan(BigInt(ACK_GAS_FLOOR));
    }
  });
});
