import { expect } from "chai";
import { Signature, TypedDataEncoder, ZeroHash, concat, toBeHex } from "ethers";
import { network } from "hardhat";

import {
  SNARK_SCALAR_FIELD,
  commitEmbeddings,
  finalMessage,
  signFinal,
  syntheticEmbeddings,
  type SignContext,
} from "./helpers/reconciliation.js";

const { ethers, networkHelpers } = await network.create();
const D_P = 100n * 10n ** 18n;
const SEED = 20260923n;
const SECP256K1_N = 0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141n;
// Schedule relative to the deployment block (block heights, not timestamps).
const TABLE_AFTER = 3;
const WINDOW_START_AFTER = 4;
const WINDOW_END_AFTER = 200;

describe("S3 Reconciliation", function () {
  let active: any, mci: any, digikala: any, maker: any, outsider: any;
  let run: any, runAddr: string, chainId: bigint, ctx: SignContext;
  // gamma = Poseidon Merkle root of the final-round embedding matrix, the value S3 signs.
  // What MCI actually sent (and signed), and a doctored matrix it registers instead.
  let gammaSent: string, gammaDoctored: string;

  before(async function () {
    const sent = syntheticEmbeddings(1);
    const doctored = sent.map((row) => [...row]);
    doctored[17][2] += 1n;
    gammaSent = await commitEmbeddings(sent);
    gammaDoctored = await commitEmbeddings(doctored);
  });

  /** Deploy a run; by default mine forward into its commit window. */
  async function deployRun({ intoWindow = true } = {}) {
    const now = await ethers.provider.getBlockNumber();
    const r = await ethers.deployContract("Reconciliation", [
      active.address, [mci.address, digikala.address, maker.address], D_P, SEED,
      now + TABLE_AFTER, now + WINDOW_START_AFTER, now + WINDOW_END_AFTER,
    ]);
    if (intoWindow) {
      const start = Number(await r.commitWindowStart());
      const cur = await ethers.provider.getBlockNumber();
      if (cur + 1 < start) await networkHelpers.mine(start - cur - 1);
    }
    return r;
  }

  beforeEach(async function () {
    [active, mci, digikala, maker, outsider] = await ethers.getSigners();
    run = await deployRun();
    runAddr = await run.getAddress();
    chainId = (await ethers.provider.getNetwork()).chainId;
    ctx = { chainId, verifyingContract: runAddr };
  });

  it("computes the same EIP-712 hash as ethers' TypedDataEncoder", async function () {
    const m = finalMessage(mci.address, gammaSent, ctx);
    expect(await run.finalMessageHash(mci.address, gammaSent)).to.equal(
      TypedDataEncoder.hash(m.domain, m.types, m.value),
    );
  });

  it("registers a commitment once, only for passive parties, and binds it", async function () {
    await expect(run.connect(mci).registerCommitment(gammaSent))
      .to.emit(run, "CommitmentRegistered").withArgs(mci.address, gammaSent);
    expect(await run.bindingCommitment(mci.address)).to.equal(gammaSent);
    await expect(run.connect(mci).registerCommitment(gammaDoctored))
      .to.be.revertedWithCustomError(run, "AlreadyRegistered");
    await expect(run.connect(outsider).registerCommitment(gammaSent))
      .to.be.revertedWithCustomError(run, "NotPassiveParty");
    await expect(run.connect(active).registerCommitment(gammaSent))
      .to.be.revertedWithCustomError(run, "NotPassiveParty");
    await expect(run.connect(digikala).registerCommitment(ZeroHash))
      .to.be.revertedWithCustomError(run, "ZeroCommitment");
  });

  // ---- The five cases required by docs/SPIKE.md (S3) -------------------------

  it("(1) proves a valid equivocation: records it, forfeits D_p, binds the signed root", async function () {
    const sig = await signFinal(mci, gammaSent, ctx);    // signed when sending h_i
    await run.connect(mci).registerCommitment(gammaDoctored); // registers something else
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, sig))
      .to.emit(run, "EquivocationProven").withArgs(mci.address, gammaDoctored, gammaSent, D_P);
    const m = await run.members(mci.address);
    expect(m.equivocated).to.equal(true);
    expect(m.deposit).to.equal(0n);
    expect(await run.forfeitedToEscrow()).to.equal(D_P);
    expect(await run.bindingCommitment(mci.address)).to.equal(gammaSent);
    expect(await run.registeredCommitment(mci.address)).to.equal(gammaDoctored); // kept for the record
    expect((await run.members(digikala.address)).deposit).to.equal(D_P);       // others untouched
  });

  it("(2) rejects a forged signature (wrong key, or corrupted bytes)", async function () {
    await run.connect(mci).registerCommitment(gammaDoctored);
    const byOther = await signFinal(outsider, gammaSent, { ...ctx }); // outsider signs its own message
    const forgedMsg = finalMessage(mci.address, gammaSent, ctx);
    const byOtherForMci = await outsider.signTypedData(forgedMsg.domain, forgedMsg.types, forgedMsg.value);
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, byOther))
      .to.be.revertedWithCustomError(run, "SignerMismatch");
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, byOtherForMci))
      .to.be.revertedWithCustomError(run, "SignerMismatch").withArgs(outsider.address);

    const good = await signFinal(mci, gammaSent, ctx);
    const flipped = good.slice(0, 20) + (good[20] === "0" ? "1" : "0") + good.slice(21); // flip a nibble of r
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, flipped)).to.revert(ethers);
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, good.slice(0, 130)))
      .to.be.revertedWithCustomError(run, "ECDSAInvalidSignatureLength");
    // Malleated twin (s -> n - s, v flipped): same signer mathematically, rejected as non-canonical.
    const s = Signature.from(good);
    const highS = concat([s.r, toBeHex(SECP256K1_N - BigInt(s.s), 32), s.v === 27 ? "0x1c" : "0x1b"]);
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, highS))
      .to.be.revertedWithCustomError(run, "ECDSAInvalidSignatureS");
    expect((await run.members(mci.address)).equivocated).to.equal(false);
  });

  it("(3) rejects a signature made for another run (replay), round or chain", async function () {
    await run.connect(mci).registerCommitment(gammaDoctored);
    const otherRun = await deployRun();
    const otherAddr = await otherRun.getAddress();

    // Replay: MCI signed gammaSent in another run; the active party submits it here.
    const sigOtherRun = await signFinal(mci, gammaSent, { chainId, verifyingContract: otherAddr });
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, sigOtherRun))
      .to.be.revertedWithCustomError(run, "SignerMismatch");
    // Struct field `run` pointing elsewhere while the domain is correct.
    const sigRunField = await signFinal(mci, gammaSent, { ...ctx, run: otherAddr });
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, sigRunField))
      .to.be.revertedWithCustomError(run, "SignerMismatch");
    // Not the final round.
    const sigRound = await signFinal(mci, gammaSent, { ...ctx, round: 7n });
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, sigRound))
      .to.be.revertedWithCustomError(run, "SignerMismatch");
    // Another chain.
    const sigChain = await signFinal(mci, gammaSent, { ...ctx, chainId: chainId + 1n });
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, sigChain))
      .to.be.revertedWithCustomError(run, "SignerMismatch");

    // The same signature *is* valid in the run it was made for.
    await otherRun.connect(mci).registerCommitment(gammaDoctored);
    await expect(otherRun.connect(active).proveEquivocation(mci.address, gammaSent, sigOtherRun))
      .to.emit(otherRun, "EquivocationProven");
  });

  it("(4) rejects when the signed root equals the registered one (no equivocation)", async function () {
    await run.connect(mci).registerCommitment(gammaSent);
    const sig = await signFinal(mci, gammaSent, ctx);
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, sig))
      .to.be.revertedWithCustomError(run, "NoEquivocation");
    expect((await run.members(mci.address)).deposit).to.equal(D_P);
  });

  it("(5) rejects a caller other than the active party", async function () {
    await run.connect(mci).registerCommitment(gammaDoctored);
    const sig = await signFinal(mci, gammaSent, ctx);
    for (const caller of [mci, digikala, outsider]) {
      await expect(run.connect(caller).proveEquivocation(mci.address, gammaSent, sig))
        .to.be.revertedWithCustomError(run, "NotActiveParty");
    }
  });

  // ---- Guards beyond the spec ---------------------------------------------------

  it("rejects a proof against a party that never registered, and a second proof", async function () {
    const sig = await signFinal(mci, gammaSent, ctx);
    await expect(run.connect(active).proveEquivocation(mci.address, gammaSent, sig))
      .to.be.revertedWithCustomError(run, "NotRegistered");

    await run.connect(mci).registerCommitment(gammaDoctored);
    await run.connect(active).proveEquivocation(mci.address, gammaSent, sig);
    const other = await commitEmbeddings(syntheticEmbeddings(99));
    const sig2 = await signFinal(mci, other, ctx);
    await expect(run.connect(active).proveEquivocation(mci.address, other, sig2))
      .to.be.revertedWithCustomError(run, "AlreadyProven");
  });

  // ---- S2b additions: block-height windows, field elements, gamma_A and C_fA -------

  it("accepts gamma_i and C_fA only inside the commit window (block heights)", async function () {
    const early = await deployRun({ intoWindow: false });
    await expect(early.connect(mci).registerCommitment(gammaSent))
      .to.be.revertedWithCustomError(early, "OutsideWindow");
    await expect(early.connect(active).registerBottomModel(gammaSent))
      .to.be.revertedWithCustomError(early, "OutsideWindow");

    await expect(run.connect(active).registerBottomModel(gammaDoctored))
      .to.emit(run, "BottomModelCommitted").withArgs(gammaDoctored);
    await expect(run.connect(active).registerBottomModel(gammaSent))
      .to.be.revertedWithCustomError(run, "AlreadyRegistered");
    await expect(run.connect(mci).registerBottomModel(gammaSent))
      .to.be.revertedWithCustomError(run, "NotActiveParty");

    const end = Number(await run.commitWindowEnd());
    await networkHelpers.mine(end - (await ethers.provider.getBlockNumber()));
    await expect(run.connect(mci).registerCommitment(gammaSent))
      .to.be.revertedWithCustomError(run, "OutsideWindow");
  });

  it("accepts gamma_A only from the active party, once, by the table deadline", async function () {
    const fresh = await deployRun({ intoWindow: false });
    await expect(fresh.connect(mci).registerTableCommitment(gammaSent))
      .to.be.revertedWithCustomError(fresh, "NotActiveParty");
    await expect(fresh.connect(active).registerTableCommitment(gammaSent))
      .to.emit(fresh, "TableCommitted").withArgs(gammaSent);
    await expect(fresh.connect(active).registerTableCommitment(gammaDoctored))
      .to.be.revertedWithCustomError(fresh, "AlreadyRegistered");
    await expect(run.connect(active).registerTableCommitment(gammaSent)) // `run` is past its deadline
      .to.be.revertedWithCustomError(run, "OutsideWindow");
  });

  it("rejects commitments that are not BN254 field elements, and a bad schedule", async function () {
    const tooBig = toBeHex(SNARK_SCALAR_FIELD, 32);
    await expect(run.connect(mci).registerCommitment(tooBig))
      .to.be.revertedWithCustomError(run, "NotAFieldElement");
    await run.connect(mci).registerCommitment(gammaDoctored);
    const sig = await signFinal(mci, tooBig, ctx);
    await expect(run.connect(active).proveEquivocation(mci.address, tooBig, sig))
      .to.be.revertedWithCustomError(run, "NotAFieldElement");
    await expect(ethers.deployContract("Reconciliation", [active.address, [mci.address], D_P, SEED, 10, 10, 20]))
      .to.be.revertedWithCustomError(run, "InvalidSchedule");
  });
});
