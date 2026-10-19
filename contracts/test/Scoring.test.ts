// S2b: scoring figures verified on-chain against the run's registered commitments.
// Uses real proofs from circuits/scripts/prove-members.mjs --export-all
// (results/s2b/proofs/N64-I1000/) and the verifier MembersVerifierN64I1000.
import { expect } from "chai";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { network } from "hardhat";

import { signFinal } from "./helpers/reconciliation.js";
import { REPO_ROOT } from "./helpers/talmud.js";

const DIR = resolve(REPO_ROOT, "results", "s2b", "proofs", "N64-I1000");
const HAVE_PROOFS = existsSync(resolve(DIR, "fused-other-fA.json"));
const load = (name: string) => JSON.parse(readFileSync(resolve(DIR, `${name}.json`), "utf8"));
const { ethers, networkHelpers } = await network.create();
const hex32 = (v: string | bigint) => ethers.toBeHex(BigInt(v), 32);
const FIGURES = ["fused", "loo1", "loo2", "loo3", "baseline"];

describe("S2b Scoring (N = 64, |I| = 1,000)", function () {
  let active: any, mci: any, digikala: any, maker: any;
  let run: any, scoring: any, pub: string[];

  before(function () {
    if (!HAVE_PROOFS) this.skip();
  });

  /** Deploy a run with the proofs' seed, register everything except `skip`, end in the commit window. */
  async function setupRun(opts: { mciRoot?: string; skipBottomModel?: boolean } = {}) {
    [active, mci, digikala, maker] = await ethers.getSigners();
    pub = load("fused").publicSignals; // [C_model, C_fA, gammaA, g1, g2, g3, seed, mask, F]
    const verifier = await ethers.deployContract("MembersVerifierN64I1000");
    const now = await ethers.provider.getBlockNumber();
    run = await ethers.deployContract("Reconciliation", [active.address,
      [mci.address, digikala.address, maker.address], 10n ** 20n, BigInt(pub[6]), now + 4, now + 5, now + 40]);
    scoring = await ethers.deployContract("Scoring", [await run.getAddress(), await verifier.getAddress()]);
    await run.connect(active).registerTableCommitment(hex32(pub[2]));
    const cur = await ethers.provider.getBlockNumber();
    if (cur + 1 < now + 5) await networkHelpers.mine(now + 5 - cur - 1);
    if (!opts.skipBottomModel) await run.connect(active).registerBottomModel(hex32(pub[1]));
    await run.connect(mci).registerCommitment(opts.mciRoot ?? hex32(pub[3]));
    await run.connect(digikala).registerCommitment(hex32(pub[4]));
    await run.connect(maker).registerCommitment(hex32(pub[5]));
  }
  const closeWindow = async () => {
    const end = Number(await run.commitWindowEnd());
    const cur = await ethers.provider.getBlockNumber();
    if (cur < end) await networkHelpers.mine(end - cur);
  };
  function submit(name: string, o: { mask?: number; F?: bigint; from?: any } = {}) {
    const p = load(name);
    const [a, b, c, ps] = p.calldata;
    return scoring.connect(o.from ?? active).submitFigure(o.mask ?? p.mask, ps[0], o.F ?? BigInt(ps[8]), a, b, c);
  }

  it("rejects figures while the commit window is still open", async function () {
    await setupRun();
    await expect(submit("fused")).to.be.revertedWithCustomError(scoring, "CommitWindowOpen");
  });

  it("accepts all five figures of a run with one verifier, all against the registered C_fA", async function () {
    await setupRun();
    await closeWindow();
    for (const name of FIGURES) {
      const p = load(name);
      await expect(submit(name)).to.emit(scoring, "FigureAccepted").withArgs(p.mask, BigInt(p.publicSignals[0]), p.F);
      const f = await scoring.figures(p.mask);
      expect(f.set).to.equal(true);
      expect(f.value).to.equal(BigInt(p.F));
    }
  });

  it("rejects a proof made with a different bottom model (C_fA)", async function () {
    await setupRun();
    await closeWindow();
    expect(load("fused-other-fA").publicSignals[1]).to.not.equal(pub[1]);
    await expect(submit("fused-other-fA")).to.be.revertedWithCustomError(scoring, "InvalidProof");
    await expect(submit("fused")).to.emit(scoring, "FigureAccepted");
  });

  it("rejects a wrong F, a mask the proof was not made for, a disallowed mask, a repeat and a non-active caller", async function () {
    await setupRun();
    await closeWindow();
    const F = BigInt(load("fused").F);
    await expect(submit("fused", { F: F + 1n })).to.be.revertedWithCustomError(scoring, "InvalidProof");
    await expect(submit("fused", { mask: 6 })).to.be.revertedWithCustomError(scoring, "InvalidProof");
    await expect(submit("fused", { mask: 1 })).to.be.revertedWithCustomError(scoring, "MaskNotAllowed");
    await expect(submit("fused", { from: mci })).to.be.revertedWithCustomError(scoring, "NotActiveParty");
    await submit("fused");
    await expect(submit("fused")).to.be.revertedWithCustomError(scoring, "AlreadyScored");
  });

  it("uses the binding commitment: after an equivocation proof, the signed root counts", async function () {
    // MCI signed its true root when sending its matrix, but registers a different value.
    const wrong = hex32(BigInt(load("fused").publicSignals[3]) ^ 1n);
    await setupRun({ mciRoot: wrong });
    const trueRoot = hex32(pub[3]);
    const chainId = (await ethers.provider.getNetwork()).chainId;
    const sig = await signFinal(mci, trueRoot, { chainId, verifyingContract: await run.getAddress() });
    await closeWindow();
    await expect(submit("fused")).to.be.revertedWithCustomError(scoring, "InvalidProof");
    await run.connect(active).proveEquivocation(mci.address, trueRoot, sig);
    await expect(submit("fused")).to.emit(scoring, "FigureAccepted");
  });

  it("refuses to score while a run-level commitment is missing", async function () {
    await setupRun({ skipBottomModel: true });
    await closeWindow();
    await expect(submit("fused")).to.be.revertedWithCustomError(scoring, "MissingCommitment");
  });
});
