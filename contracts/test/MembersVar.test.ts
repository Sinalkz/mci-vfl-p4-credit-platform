// M1b (D106): one verifier, compiled for maxDepth = 10, accepts real proofs for sizeI = 1000
// and sizeI = 700, and rejects either proof under any other public sizeI or F.
// Uses results/m1b/proofs/N64-D10-I*.json (circuits/scripts/m1b-measure.mjs) and the
// verifier MembersVarVerifierN64D10 (circuits/scripts/m1b-build.mjs).
import { expect } from "chai";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { network } from "hardhat";

import { REPO_ROOT } from "./helpers/talmud.js";

const DIR = resolve(REPO_ROOT, "results", "m1b", "proofs");
const SIZES = [1000, 700];
const file = (i: number) => resolve(DIR, `N64-D10-I${i}.json`);
const HAVE = SIZES.every((i) => existsSync(file(i)));
const { ethers } = await network.create();
const SIZE_I = 7; // public signals: C_model, C_fA, gammaA, gammaP[3], seed, sizeI, mask, F
const F_POS = 9;

describe("M1b sizeI-public scoring verifier (N = 64, maxDepth = 10)", function () {
  let verifier: any;
  before(async function () {
    if (!HAVE) this.skip();
    verifier = await ethers.deployContract("MembersVarVerifierN64D10");
  });

  for (const I of SIZES) {
    it(`accepts the proof for sizeI = ${I}, and rejects it under another sizeI or F`, async function () {
      const p = JSON.parse(readFileSync(file(I), "utf8"));
      const [a, b, c, ps] = p.calldata;
      expect(BigInt(ps[SIZE_I])).to.equal(BigInt(I));
      expect(await verifier.verifyProof(a, b, c, ps)).to.equal(true);
      const other = I === 1000 ? 700 : 1000;
      for (const [pos, v] of [[SIZE_I, I + 1], [SIZE_I, I - 1], [SIZE_I, other], [F_POS, BigInt(ps[F_POS]) + 1n]] as const) {
        const bad = [...ps];
        bad[pos] = ethers.toBeHex(BigInt(v), 32);
        expect(await verifier.verifyProof(a, b, c, bad), `position ${pos} = ${v}`).to.equal(false);
      }
    });
  }

  it("both proofs share the seed and the roots' depth, but not the roots (different aligned sets)", async function () {
    const [p1, p2] = SIZES.map((i) => JSON.parse(readFileSync(file(i), "utf8")));
    expect(p1.publicSignals[6]).to.equal(p2.publicSignals[6]);  // seed
    expect(p1.publicSignals[2]).to.not.equal(p2.publicSignals[2]); // gammaA over 1000 vs 700 rows
  });
});
