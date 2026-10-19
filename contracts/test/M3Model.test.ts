// M3 §2: the scenario data (Q6 sizes), the model configuration, and the fixed-point model
// the M1b circuit evaluates. The JS integer forward pass and F must equal the Python oracle
// (py/s2b_oracle.py, which S2b/M1b use) exactly.
import { expect } from "chai";
import { readFileSync } from "node:fs";
import { keccak256 } from "ethers";

// @ts-ignore -- plain ESM modules
import { genScenario, TIER_PRIOR } from "../offchain/data.mjs";
// @ts-ignore
import { CONFIG, CONFIG_PATH, forwardInt, macroF1, modelConfigHash, newFA, newFusion, quantFA, quantHead, softmaxCE } from "../offchain/model.mjs";
// @ts-ignore
import { prng } from "../offchain/data.mjs";
// @ts-ignore
import { oracle } from "../../circuits/scripts/members-lib.mjs";

describe("M3 model and data", function () {
  it("M01 scenario: |X_A| = 1,020 <= 2^10, |I| = 1,000, 5 tiers with the high ones rare", function () {
    const s = genScenario();
    expect(s.A.ids.length).to.equal(1020);
    expect(s.I.length).to.equal(1000);
    expect(s.pairwise.map((p: any[]) => p.length)).to.deep.equal([1010, 1005, 1005]);
    const counts = [0, 0, 0, 0, 0];
    for (const id of s.I) counts[s.A.rows.get(id).y]++;
    expect(counts.every((c) => c > 0)).to.equal(true);
    expect(counts[4]).to.be.lessThan(counts[0] / 3); // tier 4 is rare
    expect(TIER_PRIOR[4]).to.equal(0.05);
    // each passive holds 6 feature columns for every one of its IDs
    for (const p of s.passives) expect(p.ids.every((id: bigint) => p.rows.get(id).length === 6)).to.equal(true);
  });

  it("M02 modelConfigHash is keccak256 of the configuration file; T, batch size and wait time are in it", function () {
    expect(modelConfigHash()).to.equal(keccak256(readFileSync(CONFIG_PATH)));
    for (const k of ["T", "batchSize", "learningRate", "optimizer", "perBatchWaitMs", "architecture"]) expect(CONFIG).to.have.property(k);
  });

  it("M03 the JS fixed-point forward pass and F equal the Python oracle for every mask", function () {
    const rng = prng(7n);
    const g = () => (rng.float() - 0.5) * 4;
    // A briefly trained f_A and fusion model, so predictions are not constant.
    const fA = newFA();
    const fusion = newFusion();
    const n = 64;
    const xa = Array.from({ length: n }, () => Array.from({ length: 8 }, () => Math.round(g() * 4096)));
    const hp = Array.from({ length: n }, () => [0, 1, 2].map(() => Array.from({ length: 4 }, () => Math.round(g() * 4096))));
    const y = Array.from({ length: n }, (_, r) => (xa[r][0] > 0 ? 1 : 0) + (hp[r][0][0] > 0 ? 2 : 0) + (hp[r][2][1] > 3000 ? 1 : 0));
    for (let ep = 0; ep < 50; ep++) {
      const eA = fA.forward(Float64Array.from(xa.flat().map((v) => v / 4096)), n);
      const inp = new Float64Array(n * 16);
      for (let r = 0; r < n; r++) {
        for (let e = 0; e < 4; e++) inp[r * 16 + e] = eA[r * 4 + e];
        for (let p = 0; p < 3; p++) for (let e = 0; e < 4; e++) inp[r * 16 + 4 + 4 * p + e] = hp[r][p][e] / 4096;
      }
      const dInp = fusion.backward(softmaxCE(fusion.forward(inp, n), y, n).dZ);
      const deA = new Float64Array(n * 4);
      for (let r = 0; r < n; r++) for (let e = 0; e < 4; e++) deA[r * 4 + e] = dInp[r * 16 + e];
      fA.backward(deA);
    }
    const fa = quantFA(fA);
    const head = quantHead(fusion);
    for (const mask of [7, 6, 5, 3, 0]) {
      const preds = xa.map((x, r) => forwardInt(fa, head, mask, x, hp[r]));
      const py = oracle({ fA: fa, head, mask, rows: xa.map((x, r) => ({ y: y[r], xa: x, hp: hp[r] })) });
      expect(preds).to.deep.equal(py.preds);
      expect(macroF1(y, preds).F).to.equal(py.F);
    }
  });
});
