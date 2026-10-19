import { expect } from "chai";
import { network } from "hardhat";

import { big, loadVectors, propertyViolations } from "./helpers/talmud.js";

// Vectors and expected awards come from py/talmud_vectors.py (exact Fraction reference).
const vectors = loadVectors();
const { ethers } = await network.create();

describe("S4 TalmudDivision", function () {
  let talmud: Awaited<ReturnType<typeof ethers.deployContract>>;

  before(async function () {
    talmud = await ethers.deployContract("TalmudDivision");
  });

  it(`matches the exact reference on all ${vectors.cases.length} division cases`, async function () {
    const mismatches: string[] = [];
    for (const c of vectors.cases) {
      const got: bigint[] = [...(await talmud.divide(c.estate, c.claims))];
      if (got.join(",") !== c.expected.join(",")) mismatches.push(`${c.id}: got ${got} want ${c.expected}`);
    }
    expect(mismatches).to.deep.equal([]);
  });

  it("covers all three regimes, with at least 100 random cases", function () {
    const random = vectors.cases.filter((c) => c.kind === "random");
    expect(random.length).to.be.at.least(100);
    for (const r of ["surplus", "cea", "cel"]) {
      expect(random.some((c) => c.regime === r), r).to.equal(true);
    }
  });

  it("satisfies every property on every contract output", async function () {
    const failures: string[] = [];
    for (const c of [...vectors.cases, ...vectors.gas]) {
      const awards: bigint[] = [...(await talmud.divide(c.estate, c.claims))];
      const v = propertyViolations(BigInt(c.estate), big(c.claims), awards);
      if (v.length > 0) failures.push(`${c.id}: ${v.join(", ")}`);
    }
    expect(failures).to.deep.equal([]);
  });

  it("reproduces the Aumann-Maschler (1985) table, floored", async function () {
    expect([...(await talmud.divide(100, [100, 200, 300]))]).to.deep.equal([33n, 33n, 33n]);
    expect([...(await talmud.divide(200, [100, 200, 300]))]).to.deep.equal([50n, 75n, 75n]);
    expect([...(await talmud.divide(300, [100, 200, 300]))]).to.deep.equal([50n, 100n, 150n]);
  });

  it("matches the reference on budget-capped cases (D56/D65)", async function () {
    for (const c of vectors.capped) {
      const [awards, estate] = await talmud.divideCapped(c.rho, c.pool, c.estatePerf, c.claimsPerf);
      expect(estate, c.id).to.equal(BigInt(c.expectedEstate));
      expect([...awards].join(","), c.id).to.equal(c.expected.join(","));
    }
  });

  it("returns an empty award vector for no claimants", async function () {
    expect([...(await talmud.divide(10, []))]).to.deep.equal([]);
  });

  it("rejects claims that are not in non-decreasing order, at the first offending index", async function () {
    await expect(talmud.divide(200, [300, 200, 100]))
      .to.be.revertedWithCustomError(talmud, "ClaimsNotSorted").withArgs(1);
    await expect(talmud.divide(200, [100, 200, 300, 250]))
      .to.be.revertedWithCustomError(talmud, "ClaimsNotSorted").withArgs(3);
    await expect(talmud.divideCapped(10n ** 15n, 10n ** 19n, 500, [420, 180, 75]))
      .to.be.revertedWithCustomError(talmud, "ClaimsNotSorted").withArgs(1);
  });

  it("accepts equal neighbours (non-decreasing, not strictly increasing)", async function () {
    expect([...(await talmud.divide(10, [0, 7, 7, 7]))]).to.deep.equal([0n, 3n, 3n, 3n]);
  });
});
