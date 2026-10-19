// run seed k (offchain/full/run-seed.mjs): the members' seed secrets are fixed by k, distinct per member and per k.
// Usage (from contracts/):  node --test offchain/full/test/*.test.mjs
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { keccak256, toUtf8Bytes } from "ethers";

import { runSeedSecret } from "../run-seed.mjs";

describe("run seed", () => {
  it("is keccak256 of the tagged run seed and member, a uint256", () => {
    assert.equal(runSeedSecret(1, "A"), BigInt(keccak256(toUtf8Bytes("vfl-run-seed-v1:1:A"))));
    assert.equal(runSeedSecret("3", "p2"), runSeedSecret(3, "p2"));
    assert.ok(runSeedSecret(5, "p3") < 2n ** 256n);
  });
  it("differs per member and per run", () => {
    const all = [1, 2, 3, 4, 5].flatMap((k) => ["A", "p1", "p2", "p3"].map((m) => runSeedSecret(k, m)));
    assert.equal(new Set(all).size, 20);
  });
  it("refuses a malformed run seed or member", () => {
    assert.throws(() => runSeedSecret("x", "A"), /non-negative integer/);
    assert.throws(() => runSeedSecret(1, "arbiter"), /member/);
  });
});
