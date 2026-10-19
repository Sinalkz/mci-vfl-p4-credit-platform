// full-data scenario (offchain/full/recs-full-scenario.mjs): the sizes the student fixed (A holds all 15,965 cleaned
// households; each passive misses 1,597 of a common pool of 1,722 and holds 1,266 of the 2,531 imputed households),
// I = what A and every passive share, distinct national IDs, the rows each party holds, and the same draw for the same
// seed. Usage (from contracts/):  node --test offchain/full/test/*.test.mjs
import assert from "node:assert/strict";
import { before, describe, it } from "node:test";

import { PASSIVE_NAMES, openPrepared } from "../../data1/prepared.mjs";
import { FULL_SEED, openExtra, recsFullScenario } from "../recs-full-scenario.mjs";

describe("recs-full scenario", () => {
  let sc, m, P, W, e, X;
  before(() => {
    sc = recsFullScenario();
    ({ m, P, W } = openPrepared());
    ({ e, X } = openExtra());
  });

  it("A holds every cleaned household, with its rows and tier", () => {
    assert.equal(sc.A.ids.length, 15_965);
    assert.equal(new Set(sc.A.ids).size, 15_965);
    assert.equal(sc.A.rows.size, 15_965);
    assert.deepEqual([...sc.A.rows.values()].map((r) => r.y).sort(), [...m.y].sort());
  });

  it("each passive misses 1,597 of a common pool of 1,722 and holds 1,266 imputed households", () => {
    const s = sc.sample;
    assert.equal(s.seed, FULL_SEED);
    assert.equal(new Set(s.doeidPool).size, 1_722);
    const pool = new Set(s.doeidPool);
    const imputed = new Set(e.doeid);
    const aSet = new Set(sc.A.ids);
    for (let i = 0; i < 3; i++) {
      assert.equal(new Set(s.doeidMissing[i]).size, 1_597);
      assert.ok(s.doeidMissing[i].every((d) => pool.has(d)));
      assert.equal(new Set(s.doeidExtra[i]).size, 1_266);
      assert.ok(s.doeidExtra[i].every((d) => imputed.has(d)));
      const ids = sc.passives[i].ids;
      assert.equal(ids.length, 15_965 - 1_597 + 1_266);
      assert.equal(new Set(ids).size, ids.length);
      assert.equal(sc.pairwise[i].length, 15_965 - 1_597);
      assert.equal(ids.filter((x) => !aSet.has(x)).length, 1_266); // the extras are outside A: PSI drops them
    }
    assert.notDeepEqual(s.doeidExtra[0], s.doeidExtra[1]); // independent draws
  });

  it("I is what A and every passive share, sorted, in [14,243, 14,368]", () => {
    const union = new Set(sc.sample.doeidMissing.flat());
    assert.equal(sc.I.length, 15_965 - union.size);
    assert.ok(sc.I.length >= 14_243 && sc.I.length <= 14_368);
    const sets = sc.passives.map((p) => new Set(p.ids));
    const want = sc.A.ids.filter((x) => sets.every((s) => s.has(x))).sort((a, b) => (a < b ? -1 : 1));
    assert.deepEqual(sc.I, want);
  });

  it("national IDs are distinct 10-digit numbers over every household used", () => {
    const all = new Set([...sc.A.ids, ...sc.passives.flatMap((p) => p.ids)]);
    const extras = new Set(sc.sample.doeidExtra.flat());
    assert.equal(all.size, 15_965 + extras.size);
    for (const id of all) assert.ok(id >= 1_000_000_000n && id < 10_000_000_000n);
  });

  it("a passive's rows are its own columns: prepared rows for A's households, the extra binary for the others", () => {
    const pos = new Map(m.doeid.map((d, k) => [d, k]));
    const epos = new Map(e.doeid.map((d, k) => [d, k]));
    for (let i = 0; i < 3; i++) {
      const cols = m.parties[PASSIVE_NAMES[i]];
      assert.equal(sc.sample.widths[i], cols.length);
      const rows = [...sc.passives[i].rows.values()];
      assert.ok(rows.every((r) => r.length === cols.length));
      // every extra household's row is in the passive's map, as the binary holds it
      const want = new Set(sc.sample.doeidExtra[i].map((d) => JSON.stringify(cols.map((j) => X[epos.get(d) * W + j]))));
      const have = new Set(rows.map((r) => JSON.stringify(r)));
      for (const w of want) assert.ok(have.has(w));
      // a held A household (not in the pool) has its prepared row
      const k = pos.get(m.doeid.find((d) => !sc.sample.doeidPool.includes(d)));
      assert.ok(have.has(JSON.stringify(cols.map((j) => P[k * W + j]))));
    }
  });

  it("the same seed gives the same draw", () => {
    const again = recsFullScenario();
    assert.deepEqual(again.I, sc.I);
    assert.deepEqual(again.sample.doeidExtra, sc.sample.doeidExtra);
    assert.deepEqual(again.passives[2].ids, sc.passives[2].ids);
  });
});
