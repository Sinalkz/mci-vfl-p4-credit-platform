// Task 4: the ID sets of the PSI measurement (subset: E3's own sets; full data: synthetic IDs of the design's sizes).
//   node --test offchain/overhead/test/        (from contracts/)
import assert from "node:assert/strict";
import { test } from "node:test";

import { FULL, fullDataSets, psiSets } from "../id-sets.mjs";

const inter = (a, b) => { const s = new Set(b); return a.filter((x) => s.has(x)); };

for (const [option, extras] of [["a", 2531], ["b", 68], ["c", 1266]]) {
  test(`S1 full data, extras option (${option}): the design's set sizes`, () => {
    const s = fullDataSets({ option });
    assert.equal(s.aIds.length, FULL.sizeA);
    assert.equal(new Set(s.aIds).size, FULL.sizeA);
    for (const p of s.passiveIds) {
      assert.equal(new Set(p).size, p.length);
      assert.equal(p.length, FULL.sizeA - FULL.missing + extras);
      assert.equal(inter(s.aIds, p).length, FULL.sizeA - FULL.missing); // 14,368 pairwise
    }
    const held = s.passiveIds.map((p) => new Set(p));
    const inAll = s.aIds.filter((x) => held.every((h) => h.has(x)));
    assert.equal(inAll.length, s.I.length);
    assert.ok(s.I.length >= FULL.sizeA - FULL.pool && s.I.length <= FULL.sizeA - FULL.missing);
    assert.deepEqual(s.I, [...s.I].sort((x, y) => (x < y ? -1 : x > y ? 1 : 0)));
    for (const id of s.aIds) assert.ok(id >= 1_000_000_000n && id <= 9_999_999_999n);
  });
}

test("S2 extras rules: (a) the same 2,531 for every passive, (b) and (c) independent draws from those 2,531", () => {
  const a = fullDataSets({ option: "a" });
  const A = new Set(a.aIds);
  const ex = a.passiveIds.map((p) => p.filter((x) => !A.has(x)).sort());
  assert.deepEqual(ex[0], ex[1]);
  assert.deepEqual(ex[1], ex[2]);
  for (const option of ["b", "c"]) {
    const s = fullDataSets({ option });
    const universe = new Set(ex[0]);
    const exs = s.passiveIds.map((p) => p.filter((x) => !A.has(x)));
    for (const e of exs) for (const x of e) assert.ok(universe.has(x));
    assert.notDeepEqual([...exs[0]].sort(), [...exs[1]].sort());
  }
});

test("S3 deterministic for a seed, different for another", () => {
  assert.deepEqual(fullDataSets({ option: "b" }).I, fullDataSets({ option: "b" }).I);
  assert.notDeepEqual(fullDataSets({ option: "b", seed: 7n }).aIds, fullDataSets({ option: "b" }).aIds);
});

test("S4 psiSets('subset') is E3's scenario: 1,020 at A, 918 + 400 at each passive, |I| = 910", () => {
  const s = psiSets("subset");
  assert.equal(s.aIds.length, 1020);
  for (const p of s.passiveIds) { assert.equal(p.length, 1318); assert.equal(inter(s.aIds, p).length, 918); }
  assert.equal(s.I.length, 910);
});
