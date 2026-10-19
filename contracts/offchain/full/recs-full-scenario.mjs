// full-data (Task 2, items 3-4): the scenario of the five full-data on-chain runs, drawn with a recorded seed, in
// genScenario()'s shape (data.mjs) so that party.mjs runs unchanged on it; built the way data1/recs-scenario.mjs builds
// the 1,020-household subset, with the full-data sizes (the student's decisions, 2026-10-04):
//   A (power company): every cleaned household, |X_A| = 15,965 (data/recs/prepared/, prep.py), so |X_A| <= 2^14
//   each passive: misses 1,597 of A's households, drawn from a common pool of 1,722 of them (the subset's 102 of 110 of
//     1,020, scaled by 15,965 / 1,020 and rounded half up), plus 1,266 extra households A does not have, drawn
//     independently from the 2,531 households with an imputed electricity amount (prepared/passive_extra_f32.bin,
//     data/recs/prep_extra.py); the private set intersection drops them
//   |I| = 15,965 - |union of the missing sets|, in [14,243, 14,368]
// Rows: A's 8 int16 columns and tier (prepared/), each passive's encoded columns (its original party assignment) for
// both its cleaned and its extra households. National IDs: distinct 10-digit numbers drawn with the same seed, one per
// household used. One draw (seed 20265000) serves all five runs.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { prng } from "../data.mjs";
import { PASSIVE_NAMES, PREPARED, openPrepared } from "../data1/prepared.mjs";

export const FULL_SEED = 20265000;
export const FULL_SIZES = { pool: 1722, missing: 1597, extra: 1266 };

function shuffled(rng, arr) { // as data1/recs-scenario.mjs
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) { const j = Number(rng.u64() % BigInt(i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

/** The extra households' manifest and checked binary: { e (manifest), X (rows x W float32) }. */
export function openExtra() {
  const e = JSON.parse(readFileSync(resolve(PREPARED, "extra_manifest.json"), "utf8"));
  const buf = readFileSync(resolve(PREPARED, e.passive.file));
  const got = createHash("sha256").update(buf).digest("hex");
  if (got !== e.passive.sha256) throw new Error(`${e.passive.file}: SHA-256 ${got} != manifest ${e.passive.sha256}; rerun data/recs/prep_extra.py`);
  const X = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  if (X.length !== e.rows * e.passive.width) throw new Error("passive_extra_f32.bin does not match its manifest's shape");
  return { e, X };
}

export function recsFullScenario({ seed = FULL_SEED, pool = FULL_SIZES.pool, missing = FULL_SIZES.missing, extra = FULL_SIZES.extra } = {}) {
  const { m, A, P, W } = openPrepared();
  const { e, X } = openExtra();
  if (e.passive.width !== W) throw new Error(`extra width ${e.passive.width} != prepared width ${W}`);
  const rng = prng(BigInt(seed));
  const aPos = Array.from({ length: m.rows }, (_, k) => k); // A holds every cleaned household
  const poolPos = shuffled(rng, aPos).slice(0, pool);
  const missingPos = PASSIVE_NAMES.map(() => new Set(shuffled(rng, poolPos).slice(0, missing)));
  const extraPos = PASSIVE_NAMES.map(() => shuffled(rng, Array.from({ length: e.rows }, (_, k) => k)).slice(0, extra));
  // one distinct 10-digit national ID per household used: A's in DOEID order, then the extra ones in DOEID order
  const extraUsed = [...new Set(extraPos.flat())].sort((x, y) => x - y);
  const idOf = new Map(), idOfExtra = new Map();
  const taken = new Set();
  const draw = () => {
    let id;
    do { id = 1_000_000_000n + (rng.u64() % 9_000_000_000n); } while (taken.has(id));
    taken.add(id);
    return id;
  };
  for (const k of aPos) idOf.set(k, draw());
  for (const k of extraUsed) idOfExtra.set(k, draw());
  const xa = (k) => Array.from(A.subarray(k * 8, k * 8 + 8));
  const cols = PASSIVE_NAMES.map((n) => m.parties[n]);
  if (cols.some((c, i) => JSON.stringify(c) !== JSON.stringify(e.parties[PASSIVE_NAMES[i]]))) throw new Error("extra party columns differ");
  const xp = (i, k) => cols[i].map((j) => P[k * W + j]);
  const xpExtra = (i, k) => cols[i].map((j) => X[k * W + j]);
  const aIds = aPos.map((k) => idOf.get(k));
  const passives = PASSIVE_NAMES.map((_, i) => {
    const heldA = aPos.filter((k) => !missingPos[i].has(k));
    const rows = new Map([...heldA.map((k) => [idOf.get(k), xp(i, k)]), ...extraPos[i].map((k) => [idOfExtra.get(k), xpExtra(i, k)])]);
    return { ids: shuffled(rng, [...rows.keys()]), rows };
  });
  const aSet = new Set(aIds);
  const inAll = aPos.filter((k) => missingPos.every((s) => !s.has(k))).map((k) => idOf.get(k)).sort((x, y) => (x < y ? -1 : 1));
  return {
    A: { ids: shuffled(rng, aIds), rows: new Map(aPos.map((k) => [idOf.get(k), { xa: xa(k), y: m.y[k] }])) },
    passives, I: inAll, pairwise: passives.map((ps) => ps.ids.filter((x) => aSet.has(x))),
    classBoundaries: m.cuts_kwh.map((c) => BigInt(c)),
    sample: { scenario: "recs-full", seed, sizeA: m.rows, pool, missing, extra, extraPoolSize: e.rows,
      doeidPool: poolPos.map((k) => m.doeid[k]), doeidMissing: missingPos.map((s) => [...s].map((k) => m.doeid[k])),
      doeidExtra: extraPos.map((x) => x.map((k) => e.doeid[k])), widths: cols.map((c) => c.length),
      prepared: { passiveSha256: m.passive.sha256, aSha256: m.a.sha256, extraSha256: e.passive.sha256 } },
  };
}
