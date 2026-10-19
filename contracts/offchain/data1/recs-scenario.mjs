// DATA1 §5, E3: the scenario of the on-chain run, drawn from the prepared RECS households with a
// recorded seed, in genScenario()'s shape (data.mjs) so that party.mjs runs unchanged on it.
//   A (power company): 1,020 households (|X_A| <= 2^maxDepth = 1,024)
//   each passive: a random 90 % of A's households (it misses 102 of them, drawn from a common pool of
//     110 of A's households, QUESTIONS-DATA1 Q3) plus 400 households outside A, so PSI is non-trivial
//   |I| = 1,020 - |union of the missing sets|, in [910, 918]
// Rows: A's 8 int16 columns and tier (prepared/, §2), each passive's encoded columns (its original
// party assignment). National IDs: distinct 10-digit numbers drawn with the same seed, one per
// household used. Only the sampled rows are read (the party processes run next to the prover).
import { prng } from "../data.mjs";
import { PASSIVE_NAMES, openPrepared } from "./prepared.mjs";

export const E3_SEED = 20265000;

function shuffled(rng, arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) { const j = Number(rng.u64() % BigInt(i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

export function recsScenario({ seed = E3_SEED, sizeA = 1020, pool = 110, missing = 102, extra = 400 } = {}) {
  const { m, A, P, W } = openPrepared();
  const rng = prng(BigInt(seed));
  const order = shuffled(rng, Array.from({ length: m.rows }, (_, k) => k));
  const aPos = order.slice(0, sizeA);
  const outside = order.slice(sizeA);
  const poolPos = shuffled(rng, aPos).slice(0, pool);
  const missingPos = PASSIVE_NAMES.map(() => new Set(shuffled(rng, poolPos).slice(0, missing)));
  const extraPos = PASSIVE_NAMES.map(() => shuffled(rng, outside).slice(0, extra));
  // one distinct 10-digit national ID per household used
  const used = [...new Set([...aPos, ...extraPos.flat()])].sort((x, y) => x - y);
  const idOf = new Map();
  const taken = new Set();
  for (const k of used) {
    let id;
    do { id = 1_000_000_000n + (rng.u64() % 9_000_000_000n); } while (taken.has(id));
    taken.add(id);
    idOf.set(k, id);
  }
  const xa = (k) => Array.from(A.subarray(k * 8, k * 8 + 8));
  const cols = PASSIVE_NAMES.map((n) => m.parties[n]);
  const xp = (i, k) => cols[i].map((j) => P[k * W + j]);
  const aIds = aPos.map((k) => idOf.get(k));
  const passives = PASSIVE_NAMES.map((_, i) => {
    const held = [...aPos.filter((k) => !missingPos[i].has(k)), ...extraPos[i]];
    return { ids: shuffled(rng, held.map((k) => idOf.get(k))), rows: new Map(held.map((k) => [idOf.get(k), xp(i, k)])) };
  });
  const aSet = new Set(aIds);
  const inAll = aPos.filter((k) => missingPos.every((s) => !s.has(k))).map((k) => idOf.get(k)).sort((x, y) => (x < y ? -1 : 1));
  return {
    A: { ids: shuffled(rng, aIds), rows: new Map(aPos.map((k) => [idOf.get(k), { xa: xa(k), y: m.y[k] }])) },
    passives, I: inAll, pairwise: passives.map((ps) => ps.ids.filter((x) => aSet.has(x))),
    classBoundaries: m.cuts_kwh.map((c) => BigInt(c)),
    sample: { seed, sizeA, pool, missing, extra, doeidA: aPos.map((k) => m.doeid[k]),
      doeidMissing: missingPos.map((s) => [...s].map((k) => m.doeid[k])), doeidExtra: extraPos.map((e) => e.map((k) => m.doeid[k])),
      widths: cols.map((c) => c.length) },
  };
}
