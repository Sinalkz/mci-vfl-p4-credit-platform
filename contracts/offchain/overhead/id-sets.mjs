// Task 4 item 2: the ID sets of the PSI measurement.
//   subset   E3's own sets (data1/recs-scenario.mjs, seed 20265000): |X_A| = 1,020, each passive 918 of A's + 400 outside
//            A (1,318), |I| = 910.
//   full-*   the full-data setting of the five-run plan (full-data branch, results/full/design.json, read only), with
//            synthetic 10-digit IDs of the design's sizes: |X_A| = 15,965; a common pool of 1,722 of A's IDs, of which
//            each passive misses 1,597 (pairwise intersection 14,368; |I| = 15,965 - |union of the missing sets|,
//            about 14,243); extras outside A drawn from 2,531 IDs (the imputed households):
//              (a) all 2,531 for every passive (identical sets)   -> 16,899 per passive
//              (b) 68 each, independent draws                      -> 14,436 per passive
//              (c) 1,266 each, independent draws                   -> 15,634 per passive
// The PSI's cost depends on the set sizes only (one hash-to-group and scalar multiplications per ID).
import { prng } from "../data.mjs";
import { recsScenario } from "../data1/recs-scenario.mjs";

export const FULL = { sizeA: 15965, pool: 1722, missing: 1597, extraUniverse: 2531 };
export const EXTRAS = { a: 2531, b: 68, c: 1266 };
export const FULL_SEED = 20264000n;
export const SETTINGS = ["subset", "full-a", "full-b", "full-c"];

const sortIds = (ids) => [...ids].sort((x, y) => (x < y ? -1 : x > y ? 1 : 0));

function shuffled(rng, arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) { const j = Number(rng.u64() % BigInt(i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

/** `count` distinct 10-digit IDs not in `taken` (the rule of data.mjs freshIds). */
function freshIds(rng, count, taken) {
  const out = [];
  while (out.length < count) {
    const id = 1_000_000_000n + (rng.u64() % 9_000_000_000n);
    if (taken.has(id)) continue;
    taken.add(id);
    out.push(id);
  }
  return out;
}

/** { aIds, passiveIds: [3 lists], I (sorted) } for extras option "a" | "b" | "c". */
export function fullDataSets({ option, seed = FULL_SEED }) {
  if (!(option in EXTRAS)) throw new Error(`extras option ${option}: a, b or c`);
  const rng = prng(seed);
  const taken = new Set();
  const aIds = freshIds(rng, FULL.sizeA, taken);
  const universe = freshIds(rng, FULL.extraUniverse, taken);
  const pool = shuffled(rng, aIds).slice(0, FULL.pool);
  const missing = [0, 1, 2].map(() => new Set(shuffled(rng, pool).slice(0, FULL.missing)));
  const extras = [0, 1, 2].map(() => (option === "a" ? universe : shuffled(rng, universe).slice(0, EXTRAS[option])));
  const passiveIds = missing.map((m, i) => shuffled(rng, [...aIds.filter((x) => !m.has(x)), ...extras[i]]));
  const I = sortIds(aIds.filter((x) => missing.every((m) => !m.has(x))));
  return { aIds: shuffled(rng, aIds), passiveIds, I };
}

/** The sets of one PSI setting (SETTINGS). */
export function psiSets(setting) {
  if (setting === "subset") {
    const sc = recsScenario();
    return { aIds: sc.A.ids, passiveIds: sc.passives.map((p) => p.ids), I: sc.I };
  }
  const m = /^full-([abc])$/.exec(setting);
  if (!m) throw new Error(`setting ${setting}: one of ${SETTINGS.join(", ")}`);
  return fullDataSets({ option: m[1] });
}
