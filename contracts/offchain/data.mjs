// M2 §6, M3 §2: synthetic, scenario-shaped data with a fixed seed (declared as synthetic in
// the thesis). National IDs are 10-digit numbers held as uint64.
//
// Every ID in the universe has a latent tariff tier y in 0..4 drawn from TIER_PRIOR (the
// high-consumption tiers 3 and 4 are rare). Features depend on the tier:
//   A (power company): 8 features, class centroid * ACTIVE_SIGNAL + N(0, 1), stored as int16
//     at scale 2^12 (the S2b/M1b row format), and the label y;
//   passive p: its own FP = 6 feature columns, centroid * PASSIVE_SIGNAL[p] + N(0, 1), floats.
//
// Default overlap design (M2 Q6 answer: |X_A| <= 1,024; each of 3 passives 1,500; |I| = 1,000):
//   core      1,000 IDs held by A and every passive           -> I
//   E1           10 held by A and passive 1 only              -> |A ∩ p1| = 1,010
//   E2            5 held by A, passive 2 and passive 3        -> |A ∩ p2| = |A ∩ p3| = 1,005
//   E3            5 held by A only                            -> |X_A| = 1,020
//   S           200 held by every passive but not by A
//   the rest of each passive's 1,500: IDs held by that passive alone
// so some pairwise intersections are larger than I, and |I| = 1,000 exactly.

export const TIER_PRIOR = [0.35, 0.28, 0.20, 0.12, 0.05];
export const FA = 8;
export const FP = 6;
export const ACTIVE_SIGNAL = 0.5;
export const PASSIVE_SIGNAL = [0.6, 0.45, 0.8]; // MCI, the e-commerce company, device maker

const MASK64 = (1n << 64n) - 1n;

/** splitmix64: a small deterministic generator (test data only, not for secrets). */
export function prng(seed) {
  let s = BigInt(seed) & MASK64;
  const next = () => {
    s = (s + 0x9e3779b97f4a7c15n) & MASK64;
    let z = s;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  };
  return {
    u64: next,
    int: (lo, hi) => lo + Number(next() % BigInt(hi - lo + 1)), // hi - lo small
    float: () => Number(next() >> 11n) / 2 ** 53,
  };
}

/** `count` distinct 10-digit IDs (1,000,000,000 .. 9,999,999,999) not in `taken`. */
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

/** A standard normal draw (Box-Muller on the data PRNG; synthetic data only). */
export function gauss(rng) {
  let u = 0;
  while (u === 0) u = rng.float();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng.float());
}

function tierOf(rng) {
  const u = rng.float();
  let acc = 0;
  for (let k = 0; k < TIER_PRIOR.length; k++) { acc += TIER_PRIOR[k]; if (u < acc) return k; }
  return TIER_PRIOR.length - 1;
}

const centroids = (rng, dims, signal) => TIER_PRIOR.map(() => Array.from({ length: dims }, () => gauss(rng) * signal));

/** A's row: 8 int16 features at scale 2^12 (clipped to the int16 range) and the label. */
function rowFor(rng, y, centA) {
  const xa = centA[y].map((c) => Math.max(-32768, Math.min(32767, Math.round((c + gauss(rng)) * 4096))));
  return { xa, y };
}

/** Deterministic in-place shuffle with the data PRNG (only to avoid IDs grouped by origin). */
function shuffled(rng, arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) { const j = Number(rng.u64() % BigInt(i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

/**
 * The default scenario. Returns
 *   { A: { ids, rows: Map(id -> {xa, y}) }, passives: [{ ids, rows: Map(id -> xp) }...], I, pairwise }.
 * `I` and `pairwise` are the plaintext ground truth, for tests only. Each party process uses
 * only its own part (A: ids and rows with labels; passive p: its ids and feature rows).
 */
export function genScenario({ seed = 20260925n, sizeI = 1000, sizeA = 1020, sizeP = 1500, nPassives = 3,
  e1 = 10, e2 = 5, shared = 200 } = {}) {
  if (nPassives !== PASSIVE_SIGNAL.length) throw new Error(`the scenario has ${PASSIVE_SIGNAL.length} passives`);
  const rng = prng(seed);
  const centA = centroids(rng, FA, ACTIVE_SIGNAL);
  const centP = PASSIVE_SIGNAL.map((sig) => centroids(rng, FP, sig));
  const taken = new Set();
  const core = freshIds(rng, sizeI, taken);
  const E1 = freshIds(rng, e1, taken);
  const E2 = freshIds(rng, e2, taken);
  const E3 = freshIds(rng, sizeA - sizeI - e1 - e2, taken);
  const S = freshIds(rng, shared, taken);
  const aIds = [...core, ...E1, ...E2, ...E3];
  const passiveIds = [];
  for (let p = 0; p < nPassives; p++) {
    const own = [...core, ...S, ...(p === 0 ? E1 : []), ...(p === 1 || p === 2 ? E2 : [])];
    passiveIds.push([...own, ...freshIds(rng, sizeP - own.length, taken)]);
  }
  // Latent tier per ID, then the rows (in a fixed order, so the data depend on the seed only).
  const tier = new Map([...taken].sort((a, b) => (a < b ? -1 : 1)).map((id) => [id, tierOf(rng)]));
  const rows = new Map(aIds.map((id) => [id, rowFor(rng, tier.get(id), centA)]));
  const passives = passiveIds.map((ids, p) => ({
    ids: shuffled(rng, ids),
    rows: new Map(ids.map((id) => [id, centP[p][tier.get(id)].map((c) => c + gauss(rng))])),
  }));
  const aSet = new Set(aIds);
  const pairwise = passives.map((ps) => ps.ids.filter((x) => aSet.has(x)));
  return { A: { ids: shuffled(rng, aIds), rows }, passives, I: core, pairwise };
}

/** Two sets of n IDs each with round(overlap * n) in common (PSI measurements only). */
export function genPairSets({ seed = 20260926n, n, overlap = 0.8 }) {
  const rng = prng(seed);
  const taken = new Set();
  const common = freshIds(rng, Math.round(overlap * n), taken);
  const a = shuffled(rng, [...common, ...freshIds(rng, n - common.length, taken)]);
  const p = shuffled(rng, [...common, ...freshIds(rng, n - common.length, taken)]);
  return { a, p, common };
}
