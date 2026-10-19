// MCI loan instance, Step 5: the scenario of the five on-chain runs, in the shape of
// contracts/offchain/full/recs-full-scenario.mjs (genScenario()'s shape, data.mjs) so that party.mjs runs unchanged on
// it, with the loan's sizes (docs/mci/02-preregistration.md, section 4):
//   the Bank (A): every cohort applicant, |X_A| = 15,965 (data/mci_loan/prepared/, scenarios/mci_loan/prep.py) <= 2^14
//   each passive: every cohort applicant (decision (c): no passive misses one), plus 1,266 extra applicants the Bank
//     does not have, drawn independently per passive from the 2,531 of the extra pool (prepared/passive_extra_f32.bin);
//     the private set intersection drops them, so |I| = 15,965
// Rows: the Bank's 8 int16 columns and TARGET, each passive's encoded columns (manifest.parties) for both its cohort
// and its extra applicants. National IDs: distinct 10-digit numbers drawn with the same seed, one per applicant used.
// Label cut points (the proposal's classBoundaries): 1, 2, 3, 4 on TARGET, so class 0 = repaid (TARGET 0), class 1 =
// default (TARGET 1), and classes 2-4 stay empty (docs/mci/00-recon.md section 5). One draw (seed 20267005) serves
// all five runs.
// v2 (MCI as the active member): MCI_LOAN_PREPARED names data/mci_loan/v2/prepared; its manifest's passiveOrder (bank,
// ecommerce, bureau) gives the passives' columns; A's rows are MCI's. The draw itself is the same as v1's.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { prng } from "../../../contracts/offchain/data.mjs";
import { PREPARED, openPrepared, passiveNames } from "../offchain/prepared.mjs";

export const LOAN_SEED = 20267005;
export const LOAN_SIZES = { extra: 1266 };
export const LOAN_CLASS_BOUNDARIES = [1n, 2n, 3n, 4n];

function shuffled(rng, arr) { // as recs-full-scenario.mjs
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) { const j = Number(rng.u64() % BigInt(i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

/** The extra pool's manifest and checked binary: { e (manifest), X (rows x W float32) }. */
export function openExtra() {
  const e = JSON.parse(readFileSync(resolve(PREPARED, "extra_manifest.json"), "utf8"));
  const buf = readFileSync(resolve(PREPARED, e.passive.file));
  const got = createHash("sha256").update(buf).digest("hex");
  if (got !== e.passive.sha256) throw new Error(`${e.passive.file}: SHA-256 ${got} != manifest ${e.passive.sha256}; rerun scenarios/mci_loan/prep.py`);
  const X = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  if (X.length !== e.rows * e.passive.width) throw new Error("passive_extra_f32.bin does not match its manifest's shape");
  return { e, X };
}

export function loanScenario({ seed = LOAN_SEED, extra = LOAN_SIZES.extra } = {}) {
  const { m, A, P, W } = openPrepared();
  const { e, X } = openExtra();
  if (e.passive.width !== W) throw new Error(`extra width ${e.passive.width} != prepared width ${W}`);
  const rng = prng(BigInt(seed));
  const aPos = Array.from({ length: m.rows }, (_, k) => k); // the Bank holds every cohort applicant
  const names = passiveNames(m);
  const extraPos = names.map(() => shuffled(rng, Array.from({ length: e.rows }, (_, k) => k)).slice(0, extra));
  // one distinct 10-digit national ID per applicant used: the cohort's in SK_ID_CURR order, then the extra ones
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
  const cols = names.map((n) => m.parties[n]);
  const xp = (i, k) => cols[i].map((j) => P[k * W + j]);
  const xpExtra = (i, k) => cols[i].map((j) => X[k * W + j]);
  const aIds = aPos.map((k) => idOf.get(k));
  const passives = names.map((_, i) => {
    const rows = new Map([...aPos.map((k) => [idOf.get(k), xp(i, k)]), ...extraPos[i].map((k) => [idOfExtra.get(k), xpExtra(i, k)])]);
    return { ids: shuffled(rng, [...rows.keys()]), rows };
  });
  const aSet = new Set(aIds);
  const inAll = [...aIds].sort((x, y) => (x < y ? -1 : 1));
  return {
    A: { ids: shuffled(rng, aIds), rows: new Map(aPos.map((k) => [idOf.get(k), { xa: xa(k), y: m.y[k] }])) },
    passives, I: inAll, pairwise: passives.map((ps) => ps.ids.filter((x) => aSet.has(x))),
    classBoundaries: LOAN_CLASS_BOUNDARIES,
    sample: { scenario: "mci-loan", seed, sizeA: m.rows, missing: 0, extra, extraPoolSize: e.rows,
      skIdExtra: extraPos.map((x) => x.map((k) => e.ids[k])), widths: cols.map((c) => c.length), passiveOrder: names,
      prepared: { passiveSha256: m.passive.sha256, aSha256: m.a.sha256, extraSha256: e.passive.sha256 } },
  };
}
