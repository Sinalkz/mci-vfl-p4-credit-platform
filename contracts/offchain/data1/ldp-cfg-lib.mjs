// Task 5 (LDP configurations): the pieces the configuration runs add around the thesis code, none of which changes it.
//   variantConfig      a model config as configs.mjs writes it (seeds, widths), with T changed; at T = 20 its bytes are
//                      the E1 config's (results/data1/configs/e1-s<seed>.json), so the configHash is the same
//   stratifiedSubsample a training-row subsample with the class shares of I^tr, nested across fractions for one seed
//   shadowPassive      an observer of one passive's session that keeps a copy of f_i (the same init, the same batch and
//                      gradient messages, so the same float operations) and the passive's final noise draws; after the
//                      run it gives the final matrix before the int16 cap (uncapped), checked against the capped one
//                      the passive sent, entry by entry
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { keccak256 } from "ethers";

import { prng } from "../data.mjs";
import { E, SCALE, clipAndNoise, newFi, q } from "../model.mjs";
import { TMSG, decBatch, decVec } from "../training.mjs";
import { BASE_CONFIG, initSeedFor, shuffleSeedFor } from "./configs.mjs";

export const SUBSAMPLE_SEED = (seed) => 20264000 + seed; // init 20261000 + s, shuffle 20262000 + s, noise 20263000 + 10 s + i
const INT16 = [-32768, 32767];

/** The model config of configs.mjs writeConfig (seeds and widths) with T set; { bytes, hash, config }. */
export function variantConfig({ seed, widths, T }) {
  const config = JSON.parse(readFileSync(BASE_CONFIG, "utf8"));
  config.initSeed = initSeedFor(seed);
  config.shuffleSeed = shuffleSeedFor(seed);
  config.architecture.fi.in = widths;
  if (!Number.isInteger(T) || T < 1) throw new Error(`T ${T}`);
  config.T = T;
  const bytes = Buffer.from(JSON.stringify(config, null, 2) + "\n", "utf8");
  return { bytes, hash: keccak256(bytes), config };
}

/** Writes variantConfig to `path`; an existing file must hold the same bytes. Returns its hash. */
export function writeVariantConfig(path, args) {
  const v = variantConfig(args);
  let old = null;
  try { old = readFileSync(path); } catch { /* new */ }
  if (old && !old.equals(v.bytes)) throw new Error(`${path} exists with other bytes`);
  if (!old) { mkdirSync(dirname(path), { recursive: true }); writeFileSync(path, v.bytes); }
  return v.hash;
}

/**
 * A stratified subsample of the training positions: round(frac * N) rows (half up), each class's quota floor(frac n_c)
 * plus one for the classes with the largest remainders (ties: the lower class) until the total is reached. Within a
 * class the rows are taken in the order of a seeded Fisher-Yates shuffle of that class's positions (splitmix64 at
 * SUBSAMPLE_SEED(seed), classes in ascending order, as batchesFor shuffles), so for one seed a smaller fraction's rows
 * are a subset of a larger one's. Returned in trainPos order. frac = 1 returns trainPos (a copy).
 */
export function stratifiedSubsample(trainPos, y, frac, seed) {
  if (!(frac > 0 && frac <= 1)) throw new Error(`frac ${frac} not in (0, 1]`);
  if (frac === 1) return trainPos.slice();
  const byClass = new Map();
  for (const k of trainPos) { if (!byClass.has(y[k])) byClass.set(y[k], []); byClass.get(y[k]).push(k); }
  const classes = [...byClass.keys()].sort((a, b) => a - b);
  const total = Math.floor(frac * trainPos.length + 0.5);
  const quota = new Map(classes.map((c) => [c, Math.floor(frac * byClass.get(c).length)]));
  let left = total - [...quota.values()].reduce((s, v) => s + v, 0);
  const byRemainder = [...classes].sort((a, b) => {
    const ra = frac * byClass.get(a).length - quota.get(a), rb = frac * byClass.get(b).length - quota.get(b);
    return rb - ra || a - b;
  });
  for (const c of byRemainder) { if (left <= 0) break; quota.set(c, quota.get(c) + 1); left--; }
  const rng = prng(BigInt(SUBSAMPLE_SEED(seed)));
  const chosen = new Set();
  for (const c of classes) {
    const order = byClass.get(c).slice();
    for (let i = order.length - 1; i > 0; i--) { const j = Number(rng.u64() % BigInt(i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (const k of order.slice(0, quota.get(c))) chosen.add(k);
  }
  return trainPos.filter((k) => chosen.has(k));
}

/**
 * An observer of passive `index`'s side of a run. Pass `session` and `normal` to passiveTraining in place of the
 * originals; they forward every call unchanged. Meanwhile a copy of f_i (newFi(index): the same seeded init) runs the
 * same forward pass on every train/batch and the same backward pass on every matching train/grad, exactly as
 * passiveTraining does, and the noise draws after train/end (the final matrix's) are recorded.
 * uncapped(hq): the final matrix at the same 2^-12 resolution without the int16 range limit (Math.round(v * 2^12), as
 * q() rounds, no clip); throws unless q(v, 12, 16) equals the passive's hq at every entry.
 */
export function shadowPassive({ session, index, xp, sizeI, noise, normal }) {
  const fi = newFi(index);
  let current = null;
  let finalDraws = null;
  const observe = (m) => {
    if (m.type === TMSG.end) { finalDraws = []; return; }
    if (m.type === TMSG.batch) {
      const { round, batch, positions } = decBatch(m.payload);
      fi.forward(Float64Array.from(positions.flatMap((k) => xp[k])), positions.length);
      current = { round, batch };
    } else if (m.type === TMSG.grad) {
      const { round, batch, values } = decVec(m.payload);
      if (!current || current.round !== round || current.batch !== batch) return;
      fi.backward(values);
      current = null;
    }
  };
  const wrapped = {
    recv: async (ms) => { const m = await session.recv(ms); observe(m); return m; },
    send: (type, payload) => session.send(type, payload),
    close: () => session.close(),
    get failed() { return session.failed; },
  };
  const recording = () => { const z = normal(); if (finalDraws) finalDraws.push(z); return z; };
  function finalFloat() {
    if (!finalDraws || finalDraws.length !== sizeI * E) throw new Error(`passive ${index}: ${finalDraws?.length} final draws, expected ${sizeI * E}`);
    let t = 0;
    const replay = () => finalDraws[t++];
    return clipAndNoise(Float64Array.from(fi.forward(Float64Array.from(xp.flat()), sizeI)), sizeI, noise, replay);
  }
  function uncapped(hq) {
    const H = finalFloat();
    const out = Array.from({ length: sizeI }, (_, r) => Array.from({ length: E }, (_, e) => {
      const v = H[r * E + e];
      if (q(v, 12, 16) !== hq[r][e]) throw new Error(`passive ${index}: shadow row ${r} entry ${e} ${v} does not give the sent ${hq[r][e]}`);
      return Math.round(v * SCALE);
    }));
    return out;
  }
  return { session: wrapped, normal: recording, finalFloat, uncapped };
}

/** Per passive: entries beyond the int16 range, and the min and max entry. */
export function beyondInt16(hqOff) {
  return hqOff.map((m) => {
    let n = 0, lo = Infinity, hi = -Infinity;
    for (const row of m) for (const v of row) { if (v < INT16[0] || v > INT16[1]) n++; lo = Math.min(lo, v); hi = Math.max(hi, v); }
    return { beyond: n, min: lo, max: hi };
  });
}
