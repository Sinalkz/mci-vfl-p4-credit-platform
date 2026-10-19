// DATA1 evaluation helpers: predictions through the circuit's integer path, F on a set of rows, the
// protocol's evaluation draw (sampleIndices: N positions with replacement, Poseidon(seed, k) mod size),
// and int16 saturation counts.
import { sampleIndices } from "../../../circuits/scripts/members-lib.mjs";
import { E, forwardInt, macroF1 } from "../model.mjs";

const ZERO = Array(E).fill(0);

/** forwardInt for each position; hq[p] is passive p+1's |I| x 4 int16 matrix, or null for a zero slot. */
export function predictInt({ fa, head, mask, rows, hq, positions }) {
  return positions.map((k) => forwardInt(fa, head, mask, rows.xa[k], hq.map((m) => (m ? m[k] : ZERO))));
}

/** F = floor(10^4 macro-F1) of preds (aligned with positions) against the labels at those positions. */
export const fOn = (rows, positions, preds) => macroF1(positions.map((k) => rows.y[k]), preds).F;

/** The protocol's N = 64 draw (with replacement) over the test positions, for draw number j. */
export async function n64Positions(j, testPos, n = 64) {
  return (await sampleIndices(BigInt(j), n, testPos.length)).map((t) => testPos[t]);
}

/** Per passive, the number of int16 entries at the limits (32767 or -32768): the evidence of clipping. */
export const saturation = (hq) => hq.map((m) => (m ? m.reduce((s, row) => s + row.filter((v) => v === 32767 || v === -32768).length, 0) : 0));
