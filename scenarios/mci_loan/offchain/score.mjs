// MCI loan instance, decision (a): the off-chain score on the circuit's integer path. logitsInt computes the five
// logits exactly as contracts/offchain/model.mjs forwardInt does (e_A from the quantized f_A, the masked fusion head,
// all in BigInt); the predicted class is the first maximum, as forwardInt and the circuit. The score of a row is the
// margin z_1 - z_0 (default minus repaid) as a Number: higher means more likely to default. Off-chain only: the circuit
// proves macro-F1 and nothing else, so every AUC and approval-rate figure built on this score is not ZK-verified.
import { E, eAInt, forwardInt } from "../../../contracts/offchain/model.mjs";

const ZERO = Array(E).fill(0);

/** The five logits (BigInt) for one row; hp: 3 rows of 4 int16, masked as in forwardInt. */
export function logitsInt(fa, head, mask, xa, hp) {
  const inp = [...eAInt(fa, xa), ...hp.flatMap((h, p) => h.map((v) => ((mask >> p) & 1 ? v : 0)))].map(BigInt);
  const hid = head.W1.map((row, h) => {
    let z = BigInt(head.b1[h]);
    for (let d = 0; d < row.length; d++) z += BigInt(row[d]) * inp[d];
    return z > 0n ? z : 0n;
  });
  return head.W2.map((row, k) => {
    let z = BigInt(head.b2[k]);
    for (let h = 0; h < row.length; h++) z += BigInt(row[h]) * hid[h];
    return z;
  });
}

const firstMax = (z) => z.reduce((best, v, k) => (v > z[best] ? k : best), 0);

/**
 * For each position: the predicted class (checked against forwardInt) and the margin z_1 - z_0. Also how many rows
 * the head puts in classes 2-4, which the binary label never uses (docs/mci/00-recon.md section 5, point 3).
 * hq[p]: passive p+1's |I| x 4 int16 matrix, or null for a zero slot.
 */
export function scoreRows({ fa, head, mask, rows, hq, positions }) {
  const preds = [];
  const margins = [];
  let unusedClass = 0;
  for (const k of positions) {
    const hp = hq.map((m) => (m ? m[k] : ZERO));
    const z = logitsInt(fa, head, mask, rows.xa[k], hp);
    const pred = firstMax(z);
    if (pred !== forwardInt(fa, head, mask, rows.xa[k], hp)) throw new Error(`row ${k}: logitsInt and forwardInt disagree`);
    if (pred >= 2) unusedClass++;
    preds.push(pred);
    margins.push(Number(z[1] - z[0]));
  }
  return { preds, margins, unusedClass };
}
