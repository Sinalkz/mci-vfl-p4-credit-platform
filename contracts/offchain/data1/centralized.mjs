// DATA1 C4a / C1b: the same bottom models and fusion model (model.mjs MLP: same layer specs, same
// init streams, same batch order, same loss and SGD) trained jointly in one process: no split, no
// sessions, no noise. mode "f32" rounds the passives' embeddings and their gradients to float32 at
// the points where split training sends them over the wire (training.mjs encVec/decVec), so it must
// reproduce split training without noise bit for bit (the correctness check of the VFL
// implementation); "f64" does not round (C4a as the brief defines it); "aOnly" (C1b) trains f_A and
// the fusion model with every passive slot zero: what A gets with no federation at all. `exclude` (1-based
// passive indices; Q17 follow-up) trains the others as the mode says and leaves those slots at zero with no
// model: the centralized leave-one-out reference.
import { CONFIG, E, SCALE, batchesFor, newFA, newFi, newFusion, q, softmaxCE } from "../model.mjs";

export function trainCentral({ rows, xp, Itr, mode, T = CONFIG.T, exclude = [] }) {
  if (!["f64", "f32", "aOnly"].includes(mode)) throw new Error(`unknown mode ${mode}`);
  const fA = newFA();
  const fusion = newFusion();
  const fis = mode === "aOnly" ? [] : xp.map((_, i) => (exclude.includes(i + 1) ? null : newFi(i + 1)));
  const wire = mode === "f32" ? Math.fround : (v) => v;
  const D = CONFIG.architecture.fusion.in; // E * (1 + 3) = 16
  const rounds = [];
  for (let round = 1; round <= T; round++) {
    let loss = 0;
    const batches = batchesFor(Itr.length, round);
    for (const b of batches) {
      const pos = b.map((j) => Itr[j]);
      const n = pos.length;
      const eA = fA.forward(Float64Array.from(pos.flatMap((k) => rows.xa[k].map((v) => v / SCALE))), n);
      const H = fis.map((f, i) => (f ? f.forward(Float64Array.from(pos.flatMap((k) => xp[i][k])), n) : null));
      const inp = new Float64Array(n * D);
      for (let r = 0; r < n; r++) {
        for (let e = 0; e < E; e++) inp[r * D + e] = eA[r * E + e];
        H.forEach((h, i) => { if (h) for (let e = 0; e < E; e++) inp[r * D + E * (i + 1) + e] = wire(h[r * E + e]); });
      }
      const { loss: l, dZ } = softmaxCE(fusion.forward(inp, n), pos.map((k) => rows.y[k]), n);
      loss += l;
      const dInp = fusion.backward(dZ);
      const deA = new Float64Array(n * E);
      for (let r = 0; r < n; r++) for (let e = 0; e < E; e++) deA[r * E + e] = dInp[r * D + e];
      fA.backward(deA);
      fis.forEach((f, i) => {
        if (!f) return;
        const g = new Float64Array(n * E);
        for (let r = 0; r < n; r++) for (let e = 0; e < E; e++) g[r * E + e] = wire(dInp[r * D + E * (i + 1) + e]);
        f.backward(g);
      });
    }
    rounds.push({ round, loss: loss / batches.length });
  }
  return { fA, fusion, fis, rounds };
}

/** Each f_i over all rows, int16 at 2^12: what passiveTraining's final round sends when noise is off. */
export function centralHq(fis, xp, sizeI) {
  return fis.map((f, i) => {
    if (!f) return null; // an excluded passive: a zero slot (predictInt)
    const H = f.forward(Float64Array.from(xp[i].flat()), sizeI);
    return Array.from({ length: sizeI }, (_, r) => Array.from({ length: E }, (_, e) => q(H[r * E + e], 12, 16)));
  });
}
