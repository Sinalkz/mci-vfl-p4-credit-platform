// M3 §5.1 with D137 + D139 (DATA1; supersede D118): A's scoring. The bottom models are frozen.
//   FULL (F_r)  every remaining passive's slot, fusion model retrained
//   FULL_J (F_j) every remaining passive's slot, the fusion model trained jointly during split training
//               and committed at commitFA (gammaG); not retrained. The contract sets U_fed = max(F_r, F_j).
//   LOO_j       the same with passive j's slot zero, for each remaining passive j, retrained
//   BASELINE    every passive slot zero, retrained
// A retrained fusion model starts from its initial weights and uses the configuration's optimiser,
// learning rate, batch size, T epochs and batch order, on I^tr; it carries its own C_model.
// Input rows: e_A from the quantized f_A (exactly as the circuit computes it) and the binding
// final embeddings h_i. F = floor(10^4 macro-F1) over the 5 classes on I^ev, duplicates counted.
// Each figure's proof is made by circuits/scripts/m3-prove-figure.mjs in its own process, one at a time.
import { spawn } from "node:child_process";
import { freemem } from "node:os";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { faCommitment, headCommitment } from "../../circuits/scripts/members-lib.mjs";
import { membersVarInput } from "../../circuits/scripts/m1b-lib.mjs";
import { CONFIG, E, SCALE, batchesFor, eAInt, forwardInt, macroF1, newFusion, quantFA, quantHead, softmaxCE } from "./model.mjs";

export const KIND = { FULL: 0, LOO: 1, BASELINE: 2, FULL_JOINT: 3 };
const HERE = fileURLToPath(import.meta.url);

/** The required figures for a remaining-passive mask (bit i-1 = passive i remains). */
export function requiredFigures(remainingMask, n = 3) {
  const out = [{ name: "FULL", kind: KIND.FULL, memberIndex: 0, mask: remainingMask },
    { name: "FULL_J", kind: KIND.FULL_JOINT, memberIndex: 0, mask: remainingMask }]; // D139: F_r and F_j
  for (let j = 1; j <= n; j++) {
    if (remainingMask & (1 << (j - 1))) out.push({ name: `LOO_${j}`, kind: KIND.LOO, memberIndex: j, mask: remainingMask & ~(1 << (j - 1)) });
  }
  out.push({ name: "BASELINE", kind: KIND.BASELINE, memberIndex: 0, mask: 0 });
  return out;
}

/** gamma_fA (= C_fA as S2b/M1b compute it) for a trained f_A, as bytes32. */
export async function gammaFAHex(fA) {
  return "0x" + (await faCommitment(quantFA(fA))).toString(16).padStart(64, "0");
}

/** gammaG (D139): the Poseidon commitment of the jointly trained fusion model (as C_model), as bytes32. */
export async function gammaGHex(fusion) {
  return "0x" + (await headCommitment(quantHead(fusion))).toString(16).padStart(64, "0");
}

/**
 * Computes F for each figure: FULL_J (F_j) with the jointly trained fusion model `fusion`, every other
 * kind with a retrained one. rows: { xa[k], y[k] } over the positions of I; hq[p]: passive p+1's binding
 * matrix (|I| x 4 int16), or null if it left; Itr: positions of I^tr; idx: the N drawn positions (I^ev,
 * with repeats). A FULL_JOINT figure with another mask scores the joint model with those slots zeroed
 * (analysis only: the Q17 masked-LOO variant).
 */
export function scoreFigures({ fA, fusion, rows, hq, Itr, idx, figures }) {
  const fa = quantFA(fA);
  const eA = rows.xa.map((x) => eAInt(fa, x));
  const zero = Array(E).fill(0);
  const slot = (p, k) => (hq[p] ? hq[p][k] : zero);
  const out = [];
  for (const fig of figures) {
    if (fig.kind === KIND.FULL_JOINT) {
      if (!fusion) throw new Error(`${fig.name}: FULL_J (F_j, D139) needs the jointly trained fusion model`);
      const hd = quantHead(fusion);
      const preds = idx.map((k) => forwardInt(fa, hd, fig.mask, rows.xa[k], hq.map((_, p) => slot(p, k))));
      const f1 = macroF1(idx.map((k) => rows.y[k]), preds);
      out.push({ ...fig, head: hd, F: f1.F, TP: f1.TP, FP: f1.FP, FN: f1.FN, preds, trainMs: 0, retrained: false });
      continue;
    }
    const t0 = performance.now();
    const head = newFusion();
    for (let epoch = 1; epoch <= CONFIG.T; epoch++) {
      for (const b of batchesFor(Itr.length, epoch)) {
        const pos = b.map((j) => Itr[j]);
        const n = pos.length;
        const inp = Float64Array.from(pos.flatMap((k) =>
          [...eA[k], ...hq.flatMap((_, p) => ((fig.mask >> p) & 1 ? slot(p, k) : zero))].map((v) => v / SCALE)));
        head.backward(softmaxCE(head.forward(inp, n), pos.map((k) => rows.y[k]), n).dZ);
      }
    }
    const hd = quantHead(head);
    const preds = idx.map((k) => forwardInt(fa, hd, fig.mask, rows.xa[k], hq.map((_, p) => slot(p, k))));
    const f1 = macroF1(idx.map((k) => rows.y[k]), preds);
    out.push({ ...fig, head: hd, F: f1.F, TP: f1.TP, FP: f1.FP, FN: f1.FN, preds, trainMs: performance.now() - t0, retrained: true });
  }
  return { fa, figures: out };
}

/**
 * The M1b circuit input for one figure, through membersVarInput (which also runs the Python
 * oracle on the drawn rows). M4 (D128): a left passive's rows are -32768, which pack to leaf 0, so its tree is the
 * all-zero-leaf tree whose root the contract passes for it (emptyRoot); its mask bit 0 keeps the rows out of F.
 */
export async function circuitInputFor({ fa, fig, rows, hq, sizeI, maxDepth, seed, N }) {
  const pad = Array.from({ length: sizeI }, () => Array(E).fill(-32768));
  const data = {
    table: { y: rows.y, xa: rows.xa, hp: hq.map((m) => m ?? pad) },
    fA: fa,
    heads: { [fig.name]: { mask: fig.mask, ...fig.head } },
  };
  const { input, ref, idx } = await membersVarInput(N, sizeI, maxDepth, { mask: fig.mask, data, seed: BigInt(seed) });
  return { input, oracleF: ref.F, idx, cModel: "0x" + (await headCommitment(fig.head)).toString(16).padStart(64, "0") };
}

/**
 * Proves one figure in a separate process (circuits/scripts/m3-prove-figure.mjs); resolves with its result:
 * { proof8, publicSignals, witnessMs, proveMs, verifyMs, peakRssMB, freeRamMBAtStart }.
 */
export function proveInChild(input, workDir, name) {
  mkdirSync(workDir, { recursive: true });
  const inFile = resolve(workDir, `${name}.input.json`);
  const outFile = resolve(workDir, `${name}.proof.json`);
  writeFileSync(inFile, JSON.stringify(input));
  const freeRamMBAtStart = Math.round(freemem() / 2 ** 20);
  return new Promise((ok, no) => {
    const p = spawn(process.execPath, [resolve(HERE, "..", "..", "..", "circuits", "scripts", "m3-prove-figure.mjs"), inFile, outFile], { stdio: ["ignore", "inherit", "inherit"] });
    p.on("exit", (c) => (c === 0 ? ok({ ...JSON.parse(readFileSync(outFile, "utf8")), freeRamMBAtStart }) : no(new Error(`prover exited ${c}`))));
  });
}
