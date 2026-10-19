// Task 4 item 2: the Merkle commitments per member, with the protocol's own functions:
//   A        gamma_A = alignment.mjs gammaA: Poseidon tree at maxDepth over A's rows (x_A, y) in the order of I
//   passive  gamma_i = training.mjs embeddingRootHex: Poseidon tree at maxDepth over its final matrix h_i (|I| x 4 int16),
//            then its EIP-712 signature over (sender, gamma_i, "final-round") (signer.signTypedData)
//   A again  its check of each received final message (training.mjs activeTraining): the root of the received matrix
//            and the signature's signer (finalSigner)
// Both trees hash every one of the 2^maxDepth leaves (members-lib.mjs buildTree pads with zeros), so their cost is
// set by maxDepth, not by the values.
//   subset  |I| = 910, maxDepth 10: A's rows are E3's (results/data1/e3/A/I.json, data1/recs-scenario.mjs)
//   full    |I| = 14,243 (the full-data design's mode of |I|), maxDepth 14: A's rows are the first 14,243 prepared
//           households of a seeded shuffle, with synthetic IDs; every h_i is a synthetic int16 matrix of that shape
// One call of each kind first (discarded; it also builds Poseidon), then --reps timed calls; one process per setting.
//   node offchain/overhead/merkle-bench.mjs --setting subset|full [--reps 5] --out <file>
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { Wallet } from "ethers";

import { gammaA } from "../alignment.mjs";
import { prng } from "../data.mjs";
import { machine, peakRssMB } from "../data1/machine.mjs";
import { openPrepared } from "../data1/prepared.mjs";
import { recsScenario } from "../data1/recs-scenario.mjs";
import { E } from "../model.mjs";
import { FINAL_MARKER, FINAL_TYPES, embeddingRootHex, finalDomain, finalSigner } from "../training.mjs";

const ROOT = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "..");
export const MERKLE_SETTINGS = { subset: { sizeI: 910, maxDepth: 10 }, full: { sizeI: 14243, maxDepth: 14 } };

/** A's rows for a setting: { sortedI, rows: Map(id -> { xa, y }) }. */
export function activeRows(setting) {
  if (setting === "subset") {
    const dir = resolve(ROOT, "results", "data1", "e3");
    const I = JSON.parse(readFileSync(resolve(dir, "A/I.json"), "utf8")).map((x) => BigInt(x));
    const sc = recsScenario({ seed: JSON.parse(readFileSync(resolve(dir, "sample.json"), "utf8")).seed });
    return { sortedI: I, rows: new Map(I.map((id) => [id, sc.A.rows.get(id)])) };
  }
  const { m, A } = openPrepared();
  const rng = prng(20264100n);
  const order = Array.from({ length: m.rows }, (_, k) => k);
  for (let i = order.length - 1; i > 0; i--) { const j = Number(rng.u64() % BigInt(i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const picked = order.slice(0, MERKLE_SETTINGS.full.sizeI);
  const sortedI = picked.map((_, j) => 1_000_000_000n + BigInt(j) * 7919n); // ascending synthetic IDs
  return { sortedI, rows: new Map(picked.map((k, j) => [sortedI[j], { xa: Array.from(A.subarray(k * 8, k * 8 + 8)), y: m.y[k] }])) };
}

/** A synthetic final matrix: n rows of E int16 values (the range of int16 at 2^12). */
export function syntheticMatrix(n, seed) {
  const rng = prng(BigInt(seed));
  return Array.from({ length: n }, () => Array.from({ length: E }, () => rng.int(-32768, 32767)));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values: a } = parseArgs({ options: { setting: { type: "string" }, reps: { type: "string", default: "5" }, out: { type: "string" } } });
  const st = MERKLE_SETTINGS[a.setting];
  if (!st) throw new Error("--setting subset|full");
  const reps = Number(a.reps);
  const m0 = machine();
  const t0 = performance.now();
  const { sortedI, rows } = activeRows(a.setting);
  if (sortedI.length !== st.sizeI) throw new Error(`|I| ${sortedI.length} != ${st.sizeI}`);
  const hq = [1, 2, 3].map((i) => syntheticMatrix(st.sizeI, 20264200 + i));
  const ws = [1, 2, 3].map(() => Wallet.createRandom());
  const chainId = 2026n;
  const contract = Wallet.createRandom().address;
  const sign = (w, root) => w.signTypedData(finalDomain(chainId, contract), FINAL_TYPES, { sender: w.address, root, marker: FINAL_MARKER });

  const timed = async (fn) => { const s = performance.now(); const r = await fn(); return { ms: performance.now() - s, r }; };
  const series = async (label, fn) => {
    const warm = await timed(fn); // discarded
    const ms = [];
    let r;
    for (let k = 0; k < reps; k++) { const x = await timed(fn); ms.push(x.ms); r = x.r; }
    return { label, warmupMs: warm.ms, ms, result: typeof r === "string" ? r : undefined };
  };
  const poseidonSetup = await timed(() => embeddingRootHex([[0, 0, 0, 0]], 1)); // builds circomlibjs Poseidon once
  const out = { item: 2, what: "merkle", setting: a.setting, sizeI: st.sizeI, maxDepth: st.maxDepth, leaves: 2 ** st.maxDepth,
    hashesPerTree: 2 ** st.maxDepth - 1, reps, poseidonSetupMs: poseidonSetup.ms, machine: m0, series: [] };
  out.series.push(await series("A: gamma_A over I (alignment.mjs gammaA)", () => gammaA(sortedI, rows, st.maxDepth)));
  const roots = [];
  for (const [i, h] of hq.entries()) {
    const s = await series(`p${i + 1}: gamma_${i + 1} over its final matrix (training.mjs embeddingRootHex)`, () => embeddingRootHex(h, st.maxDepth));
    roots.push(s.result);
    out.series.push(s);
  }
  const sigs = [];
  for (const [i, w] of ws.entries()) {
    const s = await series(`p${i + 1}: EIP-712 signature over gamma_${i + 1} (signTypedData)`, () => sign(w, roots[i]));
    sigs.push(s.result);
    out.series.push(s);
  }
  for (const [i, w] of ws.entries()) {
    out.series.push(await series(`A: check of p${i + 1}'s final message (root of the received matrix + finalSigner)`, async () => {
      const root = await embeddingRootHex(hq[i], st.maxDepth);
      const signer = finalSigner({ chainId, contract, sender: w.address, root, sig: sigs[i] });
      if (signer !== w.address) throw new Error("signature check failed");
      return root;
    }));
  }
  out.freeRamMBAtEnd = machine().freeRamMB;
  out.wallMs = performance.now() - t0;
  out.peakRssMB = peakRssMB();
  mkdirSync(dirname(resolve(a.out)), { recursive: true });
  writeFileSync(resolve(a.out), JSON.stringify(out) + "\n");
  for (const s of out.series) console.log(JSON.stringify({ setting: a.setting, label: s.label, msMedian: [...s.ms].sort((x, y) => x - y)[Math.floor(reps / 2)] }));
  await globalThis.curve_bn128?.terminate();
  process.exit(0);
}
