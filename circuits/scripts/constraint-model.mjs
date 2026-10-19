// Exact constraint model of the S2b circuit, fitted on compiled sizes and used for
// sizes too large to compile on this machine.
// Usage (from circuits/): node scripts/constraint-model.mjs
// Reads build/members-N<N>-I<I>/compile.json; writes results/s2b/constraint_model.json.
//
// The circuit is a fixed part plus N identical row blocks, except the final
// division, whose two range checks have NB(N) = bitsFor(K (2N)^K) bits, i.e.
//   C(N) = a + b*N + 2*NB(N)
// a and b are solved from the two smallest compiled N for each |I|; every other
// compiled N is a validation point (the residual must be 0).
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { BUILD, ROOT } from "./lib.mjs";

const K = 5;
function bitsFor(a) { let r = 0; let p = 1n; while (p <= a) { p *= 2n; r++; } return r; } // as in macro_f1.circom
const NB = (n) => bitsFor(BigInt(K) * (2n * BigInt(n)) ** BigInt(K));

const compiled = {};
for (const d of readdirSync(BUILD)) {
  const m = d.match(/^members-N(\d+)-I(\d+)$/);
  const f = resolve(BUILD, d, "compile.json");
  if (!m || !existsSync(f)) continue;
  const c = JSON.parse(readFileSync(f, "utf8"));
  (compiled[m[2]] ??= []).push({ N: Number(m[1]), constraints: c.r1csConstraints, depth: c.depth });
}

const out = [];
for (const [isize, pts] of Object.entries(compiled)) {
  pts.sort((x, y) => x.N - y.N);
  if (pts.length < 2) continue;
  const [p, q] = pts;
  const b = (q.constraints - p.constraints - 2 * (NB(q.N) - NB(p.N))) / (q.N - p.N);
  const a = p.constraints - b * p.N - 2 * NB(p.N);
  const model = (n) => a + b * n + 2 * NB(n);
  const validation = pts.slice(2).map((x) => ({ N: x.N, compiled: x.constraints, model: model(x.N), residual: x.constraints - model(x.N) }));
  const predictions = [64, 128, 256].map((n) => {
    const c = pts.find((x) => x.N === n);
    return { N: n, constraints: c ? c.constraints : model(n), source: c ? "compiled" : "model" };
  });
  out.push({ isize: Number(isize), depth: p.depth, fittedOn: [p.N, q.N], fixed: a, perRow: b, validation, predictions });
  console.log(JSON.stringify(out.at(-1)));
}
const dir = resolve(ROOT, "results", "s2b");
mkdirSync(dir, { recursive: true });

// Projections for configurations that were not proven on this machine: straight
// lines through the proven configurations (peak prover memory and proving time
// against constraints). FFT work grows as n log n, so the time line is a rough guide.
const measured = existsSync(resolve(dir, "zk_members.json"))
  ? JSON.parse(readFileSync(resolve(dir, "zk_members.json"), "utf8")).filter((r) => r.proving) : [];
let projections = null;
if (measured.length >= 2) {
  const line = (pts) => {
    const [p, q] = pts;
    const slope = (q.y - p.y) / (q.x - p.x);
    return { slope, intercept: p.y - slope * p.x };
  };
  const pts = measured.map((r) => ({ x: r.constraints, mem: r.proving.peakRssMB, t: r.proving.proveMs.median / 1000 }));
  const mem = line(pts.map((p) => ({ x: p.x, y: p.mem })));
  const time = line(pts.map((p) => ({ x: p.x, y: p.t })));
  projections = {
    basis: measured.map((r) => `N=${r.N} |I|=${r.isize}`),
    memoryMiBPerKConstraints: mem.slope * 1000, memoryInterceptMiB: mem.intercept,
    secondsPerKConstraints: time.slope * 1000, timeInterceptS: time.intercept,
    configs: out.flatMap((m) => m.predictions.filter(() => m.isize !== 16).map((p) => ({
      N: p.N, isize: m.isize, constraints: p.constraints,
      ptauPower: Math.ceil(Math.log2(p.constraints + 9 + 1)),
      peakMemoryMiB: Math.round(mem.intercept + mem.slope * p.constraints),
      provingSeconds: Math.round(time.intercept + time.slope * p.constraints),
      proven: measured.some((r) => r.N === p.N && r.isize === m.isize),
    }))),
  };
  console.log(JSON.stringify(projections, null, 1));
}
writeFileSync(resolve(dir, "constraint_model.json"), JSON.stringify({ models: out, projections }, null, 2) + "\n");
