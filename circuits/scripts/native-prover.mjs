// rapidsnark (iden3's native Groth16 prover) for one witness, timed, with its peak memory from /usr/bin/time, the way
// scripts/proofcost.mjs measured it on the Mac (RESULTS-PROOFCOST.md section 6). Same zkey and witness format as
// snarkjs; the proof and public signals it writes are read back as snarkjs's JSON.
import { spawnSync } from "node:child_process";
import { readFileSync, rmSync } from "node:fs";

/** Peak resident memory in MB from /usr/bin/time's report: macOS -l (bytes) or GNU -v (kbytes); null if absent. */
export function peakRssMBFromTime(stderr) {
  const mac = stderr.match(/(\d+)\s+maximum resident set size/);
  if (mac) return Number(mac[1]) / 2 ** 20;
  const gnu = stderr.match(/Maximum resident set size \(kbytes\):\s*(\d+)/);
  return gnu ? Number(gnu[1]) / 1024 : null;
}

/** The /usr/bin/time flag whose report peakRssMBFromTime reads: -l on macOS (BSD time), -v elsewhere (GNU time). */
export const timeFlag = (platform = process.platform) => (platform === "darwin" ? "-l" : "-v");

/** Proves `wtns` with `zkey` by the rapidsnark prover binary `exe`; returns { proof, publicSignals, proveMs, peakRssMB }. */
export function rapidsnarkProve(exe, zkey, wtns) {
  const pf = `${wtns}.proof.json`, pub = `${wtns}.public.json`;
  const flag = process.platform === "darwin" ? "-l" : "-v";
  const t0 = performance.now();
  const r = spawnSync("/usr/bin/time", [flag, exe, zkey, wtns, pf, pub], { encoding: "utf8" });
  const proveMs = performance.now() - t0;
  if (r.status !== 0) throw new Error(`rapidsnark exited ${r.status}: ${r.stderr.slice(-1000)}`);
  const out = { proof: JSON.parse(readFileSync(pf, "utf8")), publicSignals: JSON.parse(readFileSync(pub, "utf8")), proveMs,
    peakRssMB: peakRssMBFromTime(r.stderr) };
  rmSync(pf, { force: true });
  rmSync(pub, { force: true });
  return out;
}
