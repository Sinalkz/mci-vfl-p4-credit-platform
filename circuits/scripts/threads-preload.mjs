// RESULTS-PROOFCOST: the prover thread count as a MEASUREMENT setting. snarkjs 0.7.6 has no thread option besides
// singleThread; ffjavascript sizes its worker pool from os.cpus().length. With PROVER_THREADS=k set, this preload makes
// os.cpus() report k CPUs inside this process only. Used by proofcost.mjs (node --import ./scripts/threads-preload.mjs);
// the protocol's prover (m3-prove-figure.mjs as called by contracts/offchain/scoring.mjs) runs without it, unchanged.
import os from "node:os";

const k = Number(process.env.PROVER_THREADS);
if (k >= 1) {
  const real = os.cpus.bind(os);
  os.cpus = () => real().slice(0, k);
}
