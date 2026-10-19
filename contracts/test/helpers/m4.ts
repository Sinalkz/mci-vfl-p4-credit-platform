// M4 test helpers: a run taken to Scored with chosen figures (MockScoreVerifier), matching one case of
// test/vectors/m4_settlement.json (py/m4_vectors.py): passives who left never commit; an equivocator registers
// another root than it signed and the equivocation is proven.
import { keccak256, toUtf8Bytes } from "ethers";
import { fieldRoot, reconciliationRun } from "./m3.js";
// @ts-ignore -- plain ESM modules
import { KIND } from "../../offchain/scoring.mjs";
// @ts-ignore
import { ALIGNED_TYPES } from "../../offchain/alignment.mjs";
// @ts-ignore
import { runDomain } from "../../offchain/encoding.mjs";

export const FAKE_PROOF = ["1", "0", "0", "0", "0", "0", "0", "0"]; // accepted by MockScoreVerifier

export type Case = {
  name: string; path: "normal" | "objection" | "backup"; B: number; rho: number; Dp: number;
  figures: { Fr: number; Fj: number; loo: number[]; base: number };
  left: boolean[]; eqv: boolean[]; acks: boolean[]; objector: number | null;
  expected: { E: number; Eprime: number; l: number[]; a: number[]; awards: number[]; owedA: number; owedP: number[]; locked: number };
};

export async function scoredRun(ethers: any, networkHelpers: any, S: { A: any; P: any[]; ARB: any }, c: Case) {
  const mineTo = async (block: number) => {
    const now = await ethers.provider.getBlockNumber();
    if (block > now) await networkHelpers.mine(block - now);
  };
  const verifier = await ethers.deployContract("MockScoreVerifier");
  const { run, finals } = await reconciliationRun(ethers, S, {
    skipFinal: c.left, registered: c.eqv.map((e, i) => (e ? fieldRoot(`Y${i}`) : null)),
    overrides: { verifier: await verifier.getAddress(), B: BigInt(c.B), rho: BigInt(c.rho), Dp: BigInt(c.Dp) },
  });
  const fed = run.fed;
  for (let i = 0; i < 3; i++) if (c.eqv[i]) await (await fed.proveEquivocation(S.P[i].address, finals[i].root, finals[i].sig)).wait();
  await mineTo(Number(await fed.reconciliationDeadline()));
  await (await fed.finalizeReconciliation()).wait();
  const A = S.A;
  await (await fed.connect(A).submitScore(KIND.FULL, 0, c.figures.Fr, fieldRoot("Fr"), FAKE_PROOF)).wait();
  await (await fed.connect(A).submitJointScore(c.figures.Fj, FAKE_PROOF)).wait();
  for (let j = 1; j <= 3; j++) {
    if (!c.left[j - 1]) await (await fed.connect(A).submitScore(KIND.LOO, j, c.figures.loo[j - 1], fieldRoot(`L${j}`), FAKE_PROOF)).wait();
  }
  await (await fed.connect(A).submitScore(KIND.BASELINE, 0, c.figures.base, fieldRoot("B"), FAKE_PROOF)).wait();
  return { run, fed, finals, mineTo };
}

/** A's AlignedSet signature for (passive, listHash) in this run's domain; the default hash is not gamma_I. */
export const alignedSig = (A: any, run: any, passive: string, listHash = keccak256(toUtf8Bytes("another list"))) =>
  A.signTypedData(runDomain(run.chainId, run.addr), ALIGNED_TYPES, { passive, listHash });
