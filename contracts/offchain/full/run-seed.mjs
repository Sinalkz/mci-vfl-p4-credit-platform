// full-data runs (Task 2, item 4): "run seed k" makes each member's seed-commitment secret r reproducible, so that run
// k's on-chain seed s = keccak256(r_A, r_1, r_2, r_3) mod p (Federation.revealSeed), and with it the 64 evaluation
// positions, is the same whenever run k is repeated. r = keccak256("vfl-run-seed-v1:<k>:<member>") as a uint256, member
// A, p1, p2, p3. Without a run seed each member draws a fresh 256-bit secret (party.mjs), as before. An experiment
// setting only: a deployment's members draw their secrets at random.
import { keccak256, toUtf8Bytes } from "ethers";

export function runSeedSecret(runSeed, member) {
  if (!/^\d+$/.test(String(runSeed))) throw new Error(`run seed ${runSeed}: a non-negative integer`);
  if (!/^(A|p[1-9])$/.test(member)) throw new Error(`member ${member}: A, p1, p2, ...`);
  return BigInt(keccak256(toUtf8Bytes(`vfl-run-seed-v1:${runSeed}:${member}`)));
}
