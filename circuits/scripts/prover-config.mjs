// The scoring prover's defaults, shared by circuits/scripts/m3-prove-figure.mjs (each figure's proof, in its own process)
// and contracts/offchain/runner3.mjs (the protocol run): the multi-party key of scripts/mpc-ceremony.mjs
// (build/members-var-N64-D10-mpc, results/mpc/setup) and rapidsnark (iden3's native prover; RAPIDSNARK = its prover
// binary). The development key and snarkjs stay selectable: VFL_KEY_DIR / VFL_PROVER, or runner3's --setup dev /
// --prover snarkjs. A missing rapidsnark binary or key file fails at once, before any work.
// full-data: the multi-party key of the maxDepth 14 circuit (mpc-ceremony.mjs --maxdepth 14) is
// build/members-var-N64-D14-mpc; the default key folder follows the circuit's maxDepth (mpcKeyDir).
import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { BUILD } from "./lib.mjs";
import { varBuildDir } from "./m1b-lib.mjs";

/** The multi-party key's folder for the N 64 circuit at `maxDepth` (mpc-ceremony.mjs's default --keys-dir). */
export const mpcKeyDir = (maxDepth) => resolve(BUILD, `members-var-N64-D${maxDepth}-mpc`);
export const KEY_DIRS = { dev: varBuildDir(64, 10), mpc: mpcKeyDir(10), "mpc-d14": mpcKeyDir(14) };
export const DEFAULT_KEY_DIR = KEY_DIRS.mpc;
export const PROVERS = ["snarkjs", "rapidsnark"];
export const DEFAULT_PROVER = "rapidsnark";

/** The rapidsnark prover binary `exe` (RAPIDSNARK), resolved; throws if it is unset or does not exist. */
export function checkRapidsnark(exe) {
  if (exe && existsSync(exe)) return resolve(exe);
  throw new Error(`rapidsnark prover not found (RAPIDSNARK ${exe ? `= ${exe}: no such file` : "is not set"}): ` +
    "pass --prover snarkjs (VFL_PROVER=snarkjs) or set RAPIDSNARK to rapidsnark's prover binary");
}

/** A proving key folder; throws unless it holds main_final.zkey and verification_key.json. */
export function checkKeyDir(dir) {
  const missing = ["main_final.zkey", "verification_key.json"].filter((f) => !existsSync(resolve(dir, f)));
  if (missing.length) {
    throw new Error(`proving key folder ${dir} lacks ${missing.join(", ")}: run circuits/scripts/mpc-ceremony.mjs or copy ` +
      "its key there, or pass --setup dev (VFL_KEY_DIR=<the development key's folder>)");
  }
  return dir;
}

/**
 * The key folder and prover from the environment (VFL_KEY_DIR, VFL_PROVER, RAPIDSNARK), with the defaults above; without
 * VFL_KEY_DIR, the multi-party key of the circuit at `maxDepth` (default 10, the deployed circuit).
 */
export function proverFromEnv(env = process.env, { maxDepth = 10 } = {}) {
  const prover = env.VFL_PROVER ?? DEFAULT_PROVER;
  if (!PROVERS.includes(prover)) throw new Error(`VFL_PROVER ${prover}: snarkjs | rapidsnark`);
  return { keyDir: env.VFL_KEY_DIR ? resolve(env.VFL_KEY_DIR) : mpcKeyDir(maxDepth), prover,
    rapidsnark: prover === "rapidsnark" ? checkRapidsnark(env.RAPIDSNARK) : null };
}
