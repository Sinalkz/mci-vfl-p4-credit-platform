// M2 runner: the proposal A creates (same values as contracts/test/helpers/federation.ts,
// plus the M2 alignmentPeriod). vkHash is keccak256 of the verification key file of the verifier the runner deploys: by
// default the multi-party key's (results/mpc/setup/verification_key.json, as runner3's default --setup mpc);
// VFL_VK_FILE names another key file (the development key: results/m1b/verification_key.json).
// Before building the proposal, A's own customer count is checked against 2^maxDepth
// (answer to QUESTIONS-M2 Q5, see checkAlignmentCapacity).
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { keccak256 } from "ethers";

import { CONFIG, modelConfigHash } from "./model.mjs";

const VK = process.env.VFL_VK_FILE ? resolve(process.env.VFL_VK_FILE)
  : resolve(fileURLToPath(import.meta.url), "..", "..", "..", "results", "mpc", "setup", "verification_key.json");

/**
 * |I| can never exceed |X_A|. If 2^maxDepth >= |X_A|, an honest A can always commit its
 * alignment; otherwise commitAlignment may revert with SizeAboveCap and A, having done nothing
 * wrong, pays the alignment-timeout compensation (D107). mode "refuse" (the default) throws;
 * mode "warn" prints a loud warning and returns { ok: false, warning }.
 */
export function checkAlignmentCapacity({ maxDepth, activeCustomers, mode = "refuse" }) {
  if (!Number.isInteger(activeCustomers) || activeCustomers < 0) throw new Error("activeCustomers (|X_A|) is required");
  if (mode !== "refuse" && mode !== "warn") throw new Error(`unknown capacity check mode ${mode}`);
  const cap = 2 ** maxDepth;
  if (activeCustomers <= cap) return { ok: true, cap };
  const warning = `|X_A| = ${activeCustomers} exceeds 2^maxDepth = ${cap}: if |I| turns out larger than ${cap}, ` +
    "commitAlignment reverts (SizeAboveCap) and A pays the alignment-timeout compensation";
  if (mode === "refuse") throw new Error(`proposal refused: ${warning}`);
  const bar = "!".repeat(100);
  console.warn(`${bar}
WARNING ${warning}
${bar}`);
  return { ok: false, cap, warning };
}

/** o.activeCustomers = |X_A| (required); o.capacityCheck = "refuse" (default) or "warn". */
export function defaultProposal(o) {
  const maxDepth = o.maxDepth ?? 10;
  checkAlignmentCapacity({ maxDepth, activeCustomers: o.activeCustomers, mode: o.capacityCheck ?? "refuse" });
  return {
    activeParty: o.activeParty,
    activeChannelKey: o.activeChannelKey,
    passives: o.passives,
    arbiter: o.arbiter,
    metricId: 1,
    classBoundaries: o.classBoundaries ?? [100n, 200n, 300n, 400n], // DATA1 E3: the RECS kWh cut points
    modelConfigHash: modelConfigHash(), // M3: keccak256 of offchain/model-config.json
    T: CONFIG.T,
    N: 64,
    maxDepth,
    verifier: o.verifier,
    vkHash: keccak256(readFileSync(VK)),
    setupPeriod: o.setupPeriod ?? 150,
    alignmentPeriod: o.alignmentPeriod ?? 60,
    revealWindow: o.revealWindow ?? 30,
    deltaR: o.deltaR ?? 20,
    deltaA: 30,
    scoringPeriod: o.scoringPeriod ?? 400,
    K: 3,
    finalWait: o.finalWait ?? 15, // M4, D129: A's wait for every commitFinal (was party-m3's FINAL_COMMIT_WAIT_BLOCKS)
    token: o.token,
    B: 1_000_003n,
    rho: 5_000n,
    Dp: 1_001n,
    Dc: 501n,
  };
}
