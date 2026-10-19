// Off-chain side of the M1 federation contract: seed commitments, the seed s, and
// a default proposal. Computed exactly as Federation.sol does; the tests compare them.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { AbiCoder, getBytes, hexlify, keccak256, randomBytes, solidityPackedKeccak256, toUtf8Bytes } from "ethers";

import { REPO_ROOT } from "./talmud.js";

export const BN254_R =
  21888242871839275222246405745257275088548364400416034343698204186575808495617n;

export enum State {
  Setup, Alignment, Revealing, Training, SetupAborted, RevealFailed, AlignmentFailed,
  Reconciliation, Scoring, Scored, ScoringFailed,
  Settled, // M4
}

/** c = keccak256(abi.encode(r, member, federation)), r a uint256. */
export function seedCommitment(r: bigint, member: string, federation: string): string {
  return keccak256(AbiCoder.defaultAbiCoder().encode(["uint256", "address", "address"], [r, member, federation]));
}

/** s = uint256(keccak256(abi.encodePacked(r_A, r_1, ..., r_n))) mod BN254_R; passives in proposal order. */
export function seedFromReveals(rA: bigint, rPassives: bigint[]): bigint {
  const rs = [rA, ...rPassives];
  return BigInt(solidityPackedKeccak256(rs.map(() => "uint256"), rs)) % BN254_R;
}

/** A fresh uniformly random uint256 (the member's secret r). */
export const randomR = (): bigint => BigInt(hexlify(randomBytes(32)));

/** A 32-byte channel or encryption public key (X25519 length; random bytes in tests). */
export const randomKey = (): string => hexlify(randomBytes(32));

/** keccak256 of the S2b verification key file (N = 64, |I| = 1,000), the verifier the tests deploy. */
export const VK_FILE = resolve(REPO_ROOT, "results", "s2b", "proofs", "N64-I1000", "verification_key.json");
export const vkHash = (): string => keccak256(getBytes(readFileSync(VK_FILE)));

export interface Proposal {
  activeParty: string;
  activeChannelKey: string;
  passives: string[];
  arbiter: string;
  metricId: number;
  classBoundaries: bigint[];
  modelConfigHash: string;
  T: number;
  N: number;
  maxDepth: number;
  verifier: string;
  vkHash: string;
  setupPeriod: number;
  alignmentPeriod: number;
  revealWindow: number;
  deltaR: number;
  deltaA: number;
  scoringPeriod: number;
  K: number;
  finalWait: number; // M4, D129: A's wait for every commitFinal (off-chain rule; stored only)
  token: string;
  B: bigint;
  rho: bigint;
  Dp: bigint;
  Dc: bigint;
}

/**
 * A valid proposal. Amounts are plain token units, chosen odd so that every
 * integer remainder in the reveal-failure rules is exercised.
 */
export function defaultProposal(o: {
  activeParty: string; passives: string[]; arbiter: string; verifier: string; token: string;
}, overrides: Partial<Proposal> = {}): Proposal {
  return {
    activeParty: o.activeParty,
    activeChannelKey: randomKey(),
    passives: o.passives,
    arbiter: o.arbiter,
    metricId: 1,
    classBoundaries: [100n, 200n, 300n, 400n], // tariff tier boundaries -> 5 classes
    modelConfigHash: keccak256(toUtf8Bytes("M1 test model configuration")),
    T: 20,
    N: 64,
    maxDepth: 10, // sizeI <= 1024
    verifier: o.verifier,
    vkHash: vkHash(),
    setupPeriod: 50,
    alignmentPeriod: 40,
    revealWindow: 10,
    deltaR: 20,
    deltaA: 30,
    scoringPeriod: 100,
    K: 3,
    finalWait: 15,
    token: o.token,
    B: 1_000_003n,
    rho: 5_000n,
    Dp: 1_001n,
    Dc: 501n,
    ...overrides,
  };
}
