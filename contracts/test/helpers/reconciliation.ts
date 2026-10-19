import { MaxUint256, toBeHex, type Signer, type TypedDataField } from "ethers";

// Commitments are computed exactly as the S2b circuit checks them (circomlibjs Poseidon).
// @ts-ignore -- plain ESM module from the circuits package
import { embeddingRoot } from "../../../circuits/scripts/members-lib.mjs";

export const DOMAIN_NAME = "VFL-Reconciliation";
export const DOMAIN_VERSION = "1";
export const FINAL_ROUND = MaxUint256;
export const SNARK_SCALAR_FIELD =
  21888242871839275222246405745257275088548364400416034343698204186575808495617n;

export const FINAL_MESSAGE_TYPES: Record<string, TypedDataField[]> = {
  FinalMessage: [
    { name: "run", type: "address" },
    { name: "party", type: "address" },
    { name: "round", type: "uint256" },
    { name: "digest", type: "bytes32" },
  ],
};

export interface SignContext {
  chainId: bigint;
  verifyingContract: string; // EIP-712 domain: the run's contract address
  run?: string;              // struct field; defaults to verifyingContract
  round?: bigint;            // defaults to FINAL_ROUND
}

export function finalMessage(party: string, digest: string, ctx: SignContext) {
  return {
    domain: { name: DOMAIN_NAME, version: DOMAIN_VERSION, chainId: ctx.chainId, verifyingContract: ctx.verifyingContract },
    types: FINAL_MESSAGE_TYPES,
    value: { run: ctx.run ?? ctx.verifyingContract, party, round: ctx.round ?? FINAL_ROUND, digest },
  };
}

/** The passive party's signature sigma_i = Sig_i(ctx_i, gamma_i), made with its own chain account key. */
export async function signFinal(signer: Signer, digest: string, ctx: SignContext): Promise<string> {
  const m = finalMessage(await signer.getAddress(), digest, ctx);
  return signer.signTypedData(m.domain, m.types, m.value);
}

/**
 * gamma_i = Poseidon Merkle root over the rows of a final-round embedding matrix
 * (leaf = packed row of int16 values, binary tree, zero-padded; see
 * circuits/scripts/members-lib.mjs). No salt (README.md, Protocol rules). Returned as bytes32.
 */
export async function commitEmbeddings(matrix: bigint[][]): Promise<string> {
  return toBeHex(await embeddingRoot(matrix), 32);
}

/** Deterministic pseudo-random embedding matrix of int16 values (fixed point, scale 2^12). */
export function syntheticEmbeddings(seed: number, rows = 64, cols = 4): bigint[][] {
  let s = BigInt(seed) + 0x9e3779b97f4a7c15n;
  const next = () => {
    s = (s * 6364136223846793005n + 1442695040888963407n) & ((1n << 64n) - 1n);
    return Number(s >> 40n) / 2 ** 24; // [0, 1)
  };
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => BigInt(Math.round((next() * 2 - 1) * 4 * 4096))),
  );
}
