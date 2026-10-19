import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const VECTORS_PATH = resolve(REPO_ROOT, "contracts", "test", "vectors", "talmud_vectors.json");

export interface DivisionCase {
  id: string;
  kind: "random" | "edge" | "textbook" | "gas";
  regime: "surplus" | "cea" | "cel";
  estate: string;
  claims: string[];
  expected: string[];
  n?: number;
}

export interface CappedCase {
  id: string;
  rho: string;
  pool: string;
  estatePerf: string;
  claimsPerf: string[];
  expectedEstate: string;
  expected: string[];
  regime: string;
}

export interface Vectors {
  meta: Record<string, unknown>;
  cases: DivisionCase[];
  capped: CappedCase[];
  gas: DivisionCase[];
}

export function loadVectors(): Vectors {
  return JSON.parse(readFileSync(VECTORS_PATH, "utf8")) as Vectors;
}

export const PROPERTIES = [
  "sumAtMostEstate",        // sum(awards) <= estate
  "roundingLossBelowN",     // min(estate, C) - sum(awards) < n; surplus pays exactly C
  "zeroClaimZeroAward",
  "equalClaimsEqualAwards",
  "awardAtMostClaim",
] as const;
export type Property = (typeof PROPERTIES)[number];

/** Returns the properties that the given contract output violates (empty = all hold). */
export function propertyViolations(estate: bigint, claims: bigint[], awards: bigint[]): Property[] {
  const v = new Set<Property>();
  const n = BigInt(claims.length);
  const total = claims.reduce((a, b) => a + b, 0n);
  const sum = awards.reduce((a, b) => a + b, 0n);
  const payable = estate < total ? estate : total;

  if (sum > estate) v.add("sumAtMostEstate");
  if (payable - sum >= (n > 0n ? n : 1n) || sum > payable) v.add("roundingLossBelowN");
  if (estate >= total && sum !== total) v.add("roundingLossBelowN");
  for (let i = 0; i < claims.length; i++) {
    if (claims[i] === 0n && awards[i] !== 0n) v.add("zeroClaimZeroAward");
    if (awards[i] > claims[i]) v.add("awardAtMostClaim");
    for (let j = i + 1; j < claims.length; j++) {
      if (claims[i] === claims[j] && awards[i] !== awards[j]) v.add("equalClaimsEqualAwards");
    }
  }
  return [...v];
}

export const big = (xs: string[]): bigint[] => xs.map((x) => BigInt(x));
