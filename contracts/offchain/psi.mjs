// M2 §5.2: ECDH-based PSI over ristretto255 (@noble/curves). The ID-to-point map is the
// RFC 9380 hash-to-group of ristretto255 (ristretto255_hasher.hashToCurve, expand_message_xmd
// with SHA-512) with DST = "vfl-psi-v1" || contract address. Scalars are fresh per run and
// per pair, drawn from the OS CSPRNG, and never the channel keys.
// blindOwn and reblind yield to the event loop every YIELD_EVERY points, so that a process
// computing one pair still reads and acknowledges frames of the others (session.mjs): a
// multi-second synchronous loop would let the peers' retransmission timers expire.
import { randomBytes, randomInt } from "node:crypto";
import { ristretto255, ristretto255_hasher } from "@noble/curves/ed25519.js";

import { addressBytes, concatBytes, idBytes, utf8 } from "./encoding.mjs";

const Point = ristretto255.Point;
export const GROUP_ORDER = Point.Fn.ORDER;
export const POINT_BYTES = 32;

export const psiDST = (contract) => concatBytes(utf8("vfl-psi-v1"), addressBytes(contract));

/** A uniform non-zero scalar mod the group order (64 random bytes, wide reduction). */
export function randomScalar() {
  for (;;) {
    const x = BigInt("0x" + randomBytes(64).toString("hex")) % GROUP_ORDER;
    if (x !== 0n) return x;
  }
}

export const hashId = (id, dst) => ristretto255_hasher.hashToCurve(idBytes(id), { DST: dst });

/** Unbiased Fisher-Yates shuffle with the OS CSPRNG. Returns a new array. */
export function secureShuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) { const j = randomInt(i + 1); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

const pack = (points) => concatBytes(...points.map((P) => P.toBytes()));
export function unpack(bytes) {
  if (bytes.length % POINT_BYTES !== 0) throw new Error("point list length is not a multiple of 32");
  return Array.from({ length: bytes.length / POINT_BYTES },
    (_, i) => Point.fromBytes(bytes.subarray(POINT_BYTES * i, POINT_BYTES * (i + 1)))); // rejects invalid encodings
}

const YIELD_EVERY = 32;
async function mapYielding(n, f) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    if (i > 0 && i % YIELD_EVERY === 0) await new Promise((r) => setImmediate(r));
    out[i] = f(i);
  }
  return out;
}

/** Steps 1 and 3: k·H(x) for each own ID, in a fresh random order. Returns the order kept privately. */
export async function blindOwn(ids, k, dst) {
  const order = secureShuffle(ids);
  return { order, bytes: pack(await mapYielding(order.length, (i) => hashId(order[i], dst).multiply(k))) };
}

/** Step 2 and the first half of step 4: k·P for each received point, in the same order. */
export async function reblind(bytes, k) {
  if (bytes.length % POINT_BYTES !== 0) throw new Error("point list length is not a multiple of 32");
  return pack(await mapYielding(bytes.length / POINT_BYTES,
    (i) => Point.fromBytes(bytes.subarray(POINT_BYTES * i, POINT_BYTES * (i + 1))).multiply(k))); // rejects invalid encodings
}

/**
 * Step 4: A's IDs whose double-locked value b·a·H(x) (from p, in A's shuffled order) is
 * among the a·b·H(y) values computed from p's list.
 */
export function matchIntersection(orderA, doubleOnA, doubleOnP) {
  const theirs = new Set();
  for (let i = 0; i < doubleOnP.length; i += POINT_BYTES) theirs.add(Buffer.from(doubleOnP.subarray(i, i + POINT_BYTES)).toString("hex"));
  const out = [];
  for (let i = 0; i < orderA.length; i++) {
    if (theirs.has(Buffer.from(doubleOnA.subarray(POINT_BYTES * i, POINT_BYTES * (i + 1))).toString("hex"))) out.push(orderA[i]);
  }
  return out;
}
