// M2 encodings shared by the channel, the PSI and the commitments.
// IDs are 10-digit national IDs held as uint64 (BigInt in JS).
import { keccak256 } from "ethers";

export const utf8 = (s) => new TextEncoder().encode(s);

export function concatBytes(...parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/** 20 raw bytes of a 0x-prefixed address. */
export function addressBytes(addr) {
  const h = addr.toLowerCase().replace(/^0x/, "");
  if (h.length !== 40) throw new Error(`bad address ${addr}`);
  return Uint8Array.from(Buffer.from(h, "hex"));
}

/** Unsigned integer as a fixed-width big-endian byte string. */
export function uintBE(v, width) {
  let x = BigInt(v);
  if (x < 0n || x >= 1n << BigInt(8 * width)) throw new Error(`${v} does not fit in ${width} bytes`);
  const out = new Uint8Array(width);
  for (let i = width - 1; i >= 0; i--) { out[i] = Number(x & 0xffn); x >>= 8n; }
  return out;
}
export const readUintBE = (b, off, width) => {
  let x = 0n;
  for (let i = 0; i < width; i++) x = (x << 8n) | BigInt(b[off + i]);
  return x;
};

/** An ID as 8 bytes big-endian (uint64): the input to hash-to-group and to gamma_I. */
export const idBytes = (id) => uintBE(id, 8);

/** Concatenated uint64 big-endian IDs, and back. */
export const packIds = (ids) => concatBytes(...ids.map(idBytes));
export function unpackIds(b) {
  if (b.length % 8 !== 0) throw new Error("ID list length is not a multiple of 8");
  return Array.from({ length: b.length / 8 }, (_, i) => readUintBE(b, 8 * i, 8));
}

/** The public ordering rule: ascending by ID. */
/** The run's EIP-712 signing domain (final-round message; M4: A's AlignedSet signature). */
export const runDomain = (chainId, contract) => ({ name: "VFLFederation", version: "1", chainId, verifyingContract: contract });

export const sortIds = (ids) => [...ids].sort((x, y) => (x < y ? -1 : x > y ? 1 : 0));

/** gamma_I = keccak256(abi.encodePacked(uint64[] I_sorted)): each ID 8 bytes big-endian. */
export function gammaI(sortedIds) {
  return keccak256(packIds(sortedIds));
}
