// M3 §4: the arbiter relay for a final message A says it did not receive (D110).
//   1. The passive seals h_i and its signature to the arbiter's registered X25519 key.
//   2. It hands the sealed box to the arbiter.
//   3. The arbiter opens it, checks the signature and that the root equals the member's
//      current gamma on-chain.
//   4. It seals the same final message to A's registered key (activeChannelKey), delivers
//      it, and certifies the delivery on-chain (certifyDelivery), done by the caller.
//
// Sealed box (built from the M2 primitives; nothing new is implemented):
//   eph      = fresh X25519 key pair
//   key      = HKDF-SHA256(ikm = X25519(eph.sk, recipientPk), salt = contract (20 bytes),
//                          info = "vfl-relay-v1" || chainId (32, BE) || member (20) || eph.pk || recipientPk, L = 32)
//   box      = eph.pk (32) || ChaCha20-Poly1305(key, nonce = 12 zero bytes, AD = "vfl-relay-v1").encrypt(message)
// The key is used once, so the fixed nonce is safe.
// Transport: every hop is a Session (session.mjs) keyed as in M2 (deriveChannelKeys) from the two
// registered keys, with the arbiter in the "active" slot for the passive -> arbiter hop and in the
// "passive" slot for the arbiter -> A hop.
import { chacha20poly1305 } from "@noble/ciphers/chacha.js";
import { x25519 } from "@noble/curves/ed25519.js";
import { hkdf } from "@noble/hashes/hkdf.js";
import { sha256 } from "@noble/hashes/sha2.js";

import { channelKeyPair, hexToBytes } from "./channel.mjs";
import { addressBytes, concatBytes, uintBE, utf8 } from "./encoding.mjs";
import { decFinal, embeddingRootHex, finalSigner } from "./training.mjs";

export const RMSG = { box: "relay/box" };
const TAG = utf8("vfl-relay-v1");

function boxKey({ ss, contract, chainId, member, ephPk, recipientPk }) {
  if (ss.every((b) => b === 0)) throw new Error("X25519 shared secret is zero");
  const info = concatBytes(TAG, uintBE(chainId, 32), addressBytes(member), ephPk, recipientPk);
  return hkdf(sha256, ss, addressBytes(contract), info, 32);
}

/** Seals `message` to `recipientPk` (bytes or hex) for this run and member. */
export function seal({ message, recipientPk, contract, chainId, member }) {
  const rpk = typeof recipientPk === "string" ? hexToBytes(recipientPk) : recipientPk;
  const eph = channelKeyPair();
  const key = boxKey({ ss: x25519.getSharedSecret(eph.sk, rpk), contract, chainId, member, ephPk: eph.pk, recipientPk: rpk });
  return concatBytes(eph.pk, chacha20poly1305(key, new Uint8Array(12), TAG).encrypt(message));
}

/** Opens a sealed box with the recipient's secret key; throws if it does not authenticate. */
export function open({ box, recipientSk, contract, chainId, member }) {
  const ephPk = box.subarray(0, 32);
  const rpk = x25519.getPublicKey(recipientSk);
  const key = boxKey({ ss: x25519.getSharedSecret(recipientSk, ephPk), contract, chainId, member, ephPk, recipientPk: rpk });
  return chacha20poly1305(key, new Uint8Array(12), TAG).decrypt(box.subarray(32));
}

/**
 * The arbiter's checks and re-sealing for one box from `member`. fed: the run's contract
 * (read-only). Returns { ok, reason?, root, boxForA }.
 */
export async function arbiterRelay({ fed, box, arbiterSk, member, chainId }) {
  const contract = await fed.getAddress();
  let message;
  try { message = open({ box, recipientSk: arbiterSk, contract, chainId, member }); } catch { return { ok: false, reason: "box does not open" }; }
  const { sig, hq } = decFinal(message);
  const p = await fed.proposal();
  const sizeI = Number(await fed.sizeI());
  if (hq.length !== sizeI) return { ok: false, reason: `matrix has ${hq.length} rows, sizeI is ${sizeI}` };
  const root = await embeddingRootHex(hq, Number(p.maxDepth));
  const signer = finalSigner({ chainId, contract, sender: member, root, sig });
  if (!signer || signer.toLowerCase() !== member.toLowerCase()) return { ok: false, reason: "signature does not verify", root };
  const gamma = (await fed.finalStatus(member)).gamma;
  if (gamma.toLowerCase() !== root.toLowerCase()) return { ok: false, reason: "root differs from the member's gamma", root };
  const boxForA = seal({ message, recipientPk: p.activeChannelKey, contract, chainId, member });
  return { ok: true, root, boxForA };
}

/** A's side: opens the relayed box with its channel secret key and checks it like a direct final message. */
export async function activeOpenRelayed({ box, activeSk, contract, chainId, member, maxDepth }) {
  const { sig, hq } = decFinal(open({ box, recipientSk: activeSk, contract, chainId, member }));
  const root = await embeddingRootHex(hq, maxDepth);
  const signer = finalSigner({ chainId, contract, sender: member, root, sig });
  return { hq, sig, root, signatureValid: signer !== null && signer.toLowerCase() === member.toLowerCase() };
}
