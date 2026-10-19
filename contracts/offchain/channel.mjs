// M2 §5.1: the secure channel between A and one passive.
// Keys: X25519 (the members' channel keys registered on-chain), HKDF-SHA256, one key per
// direction; messages: ChaCha20-Poly1305 with a per-direction 96-bit big-endian counter
// nonce and the message type as associated data. Libraries: @noble/curves, @noble/hashes,
// @noble/ciphers (no primitive is implemented here).
//
// Wire body of one message (framing adds a 4-byte length in link.mjs):
//   kind (1 byte: 0 clear, 1 sealed, 2 ack) | typeLen (1) | type (utf8) | [counter (8, BE) | ciphertext+tag]   (sealed)
//                                                                      | [payload]                          (clear)
//                                                                      | [counter (8, BE) | tag (16)]        (ack)
// Nonces: 4-byte prefix || counter (8, BE); prefix 0 for sealed messages, 1 for acks, so the
// two never share a nonce under the same key. An ack of counter c is the ChaCha20-Poly1305
// encryption of the empty message under the acking side's send key, with type "ack" as AD.
// The retransmission layer on top of this (acks, timers, retries) is session.mjs.
import { randomBytes } from "node:crypto";
import { chacha20poly1305 } from "@noble/ciphers/chacha.js";
import { x25519 } from "@noble/curves/ed25519.js";
import { hkdf } from "@noble/hashes/hkdf.js";
import { sha256 } from "@noble/hashes/sha2.js";

import { addressBytes, concatBytes, readUintBE, uintBE, utf8 } from "./encoding.mjs";

export class ChannelError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}

/** A fresh X25519 key pair; the public key is what a member registers on-chain. */
export function channelKeyPair() {
  const sk = Uint8Array.from(randomBytes(32));
  return { sk, pk: x25519.getPublicKey(sk) };
}

export const hexToBytes = (h) => Uint8Array.from(Buffer.from(h.replace(/^0x/, ""), "hex"));
export const bytesToHex = (b) => "0x" + Buffer.from(b).toString("hex");

/**
 * Both directional keys of the pair (A, passive).
 *   ss   = X25519(sk_self, pk_other_from_chain)
 *   okm  = HKDF-SHA256(ikm = ss, salt = contract address (20 bytes),
 *          info = "vfl-channel-v1" || chainId (32 bytes BE) || A (20) || passive (20), L = 64)
 *   kAp  = okm[0..32), kpA = okm[32..64)
 */
export function deriveChannelKeys({ selfSk, peerPk, contract, chainId, activeParty, passive }) {
  const ss = x25519.getSharedSecret(selfSk, typeof peerPk === "string" ? hexToBytes(peerPk) : peerPk);
  if (ss.every((b) => b === 0)) throw new ChannelError("key", "X25519 shared secret is zero (low-order public key)");
  const info = concatBytes(utf8("vfl-channel-v1"), uintBE(chainId, 32), addressBytes(activeParty), addressBytes(passive));
  const okm = hkdf(sha256, ss, addressBytes(contract), info, 64);
  return { kAp: okm.slice(0, 32), kpA: okm.slice(32, 64) };
}

const NONCE_DATA = 0;
const NONCE_ACK = 1;
const nonceOf = (counter, prefix = NONCE_DATA) => concatBytes(uintBE(prefix, 4), uintBE(counter, 8));
const ACK_TYPE = utf8("ack");

/** One side's view of a channel: sends with its direction's key, receives with the other. */
export class Channel {
  /** role "A" or "p"; keys from deriveChannelKeys. */
  constructor(role, { kAp, kpA }) {
    this.sendKey = role === "A" ? kAp : kpA;
    this.recvKey = role === "A" ? kpA : kAp;
    this.sendCounter = 0n;
    this.lastAccepted = -1n;
    this.stats = { sent: 0, received: 0, rejected: 0, plainBytesOut: 0, wireBytesOut: 0 };
  }

  /** Encrypts one message; returns its wire body. */
  seal(type, plaintext) {
    const t = utf8(type);
    if (t.length > 255) throw new Error("type too long");
    const counter = this.sendCounter++;
    const ct = chacha20poly1305(this.sendKey, nonceOf(counter), t).encrypt(plaintext);
    const body = concatBytes(Uint8Array.of(1, t.length), t, uintBE(counter, 8), ct);
    this.stats.sent++;
    this.stats.plainBytesOut += plaintext.length;
    this.stats.wireBytesOut += body.length + 4;
    return body;
  }

  /** Decrypts one wire body; accepts it only if its counter is last_accepted + 1. */
  open(body) {
    const { kind, type, rest } = parseBody(body);
    if (kind !== 1) throw this.reject("format", `expected a sealed message, got clear "${type}"`);
    if (rest.length < 8 + 16) throw this.reject("format", "sealed message too short");
    const counter = readUintBE(rest, 0, 8);
    if (counter !== this.lastAccepted + 1n) {
      throw this.reject("counter", `counter ${counter}, expected ${this.lastAccepted + 1n}`);
    }
    let pt;
    try {
      pt = chacha20poly1305(this.recvKey, nonceOf(counter), utf8(type)).decrypt(rest.subarray(8));
    } catch {
      throw this.reject("tag", `authentication failed (type "${type}", counter ${counter})`);
    }
    this.lastAccepted = counter;
    this.stats.received++;
    return { type, payload: pt };
  }

  reject(code, msg) { this.stats.rejected++; return new ChannelError(code, msg); }

  /**
   * Receive side for session.mjs: classifies one sealed body without throwing.
   *   next       counter is last_accepted + 1 and the tag verifies: accepted (payload returned)
   *   duplicate  counter <= last_accepted and the tag verifies: an earlier message sent again
   *   future     counter > last_accepted + 1: not accepted (the tag is not checked)
   *   tag        the tag does not verify
   *   format     not a well-formed sealed body
   * Only "next" advances last_accepted, so the replay rule is the one open() enforces.
   */
  inspect(body) {
    let parsed;
    try { parsed = parseBody(body); } catch { return { status: "format" }; }
    const { kind, type, rest } = parsed;
    if (kind !== 1 || rest.length < 8 + 16) return { status: "format" };
    const counter = readUintBE(rest, 0, 8);
    if (counter > this.lastAccepted + 1n) return { status: "future", counter };
    let payload;
    try {
      payload = chacha20poly1305(this.recvKey, nonceOf(counter), utf8(type)).decrypt(rest.subarray(8));
    } catch {
      return { status: "tag", counter };
    }
    if (counter <= this.lastAccepted) return { status: "duplicate", counter };
    this.lastAccepted = counter;
    this.stats.received++;
    return { status: "next", counter, type, payload };
  }

  /** The acknowledgement of a received counter, under this side's send key. */
  sealAck(counter) {
    const tag = chacha20poly1305(this.sendKey, nonceOf(counter, NONCE_ACK), ACK_TYPE).encrypt(new Uint8Array(0));
    return concatBytes(Uint8Array.of(2, ACK_TYPE.length), ACK_TYPE, uintBE(counter, 8), tag);
  }

  /** The counter an ack body acknowledges, or null if it is malformed or its tag does not verify. */
  openAck(body) {
    let parsed;
    try { parsed = parseBody(body); } catch { return null; }
    const { kind, type, rest } = parsed;
    if (kind !== 2 || type !== "ack" || rest.length !== 8 + 16) return null;
    const counter = readUintBE(rest, 0, 8);
    try {
      chacha20poly1305(this.recvKey, nonceOf(counter, NONCE_ACK), ACK_TYPE).decrypt(rest.subarray(8));
    } catch {
      return null;
    }
    return counter;
  }
}

/** A clear (unencrypted) message: only the connection hello uses it. */
export function clearBody(type, payload) {
  const t = utf8(type);
  return concatBytes(Uint8Array.of(0, t.length), t, payload);
}

export function parseBody(body) {
  if (body.length < 2) throw new ChannelError("format", "message too short");
  const kind = body[0];
  const tl = body[1];
  if (body.length < 2 + tl) throw new ChannelError("format", "truncated type");
  const type = new TextDecoder().decode(body.subarray(2, 2 + tl));
  return { kind, type, rest: body.subarray(2 + tl) };
}
