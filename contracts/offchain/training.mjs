// M3 §3: split training over the M2 session layer (D108), the zero rule (D109) and the final
// round (D75, D83). A holds f_A, the fusion model and the labels; passive i holds f_i and its
// feature columns. Rows are addressed by their position in I (sorted ascending, as committed).
//
// Messages (every one sealed and carried by a Session, session.mjs):
//   train/batch  A -> p  round u32 | batch u32 | count u16 | positions u16[count]
//   train/emb    p -> A  round u32 | batch u32 | float32 BE [count x 4]  (clipped and noised)
//   train/grad   A -> p  round u32 | batch u32 | float32 BE [count x 4]  (not sent to a zero-filled slot)
//   train/end    A -> p  (empty) after the last batch of round T
//   train/final  p -> A  EIP-712 signature (65 bytes) | int16 BE [|I| x 4]  (h_i, in I order)
import { randomBytes } from "node:crypto";
import { Signature, verifyTypedData } from "ethers";

import { buildTree, leafP } from "../../circuits/scripts/members-lib.mjs";
import { concatBytes, readUintBE, runDomain, uintBE } from "./encoding.mjs";
import { CONFIG, E, SCALE, batchesFor, clipAndNoise, newFA, newFi, newFusion, q, softmaxCE } from "./model.mjs";

export const TMSG = { batch: "train/batch", emb: "train/emb", grad: "train/grad", end: "train/end", final: "train/final" };
export const FINAL_MARKER = "final-round";
export const FINAL_TYPES = { FinalMessage: [
  { name: "sender", type: "address" }, { name: "root", type: "bytes32" }, { name: "marker", type: "string" },
] };
export const finalDomain = runDomain;

// ---- Encodings ---------------------------------------------------------------------

const f32 = (arr) => { const b = Buffer.alloc(arr.length * 4); arr.forEach((v, i) => b.writeFloatBE(v, i * 4)); return new Uint8Array(b); };
const fromF32 = (bytes) => { const b = Buffer.from(bytes); return Float64Array.from({ length: b.length / 4 }, (_, i) => b.readFloatBE(i * 4)); };
const head = (round, batch) => concatBytes(uintBE(round, 4), uintBE(batch, 4));
const readHead = (p) => ({ round: Number(readUintBE(p, 0, 4)), batch: Number(readUintBE(p, 4, 4)) });

export const encBatch = (round, batch, positions) => {
  const b = Buffer.alloc(2 + 2 * positions.length);
  b.writeUInt16BE(positions.length, 0);
  positions.forEach((v, i) => b.writeUInt16BE(v, 2 + 2 * i));
  return concatBytes(head(round, batch), new Uint8Array(b));
};
export const decBatch = (p) => {
  const b = Buffer.from(p.subarray(8));
  return { ...readHead(p), positions: Array.from({ length: b.readUInt16BE(0) }, (_, i) => b.readUInt16BE(2 + 2 * i)) };
};
export const encVec = (round, batch, values) => concatBytes(head(round, batch), f32(values));
export const decVec = (p) => ({ ...readHead(p), values: fromF32(p.subarray(8)) });

export function encFinal(sig, hq) {
  const b = Buffer.alloc(hq.length * E * 2);
  hq.forEach((row, r) => row.forEach((v, e) => b.writeInt16BE(v, (r * E + e) * 2)));
  return concatBytes(Buffer.from(Signature.from(sig).serialized.slice(2), "hex"), new Uint8Array(b)); // r | s | v, 65 bytes
}
export function decFinal(p) {
  const sig = "0x" + Buffer.from(p.subarray(0, 65)).toString("hex");
  const b = Buffer.from(p.subarray(65));
  const n = b.length / (2 * E);
  return { sig, hq: Array.from({ length: n }, (_, r) => Array.from({ length: E }, (_, e) => b.readInt16BE((r * E + e) * 2))) };
}

/** gamma_i: Poseidon Merkle root at depth maxDepth over the rows of h_i (M1b leaf packing), as bytes32. */
export async function embeddingRootHex(hq, maxDepth) {
  const tree = await buildTree(hq.map(leafP), maxDepth);
  return "0x" + tree.root.toString(16).padStart(64, "0");
}

/** The address that signed (sender, root, "final-round") for this run, or null. */
export function finalSigner({ chainId, contract, sender, root, sig }) {
  try { return verifyTypedData(finalDomain(chainId, contract), FINAL_TYPES, { sender, root, marker: FINAL_MARKER }, sig); } catch { return null; }
}

/** A standard normal draw from the OS CSPRNG (the passives' embedding noise). */
export function cryptoNormal() {
  const b = randomBytes(16);
  const u1 = (Number(b.readBigUInt64BE(0) >> 11n) + 1) / 2 ** 53;
  const u2 = Number(b.readBigUInt64BE(8) >> 11n) / 2 ** 53;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const sendQuiet = (session, type, payload) => session.send(type, payload).catch(() => {}); // failures surface as a failed session

/**
 * Waits until deadline (ms, performance.now) for a message of `type` for (round, batch) on a
 * session; older ones are discarded. Returns the payload, or null (late, or the session failed).
 */
async function awaitFor(peer, type, round, batch, deadline) {
  for (;;) {
    if (peer.session.failed) return null;
    const left = deadline - performance.now();
    if (left <= 0) return null;
    let m;
    try { m = await peer.session.recv(Math.max(1, Math.round(left))); } catch { return null; }
    if (m.type === TMSG.final) { peer.final = m.payload; if (type === TMSG.final) return m.payload; continue; }
    if (m.type !== type) { peer.unexpected = (peer.unexpected ?? 0) + 1; continue; }
    if (type === TMSG.final) return m.payload;
    const h = readHead(m.payload);
    if (h.round === round && h.batch === batch) return m.payload;
    peer.stale = (peer.stale ?? 0) + 1; // a late embedding of an earlier batch
  }
}

/**
 * A's side. peers: [{ index (1..n), address, session }] in member order; rows: { xa[k], y[k] }
 * for each position k of I; Itr: positions of I^tr. Returns the trained f_A and fusion model,
 * each passive's final message check, and per-round statistics.
 * hooks.beforeGrad(round, batch, present) is for tests.
 */
export async function activeTraining({ peers, rows, Itr, sizeI, chainId, contract, maxDepth, waitMs = CONFIG.perBatchWaitMs, T = CONFIG.T, hooks = {} }) {
  const fA = newFA();
  const fusion = newFusion();
  const P = peers.length;
  const rounds = [];
  const zeroFilled = [];
  for (let round = 1; round <= T; round++) {
    const t0 = performance.now();
    const bytes0 = peers.map((p) => ({ out: p.session.bytesOut, in: p.session.bytesIn }));
    let loss = 0;
    const batches = batchesFor(Itr.length, round);
    for (let b = 0; b < batches.length; b++) {
      const pos = batches[b].map((j) => Itr[j]);
      const n = pos.length;
      for (const p of peers) if (!p.session.failed) sendQuiet(p.session, TMSG.batch, encBatch(round, b, pos));
      const deadline = performance.now() + waitMs;
      const embs = await Promise.all(peers.map(async (p) => {
        const m = await awaitFor(p, TMSG.emb, round, b, deadline);
        return m ? decVec(m).values : null;
      }));
      embs.forEach((v, i) => { if (!v) zeroFilled.push({ round, batch: b, passive: peers[i].index }); });
      const XA = Float64Array.from(pos.flatMap((k) => rows.xa[k].map((v) => v / SCALE)));
      const eA = fA.forward(XA, n);
      const D = E * (P + 1);
      const inp = new Float64Array(n * D);
      for (let r = 0; r < n; r++) {
        for (let e = 0; e < E; e++) inp[r * D + e] = eA[r * E + e];
        for (let i = 0; i < P; i++) if (embs[i]) for (let e = 0; e < E; e++) inp[r * D + E * (i + 1) + e] = embs[i][r * E + e];
      }
      const { loss: l, dZ } = softmaxCE(fusion.forward(inp, n), pos.map((k) => rows.y[k]), n);
      loss += l;
      const dInp = fusion.backward(dZ);
      const deA = new Float64Array(n * E);
      for (let r = 0; r < n; r++) for (let e = 0; e < E; e++) deA[r * E + e] = dInp[r * D + e];
      fA.backward(deA);
      await hooks.beforeGrad?.(round, b, embs.map(Boolean));
      peers.forEach((p, i) => {
        if (!embs[i] || p.session.failed) return; // D109: no gradient to a zero-filled slot
        const g = new Float64Array(n * E);
        for (let r = 0; r < n; r++) for (let e = 0; e < E; e++) g[r * E + e] = dInp[r * D + E * (i + 1) + e];
        sendQuiet(p.session, TMSG.grad, encVec(round, b, g));
      });
    }
    rounds.push({ round, ms: performance.now() - t0, loss: loss / batches.length, batches: batches.length,
      bytes: peers.map((p, i) => ({ passive: p.index, out: p.session.bytesOut - bytes0[i].out, in: p.session.bytesIn - bytes0[i].in })) });
  }
  // Final round: every passive sends h_i with its signature; wait for all, or waitMs for the rest.
  for (const p of peers) if (!p.session.failed) sendQuiet(p.session, TMSG.end, new Uint8Array(0));
  const deadline = performance.now() + waitMs;
  const finals = await Promise.all(peers.map(async (p) => {
    const m = p.final ?? (await awaitFor(p, TMSG.final, 0, 0, deadline));
    if (!m) return { passive: p.index, address: p.address, received: false };
    const { sig, hq } = decFinal(m);
    const root = hq.length === sizeI ? await embeddingRootHex(hq, maxDepth) : null;
    const signer = root ? finalSigner({ chainId, contract, sender: p.address, root, sig }) : null;
    return { passive: p.index, address: p.address, received: true, hq, sig, root, bytes: m.length,
      signatureValid: signer !== null && signer.toLowerCase() === p.address.toLowerCase() };
  }));
  return { fA, fusion, rounds, zeroFilled, finals };
}

/**
 * M4 (D129): A's wait before commitFA, an off-chain rule of A's process: until every listed passive has a gamma
 * on-chain, or `finalWait` blocks (the proposal field) after the call. Returns { allCommitted, atBlock }.
 */
export async function waitForFinalCommits({ provider, fed, addrs, finalWait, pollMs = 500 }) {
  const until = (await provider.getBlockNumber()) + finalWait;
  for (;;) {
    const done = (await Promise.all(addrs.map(async (a) => (await fed.finalStatus(a)).finalCommitted))).every(Boolean);
    const atBlock = await provider.getBlockNumber();
    if (done) return { allCommitted: true, atBlock };
    if (atBlock >= until) return { allCommitted: false, atBlock };
    await new Promise((r) => setTimeout(r, pollMs));
  }
}

/**
 * A passive's side, until its final message is sent. xp[k]: its feature row for position k of I.
 * noise: its local { clip, sigma }. normal: the noise source (default: the OS CSPRNG; DATA1 passes a
 * seeded one for reproducible experiments). signer: an ethers signer for its account key (EIP-712).
 * hooks.stopAfterRound: close the session when the first batch of that round + 1 arrives (T03).
 * hooks.beforeSend({ root, hq }): awaited before the final message is sent. M4 (D129): the passive registers its
 * gamma_i (commitFinal) here, so its commitment is on-chain before A can hold its message.
 */
export async function passiveTraining({ session, index, xp, sizeI, chainId, contract, maxDepth, signer, noise, normal = cryptoNormal, hooks = {}, timeoutMs = 600_000 }) {
  const fi = newFi(index);
  let current = null; // { round, batch }: the batch whose forward pass f_i holds
  const t0 = performance.now();
  const stats = { batches: 0, grads: 0, missedGrads: 0 };
  for (;;) {
    const m = await session.recv(timeoutMs);
    if (m.type === TMSG.end) break;
    if (m.type === TMSG.batch) {
      const { round, batch, positions } = decBatch(m.payload);
      if (hooks.stopAfterRound && round > hooks.stopAfterRound) { session.close(); return { stopped: true, stats }; }
      if (current) stats.missedGrads++;
      const n = positions.length;
      const H = fi.forward(Float64Array.from(positions.flatMap((k) => xp[k])), n);
      const Hn = clipAndNoise(Float64Array.from(H), n, noise, normal);
      current = { round, batch };
      stats.batches++;
      sendQuiet(session, TMSG.emb, encVec(round, batch, Hn));
    } else if (m.type === TMSG.grad) {
      const { round, batch, values } = decVec(m.payload);
      if (!current || current.round !== round || current.batch !== batch) continue; // not the batch f_i holds
      fi.backward(values); // straight-through: clipping and noise treated as the identity
      current = null;
      stats.grads++;
    }
  }
  // Final round (D75, D83): f_i over all rows of I in I order, noise, int16 at 2^12.
  const H = clipAndNoise(Float64Array.from(fi.forward(Float64Array.from(xp.flat()), sizeI)), sizeI, noise, normal);
  const hq = Array.from({ length: sizeI }, (_, r) => Array.from({ length: E }, (_, e) => q(H[r * E + e], 12, 16)));
  const root = await embeddingRootHex(hq, maxDepth);
  const signedRoot = hooks.signRoot ?? root; // tests: sign another root than the one registered (R01)
  const sig = await signer.signTypedData(finalDomain(chainId, contract), FINAL_TYPES,
    { sender: await signer.getAddress(), root: signedRoot, marker: FINAL_MARKER });
  const payload = encFinal(sig, hooks.sendMatrix ?? hq);
  await hooks.beforeSend?.({ root, hq });
  if (!hooks.withholdFinal) await session.send(TMSG.final, payload);
  return { hq, root, sig, signedRoot, finalPayload: payload, stats, ms: performance.now() - t0 };
}
