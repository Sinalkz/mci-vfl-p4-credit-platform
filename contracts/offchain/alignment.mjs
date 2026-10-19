// M2 §5.2-§5.5: the alignment protocol for A and a passive, over any Link (TCP, the relay,
// or an in-memory pair), plus the commitments and the evaluation rows.
// Each side reads the other's channel public key from the contract, never from the peer.
// Sealed messages go through a Session (session.mjs): acknowledged, resent until acknowledged.
import { Channel, clearBody, deriveChannelKeys, hexToBytes, parseBody } from "./channel.mjs";
import { Session } from "./session.mjs";
import { getBytes, hexlify, verifyTypedData } from "ethers";

import { gammaI, packIds, runDomain, sortIds, unpackIds, utf8 } from "./encoding.mjs";
import { blindOwn, matchIntersection, POINT_BYTES, psiDST, randomScalar, reblind } from "./psi.mjs";
// M1b helpers (the scoring circuit's row packing, depth-maxDepth trees, index rule).
import { varTrees } from "../../circuits/scripts/m1b-lib.mjs";
import { sampleIndices } from "../../circuits/scripts/members-lib.mjs";

export const MSG = {
  hello: "hello",
  aBlinded: "psi/a-blinded",   // A -> p: a·H(x), A's IDs shuffled
  pOnA: "psi/p-on-a",          // p -> A: b·(a·H(x)), same order
  pBlinded: "psi/p-blinded",   // p -> A: b·H(y), p's IDs shuffled
  I: "align/I",                // A -> p: I sorted ascending, uint64 big-endian each
  Isig: "align/I-sig",         // A -> p: M4 (D120/D121): A's EIP-712 AlignedSet signature over the hash of that list (65 bytes)
};

/** M4 (D120/D121): A signs, for each passive, the hash of the list I it sent (gamma_I's encoding), in the run's domain. */
export const ALIGNED_TYPES = { AlignedSet: [{ name: "passive", type: "address" }, { name: "listHash", type: "bytes32" }] };

export class ProtocolError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}

/** What each side reads from the contract. `fed` is an ethers Contract on Federation. */
export async function readChain(fed) {
  const p = await fed.proposal();
  const contract = await fed.getAddress();
  const chainId = (await fed.runner.provider.getNetwork()).chainId;
  return {
    contract, chainId, activeParty: p.activeParty, activeKey: p.activeChannelKey, passives: [...p.passives],
    maxDepth: Number(p.maxDepth), N: Number(p.N),
    passiveKey: async (addr) => (await fed.member(addr)).channelKey,
  };
}

const now = () => performance.now();

async function expectMsg(session, type, timeoutMs) {
  const m = await session.recv(timeoutMs);
  if (m.type !== type) throw new ProtocolError("type", `expected ${type}, got ${m.type}`);
  return m.payload;
}

/**
 * A's side of one pair, up to its pairwise intersection. Waits for the passive's clear hello
 * (which only names the passive; any key offered in it is ignored).
 * hooks.onScalar(passive, a) exposes the PSI scalar to tests (P06). sessionOpts: see session.mjs.
 */
export async function activePair({ link, chain, sk, ids, hooks = {}, timeoutMs = 120_000, sessionOpts = {} }) {
  const t0 = now();
  const hello = parseBody(await link.recv(timeoutMs));
  if (hello.kind !== 0 || hello.type !== MSG.hello) throw new ProtocolError("hello", "expected a clear hello");
  const passive = JSON.parse(new TextDecoder().decode(hello.rest)).address;
  if (!chain.passives.some((x) => x.toLowerCase() === passive.toLowerCase())) throw new ProtocolError("hello", `${passive} is not a listed passive`);
  const peerPk = hexToBytes(await chain.passiveKey(passive));
  const session = new Session(link, new Channel("A", deriveChannelKeys({ selfSk: sk, peerPk, contract: chain.contract,
    chainId: chain.chainId, activeParty: chain.activeParty, passive })), sessionOpts);
  const dst = psiDST(chain.contract);
  const a = randomScalar();
  hooks.onScalar?.(passive, a);
  const t1 = now();
  const { order, bytes } = await blindOwn(ids, a, dst);
  hooks.onOrder?.(passive, order);
  await session.send(MSG.aBlinded, bytes);
  const t2 = now();
  const onA = await expectMsg(session, MSG.pOnA, timeoutMs);
  if (onA.length !== order.length * POINT_BYTES) throw new ProtocolError("length", "p returned a list of the wrong length");
  const pBlinded = await expectMsg(session, MSG.pBlinded, timeoutMs);
  const t3 = now();
  const onP = await reblind(pBlinded, a);
  const intersection = matchIntersection(order, onA, onP);
  const t4 = now();
  return { passive, chain, session, link, intersection, pSetSize: pBlinded.length / POINT_BYTES,
    timings: { helloMs: t1 - t0, blindMs: t2 - t1, waitMs: t3 - t2, finishMs: t4 - t3, computeMs: (t2 - t1) + (t4 - t3) } };
}

/**
 * A sends the sorted I to one passive, then its AlignedSet signature over the hash of exactly that list (M4, D120/D121).
 * signer: A's wallet. Tests: tamper(ids) may alter I (A02, A05); tamperSigHash(h) may sign another hash (A13).
 */
export async function sendI(pair, sortedI, tamper, signer, tamperSigHash) {
  if (!signer) throw new ProtocolError("sig", "A must sign the list it sends (AlignedSet)");
  const ids = tamper ? tamper([...sortedI]) : sortedI;
  const listHash = gammaI(ids);
  const sig = await signer.signTypedData(runDomain(pair.chain.chainId, pair.chain.contract), ALIGNED_TYPES,
    { passive: pair.passive, listHash: tamperSigHash ? tamperSigHash(listHash) : listHash });
  // both frames leave together: a passive that refuses I (A05) closes its link right after reading it
  const sent = [pair.session.send(MSG.I, packIds(ids)), pair.session.send(MSG.Isig, getBytes(sig))];
  await Promise.all(sent);
  return { listHash, sig };
}

/** A's whole alignment computation: I = the intersection of the pairwise intersections, sorted. */
export function buildI(pairwise) {
  let acc = new Set(pairwise[0]);
  for (const s of pairwise.slice(1)) { const t = new Set(s); acc = new Set([...acc].filter((x) => t.has(x))); }
  return sortIds([...acc]);
}

/**
 * A passive's side: hello, PSI, then receive I and check every ID is its own.
 * hooks.onScalar(b) and hooks.onOrder(order) expose the scalar and the shuffle to tests.
 */
export async function passiveSide({ link, chain, sk, self, ids, hooks = {}, timeoutMs = 120_000, offeredPk = null, sessionOpts = {} }) {
  const t0 = now();
  await link.send(clearBody(MSG.hello, utf8(JSON.stringify({ address: self, ...(offeredPk ? { pk: offeredPk } : {}) }))));
  const session = new Session(link, new Channel("p", deriveChannelKeys({ selfSk: sk, peerPk: hexToBytes(chain.activeKey),
    contract: chain.contract, chainId: chain.chainId, activeParty: chain.activeParty, passive: self })), sessionOpts);
  const dst = psiDST(chain.contract);
  const aBlinded = await expectMsg(session, MSG.aBlinded, timeoutMs);
  const t1 = now();
  const b = randomScalar();
  hooks.onScalar?.(b);
  await session.send(MSG.pOnA, await reblind(aBlinded, b));
  const { order, bytes } = await blindOwn(ids, b, dst);
  hooks.onOrder?.(order);
  await session.send(MSG.pBlinded, bytes);
  const t2 = now();
  const I = unpackIds(await expectMsg(session, MSG.I, timeoutMs));
  const t3 = now();
  const own = new Set(ids);
  const foreign = I.filter((x) => !own.has(x));
  if (foreign.length > 0) throw new ProtocolError("foreign-id", `received ${foreign.length} ID(s) that are not this passive's customers`);
  // M4 (D120/D121): A's signature over the hash of the list received, for this passive; the evidence of objectAlignment
  const alignSig = hexlify(await expectMsg(session, MSG.Isig, timeoutMs));
  const listHash = gammaI(I);
  let signer = null;
  try { signer = verifyTypedData(runDomain(chain.chainId, chain.contract), ALIGNED_TYPES, { passive: self, listHash }, alignSig); } catch { /* malformed */ }
  if (!signer || signer.toLowerCase() !== chain.activeParty.toLowerCase()) {
    throw new ProtocolError("sig", "the list is not signed by A for this passive and this list (AlignedSet)");
  }
  return { I, alignSig, listHash, aSetSize: aBlinded.length / POINT_BYTES, session, link,
    timings: { waitAMs: t1 - t0, computeMs: t2 - t1, waitIMs: t3 - t2 } };
}

/** γ_A: Poseidon Merkle root at depth maxDepth over A's rows (X_A, y) in the order of I (M1b varTrees). */
export async function gammaA(sortedI, rows, maxDepth) {
  const table = { y: sortedI.map((id) => rows.get(id).y), xa: sortedI.map((id) => rows.get(id).xa), hp: [] };
  const trees = await varTrees({ table }, sortedI.length, maxDepth);
  return "0x" + trees.A.root.toString(16).padStart(64, "0");
}

/** A passive's check of γ_I against the chain. Prints a clear PASS or FAIL line. */
export async function checkGammaI(fed, receivedI, label = "passive") {
  const onChain = (await fed.gammaI()).toLowerCase();
  const local = gammaI(receivedI).toLowerCase();
  const pass = onChain === local;
  console.log(`[${label}] gamma_I check ${pass ? "PASS" : "FAIL"}: local ${local}, on-chain ${onChain}`);
  return { pass, onChain, local };
}

/** §5.5: I^ev (one ID per draw, with repeats) and I^tr (the rows of I no draw picked), from s and sizeI on-chain. */
export async function evaluationRows(fed, sortedI) {
  const s = await fed.seed();
  const sizeI = Number(await fed.sizeI());
  const N = Number((await fed.proposal()).N);
  if (sizeI !== sortedI.length) throw new ProtocolError("size", `on-chain sizeI ${sizeI} != |I| ${sortedI.length}`);
  const idx = await sampleIndices(s, N, sizeI);
  const picked = new Set(idx);
  return { s, sizeI, N, idx, Iev: idx.map((k) => sortedI[k]), Itr: sortedI.filter((_, k) => !picked.has(k)) };
}
