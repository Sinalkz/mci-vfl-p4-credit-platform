// Task 4 item 1, attribution: the protocol's per-batch extra steps timed in isolation, one batch of 64 rows and one
// passive, with the protocol's own functions. It explains the end-to-end differences of train-bench.mjs; it does not
// replace them.
//   noise   model.mjs clipAndNoise on a 64 x 4 batch of embeddings:
//             plain's call     clip = infinity, sigma = 0, a zero noise source (what train-variants.mjs's plain runs)
//             clipping only    the passive's clip from LOCAL_NOISE, sigma = 0
//             clip and noise   LOCAL_NOISE (clip and sigma) with the OS CSPRNG (training.mjs cryptoNormal), as the protocol
//           each minus the copy of the batch it works on
//   frames  the three messages each passive exchanges per batch, with their real sizes (training.mjs encBatch / encVec):
//           train/batch (A -> p, 8 + 2 + 2 x 64 bytes), train/emb (p -> A) and train/grad (A -> p) (8 + 64 x 4 x 4 bytes):
//             clear frame      channel.mjs clearBody + parseBody (what PlainSession does)
//             secure channel   Channel.seal + Channel.inspect + the receiver's sealAck + the sender's openAck (Session)
// Median over --blocks blocks of --iters calls each, after one discarded block.
//   node offchain/overhead/steps-bench.mjs [--blocks 21] [--iters 2000] --out <file>
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { Channel, channelKeyPair, clearBody, deriveChannelKeys, parseBody } from "../channel.mjs";
import { gauss, prng } from "../data.mjs";
import { machine, peakRssMB } from "../data1/machine.mjs";
import { E, clipAndNoise } from "../model.mjs";
import { LOCAL_NOISE } from "../party-m3.mjs";
import { cryptoNormal, encBatch, encVec, TMSG } from "../training.mjs";

const { values: a } = parseArgs({ options: { blocks: { type: "string", default: "21" }, iters: { type: "string", default: "2000" },
  out: { type: "string" } } });
const BLOCKS = Number(a.blocks);
const ITERS = Number(a.iters);
const N = 64;
const m0 = machine();

/** Median microseconds per call of fn over BLOCKS blocks of `iters` calls, after one discarded block. */
function perCall(fn, iters = ITERS) {
  const us = [];
  for (let b = 0; b <= BLOCKS; b++) {
    const s = performance.now();
    for (let k = 0; k < iters; k++) fn(k);
    if (b > 0) us.push(((performance.now() - s) * 1000) / iters);
  }
  const sorted = [...us].sort((x, y) => x - y);
  return { usMedian: sorted[Math.floor(sorted.length / 2)], usMean: us.reduce((s, x) => s + x, 0) / us.length, usMin: sorted[0], usMax: sorted.at(-1), blocks: BLOCKS, iters };
}

const rng = prng(20264300n);
const H0 = Float64Array.from({ length: N * E }, () => gauss(rng) * 2); // embedding-like values, some rows above the clip
const zero = () => 0;
let sink = 0;
const copy = perCall(() => { const H = Float64Array.from(H0); sink += H[0]; });
const noise = LOCAL_NOISE.map((ln, i) => {
  const plainCall = perCall(() => { const H = clipAndNoise(Float64Array.from(H0), N, { clip: Infinity, sigma: 0 }, zero); sink += H[0]; });
  const clipOnly = perCall(() => { const H = clipAndNoise(Float64Array.from(H0), N, { clip: ln.clip, sigma: 0 }, zero); sink += H[0]; });
  const clipNoise = perCall(() => { const H = clipAndNoise(Float64Array.from(H0), N, ln, cryptoNormal); sink += H[0]; }, ITERS / 4);
  const net = (x) => ({ ...x, usNetMedian: x.usMedian - copy.usMedian });
  return { passive: i + 1, setting: ln, plainCall: net(plainCall), clipOnly: net(clipOnly), clipNoise: net(clipNoise) };
});

const kp = channelKeyPair();
const kq = channelKeyPair();
const ctx = { contract: "0x" + "c0".repeat(20), chainId: 2026n, activeParty: "0x" + "a0".repeat(20), passive: "0x" + "b0".repeat(20) };
const kA = deriveChannelKeys({ selfSk: kp.sk, peerPk: kq.pk, ...ctx });
const kP = deriveChannelKeys({ selfSk: kq.sk, peerPk: kp.pk, ...ctx });
const positions = Array.from({ length: N }, (_, k) => k * 13);
const vec = Float64Array.from({ length: N * E }, () => gauss(rng));
const messages = [
  { type: TMSG.batch, from: "A", payload: encBatch(3, 7, positions) },
  { type: TMSG.emb, from: "p", payload: encVec(3, 7, vec) },
  { type: TMSG.grad, from: "A", payload: encVec(3, 7, vec) },
];
const frames = messages.map(({ type, from, payload }) => {
  const clear = perCall(() => { const m = parseBody(clearBody(type, payload)); sink += m.rest.length; });
  const sender = new Channel(from, from === "A" ? kA : kP);
  const receiver = new Channel(from === "A" ? "p" : "A", from === "A" ? kP : kA);
  const channel = perCall(() => {
    const body = sender.seal(type, payload);
    const r = receiver.inspect(body);
    if (r.status !== "next") throw new Error(`inspect ${r.status}`);
    const ack = receiver.sealAck(r.counter);
    if (sender.openAck(ack) !== r.counter) throw new Error("ack");
    sink += r.payload.length;
  }, ITERS / 4);
  return { type, payloadBytes: payload.length, clearFrameBytes: clearBody(type, payload).length + 4,
    sealedFrameBytes: sender.seal(type, payload).length + 4, ackFrameBytes: receiver.sealAck(0n).length + 4, clear, channel };
});

const out = { item: 1, what: "steps", batchRows: N, embeddingWidth: E, copyOfBatch: copy, noise, frames, sink: Number.isFinite(sink),
  machine: m0, freeRamMBAtEnd: machine().freeRamMB, peakRssMB: peakRssMB() };
mkdirSync(dirname(resolve(a.out)), { recursive: true });
writeFileSync(resolve(a.out), JSON.stringify(out) + "\n");
console.log(JSON.stringify({ noise: noise.map((n) => [n.plainCall.usNetMedian, n.clipOnly.usNetMedian, n.clipNoise.usNetMedian].map((x) => x.toFixed(2))),
  frames: frames.map((f) => [f.type, f.clear.usMedian.toFixed(2), f.channel.usMedian.toFixed(2)]) }));
process.exit(0);
