// M2 §8: channel overhead per message against plaintext: bytes on the wire and CPU time to
// seal and open (ChaCha20-Poly1305 from @noble/ciphers), for the message sizes that occur
// (a small control message, an ID list, the 1k/10k/50k PSI point lists), and the cost of the
// acknowledgement the receiver returns for every sealed message (session.mjs).
// Usage (from contracts/):  node offchain/bench-channel.mjs    -> results/m2/channel_bench.json
import { randomBytes } from "node:crypto";
import { freemem } from "node:os";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { Channel, channelKeyPair, deriveChannelKeys } from "./channel.mjs";

const a = channelKeyPair();
const p = channelKeyPair();
const base = { contract: "0x" + "c0".repeat(20), chainId: 2026n, activeParty: "0x" + "a0".repeat(20), passive: "0x" + "b0".repeat(20) };
const kA = deriveChannelKeys({ selfSk: a.sk, peerPk: p.pk, ...base });
const kP = deriveChannelKeys({ selfSk: p.sk, peerPk: a.pk, ...base });
const type = "psi/a-blinded";
const cases = [
  ["control message", 32],
  ["1,000 IDs (I, 8 bytes each)", 8_000],
  ["1k points (32 bytes each)", 32_000],
  ["10k points", 320_000],
  ["50k points", 1_600_000],
];
const freeRamMBAtStart = Math.round(freemem() / 2 ** 20);
const rows = [];
for (const [label, size] of cases) {
  const msg = randomBytes(size);
  const reps = Math.max(5, Math.min(2000, Math.round(4e7 / size / 10)));
  const sender = new Channel("A", kA);
  const receiver = new Channel("p", kP);
  let t = performance.now();
  let wire = 0;
  for (let i = 0; i < reps; i++) { const body = sender.seal(type, msg); wire = body.length + 4; receiver.open(body); }
  const sealOpenMs = (performance.now() - t) / reps;
  t = performance.now();
  for (let i = 0; i < reps; i++) Buffer.from(msg); // plaintext baseline: one copy
  const copyMs = (performance.now() - t) / reps;
  const plainWire = size + 4; // the same framing without the channel
  rows.push({ label, plaintextBytes: size, wireBytes: wire, overheadBytes: wire - plainWire,
    sealPlusOpenMs: sealOpenMs, plaintextCopyMs: copyMs, throughputMBps: size / 2 ** 20 / (sealOpenMs / 1000), reps });
  console.log(JSON.stringify(rows.at(-1)));
}
// One ack per sealed message, in the opposite direction.
const ackReps = 20_000;
const acker = new Channel("p", kP);
const ackReader = new Channel("A", kA);
let t = performance.now();
let ackWire = 0;
for (let i = 0; i < ackReps; i++) { const body = acker.sealAck(BigInt(i)); ackWire = body.length + 4; if (ackReader.openAck(body) !== BigInt(i)) throw new Error("ack"); }
const ack = { wireBytesPerSealedMessage: ackWire, sealPlusOpenMs: (performance.now() - t) / ackReps, reps: ackReps };
console.log(JSON.stringify({ ack }));
const out = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "results", "m2");
mkdirSync(out, { recursive: true });
writeFileSync(resolve(out, "channel_bench.json"), JSON.stringify({ machine: "Windows 11 laptop, i7-9750H, 16 GB", freeRamMBAtStart,
  node: process.version, library: "@noble/ciphers 2.4.0 chacha20poly1305", type, rows, ack }, null, 2) + "\n");
