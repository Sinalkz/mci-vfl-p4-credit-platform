// Task 4 item 1: one split-training run of the thesis's own code, activeTraining + 3 passiveTraining (training.mjs,
// unchanged), with A and the three passives in one process over loopback TCP (as DATA1's vfl-run.mjs), in one of
// three variants. Same rows, model config (seeds), batch order and float32 transport encoding in every variant.
//   plain      plain split learning: clear frames (PlainSession: no encryption, no tag, no ack); the passives send
//              f_i's output as it is (clip = infinity, sigma = 0, a zero noise source)
//   clipnoise  plain + the passives' clipping and noise: LOCAL_NOISE (party-m3.mjs: per-row L2 clip 4 / 4 / 3, then
//              N(0, sigma^2) with sigma 0.05 / 0.1 / 0.05) drawn from the protocol's default source, the OS CSPRNG
//              (training.mjs cryptoNormal)
//   protocol   clipnoise + the secure channel: every frame sealed with ChaCha20-Poly1305 and acknowledged with an
//              authenticated ack, with the retransmission timers (Session over data1/inproc.mjs sessionPairs)
// In plain, training.mjs still calls clipAndNoise, with clip = infinity and a noise source returning 0: it computes
// each row's norm and leaves the row as it is (64 x 4 multiply-adds per batch; measured separately in steps-bench).
// The final round (h_i as int16, its Poseidon root at maxDepth, the EIP-712 signature, A's root and signature
// check) is the same code in every variant; it is timed apart from the T rounds (finalMs).
import { Wallet } from "ethers";

import { sessionPairs } from "../data1/inproc.mjs";
import { LOCAL_NOISE } from "../party-m3.mjs";
import { activeTraining, cryptoNormal, passiveTraining } from "../training.mjs";
import { plainPairs } from "./plain-session.mjs";

export const VARIANTS = {
  plain: { transport: "plain", noise: false },
  clipnoise: { transport: "plain", noise: true },
  protocol: { transport: "session", noise: true },
};
export const NO_NOISE = { clip: Infinity, sigma: 0 };
const zero = () => 0;

/**
 * One run. rows, xp, Itr, sizeI, maxDepth as activeTraining/passiveTraining take them; T rounds.
 * keep: also return the trained fA, fusion model and final matrices. noiseOff (tests only): the variant's transport
 * with plain's noise setting.
 */
export async function trainOnce({ variant, rows, xp, Itr, sizeI, maxDepth, T, waitMs = 120_000, keep = false, noiseOff = false }) {
  const v = VARIANTS[variant];
  if (!v) throw new Error(`variant ${variant}: one of ${Object.keys(VARIANTS).join(", ")}`);
  const noisy = v.noise && !noiseOff;
  const ws = [0, 1, 2, 3].map(() => Wallet.createRandom());
  const chainId = 2026n;
  const contract = ws[0].address; // any address: the EIP-712 domain of this off-chain run
  const { pairs, close } = v.transport === "session"
    ? await sessionPairs(3, { chainId, contract, activeParty: ws[0].address, passives: ws.slice(1).map((w) => w.address) })
    : await plainPairs(3);
  const t0 = performance.now();
  const [act, ...pas] = await Promise.all([
    activeTraining({ peers: pairs.map((x, i) => ({ index: i + 1, address: ws[i + 1].address, session: x.a })), rows, Itr, sizeI,
      chainId, contract, maxDepth, waitMs, T }),
    ...pairs.map((x, i) => passiveTraining({ session: x.p, index: i + 1, xp: xp[i], sizeI, chainId, contract, maxDepth,
      signer: ws[i + 1], noise: noisy ? LOCAL_NOISE[i] : NO_NOISE, normal: noisy ? cryptoNormal : zero })),
  ]);
  const totalMs = performance.now() - t0;
  const sessions = pairs.flatMap((x) => [x.a, x.p]);
  const acks = sessions.reduce((s, x) => s + x.stats.acksSent, 0);
  const retransmitted = sessions.reduce((s, x) => s + (x.stats.retransmitted ?? 0), 0);
  close();
  if (act.zeroFilled.length > 0) throw new Error(`${variant}: ${act.zeroFilled.length} zero-filled batches`);
  if (!act.finals.every((f) => f.received && f.signatureValid)) throw new Error(`${variant}: a final message is missing or badly signed`);
  const rounds = act.rounds.map(({ round, ms, loss, batches, bytes }) => ({ round, ms, loss, batches,
    bytesOut: bytes.reduce((s, b) => s + b.out, 0), bytesIn: bytes.reduce((s, b) => s + b.in, 0) }));
  const out = { variant, T, rounds, totalMs, finalMs: totalMs - rounds.reduce((s, r) => s + r.ms, 0), acks, retransmitted,
    finalBytes: act.finals.map((f) => f.bytes) };
  if (keep) Object.assign(out, { fA: act.fA, fusion: act.fusion, hq: pas.map((p) => p.hq) });
  return out;
}
