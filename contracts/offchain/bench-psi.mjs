// M2 §8: PSI per pair, A and one passive as two separate processes over TCP (no chain: the
// channel keys and contract address are generated for the benchmark; the protocol code is
// the same as in the runner). Sizes: n customers per member, 80% overlap.
// Usage (from contracts/):  node offchain/bench-psi.mjs [--sizes 1000,10000,50000]
// Writes results/m2/psi_bench.json. One size at a time; each side's peak memory is its own.
import { spawn } from "node:child_process";
import { freemem, totalmem, cpus } from "node:os";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { activePair, buildI, passiveSide, sendI } from "./alignment.mjs";
import { bytesToHex, channelKeyPair, hexToBytes } from "./channel.mjs";
import { genPairSets } from "./data.mjs";
import { connect, listen } from "./link.mjs";

const HERE = fileURLToPath(import.meta.url);
const { values: args } = parseArgs({ options: {
  sizes: { type: "string", default: "1000,10000,50000" },
  side: { type: "string" }, n: { type: "string" }, port: { type: "string" }, keys: { type: "string" },
} });

const CONTRACT = "0x" + "c0".repeat(20);
const A_ADDR = "0x" + "a0".repeat(20);
const P_ADDR = "0x" + "b0".repeat(20);

if (args.side) {
  // ---- one side (child process) ----
  const k = JSON.parse(args.keys);
  const chain = { contract: CONTRACT, chainId: 2026n, activeParty: A_ADDR, activeKey: k.pkA, passives: [P_ADDR], passiveKey: async () => k.pkP };
  const n = Number(args.n);
  const sets = genPairSets({ n });
  const t0 = performance.now();
  let res;
  if (args.side === "A") {
    let got;
    const linkP = new Promise((ok) => { got = ok; });
    const server = await listen(0, (l) => got(l));
    console.log(`PORT ${server.address().port}`);
    const link = await linkP;
    const pr = await activePair({ link, chain, sk: hexToBytes(k.skA), ids: sets.a, timeoutMs: 3_600_000 });
    const psiMs = performance.now() - t0;
    await sendI(pr, buildI([pr.intersection]));
    res = { side: "A", n, intersection: pr.intersection.length, expected: sets.common.length, psiMs, timings: pr.timings,
      bytesOut: link.bytesOut, bytesIn: link.bytesIn, session: pr.session.stats };
    await new Promise((ok) => setTimeout(ok, 200));
    link.destroy(); server.close();
  } else {
    const link = await connect(Number(args.port));
    const r = await passiveSide({ link, chain, sk: hexToBytes(k.skP), self: P_ADDR, ids: sets.p, timeoutMs: 3_600_000 });
    res = { side: "p", n, received: r.I.length, psiMs: performance.now() - t0 - r.timings.waitIMs, timings: r.timings,
      bytesOut: link.bytesOut, bytesIn: link.bytesIn, session: r.session.stats };
    link.destroy();
  }
  res.peakRssMB = process.resourceUsage().maxRSS / 1024;
  console.log("RESULT " + JSON.stringify(res));
  process.exit(0);
}

// ---- coordinator ----
const run = (argv) => {
  const p = spawn(process.execPath, [HERE, ...argv], { stdio: ["ignore", "pipe", "inherit"] });
  let out = "";
  const port = new Promise((ok) => p.stdout.on("data", (d) => { out += d; const m = out.match(/PORT (\d+)/); if (m) ok(Number(m[1])); }));
  const done = new Promise((ok, no) => p.on("exit", (c) => (c === 0 ? ok(JSON.parse(out.match(/RESULT (.*)/)[1])) : no(new Error(`side exited ${c}`)))));
  return { port, done };
};
const rows = [];
for (const n of args.sizes.split(",").map(Number)) {
  const a = channelKeyPair();
  const p = channelKeyPair();
  const keys = JSON.stringify({ skA: bytesToHex(a.sk), pkA: bytesToHex(a.pk), skP: bytesToHex(p.sk), pkP: bytesToHex(p.pk) });
  const freeRamMBAtStart = Math.round(freemem() / 2 ** 20);
  const t0 = Date.now();
  const A = run(["--side", "A", "--n", String(n), "--keys", keys]);
  const P = run(["--side", "p", "--n", String(n), "--keys", keys, "--port", String(await A.port)]);
  const [ra, rp] = await Promise.all([A.done, P.done]);
  const row = { n, overlap: 0.8, freeRamMBAtStart, wallMs: Date.now() - t0, A: ra, p: rp, correct: ra.intersection === ra.expected && rp.received === ra.expected };
  rows.push(row);
  console.log(JSON.stringify({ n, wallMs: row.wallMs, A_ms: Math.round(ra.psiMs), p_ms: Math.round(rp.psiMs), A_out: ra.bytesOut, p_out: rp.bytesOut,
    A_MB: Math.round(ra.peakRssMB), p_MB: Math.round(rp.peakRssMB), resent: ra.session.retransmitted + rp.session.retransmitted, correct: row.correct }));
}
const outDir = resolve(HERE, "..", "..", "..", "results", "m2");
mkdirSync(outDir, { recursive: true });
writeFileSync(resolve(outDir, "psi_bench.json"), JSON.stringify({
  machine: { host: "Windows 11 laptop, i7-9750H, 16 GB", cpus: cpus().length, totalRamMB: Math.round(totalmem() / 2 ** 20) },
  library: "@noble/curves 2.4.0 (ristretto255, constant-time multiply), @noble/ciphers 2.4.0, @noble/hashes 2.4.0; Node " + process.version,
  rows,
}, null, 2) + "\n");
console.log("wrote results/m2/psi_bench.json");
