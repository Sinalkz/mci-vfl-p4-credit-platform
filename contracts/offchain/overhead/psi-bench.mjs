// Task 4 item 2: the private set intersection of the alignment phase with the three passives, as in a run: A and each
// passive are their own OS processes over loopback TCP, and the protocol's own code runs (alignment.mjs activePair,
// passiveSide, buildI, sendI with A's EIP-712 AlignedSet signature; psi.mjs ristretto255). As in party.mjs, A
// accepts the three connections and runs the three pairs in parallel on its one thread, builds I, then sends I and
// its signature to each passive in turn. No chain: the channel keys, A's account and the contract address are made
// for the measurement (as bench-psi.mjs, M2 §8). The ID sets are id-sets.mjs's.
// Timers (performance.now in each process):
//   A: psiMs from the first accepted connection until all three pairwise intersections are known; buildIMs;
//      sendIMs (I and the signature to each passive, acknowledged)
//   passive: psiMs from its connection until it holds I and has checked A's signature
//   coordinator: wallMs from the go signal until every process has reported
// Started by overhead/run.mjs (one setting at a time):
//   node offchain/overhead/psi-bench.mjs --setting subset|full-a|full-b|full-c --out <file>
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { Wallet } from "ethers";

import { activePair, buildI, passiveSide, sendI } from "../alignment.mjs";
import { bytesToHex, channelKeyPair, hexToBytes } from "../channel.mjs";
import { machine, peakRssMB } from "../data1/machine.mjs";
import { connect, listen } from "../link.mjs";
import { psiSets } from "./id-sets.mjs";

const HERE = fileURLToPath(import.meta.url);
const CONTRACT = "0x" + "c0".repeat(20);
const { values: args } = parseArgs({ options: {
  setting: { type: "string" }, out: { type: "string" }, side: { type: "string" }, index: { type: "string" },
  port: { type: "string" }, keys: { type: "string" },
} });
const same = (x, y) => x.length === y.length && x.every((v, i) => v === y[i]);

if (args.side) {
  // ---- one party (child process) ----
  const k = JSON.parse(args.keys);
  const chain = { contract: CONTRACT, chainId: 2026n, activeParty: k.aAddr, activeKey: k.pkA, passives: k.pAddrs,
    passiveKey: async (addr) => k.pkP[k.pAddrs.findIndex((x) => x.toLowerCase() === addr.toLowerCase())] };
  const sets = psiSets(args.setting);
  let res;
  if (args.side === "A") {
    const pairsP = [];
    let tFirst = null;
    const server = await listen(0, (link) => {
      tFirst ??= performance.now();
      pairsP.push(activePair({ link, chain, sk: hexToBytes(k.skA), ids: sets.aIds, timeoutMs: 3_600_000 }));
    });
    console.log(`PORT ${server.address().port}`);
    while (pairsP.length < 3) await new Promise((ok) => setTimeout(ok, 5));
    const pairs = await Promise.all(pairsP);
    const t1 = performance.now();
    const I = buildI(pairs.map((x) => x.intersection));
    const t2 = performance.now();
    const wallet = new Wallet(k.aKey);
    for (const pr of pairs) await sendI(pr, I, undefined, wallet);
    const t3 = performance.now();
    res = { side: "A", setSize: sets.aIds.length, sizeI: I.length, correct: same(I, sets.I), psiMs: t1 - tFirst, buildIMs: t2 - t1, sendIMs: t3 - t2,
      pairs: pairs.map((x) => ({ passive: x.passive, pairwise: x.intersection.length, pSetSize: x.pSetSize, timings: x.timings,
        bytesOut: x.link.bytesOut, bytesIn: x.link.bytesIn, retransmitted: x.session.stats.retransmitted })) };
    await new Promise((ok) => setTimeout(ok, 200));
    for (const pr of pairs) pr.link.destroy();
    server.close();
  } else {
    const i = Number(args.index);
    const ids = sets.passiveIds[i];
    console.log("READY");
    await new Promise((ok) => createInterface({ input: process.stdin }).once("line", ok));
    const t0 = performance.now();
    const link = await connect(Number(args.port));
    const r = await passiveSide({ link, chain, sk: hexToBytes(k.skP[i]), self: k.pAddrs[i], ids, timeoutMs: 3_600_000 });
    res = { side: `p${i + 1}`, setSize: ids.length, received: r.I.length, correct: same(r.I, sets.I), psiMs: performance.now() - t0, timings: r.timings,
      bytesOut: link.bytesOut, bytesIn: link.bytesIn, retransmitted: r.session.stats.retransmitted };
    link.destroy();
  }
  res.peakRssMB = peakRssMB();
  console.log("RESULT " + JSON.stringify(res));
  process.exit(0);
}

// ---- coordinator: one run of one setting ----
function child(argv) {
  const p = spawn(process.execPath, [HERE, ...argv], { stdio: ["pipe", "pipe", "inherit"] });
  let out = "";
  const seen = (re) => new Promise((ok) => { const f = () => { const m = out.match(re); if (m) ok(m); else p.stdout.once("data", f); }; f(); });
  p.stdout.on("data", (d) => { out += d; });
  const done = new Promise((ok, no) => p.on("exit", (c) => (c === 0 ? ok(JSON.parse(out.match(/RESULT (.*)/)[1])) : no(new Error(`${argv[1]} exited ${c}`)))));
  return { p, seen, done };
}
const a = channelKeyPair();
const ps = [0, 1, 2].map(() => channelKeyPair());
const aw = Wallet.createRandom();
const keys = JSON.stringify({ skA: bytesToHex(a.sk), pkA: bytesToHex(a.pk), aKey: aw.privateKey, aAddr: aw.address,
  skP: ps.map((x) => bytesToHex(x.sk)), pkP: ps.map((x) => bytesToHex(x.pk)), pAddrs: [0, 1, 2].map(() => Wallet.createRandom().address) });
const m0 = machine();
const A = child(["--side", "A", "--setting", args.setting, "--keys", keys]);
const port = (await A.seen(/PORT (\d+)/))[1];
const P = [0, 1, 2].map((i) => child(["--side", "p", "--index", String(i), "--setting", args.setting, "--keys", keys, "--port", port]));
await Promise.all(P.map((x) => x.seen(/READY/)));
const t0 = performance.now();
for (const x of P) x.p.stdin.write("go\n");
const [ra, ...rp] = await Promise.all([A.done, ...P.map((x) => x.done)]);
const wallMs = performance.now() - t0;
const out = { item: 2, what: "psi", setting: args.setting, machine: m0, wallMs, A: ra, passives: rp,
  correct: ra.correct && rp.every((x) => x.correct) };
if (!out.correct) throw new Error(`${args.setting}: I differs from the expected intersection`);
mkdirSync(dirname(resolve(args.out)), { recursive: true });
writeFileSync(resolve(args.out), JSON.stringify(out) + "\n");
console.log(JSON.stringify({ setting: args.setting, wallS: (wallMs / 1000).toFixed(1), A_psiS: (ra.psiMs / 1000).toFixed(1),
  p_psiS: rp.map((x) => (x.psiMs / 1000).toFixed(1)), sizeI: ra.sizeI }));
process.exit(0);
