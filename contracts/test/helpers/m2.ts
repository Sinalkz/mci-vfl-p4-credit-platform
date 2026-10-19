// M2 test helpers: a federation run whose members registered real X25519 channel keys,
// and a relay process.
import { spawn, type ChildProcess } from "node:child_process";
import { resolve } from "node:path";

import { defaultProposal, seedCommitment, randomR, type Proposal } from "./federation.js";
import { REPO_ROOT } from "./talmud.js";
// @ts-ignore -- plain ESM module
import { bytesToHex, channelKeyPair } from "../../offchain/channel.mjs";

export type KeyPair = { sk: Uint8Array; pk: Uint8Array };

/**
 * Creates a run on the given Hardhat network through a fresh factory, registers every
 * passive with its own X25519 channel key (A's key is in the proposal), and commits all
 * seeds; the arbiter's acceptance starts the run. Returns the contract and the key pairs.
 */
export async function startedRunWithKeys(ethers: any, signers: { A: any; P: any[]; ARB: any },
  overrides: Partial<Proposal> = {}) {
  const token = await ethers.deployContract("MockERC20");
  const verifier = await ethers.deployContract("MembersVarVerifierN64D10");
  const factory = await ethers.deployContract("FederationFactory");
  const keysA: KeyPair = channelKeyPair();
  const keysP: KeyPair[] = signers.P.map(() => channelKeyPair());
  const p = defaultProposal({
    activeParty: signers.A.address, passives: signers.P.map((s: any) => s.address), arbiter: signers.ARB.address,
    verifier: await verifier.getAddress(), token: await token.getAddress(),
  }, { activeChannelKey: bytesToHex(keysA.pk), ...overrides });
  await token.mint(signers.A.address, p.B);
  await token.connect(signers.A).approve(await factory.getAddress(), p.B);
  const rc = await (await factory.connect(signers.A).create(p)).wait();
  const ev = rc.logs.map((l: any) => factory.interface.parseLog(l)).find((e: any) => e?.name === "FederationCreated");
  const fed = await ethers.getContractAt("Federation", ev.args.federation);
  const addr = await fed.getAddress();
  const rA = randomR();
  const rP = signers.P.map(() => randomR());
  await fed.connect(signers.A).commitSeed(seedCommitment(rA, signers.A.address, addr));
  for (let i = 0; i < signers.P.length; i++) {
    const s = signers.P[i];
    await token.mint(s.address, p.Dp);
    await token.connect(s).approve(addr, p.Dp);
    await fed.connect(s).register(bytesToHex(keysP[i].pk), signers.ARB.address);
    await fed.connect(s).commitSeed(seedCommitment(rP[i], s.address, addr));
  }
  const keysArb: KeyPair = channelKeyPair(); // the arbiter's X25519 key (M3 relay)
  await fed.connect(signers.ARB).acceptArbiter(bytesToHex(keysArb.pk));
  return { fed, addr, p, keysA, keysP, keysArb, token, rA, rP, chainId: (await ethers.provider.getNetwork()).chainId as bigint };
}

/** Starts offchain/relay.mjs towards `targetPort`; resolves with its port and the process. */
export function startRelay(targetPort: number, policy: object = {}, capture?: string): Promise<{ port: number; proc: ChildProcess }> {
  const args = [resolve(REPO_ROOT, "contracts", "offchain", "relay.mjs"), "--target", `127.0.0.1:${targetPort}`,
    "--policy", JSON.stringify(policy), ...(capture ? ["--capture", capture] : [])];
  const proc = spawn(process.execPath, args, { stdio: ["ignore", "pipe", "inherit"] });
  return new Promise((res, rej) => {
    proc.stdout!.on("data", (d: Buffer) => {
      const m = d.toString().match(/RELAY (\d+)/);
      if (m) res({ port: Number(m[1]), proc });
    });
    proc.once("exit", (c) => rej(new Error(`relay exited ${c}`)));
  });
}

// @ts-ignore -- plain ESM modules
import { activePair, buildI, gammaA, passiveSide, readChain, sendI } from "../../offchain/alignment.mjs";
// @ts-ignore
import { gammaI } from "../../offchain/encoding.mjs";
// @ts-ignore
import { connect, listen } from "../../offchain/link.mjs";

/**
 * One whole off-chain alignment in this process over real TCP: A listens; each passive
 * connects directly, or through its own relay process (capture = one wire capture file
 * per passive). Returns A's pairwise results, I, each passive's result or error, and the
 * values A would register. Does not send any transaction.
 */
export async function runAlignmentOffChain(opts: {
  fed: any; run: { keysA: KeyPair; keysP: KeyPair[] }; A: any; P: any[]; scenario: any; // A signs each list (M4 AlignedSet)
  relay?: boolean; captureDir?: string; relayPolicy?: object; relayPolicies?: object[]; keepOpen?: boolean;
  hooksA?: any; hooksP?: any[]; tamperI?: (i: number, ids: bigint[]) => bigint[] | undefined;
  tamperSigHash?: (i: number, h: string) => string | undefined;
}) {
  const chain = await readChain(opts.fed);
  const pairsP: Promise<any>[] = [];
  const server = await listen(0, (link: any) => pairsP.push(activePair({ link, chain, sk: opts.run.keysA.sk, ids: opts.scenario.A.ids, hooks: opts.hooksA ?? {} })));
  const relays: any[] = [];
  const passiveP = opts.P.map(async (s: any, i: number) => {
    let port = server.address().port;
    if (opts.relay) {
      const r = await startRelay(port, opts.relayPolicies?.[i] ?? opts.relayPolicy ?? {}, opts.captureDir ? `${opts.captureDir}/wire-p${i + 1}.jsonl` : undefined);
      relays.push(r.proc);
      port = r.port;
    }
    const link = await connect(port);
    try {
      return { ok: true, ...(await passiveSide({ link, chain, sk: opts.run.keysP[i].sk, self: s.address,
        ids: opts.scenario.passives[i].ids, hooks: opts.hooksP?.[i] ?? {} })) };
    } catch (e: any) {
      return { ok: false, error: e };
    } finally { if (!opts.keepOpen) link.destroy(); }
  });
  while (pairsP.length < opts.P.length) await new Promise((r) => setTimeout(r, 20));
  const pairs = await Promise.all(pairsP);
  const I = buildI(pairs.map((p: any) => p.intersection));
  for (const pr of pairs) {
    const i = opts.P.findIndex((s: any) => s.address.toLowerCase() === pr.passive.toLowerCase());
    await sendI(pr, I, opts.tamperI ? (ids: bigint[]) => opts.tamperI!(i, ids) ?? ids : undefined, opts.A,
      opts.tamperSigHash ? (h: string) => opts.tamperSigHash!(i, h) ?? h : undefined);
  }
  const passives = await Promise.all(passiveP);
  server.close();
  if (!opts.keepOpen) {
    for (const pr of pairs) pr.link.destroy();
    for (const r of relays) r.kill();
  }
  const byPassive = opts.P.map((s: any) => pairs.find((p: any) => p.passive.toLowerCase() === s.address.toLowerCase()));
  return { chain, pairs: byPassive, I, passives, relays, gammaI: gammaI(I) as string,
    gammaA: I.length > 0 ? await gammaA(I, opts.scenario.A.rows, chain.maxDepth) as string : null }; // empty I: nothing to commit
}
