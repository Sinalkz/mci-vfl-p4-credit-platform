// M3 test helpers: a run taken through setup, alignment (off-chain PSI over TCP, sessions kept
// open) and the seed reveals into Training; then the split training in this process.
import { startedRunWithKeys, runAlignmentOffChain } from "./m2.js";
// @ts-ignore -- plain ESM modules
import { evaluationRows } from "../../offchain/alignment.mjs";
// @ts-ignore
import { activeTraining, embeddingRootHex, FINAL_MARKER, FINAL_TYPES, finalDomain, passiveTraining } from "../../offchain/training.mjs";
// @ts-ignore
import { prng } from "../../offchain/data.mjs";
import { keccak256, toBeHex, toUtf8Bytes } from "ethers";
import { BN254_R, State } from "./federation.js";

export const NOISE = [{ clip: 4, sigma: 0.05 }, { clip: 4, sigma: 0.1 }, { clip: 3, sigma: 0.05 }]; // each passive's own

/** Setup -> alignment -> reveals; state Training. Sessions of the PSI stay open for training. */
export async function trainingRun(ethers: any, signers: { A: any; P: any[]; ARB: any }, scenario: any,
  opts: { overrides?: any; relayPolicies?: object[] } = {}) {
  const run = await startedRunWithKeys(ethers, signers, opts.overrides ?? {});
  const r = await runAlignmentOffChain({ fed: run.fed, run, A: signers.A, P: signers.P, scenario, keepOpen: true,
    relay: !!opts.relayPolicies, relayPolicies: opts.relayPolicies });
  await (await run.fed.connect(signers.A).commitAlignment(r.gammaI, r.I.length, r.gammaA)).wait();
  await (await run.fed.connect(signers.A).revealSeed(run.rA)).wait();
  for (let i = 0; i < signers.P.length; i++) await (await run.fed.connect(signers.P[i]).revealSeed(run.rP[i])).wait();
  const ev = await evaluationRows(run.fed, r.I);
  const picked = new Set(ev.idx);
  const Itr = [...r.I.keys()].filter((k: number) => !picked.has(k));
  const rowsA = { xa: r.I.map((id: bigint) => scenario.A.rows.get(id).xa), y: r.I.map((id: bigint) => scenario.A.rows.get(id).y) };
  const xp = scenario.passives.map((p: any) => r.I.map((id: bigint) => p.rows.get(id)));
  const close = () => { for (const pr of r.pairs) pr.link.destroy(); for (const x of r.passives) x.link?.destroy(); for (const k of r.relays) k.kill(); };
  return { run, r, ev, Itr, rowsA, xp, close };
}

/**
 * A and the 3 passives train in this process. hooksP[i] go to passive i+1. commitFirst (M4, D129): each passive
 * registers its gamma_i before sending its final message.
 */
export async function trainAll(ctx: Awaited<ReturnType<typeof trainingRun>>, P: any[],
  opts: { hooksP?: any[]; hooksA?: any; T?: number; waitMs?: number; commitFirst?: boolean } = {}) {
  const chainId = ctx.run.chainId;
  const contract = ctx.run.addr;
  const maxDepth = ctx.r.chain.maxDepth;
  const sizeI = ctx.r.I.length;
  const peers = ctx.r.pairs.map((pr: any, i: number) => ({ index: i + 1, address: P[i].address, session: pr.session }));
  const aP = activeTraining({ peers, rows: ctx.rowsA, Itr: ctx.Itr, sizeI, chainId, contract, maxDepth, T: opts.T, waitMs: opts.waitMs, hooks: opts.hooksA });
  const pP = ctx.r.passives.map((x: any, i: number) => passiveTraining({ session: x.session, index: i + 1, xp: ctx.xp[i], sizeI,
    chainId, contract, maxDepth, signer: P[i], noise: NOISE[i], hooks: { ...(opts.hooksP?.[i] ?? {}),
      ...(opts.commitFirst ? { beforeSend: async ({ root }: any) => (await ctx.run.fed.connect(P[i]).commitFinal(root)).wait() } : {}) } }));
  const [active, ...passives] = await Promise.all([aP, ...pP]);
  return { active, passives };
}

export const fieldRoot = (tag: string) => toBeHex(BigInt(keccak256(toUtf8Bytes(tag))) % BN254_R, 32);
export const signFinal = (signer: any, chainId: bigint, contract: string, root: string, marker = FINAL_MARKER) =>
  signer.signTypedData(finalDomain(chainId, contract), FINAL_TYPES, { sender: signer.address, root, marker });

/** A run in Training without the off-chain alignment (sizeI = 1000, arbitrary valid commitments). */
export async function quickTraining(ethers: any, S: { A: any; P: any[]; ARB: any }, overrides: any = {}) {
  const run = await startedRunWithKeys(ethers, S, overrides);
  await (await run.fed.connect(S.A).commitAlignment(keccak256(toUtf8Bytes("I")), 1000n, fieldRoot("gammaA"))).wait();
  await (await run.fed.connect(S.A).revealSeed(run.rA)).wait();
  for (let i = 0; i < S.P.length; i++) await (await run.fed.connect(S.P[i]).revealSeed(run.rP[i])).wait();
  if ((await run.fed.state()) !== BigInt(State.Training)) throw new Error("not in Training");
  return run;
}

/** A deterministic 1000 x 4 int16 matrix (a stand-in for h_i) and its root at depth 10. */
export async function fakeMatrix(tag: number) {
  const rng = prng(BigInt(1000 + tag));
  const hq = Array.from({ length: 1000 }, () => Array.from({ length: 4 }, () => rng.int(0, 16000) - 8000));
  return { hq, root: await embeddingRootHex(hq, 10) as string };
}

/**
 * A run in Reconciliation: each passive committed the root of its own matrix and signed it
 * (unless `registered[i]` overrides the registered root); A committed C_fA.
 */
export async function reconciliationRun(ethers: any, S: { A: any; P: any[]; ARB: any },
  opts: { registered?: (string | null)[]; skipFinal?: boolean[]; overrides?: any } = {}) {
  const run = await quickTraining(ethers, S, opts.overrides);
  const finals = [];
  for (let i = 0; i < S.P.length; i++) {
    const m = await fakeMatrix(i + 1);
    const sig = await signFinal(S.P[i], run.chainId, run.addr, m.root);
    const reg = opts.registered?.[i] ?? m.root;
    if (!opts.skipFinal?.[i]) await (await run.fed.connect(S.P[i]).commitFinal(reg)).wait(); // skipped: left at commitFA
    finals.push({ ...m, sig, registered: reg });
  }
  await (await run.fed.connect(S.A).commitFA(fieldRoot("C_fA"), fieldRoot("gamma_g"))).wait();
  return { run, finals };
}
