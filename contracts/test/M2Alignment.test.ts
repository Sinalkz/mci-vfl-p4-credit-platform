// M2 §5.3-§5.5: protocol tests A01..A05 (M2 alignment spec §7): the off-chain
// alignment over TCP, then commitAlignment on the in-process chain, the passives' gamma_I
// check, the reveals and the evaluation rows.
import { expect } from "chai";
import { network } from "hardhat";

import { runAlignmentOffChain, startedRunWithKeys } from "./helpers/m2.js";
// @ts-ignore -- plain ESM modules
import { ALIGNED_TYPES, checkGammaI, evaluationRows, ProtocolError } from "../offchain/alignment.mjs";
// @ts-ignore
import { genScenario } from "../offchain/data.mjs";
// @ts-ignore
import { gammaI, runDomain, sortIds } from "../offchain/encoding.mjs";
import { verifyTypedData } from "ethers";
// @ts-ignore
import { checkAlignmentCapacity, defaultProposal } from "../offchain/proposal.mjs";
// @ts-ignore
import { varTrees, membersVarInput } from "../../circuits/scripts/m1b-lib.mjs";
// @ts-ignore
import { loadS2b, sampleIndices } from "../../circuits/scripts/members-lib.mjs";

const { ethers } = await network.create();
const signers = await ethers.getSigners();
const A = signers[0];
const P = signers.slice(1, 4);
const ARB = signers[11];
const scenario = genScenario();

async function alignedOnChain(opts: { tamperI?: (i: number, ids: bigint[]) => bigint[] | undefined; signerA?: any;
  tamperSigHash?: (i: number, h: string) => string | undefined } = {}) {
  const run = await startedRunWithKeys(ethers, { A, P, ARB });
  const r = await runAlignmentOffChain({ fed: run.fed, run, A: opts.signerA ?? A, P, scenario, tamperI: opts.tamperI, tamperSigHash: opts.tamperSigHash });
  await (await run.fed.connect(A).commitAlignment(r.gammaI, r.I.length, r.gammaA)).wait();
  return { run, r };
}

describe("M2 alignment protocol (A + 3 passives off-chain, commitments on-chain)", function () {
  it("A01 each passive's recomputed gamma_I equals the on-chain value", async function () {
    const { run, r } = await alignedOnChain();
    expect(await run.fed.sizeI()).to.equal(1000n);
    for (let i = 0; i < 3; i++) {
      const c = await checkGammaI(run.fed, r.passives[i].I, `p${i + 1}`);
      expect(c.pass).to.equal(true);
    }
  });

  it("A02 A sends one passive a modified I: that passive's check fails against the chain", async function () {
    // For passive 2, one ID of I is replaced by one of passive 2's own customers outside I,
    // so the ownership check passes and only the gamma_I check can catch it.
    const inI = new Set(scenario.I.map(String));
    const swapIn = scenario.passives[1].ids.find((x: bigint) => !inI.has(String(x)));
    const { run, r } = await alignedOnChain({
      tamperI: (i, ids) => (i === 1 ? sortIds([...ids.slice(1), swapIn]) : undefined),
    });
    const results = [];
    for (let i = 0; i < 3; i++) results.push((await checkGammaI(run.fed, r.passives[i].I, `p${i + 1}`)).pass);
    expect(results).to.deep.equal([true, false, true]);
  });

  it("A03 gamma_A equals the M1b varTrees root for the same rows and maxDepth", async function () {
    const { run, r } = await alignedOnChain();
    const I = sortIds(scenario.I);
    const table = { y: I.map((id: bigint) => scenario.A.rows.get(id).y), xa: I.map((id: bigint) => scenario.A.rows.get(id).xa), hp: [] };
    const trees = await varTrees({ table }, I.length, 10);
    expect(BigInt(await run.fed.gammaA())).to.equal(trees.A.root);
    expect(BigInt(r.gammaA)).to.equal(trees.A.root);
    expect(trees.depth).to.equal(Number((await run.fed.proposal()).maxDepth));
  });

  it("A04 I^ev from on-chain s and sizeI equals sampleIndices and the M1b circuit's input rows", async function () {
    const { run, r } = await alignedOnChain();
    await run.fed.connect(A).revealSeed(run.rA);
    for (let i = 0; i < 3; i++) await run.fed.connect(P[i]).revealSeed(run.rP[i]);
    expect(await run.fed.state()).to.equal(3n); // Training (the state once s is fixed; M2 called it Seeded)
    const ev = await evaluationRows(run.fed, r.I);
    expect(ev.idx).to.deep.equal(await sampleIndices(await run.fed.seed(), 64, 1000));
    expect(ev.Iev.length).to.equal(64);
    expect(ev.Itr.length).to.equal(1000 - new Set(ev.idx).size);
    expect(new Set([...ev.Iev, ...ev.Itr].map(String)).size).to.equal(1000); // I^ev and I^tr cover I
    // The M1b circuit input built from the same rows, s and sizeI draws exactly these rows.
    const I = r.I;
    const s2b = loadS2b();
    const data = { ...s2b, table: {
      y: I.map((id: bigint) => scenario.A.rows.get(id).y), xa: I.map((id: bigint) => scenario.A.rows.get(id).xa),
      hp: [0, 1, 2].map(() => I.map(() => [0, 0, 0, 0])),
    } };
    const { input, idx } = await membersVarInput(64, 1000, 10, { data, seed: await run.fed.seed() });
    expect(idx).to.deep.equal(ev.idx);
    expect(input.sizeI).to.equal("1000");
    expect(BigInt(input.gammaA)).to.equal(BigInt(await run.fed.gammaA()));
    ev.Iev.forEach((id: bigint, k: number) => {
      expect(input.XA[k]).to.deep.equal(scenario.A.rows.get(id).xa.map(String));
      expect(input.Y[k]).to.equal(String(scenario.A.rows.get(id).y));
    });
  });

  it("A05 a received ID that is not the passive's own: the passive stops with an error", async function () {
    const onlyA = scenario.A.ids.find((x: bigint) => !scenario.passives[2].ids.some((y: bigint) => y === x));
    const run = await startedRunWithKeys(ethers, { A, P, ARB });
    const r = await runAlignmentOffChain({ fed: run.fed, run, A, P, scenario,
      tamperI: (i, ids) => (i === 2 ? sortIds([...ids, onlyA]) : undefined) });
    expect(r.passives[0].ok && r.passives[1].ok).to.equal(true);
    expect(r.passives[2].ok).to.equal(false);
    expect(r.passives[2].error).to.be.instanceOf(ProtocolError);
    expect(r.passives[2].error.code).to.equal("foreign-id");
  });

  it("Q5 the proposal helper refuses 2^maxDepth < |X_A| (warns instead only when asked)", async function () {
    const base = { activeParty: A.address, passives: P.map((s: any) => s.address), arbiter: ARB.address,
      verifier: ARB.address, token: ARB.address, activeChannelKey: "0x" + "11".repeat(32) };
    expect(() => defaultProposal(base)).to.throw(/activeCustomers/);              // |X_A| must be given
    expect(defaultProposal({ ...base, activeCustomers: 1024 }).maxDepth).to.equal(10); // 2^10 = 1024: accepted
    expect(() => defaultProposal({ ...base, activeCustomers: 1025 })).to.throw(/proposal refused.*1025.*1024/);
    expect(() => defaultProposal({ ...base, activeCustomers: 1200 })).to.throw(/proposal refused/); // the first M2 scenario
    expect(defaultProposal({ ...base, activeCustomers: 1020 }).maxDepth).to.equal(10);                 // the scenario since Q6
    expect(defaultProposal({ ...base, activeCustomers: 1200, maxDepth: 11 }).maxDepth).to.equal(11);
    const w = checkAlignmentCapacity({ maxDepth: 10, activeCustomers: 1200, mode: "warn" });
    expect(w.ok).to.equal(false);
    expect(w.warning).to.match(/1200 exceeds 2\^maxDepth = 1024/);
    expect(defaultProposal({ ...base, activeCustomers: 1200, capacityCheck: "warn" }).maxDepth).to.equal(10);
  });

  // ---- M4 (D120/D121): A signs the hash of the list I it sent to each passive -------------------

  it("A12 M4: each passive holds A's AlignedSet signature over the hash of the exact list it received, bound to its address", async function () {
    const inI = new Set(scenario.I.map(String));
    const swapIn = scenario.passives[1].ids.find((x: bigint) => !inI.has(String(x)));
    const { run, r } = await alignedOnChain({ tamperI: (i, ids) => (i === 1 ? sortIds([...ids.slice(1), swapIn]) : undefined) });
    const domain = runDomain(run.chainId, run.addr);
    for (let i = 0; i < 3; i++) {
      const res = r.passives[i];
      expect(res.ok, `p${i + 1}`).to.equal(true);
      expect(res.listHash).to.equal(gammaI(res.I));
      expect(verifyTypedData(domain, ALIGNED_TYPES, { passive: P[i].address, listHash: res.listHash }, res.alignSig)).to.equal(A.address);
      expect(verifyTypedData(domain, ALIGNED_TYPES, { passive: P[(i + 1) % 3].address, listHash: res.listHash }, res.alignSig)).to.not.equal(A.address);
    }
    const onChain = (await run.fed.gammaI()).toLowerCase();
    expect(r.passives.map((x: any) => x.listHash.toLowerCase() === onChain)).to.deep.equal([true, false, true]); // the evidence for passive 2
  });

  it("A13 M4: a list signed by another key, or a signature over another hash, is refused (ProtocolError sig)", async function () {
    const a = await alignedOnChain({ signerA: ARB });
    for (const x of a.r.passives) {
      expect(x.ok).to.equal(false);
      expect(x.error).to.be.instanceOf(ProtocolError);
      expect(x.error.code).to.equal("sig");
    }
    const b = await alignedOnChain({ tamperSigHash: (i, h) => (i === 2 ? gammaI([1n, 2n, 3n]) : undefined) });
    expect(b.r.passives.map((x: any) => x.ok)).to.deep.equal([true, true, false]);
    expect(b.r.passives[2].error.code).to.equal("sig");
  });
});
