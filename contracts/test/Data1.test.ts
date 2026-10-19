// DATA1: the thesis code's new options keep the M3 default path unchanged; the in-process harness; the
// centralized reference and its bit-exact equivalence with split training; evaluation helpers.
import { expect } from "chai";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { keccak256 } from "ethers";
// @ts-ignore -- plain ESM modules
import { CONFIG_PATH, fiSpec, modelConfigHash } from "../offchain/model.mjs";
import { RECS_MANIFEST, skipWithoutRecs } from "./helpers/recs.js";

const MODEL = resolve("offchain", "model.mjs");

/** One split run of the M3 code (activeTraining + 3 passiveTraining over in-process sessions) on the
 * M3 synthetic scenario (6-wide passives: the default config applies). Reused by D05, D06, D06b. */
async function splitRun(T: number, noise: any = { clip: Infinity, sigma: 0 }, normals: (() => number)[] | null = null) {
  // @ts-ignore
  const { genScenario } = await import("../offchain/data.mjs");
  // @ts-ignore
  const { activeTraining, passiveTraining } = await import("../offchain/training.mjs");
  // @ts-ignore
  const { sessionPairs } = await import("../offchain/data1/inproc.mjs");
  const { Wallet } = await import("ethers");
  const s = genScenario();
  const I = [...s.I].sort((a: bigint, b: bigint) => (a < b ? -1 : 1));
  const rows = { xa: I.map((id: bigint) => s.A.rows.get(id).xa), y: I.map((id: bigint) => s.A.rows.get(id).y) };
  const xp = s.passives.map((p: any) => I.map((id: bigint) => p.rows.get(id)));
  const Itr = [...Array(800).keys()];
  const ws = [0, 1, 2, 3].map(() => Wallet.createRandom());
  const ctx = { chainId: 2026n, contract: ws[0].address, activeParty: ws[0].address, passives: ws.slice(1).map((w) => w.address) };
  const { pairs, close } = await sessionPairs(3, ctx);
  const [act, ...pas] = await Promise.all([
    activeTraining({ peers: pairs.map((x: any, i: number) => ({ index: i + 1, address: ws[i + 1].address, session: x.a })),
      rows, Itr, sizeI: I.length, chainId: 2026n, contract: ctx.contract, maxDepth: 10, T, waitMs: 60_000 }),
    ...pairs.map((x: any, i: number) => passiveTraining({ session: x.p, index: i + 1, xp: xp[i], sizeI: I.length,
      chainId: 2026n, contract: ctx.contract, maxDepth: 10, signer: ws[i + 1], noise, ...(normals ? { normal: normals[i] } : {}) })),
  ]);
  close();
  return { act, pas, rows, xp, Itr, sizeI: I.length };
}

describe("DATA1 thesis-code options", function () {
  it("D01 no override: the M3 config file and hash are unchanged", function () {
    expect(CONFIG_PATH.endsWith(join("offchain", "model-config.json"))).to.equal(true);
    expect(modelConfigHash()).to.equal(keccak256(readFileSync(resolve("offchain", "model-config.json"))));
    expect(fiSpec(1).in).to.equal(6);
  });

  it("D02 VFL_MODEL_CONFIG with a per-passive fi.in gives each passive its own width", function () {
    const cfg = JSON.parse(readFileSync(resolve("offchain", "model-config.json"), "utf8"));
    cfg.architecture.fi.in = [3, 5, 7];
    const dir = mkdtempSync(join(tmpdir(), "d1cfg-"));
    const f = join(dir, "c.json");
    writeFileSync(f, JSON.stringify(cfg));
    const out = execFileSync(process.execPath, ["--input-type=module", "-e",
      `import { newFi, CONFIG_PATH } from ${JSON.stringify(pathToFileURL(MODEL).href)};` +
      `console.log(JSON.stringify([CONFIG_PATH, [1,2,3].map((i) => newFi(i).inDim)]));`],
      { env: { ...process.env, VFL_MODEL_CONFIG: f } }).toString();
    const [path, dims] = JSON.parse(out);
    expect(path).to.equal(f);
    expect(dims).to.deep.equal([3, 5, 7]);
  });

  it("D139a scoreFigures: F_r (FULL) is retrained, F_j (FULL_J) is the jointly trained fusion model; LOO and BASELINE retrained", async function () {
    this.timeout(120_000);
    // @ts-ignore
    const { genScenario } = await import("../offchain/data.mjs");
    // @ts-ignore
    const { newFA, newFusion, quantHead } = await import("../offchain/model.mjs");
    // @ts-ignore
    const { KIND, requiredFigures, scoreFigures } = await import("../offchain/scoring.mjs");
    const s = genScenario();
    const I = [...s.I].slice(0, 200);
    const rows = { xa: I.map((id: bigint) => s.A.rows.get(id).xa), y: I.map((id: bigint) => s.A.rows.get(id).y) };
    const hq = [0, 1, 2].map((p) => I.map((_: bigint, k: number) => [0, 1, 2, 3].map((e) => ((k * 7 + p * 13 + e * 29) % 2000) - 1000)));
    const fusion = newFusion();
    fusion.layers[0].W[0] += 0.5; // "trained": any model other than the initial one
    const figs = requiredFigures(7);
    expect(figs.map((f: any) => f.name)).to.deep.equal(["FULL", "FULL_J", "LOO_1", "LOO_2", "LOO_3", "BASELINE"]);
    expect(figs.map((f: any) => f.kind)).to.deep.equal([KIND.FULL, KIND.FULL_JOINT, KIND.LOO, KIND.LOO, KIND.LOO, KIND.BASELINE]);
    const args = { fA: newFA(), fusion, rows, hq, Itr: [...Array(150).keys()], idx: [...Array(50).keys()].map((k) => 150 + k), figures: figs };
    const out = Object.fromEntries(scoreFigures(args).figures.map((f: any) => [f.name, f]));
    expect(out.FULL_J.head).to.deep.equal(quantHead(fusion));
    expect(out.FULL_J.retrained).to.equal(false);
    for (const n of ["FULL", "LOO_1", "LOO_2", "LOO_3", "BASELINE"]) expect(out[n].retrained, n).to.equal(true);
    expect(out.FULL.head).to.not.deep.equal(quantHead(fusion));
    expect(out.FULL.mask).to.equal(out.FULL_J.mask);
    expect(() => scoreFigures({ ...args, fusion: undefined })).to.throw(/FULL_J/);
  });

  it("D139b the Federation ABI: commitFA(C_fA, gammaG), gammaG, uFed, submitJointScore(F, proof); TrainingEnded carries gammaG", function () {
    const abi = JSON.parse(readFileSync(resolve("artifacts", "contracts", "Federation.sol", "Federation.json"), "utf8")).abi;
    const fn = (n: string) => abi.find((x: any) => x.type === "function" && x.name === n);
    expect(fn("commitFA").inputs.map((i: any) => i.type)).to.deep.equal(["bytes32", "bytes32"]);
    expect(fn("gammaG").outputs.map((i: any) => i.type)).to.deep.equal(["bytes32"]);
    expect(fn("uFed").outputs.map((i: any) => i.type)).to.deep.equal(["uint256"]);
    expect(fn("submitJointScore").inputs.map((i: any) => i.type)).to.deep.equal(["uint256", "uint256[8]"]);
    expect(fn("submitFullScore")).to.equal(undefined);
    expect(fn("cFusion")).to.equal(undefined);
    expect(abi.find((x: any) => x.type === "event" && x.name === "TrainingEnded").inputs.map((i: any) => i.name)).to.include("gammaG");
  });
});

describe("DATA1 harness", function () {
  it("D03 sessionPairs: A and passive exchange sealed messages in order", async function () {
    // @ts-ignore
    const { sessionPairs } = await import("../offchain/data1/inproc.mjs");
    const { Wallet } = await import("ethers");
    const ws = [0, 1, 2, 3].map(() => Wallet.createRandom());
    const { pairs, close } = await sessionPairs(3, { chainId: 2026n, contract: ws[0].address, activeParty: ws[0].address, passives: ws.slice(1).map((w) => w.address) });
    await pairs[1].a.send("t/x", new Uint8Array([1, 2, 3]));
    const m = await pairs[1].p.recv(5000);
    expect(m.type).to.equal("t/x");
    expect(Array.from(m.payload)).to.deep.equal([1, 2, 3]);
    await pairs[2].p.send("t/y", new Uint8Array([9]));
    expect(Array.from((await pairs[2].a.recv(5000)).payload)).to.deep.equal([9]);
    close();
  });

  it("D04 writeConfig changes only the seeds and widths", async function () {
    // @ts-ignore
    const { writeConfig } = await import("../offchain/data1/configs.mjs");
    const dir = mkdtempSync(join(tmpdir(), "d1c-"));
    const { config, hash, path } = writeConfig({ name: "t", seed: 2, widths: [4, 5, 6], dir });
    const base = JSON.parse(readFileSync(resolve("offchain", "model-config.json"), "utf8"));
    expect(config.initSeed).to.equal(20261002);
    expect(config.shuffleSeed).to.equal(20262002);
    expect(config.architecture.fi.in).to.deep.equal([4, 5, 6]);
    const strip = (c: any) => ({ ...c, initSeed: 0, shuffleSeed: 0, architecture: { ...c.architecture, fi: { ...c.architecture.fi, in: 0 } } });
    expect(strip(config)).to.deep.equal(strip(base));
    expect(hash).to.equal(keccak256(readFileSync(path)));
  });

  it("D04b loadPrepared: 15,965 rows in DOEID order, A int16 rows of 8, each passive's width from the manifest, disjoint split", async function () {
    skipWithoutRecs(this, RECS_MANIFEST);
    // @ts-ignore
    const { loadPrepared } = await import("../offchain/data1/prepared.mjs");
    const d = loadPrepared({});
    expect(d.rows.xa.length).to.equal(15965);
    expect(d.rows.xa[0].length).to.equal(8);
    expect(d.rows.xa.every((r: number[]) => r.every((v) => Number.isInteger(v) && v >= -32768 && v <= 32767))).to.equal(true);
    expect(d.widths).to.deep.equal(["telecom", "retailer", "device"].map((p) => d.manifest.parties[p].length));
    d.xp.forEach((m: number[][], i: number) => expect(Array.isArray(m[0]) && m[0].length === d.widths[i]).to.equal(true));
    expect(d.trainPos.length + d.testPos.length).to.equal(15965);
    const test = new Set(d.testPos);
    expect(d.trainPos.some((k: number) => test.has(k))).to.equal(false);
    expect(d.rows.y.length).to.equal(15965);
  });

  it("D13s --seeds: a range or a list; the default 1-5 keeps the old file names, any other set gets its own tag", async function () {
    // @ts-ignore
    const { SEEDS, parseSeeds, seedTag } = await import("../offchain/data1/e1.mjs");
    expect(SEEDS).to.deep.equal([1, 2, 3, 4, 5]);
    expect(parseSeeds("1-5")).to.deep.equal(SEEDS);
    expect(parseSeeds("6-10")).to.deep.equal([6, 7, 8, 9, 10]);
    expect(parseSeeds("8,6,7")).to.deep.equal([6, 7, 8]);
    expect(parseSeeds("3")).to.deep.equal([3]);
    for (const bad of ["", "0-2", "5-3", "a", "1,1", "1.5", "-2"]) expect(() => parseSeeds(bad), bad).to.throw();
    expect(seedTag(SEEDS)).to.equal("");
    expect(seedTag([6, 7, 8, 9, 10])).to.equal("-s6-10");
    expect(seedTag([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).to.equal("-s1-10");
    expect(seedTag([6, 8])).to.equal("-s6_8");
    expect(seedTag([7])).to.equal("-s7");
  });
});

describe("DATA1 centralized reference and equivalence", function () {
  it("D05 split training (M3 code, no noise) equals centralized training with float32 transport, bit for bit", async function () {
    this.timeout(300_000);
    // @ts-ignore
    const { trainCentral, centralHq } = await import("../offchain/data1/centralized.mjs");
    // @ts-ignore
    const { quantFA, quantHead } = await import("../offchain/model.mjs");
    const { act, pas, rows, xp, Itr, sizeI } = await splitRun(2);
    expect(act.zeroFilled).to.deep.equal([]);
    expect(act.finals.every((f: any) => f.received && f.signatureValid)).to.equal(true);
    const c = trainCentral({ rows, xp, Itr, mode: "f32", T: 2 });
    expect(c.rounds.map((r: any) => r.loss)).to.deep.equal(act.rounds.map((r: any) => r.loss)); // exact
    expect(quantFA(c.fA)).to.deep.equal(quantFA(act.fA));
    expect(quantHead(c.fusion)).to.deep.equal(quantHead(act.fusion));
    expect(centralHq(c.fis, xp, sizeI)).to.deep.equal(pas.map((p: any) => p.hq));
  });

  it("D06 pure float64 centralized training differs from the split run only by float32 transport rounding", async function () {
    this.timeout(300_000);
    // @ts-ignore
    const { trainCentral } = await import("../offchain/data1/centralized.mjs");
    const { act, rows, xp, Itr } = await splitRun(2);
    const c = trainCentral({ rows, xp, Itr, mode: "f64", T: 2 });
    c.rounds.forEach((r: any, k: number) => expect(Math.abs(r.loss - act.rounds[k].loss)).to.be.lessThan(1e-6));
    expect(c.rounds.map((r: any) => r.loss)).to.not.deep.equal(act.rounds.map((r: any) => r.loss)); // f64 is not bit-equal
  });

  it("D06b noise: two runs with the default CSPRNG differ; two with the same seeded source are identical", async function () {
    this.timeout(300_000);
    // @ts-ignore
    const { gauss, prng } = await import("../offchain/data.mjs");
    const noise = { clip: 4, sigma: 0.5 };
    const seeded = () => [1, 2, 3].map((i) => { const r = prng(700n + BigInt(i)); return () => gauss(r); });
    const a = await splitRun(1, noise, seeded());
    const b = await splitRun(1, noise, seeded());
    expect(b.pas.map((p: any) => p.hq)).to.deep.equal(a.pas.map((p: any) => p.hq));
    const c = await splitRun(1, noise);
    const d = await splitRun(1, noise);
    expect(d.pas.map((p: any) => p.hq)).to.not.deep.equal(c.pas.map((p: any) => p.hq));
  });

  it("D06c trainCentral aOnly (C1b) keeps every passive slot at zero: no passive model, and its fusion never sees them", async function () {
    this.timeout(120_000);
    // @ts-ignore
    const { trainCentral } = await import("../offchain/data1/centralized.mjs");
    // @ts-ignore
    const { genScenario } = await import("../offchain/data.mjs");
    const s = genScenario();
    const I = [...s.I].slice(0, 300);
    const rows = { xa: I.map((id: bigint) => s.A.rows.get(id).xa), y: I.map((id: bigint) => s.A.rows.get(id).y) };
    const xp = s.passives.map((p: any) => I.map((id: bigint) => p.rows.get(id)));
    const c = trainCentral({ rows, xp, Itr: [...Array(256).keys()], mode: "aOnly", T: 2 });
    expect(c.fis).to.deep.equal([]);
    // the weights on the 12 passive inputs of the fusion's first layer never receive a gradient: still their init values
    // @ts-ignore
    const { newFusion } = await import("../offchain/model.mjs");
    const init = newFusion().layers[0];
    const L = c.fusion.layers[0];
    for (let o = 0; o < L.outDim; o++) for (let i = 4; i < L.inDim; i++) expect(L.W[o * L.inDim + i]).to.equal(init.W[o * L.inDim + i]);
  });
});

describe("DATA1 evaluation", function () {
  it("D07 n64Positions draws with replacement exactly like the protocol and fOn equals macroF1 on those rows", async function () {
    // @ts-ignore
    const { n64Positions, fOn } = await import("../offchain/data1/evaluate.mjs");
    // @ts-ignore
    const { sampleIndices } = await import("../../circuits/scripts/members-lib.mjs");
    // @ts-ignore
    const { macroF1 } = await import("../offchain/model.mjs");
    const testPos = [5, 9, 11, 20, 33, 40, 41, 58];
    const pos = await n64Positions(7, testPos);
    const raw = await sampleIndices(7n, 64, testPos.length);
    expect(pos).to.deep.equal(raw.map((t: number) => testPos[t]));
    expect(new Set(pos).size).to.be.lessThan(64); // with replacement
    const rows = { y: Array.from({ length: 60 }, (_, k) => k % 5) };
    const preds = pos.map((k: number) => (k + 1) % 5);
    expect(fOn(rows, pos, preds)).to.equal(macroF1(pos.map((k: number) => rows.y[k]), preds).F);
  });

  it("D07b saturation counts int16-clipped embedding entries per passive", async function () {
    // @ts-ignore
    const { saturation } = await import("../offchain/data1/evaluate.mjs");
    const hq = [[[0, 32767, 5, -32768], [1, 2, 3, 4]], [[0, 0, 0, 0], [0, 0, 0, 0]], [[-32768, -32768, 0, 0], [32767, 0, 0, 0]]];
    expect(saturation(hq)).to.deep.equal([2, 0, 3]);
  });

  it("D07c predictInt with a null slot equals forwardInt with that slot zero", async function () {
    // @ts-ignore
    const { predictInt } = await import("../offchain/data1/evaluate.mjs");
    // @ts-ignore
    const { forwardInt, newFA, newFusion, quantFA, quantHead } = await import("../offchain/model.mjs");
    const fa = quantFA(newFA()), head = quantHead(newFusion());
    const rows = { xa: [[100, -200, 300, 0, 50, 60, -70, 80], [0, 0, 0, 0, 0, 0, 0, 4096]] };
    const hq = [[[10, 20, 30, 40], [1, 2, 3, 4]], null, [[5, 6, 7, 8], [9, 9, 9, 9]]];
    const got = predictInt({ fa, head, mask: 7, rows, hq, positions: [0, 1] });
    const zero = [0, 0, 0, 0];
    expect(got).to.deep.equal([0, 1].map((k) => forwardInt(fa, head, 7, rows.xa[k], [hq[0]![k], zero, hq[2]![k]])));
  });
});

describe("DATA1 E3 scenario and proposal switches", function () {
  it("D08 recsScenario: |X_A| = 1,020, each passive holds 90% of A plus 400 others, |I| in [910, 918], deterministic", async function () {
    this.timeout(60_000);
    skipWithoutRecs(this, RECS_MANIFEST);
    // @ts-ignore
    const { recsScenario } = await import("../offchain/data1/recs-scenario.mjs");
    const s = recsScenario({});
    expect(s.A.ids.length).to.equal(1020);
    const aSet = new Set(s.A.ids);
    for (const p of s.passives) {
      expect(p.ids.filter((x: bigint) => aSet.has(x)).length).to.equal(918);
      expect(p.ids.length).to.equal(918 + 400);
      expect(p.ids.every((id: bigint) => p.rows.get(id).length === p.rows.get(p.ids[0]).length)).to.equal(true);
    }
    expect(s.I.length).to.be.within(910, 918);
    expect(s.I.every((id: bigint) => s.passives.every((p: any) => p.rows.has(id)) && s.A.rows.has(id))).to.equal(true);
    expect(s.A.ids.every((id: bigint) => id >= 1_000_000_000n && id <= 9_999_999_999n)).to.equal(true);
    expect(s.classBoundaries).to.deep.equal([7400n, 11600n, 16800n, 24300n]);
    const again = recsScenario({});
    expect(again.I).to.deep.equal(s.I);
    expect(again.A.rows.get(s.I[0])).to.deep.equal(s.A.rows.get(s.I[0]));
  });

  it("D09 defaultProposal takes classBoundaries and keeps [100..400] by default", async function () {
    // @ts-ignore
    const { defaultProposal } = await import("../offchain/proposal.mjs");
    const base = { activeParty: "0x" + "1".repeat(40), activeChannelKey: "0x00", passives: [], arbiter: "0x" + "2".repeat(40),
      verifier: "0x" + "3".repeat(40), token: "0x" + "4".repeat(40), activeCustomers: 1020 };
    expect(defaultProposal(base).classBoundaries).to.deep.equal([100n, 200n, 300n, 400n]);
    expect(defaultProposal({ ...base, classBoundaries: [7400n, 11600n, 16800n, 24300n] }).classBoundaries)
      .to.deep.equal([7400n, 11600n, 16800n, 24300n]);
  });
});

describe("DATA1 run-config guards (review finding 2)", function () {
  it("D10 assertFiWidth: a passive row whose width differs from fi.in is refused", async function () {
    // @ts-ignore
    const { assertFiWidth, fiSpec } = await import("../offchain/model.mjs");
    const w = fiSpec(1).in; // default M3 config
    expect(() => assertFiWidth(1, w)).to.not.throw();
    expect(() => assertFiWidth(1, 46)).to.throw(/width/);
  });

  it("D11 checkRunConfig: e3 needs VFL_MODEL_CONFIG with the data's widths; i6-i9 refuse an override", async function () {
    // @ts-ignore
    const { checkRunConfig } = await import("../offchain/model.mjs");
    const fiIn = [46, 225, 358];
    expect(() => checkRunConfig({ run: "e3", configEnv: undefined, fiIn, dataWidths: fiIn })).to.throw(/VFL_MODEL_CONFIG/);
    expect(() => checkRunConfig({ run: "e3", configEnv: "x.json", fiIn: [6, 6, 6], dataWidths: fiIn })).to.throw(/width/);
    expect(() => checkRunConfig({ run: "e3", configEnv: "x.json", fiIn, dataWidths: fiIn })).to.not.throw();
    for (const run of ["i6", "i7", "i8", "i9"]) {
      expect(() => checkRunConfig({ run, configEnv: "x.json", fiIn, dataWidths: null })).to.throw(/VFL_MODEL_CONFIG/);
      expect(() => checkRunConfig({ run, configEnv: undefined, fiIn: [6, 6, 6], dataWidths: null })).to.not.throw();
    }
  });
});

describe("Q17 follow-up: centralized leave-one-out reference", function () {
  it("D12 trainCentral exclude: the excluded passive has no model and a zero slot; [] = f32, [1,2,3] = aOnly", async function () {
    this.timeout(180_000);
    // @ts-ignore
    const { trainCentral, centralHq } = await import("../offchain/data1/centralized.mjs");
    // @ts-ignore
    const { genScenario } = await import("../offchain/data.mjs");
    // @ts-ignore
    const { newFusion } = await import("../offchain/model.mjs");
    const s = genScenario();
    const I = [...s.I].slice(0, 300);
    const rows = { xa: I.map((id: bigint) => s.A.rows.get(id).xa), y: I.map((id: bigint) => s.A.rows.get(id).y) };
    const xp = s.passives.map((p: any) => I.map((id: bigint) => p.rows.get(id)));
    const Itr = [...Array(256).keys()];
    const w = (c: any) => c.fusion.layers.map((l: any) => [Array.from(l.W), Array.from(l.b)]);
    const c2 = trainCentral({ rows, xp, Itr, mode: "f32", exclude: [2], T: 2 });
    expect(c2.fis[1]).to.equal(null);
    expect(c2.fis[0]).to.not.equal(null);
    const hq = centralHq(c2.fis, xp, I.length);
    expect(hq[1]).to.equal(null);
    expect(hq[0].length).to.equal(I.length);
    const init = newFusion().layers[0];
    const L = c2.fusion.layers[0];
    const inDim = init.W.length / init.b.length; // W is out x in, row-major
    for (let o = 0; o < init.b.length; o++) for (let e = 8; e < 12; e++) expect(L.W[o * inDim + e]).to.equal(init.W[o * inDim + e]);
    expect(w(trainCentral({ rows, xp, Itr, mode: "f32", exclude: [], T: 2 }))).to.deep.equal(w(trainCentral({ rows, xp, Itr, mode: "f32", T: 2 })));
    expect(w(trainCentral({ rows, xp, Itr, mode: "f32", exclude: [1, 2, 3], T: 2 }))).to.deep.equal(w(trainCentral({ rows, xp, Itr, mode: "aOnly", T: 2 })));
  });
});
