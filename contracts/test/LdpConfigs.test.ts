// Task 5 (LDP configurations): the config variant, the stratified subsample and the shadow passive (the final matrix
// before the int16 cap) add nothing to the thesis code's path.
import { expect } from "chai";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

// @ts-ignore -- plain ESM modules
import { beyondInt16, shadowPassive, stratifiedSubsample, variantConfig, writeVariantConfig } from "../offchain/data1/ldp-cfg-lib.mjs";
import { RECS_MANIFEST, skipWithoutRecs } from "./helpers/recs.js";

const E1_CONFIG = (s: number) => resolve("..", "results", "data1", "configs", `e1-s${s}.json`);
const WIDTHS = [46, 225, 358];

/** A small split run of the thesis code on the M3 synthetic scenario (default config), with seeded noise; with
 * shadow = true every passive runs through shadowPassive. */
async function smallRun(shadow: boolean, noise: { clip: number; sigma: number }, T = 2) {
  // @ts-ignore
  const { genScenario, gauss, prng } = await import("../offchain/data.mjs");
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
  const sizeI = I.length;
  const normals = [1, 2, 3].map((i) => { const r = prng(BigInt(777 + i)); return () => gauss(r); });
  const ws = [0, 1, 2, 3].map(() => Wallet.createRandom());
  const ctx = { chainId: 2026n, contract: ws[0].address, activeParty: ws[0].address, passives: ws.slice(1).map((w) => w.address) };
  const { pairs, close } = await sessionPairs(3, ctx);
  const shadows = shadow ? pairs.map((x: any, i: number) => shadowPassive({ session: x.p, index: i + 1, xp: xp[i], sizeI, noise, normal: normals[i] })) : null;
  const [act, ...pas] = await Promise.all([
    activeTraining({ peers: pairs.map((x: any, i: number) => ({ index: i + 1, address: ws[i + 1].address, session: x.a })),
      rows, Itr, sizeI, chainId: 2026n, contract: ctx.contract, maxDepth: 10, T, waitMs: 60_000 }),
    ...pairs.map((x: any, i: number) => passiveTraining({ session: shadows ? shadows[i].session : x.p, index: i + 1, xp: xp[i], sizeI,
      chainId: 2026n, contract: ctx.contract, maxDepth: 10, signer: ws[i + 1], noise, normal: shadows ? shadows[i].normal : normals[i] })),
  ]);
  close();
  return { act, pas, shadows };
}

describe("Task 5 LDP configurations", function () {
  it("L01 variantConfig at T = 20 is the E1 config byte for byte; another T changes only T", function () {
    skipWithoutRecs(this, E1_CONFIG(1), E1_CONFIG(5));
    for (const s of [1, 5]) {
      const v = variantConfig({ seed: s, widths: WIDTHS, T: 20 });
      expect(v.bytes.equals(readFileSync(E1_CONFIG(s)))).to.equal(true);
    }
    expect(variantConfig({ seed: 1, widths: WIDTHS, T: 20 }).hash).to.equal("0x52bd91d1417e7ec99a7aa22bac26846022c10d21e120ce0b420e5ccaf18eb3f8");
    const a = variantConfig({ seed: 1, widths: WIDTHS, T: 20 }).config;
    const b = variantConfig({ seed: 1, widths: WIDTHS, T: 5 }).config;
    expect(b.T).to.equal(5);
    expect(JSON.stringify({ ...b, T: 20 })).to.equal(JSON.stringify(a));
    expect(() => variantConfig({ seed: 1, widths: WIDTHS, T: 0 })).to.throw();
    const dir = mkdtempSync(join(tmpdir(), "ldpcfg-"));
    const f = join(dir, "c.json");
    writeVariantConfig(f, { seed: 1, widths: WIDTHS, T: 5 });
    expect(writeVariantConfig(f, { seed: 1, widths: WIDTHS, T: 5 })).to.equal(variantConfig({ seed: 1, widths: WIDTHS, T: 5 }).hash);
    writeFileSync(f, "{}");
    expect(() => writeVariantConfig(f, { seed: 1, widths: WIDTHS, T: 5 })).to.throw(/other bytes/);
  });

  it("L02 stratifiedSubsample: the class shares, round(frac N) rows, nested across fractions, seeded, in trainPos order", async function () {
    skipWithoutRecs(this, RECS_MANIFEST);
    // @ts-ignore
    const { loadManifest } = await import("../offchain/data1/prepared.mjs");
    const m = loadManifest();
    const tr: number[] = m.split.train_pos;
    const y: number[] = m.y;
    const counts = (pos: number[]) => pos.reduce((c: number[], k) => { c[y[k]]++; return c; }, [0, 0, 0, 0, 0]);
    const all = counts(tr);
    expect(all).to.deep.equal([4512, 3498, 2573, 1548, 641]);
    const full = stratifiedSubsample(tr, y, 1, 1);
    expect(full).to.deep.equal(tr);
    expect(full).to.not.equal(tr);
    for (const s of [1, 2, 3, 4, 5]) {
      const q = stratifiedSubsample(tr, y, 0.25, s);
      const h = stratifiedSubsample(tr, y, 0.5, s);
      expect(q.length).to.equal(3193);
      expect(h.length).to.equal(6386);
      for (const [sub, f] of [[q, 0.25], [h, 0.5]] as [number[], number][]) {
        counts(sub).forEach((c, k) => expect(Math.abs(c - f * all[k])).to.be.below(1));
        const at = sub.map((k) => tr.indexOf(k));
        expect(at.every((v, i) => v >= 0 && (i === 0 || at[i - 1] < v))).to.equal(true);
      }
      const hs = new Set(h);
      expect(q.every((k) => hs.has(k))).to.equal(true);
      expect(stratifiedSubsample(tr, y, 0.25, s)).to.deep.equal(q);
    }
    expect(stratifiedSubsample(tr, y, 0.25, 1)).to.not.deep.equal(stratifiedSubsample(tr, y, 0.25, 2));
    expect(() => stratifiedSubsample(tr, y, 0, 1)).to.throw();
    expect(() => stratifiedSubsample(tr, y, 1.5, 1)).to.throw();
  });

  it("L03 shadowPassive: the run is unchanged; the uncapped matrix gives the sent int16 matrix and exceeds it where it saturates", async function () {
    const noise = { clip: 4, sigma: 26 };
    const plain = await smallRun(false, noise);
    const shadowed = await smallRun(true, noise);
    for (let i = 0; i < 3; i++) expect(shadowed.pas[i].hq).to.deep.equal(plain.pas[i].hq);
    const w = (f: any) => f.layers.map((l: any) => [...l.W, ...l.b]);
    expect(w(shadowed.act.fusion)).to.deep.equal(w(plain.act.fusion));
    const off = shadowed.shadows!.map((s: any, i: number) => s.uncapped(shadowed.pas[i].hq)); // throws on any mismatch
    const b = beyondInt16(off);
    for (let i = 0; i < 3; i++) {
      const hq = shadowed.pas[i].hq;
      let sat = 0;
      hq.forEach((row: number[], r: number) => row.forEach((v: number, e: number) => {
        if (v === 32767 || v === -32768) { sat++; expect(Math.sign(off[i][r][e])).to.equal(Math.sign(v)); }
        else expect(off[i][r][e]).to.equal(v);
      }));
      expect(sat).to.be.above(0);
      expect(b[i].beyond).to.be.above(0);
      expect(b[i].beyond).to.be.at.most(sat);           // only a saturated entry can lie beyond the int16 range
      expect(Math.max(-b[i].min, b[i].max)).to.be.above(32767);
    }
    // a tampered matrix is refused
    const bad = shadowed.pas[0].hq.map((row: number[]) => row.slice());
    bad[0][0] = bad[0][0] === 0 ? 1 : 0;
    expect(() => shadowed.shadows![0].uncapped(bad)).to.throw(/does not give the sent/);
  });
});
