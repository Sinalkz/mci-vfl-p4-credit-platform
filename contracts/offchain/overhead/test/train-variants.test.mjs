// Task 4: the three training variants run the thesis's own training code (training.mjs) and differ only in the
// protocol's extra steps. On the M3 synthetic scenario with the default model config (6-wide passives).
//   node --test offchain/overhead/test/        (from contracts/)
import assert from "node:assert/strict";
import { test } from "node:test";

import { genScenario } from "../../data.mjs";
import { trainCentral } from "../../data1/centralized.mjs";
import { quantFA, quantHead } from "../../model.mjs";
import { trainOnce, VARIANTS } from "../train-variants.mjs";

function m3Rows() {
  const s = genScenario();
  const I = [...s.I].sort((a, b) => (a < b ? -1 : 1));
  return {
    rows: { xa: I.map((id) => s.A.rows.get(id).xa), y: I.map((id) => s.A.rows.get(id).y) },
    xp: s.passives.map((p) => I.map((id) => p.rows.get(id))), Itr: [...Array(800).keys()], sizeI: I.length, maxDepth: 10,
  };
}

test("V1 the variants: plain = clear frames, no clipping, no noise; clipnoise adds clipping and noise; protocol adds the channel", () => {
  assert.deepEqual(Object.keys(VARIANTS), ["plain", "clipnoise", "protocol"]);
  assert.deepEqual(VARIANTS.plain, { transport: "plain", noise: false });
  assert.deepEqual(VARIANTS.clipnoise, { transport: "plain", noise: true });
  assert.deepEqual(VARIANTS.protocol, { transport: "session", noise: true });
});

test("V2 plain equals centralized float32 training bit for bit (the same arithmetic as split training without noise)", async () => {
  const d = m3Rows();
  const r = await trainOnce({ variant: "plain", ...d, T: 2, keep: true });
  const c = trainCentral({ rows: d.rows, xp: d.xp, Itr: d.Itr, mode: "f32", T: 2 });
  assert.deepEqual(r.rounds.map((x) => x.loss), c.rounds.map((x) => x.loss));
  assert.deepEqual(quantFA(r.fA), quantFA(c.fA));
  assert.deepEqual(quantHead(r.fusion), quantHead(c.fusion));
  assert.equal(r.rounds.length, 2);
  assert.ok(r.rounds.every((x) => x.ms > 0 && x.batches === Math.ceil(800 / 64)));
  assert.ok(r.finalMs > 0 && r.totalMs >= r.finalMs);
  assert.equal(r.acks, 0); // clear frames: nothing is acknowledged
});

test("V3 the channel changes no number: protocol with noise switched off equals plain", async () => {
  const d = m3Rows();
  const p = await trainOnce({ variant: "plain", ...d, T: 1, keep: true });
  const s = await trainOnce({ variant: "protocol", ...d, T: 1, keep: true, noiseOff: true });
  assert.deepEqual(s.rounds.map((x) => x.loss), p.rounds.map((x) => x.loss));
  assert.deepEqual(s.hq, p.hq);
  assert.ok(s.acks > 0); // every sealed frame is acknowledged
});

test("V4 clipnoise and protocol add noise (the OS CSPRNG): their final matrices differ from plain's", async () => {
  const d = m3Rows();
  const p = await trainOnce({ variant: "plain", ...d, T: 1, keep: true });
  for (const variant of ["clipnoise", "protocol"]) {
    const n = await trainOnce({ variant, ...d, T: 1, keep: true });
    assert.notDeepEqual(n.hq, p.hq);
    assert.equal(n.rounds.length, 1);
  }
});
