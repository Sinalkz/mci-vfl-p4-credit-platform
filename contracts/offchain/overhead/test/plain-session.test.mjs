// Task 4: PlainSession, the clear-frame transport of the plain split-learning baseline.
//   node --test offchain/overhead/test/        (from contracts/)
import assert from "node:assert/strict";
import { test } from "node:test";

import { Session } from "../../session.mjs";
import { PlainSession, plainPairs } from "../plain-session.mjs";

const bytes = (...v) => Uint8Array.from(v);

test("P1 messages arrive in order with their types and payloads", async () => {
  const { pairs, close } = await plainPairs(2);
  try {
    for (const [k, x] of pairs.entries()) {
      await x.a.send("train/batch", bytes(1, 2, k));
      await x.a.send("train/end", new Uint8Array(0));
      await x.p.send("train/emb", bytes(9, 8, 7, 6));
      const m1 = await x.p.recv(5_000);
      const m2 = await x.p.recv(5_000);
      const m3 = await x.a.recv(5_000);
      assert.equal(m1.type, "train/batch");
      assert.deepEqual([...m1.payload], [1, 2, k]);
      assert.equal(m2.type, "train/end");
      assert.equal(m2.payload.length, 0);
      assert.equal(m3.type, "train/emb");
      assert.deepEqual([...m3.payload], [9, 8, 7, 6]);
    }
  } finally { close(); }
});

test("P2 a frame is the clear body only: no counter, no tag, no acknowledgement", async () => {
  const { pairs, close } = await plainPairs(1);
  try {
    const { a, p } = pairs[0];
    const payload = new Uint8Array(1032);
    await a.send("train/emb", payload);
    await p.recv(5_000);
    // 4-byte length | kind | typeLen | "train/emb" (9) | payload
    assert.equal(a.bytesOut, 4 + 2 + 9 + 1032);
    assert.equal(p.bytesIn, a.bytesOut);
    await new Promise((ok) => setTimeout(ok, 50));
    assert.equal(p.bytesOut, 0); // nothing sent back
  } finally { close(); }
});

test("P3 recv times out, and a closed peer fails the session", async () => {
  const { pairs, close } = await plainPairs(1);
  try {
    const { a, p } = pairs[0];
    await assert.rejects(p.recv(30), /timeout/);
    a.close();
    await assert.rejects(p.recv(5_000), (e) => e.code === "link");
    assert.ok(p.failed);
    await assert.rejects(p.send("x", new Uint8Array(0)));
  } finally { close(); }
});

test("P4 PlainSession has the members of Session that training.mjs uses", () => {
  for (const m of ["send", "recv", "close"]) assert.equal(typeof PlainSession.prototype[m], "function");
  for (const g of ["bytesOut", "bytesIn"]) {
    assert.ok(Object.getOwnPropertyDescriptor(Session.prototype, g).get);
    assert.ok(Object.getOwnPropertyDescriptor(PlainSession.prototype, g).get);
  }
});
