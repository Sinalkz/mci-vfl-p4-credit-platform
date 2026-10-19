// M2 test relay: a separate process between a passive (client) and A (server) that forwards
// frames and can drop, modify, replay, reorder or substitute them.
// Usage: node offchain/relay.mjs --target 127.0.0.1:<A port> [--policy '<json>'] [--capture <file>]
// Prints "RELAY <port>" once listening (an OS-assigned port).
//
// Policy: { "p2a": [action...], "a2p": [action...] }. Frame indices count per direction
// from 0, acks included; in "p2a" frame 0 is the passive's clear hello.
//   {"op":"drop","at":i}                         do not forward frame i
//   {"op":"dropFrom","at":i}                     do not forward frame i or any later frame
//   {"op":"dropType","type":t,"nth":k}           do not forward the k-th (from 0) sealed frame of message
//                                                type t; a retransmission is a later frame and passes
//   {"op":"flip","at":i,"byte":k,"bit":b}        flip bit b of byte k of frame i (k < 0 counts from the end)
//   {"op":"replay","at":i,"of":j}                after forwarding frame i, send frame j again
//   {"op":"swap","at":i}                         hold frame i and forward it after frame i+1
//   {"op":"substituteHelloPk","pk":"0x.."}       replace the pk offered in the clear hello
// --capture appends one JSON line per forwarded frame: {"dir","index","hex"}.
import { appendFileSync } from "node:fs";
import net from "node:net";
import { parseArgs } from "node:util";

import { Link } from "./link.mjs";
import { clearBody, parseBody } from "./channel.mjs";

const { values: args } = parseArgs({ options: {
  target: { type: "string" }, policy: { type: "string", default: "{}" }, capture: { type: "string" },
} });
const [thost, tport] = args.target.split(":");
const policy = JSON.parse(args.policy);

function pipe(from, to, dir) {
  const actions = policy[dir] ?? [];
  const seen = [];
  let held = null;
  let index = 0;
  const typeCount = new Map();
  const forward = async (body, i) => {
    if (args.capture) appendFileSync(args.capture, JSON.stringify({ dir, index: i, hex: Buffer.from(body).toString("hex") }) + "\n");
    await to.send(body);
  };
  (async () => {
    for (;;) {
      let body;
      try { body = await from.recv(24 * 3600 * 1000); } catch { to.close(); return; }
      const i = index++;
      seen[i] = body;
      const mine = actions.filter((a) => a.at === i);
      if (body[0] === 1) {
        const { type } = parseBody(body);
        const k = typeCount.get(type) ?? 0;
        typeCount.set(type, k + 1);
        if (actions.some((a) => a.op === "dropType" && a.type === type && a.nth === k)) continue;
      }
      if (mine.some((a) => a.op === "drop") || actions.some((a) => a.op === "dropFrom" && i >= a.at)) continue;
      for (const a of mine.filter((x) => x.op === "flip")) {
        body = Uint8Array.from(body);
        const k = a.byte < 0 ? body.length + a.byte : a.byte;
        body[k] ^= 1 << a.bit;
      }
      for (const a of mine.filter((x) => x.op === "substituteHelloPk")) {
        const { type, rest } = parseBody(body);
        const hello = JSON.parse(new TextDecoder().decode(rest));
        body = clearBody(type, new TextEncoder().encode(JSON.stringify({ ...hello, pk: a.pk })));
      }
      if (mine.some((a) => a.op === "swap")) { held = { body, i }; continue; }
      await forward(body, i);
      if (held) { const h = held; held = null; await forward(h.body, h.i); }
      for (const a of mine.filter((x) => x.op === "replay")) await forward(seen[a.of], a.of);
    }
  })();
}

const server = net.createServer(async (client) => {
  const down = new Link(client);
  const up = await new Promise((resolve, reject) => {
    const s = net.connect(Number(tport), thost, () => resolve(new Link(s)));
    s.once("error", reject);
  });
  pipe(down, up, "p2a");
  pipe(up, down, "a2p");
});
server.listen(0, "127.0.0.1", () => console.log(`RELAY ${server.address().port}`));
