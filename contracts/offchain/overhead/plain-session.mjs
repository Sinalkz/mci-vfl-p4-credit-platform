// Task 4 (off-chain overhead), the plain split-learning baseline's transport. Not used by the protocol.
// PlainSession has the members of Session (session.mjs) that training.mjs uses: send, recv, failed, bytesOut,
// bytesIn, close. It runs over the same Link (link.mjs: TCP, 4-byte length framing), but every frame is a clear
// body (channel.mjs clearBody: kind 0 | typeLen | type | payload). There is no encryption, no tag, no counter, no
// acknowledgement and no retransmission; TCP alone carries the frames in order.
import { ChannelError, clearBody, parseBody } from "../channel.mjs";
import { connect, listen } from "../link.mjs";

export class PlainSession {
  constructor(link) {
    this.link = link;
    this.inbox = [];
    this.waiters = [];
    this.failed = null;
    this.stats = { sent: 0, delivered: 0, droppedFormat: 0, acksSent: 0, acksReceived: 0 };
    this.readLoop();
  }

  get bytesOut() { return this.link.bytesOut; }
  get bytesIn() { return this.link.bytesIn; }

  /** Sends one clear frame; resolves once it is written to the socket. */
  send(type, payload) {
    if (this.failed) return Promise.reject(this.failed);
    this.stats.sent++;
    return this.link.send(clearBody(type, payload));
  }

  /** The next message { type, payload }, in arrival order (as Session.recv). */
  recv(timeoutMs = 120_000) {
    const m = this.inbox.shift();
    if (m) return Promise.resolve(m);
    if (this.failed) return Promise.reject(this.failed);
    return new Promise((resolve, reject) => {
      const w = { resolve: (x) => { clearTimeout(t); resolve(x); }, reject: (e) => { clearTimeout(t); reject(e); } };
      const t = setTimeout(() => { this.waiters.splice(this.waiters.indexOf(w), 1); reject(new ChannelError("timeout", "recv timeout")); }, timeoutMs);
      this.waiters.push(w);
    });
  }

  close() {
    this.fail(new ChannelError("closed", "session closed"));
    this.link.destroy();
  }

  fail(err) {
    if (this.failed) return;
    this.failed = err;
    for (const w of this.waiters.splice(0)) w.reject(err);
  }

  deliver(m) {
    this.stats.delivered++;
    const w = this.waiters.shift();
    if (w) w.resolve(m); else this.inbox.push(m);
  }

  async readLoop() {
    for (;;) {
      let body;
      try {
        body = await this.link.recv(0);
      } catch (e) {
        this.fail(new ChannelError("link", e.message));
        return;
      }
      if (this.failed) return;
      let m;
      try { m = parseBody(body); } catch { this.stats.droppedFormat++; continue; }
      if (m.kind !== 0) { this.stats.droppedFormat++; continue; }
      this.deliver({ type: m.type, payload: m.rest });
    }
  }
}

/** n A<->passive pairs over loopback TCP, as data1/inproc.mjs sessionPairs, with PlainSession instead of Session. */
export async function plainPairs(n) {
  const accepted = [];
  const server = await listen(0, (link) => accepted.push(link));
  const pairs = [];
  for (let i = 0; i < n; i++) {
    const pLink = await connect(server.address().port);
    while (accepted.length < i + 1) await new Promise((ok) => setImmediate(ok));
    pairs.push({ a: new PlainSession(accepted[i]), p: new PlainSession(pLink) });
  }
  const close = () => {
    for (const x of pairs) { x.a.link.destroy(); x.p.link.destroy(); }
    server.close();
  };
  return { pairs, close };
}
