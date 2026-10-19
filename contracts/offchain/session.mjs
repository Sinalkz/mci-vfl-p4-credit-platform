// M2 (answer to QUESTIONS-M2 Q2): retransmission on top of the secure channel.
// A Session owns one Link and one Channel and reads every frame in the background.
//
// Sender: keeps each sealed frame until it is acknowledged. When the oldest unacknowledged
// frame's timer expires, it resends every unacknowledged frame byte for byte (same counter,
// same ciphertext), doubling the timeout up to maxRtoMs. After maxRetries resends without
// progress, the session fails with ChannelError "retries".
// Receiver: acknowledges each accepted counter. A frame is accepted only when its counter is
// last_accepted + 1 and its tag verifies (Channel.inspect, the same rule as Channel.open).
//   duplicate of an accepted counter (tag verifies)  -> acknowledged again, dropped
//   tag fails                                        -> dropped silently; the sender resends
//   counter ahead of the next one                    -> dropped silently; the sender resends
// Acceptance is in order, so an ack of c also covers every counter below c.
// Acks are authenticated (Channel.sealAck/openAck); an ack that does not verify is dropped.
// A broken TCP connection is not recovered: the session fails and the protocol's on-chain
// timeout applies.
import { ChannelError } from "./channel.mjs";

export const SESSION_DEFAULTS = { rtoMs: 3_000, maxRtoMs: 60_000, maxRetries: 8 };

export class Session {
  constructor(link, channel, opts = {}) {
    this.link = link;
    this.ch = channel;
    this.opts = { ...SESSION_DEFAULTS, ...opts };
    this.unacked = new Map(); // counter -> { body, resolve, reject }, in counter order
    this.inbox = [];
    this.waiters = [];
    this.failed = null;
    this.timer = null;
    this.rto = this.opts.rtoMs;
    this.retries = 0;
    this.stats = { sent: 0, delivered: 0, retransmitted: 0, acksSent: 0, acksReceived: 0,
      duplicates: 0, droppedTag: 0, droppedFuture: 0, droppedFormat: 0, droppedAck: 0 };
    this.readLoop();
  }

  get bytesOut() { return this.link.bytesOut; }
  get bytesIn() { return this.link.bytesIn; }

  /** Seals and sends one message; resolves once the peer has acknowledged it. */
  send(type, payload) {
    if (this.failed) return Promise.reject(this.failed);
    const body = this.ch.seal(type, payload);
    const counter = this.ch.sendCounter - 1n;
    this.stats.sent++;
    return new Promise((resolve, reject) => {
      this.unacked.set(counter, { body, resolve, reject });
      this.transmit(body);
      if (!this.timer) this.arm();
    });
  }

  /** The next accepted message { type, payload }, in counter order. */
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

  // ---- internals ----

  // A write error closes the socket; readLoop then fails the session.
  transmit(body) { this.link.send(body).catch(() => {}); }

  arm() { this.timer = setTimeout(() => this.onTimeout(), this.rto); }

  onTimeout() {
    this.timer = null;
    if (this.failed || this.unacked.size === 0) return;
    if (this.retries >= this.opts.maxRetries) {
      const oldest = this.unacked.keys().next().value;
      this.fail(new ChannelError("retries", `no acknowledgement of counter ${oldest} after ${this.retries} retransmissions`));
      return;
    }
    this.retries++;
    for (const f of this.unacked.values()) { this.transmit(f.body); this.stats.retransmitted++; }
    this.rto = Math.min(this.rto * 2, this.opts.maxRtoMs);
    this.arm();
  }

  onAck(counter) {
    let advanced = false;
    for (const [c, f] of this.unacked) {
      if (c > counter) break;
      this.unacked.delete(c);
      f.resolve();
      advanced = true;
    }
    if (!advanced) return; // an ack of something already acknowledged
    clearTimeout(this.timer);
    this.timer = null;
    this.retries = 0;
    this.rto = this.opts.rtoMs;
    if (this.unacked.size > 0) this.arm();
  }

  ack(counter) {
    this.stats.acksSent++;
    this.transmit(this.ch.sealAck(counter));
  }

  deliver(m) {
    this.stats.delivered++;
    const w = this.waiters.shift();
    if (w) w.resolve(m); else this.inbox.push(m);
  }

  fail(err) {
    if (this.failed) return;
    this.failed = err;
    clearTimeout(this.timer);
    this.timer = null;
    for (const f of this.unacked.values()) f.reject(err);
    this.unacked.clear();
    for (const w of this.waiters.splice(0)) w.reject(err);
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
      if (body[0] === 2) {
        const c = this.ch.openAck(body);
        if (c === null) { this.stats.droppedAck++; continue; }
        this.stats.acksReceived++;
        this.onAck(c);
        continue;
      }
      const r = this.ch.inspect(body);
      if (r.status === "next") { this.ack(r.counter); this.deliver({ type: r.type, payload: r.payload }); }
      else if (r.status === "duplicate") { this.stats.duplicates++; this.ack(r.counter); }
      else if (r.status === "tag") this.stats.droppedTag++;
      else if (r.status === "future") this.stats.droppedFuture++;
      else this.stats.droppedFormat++;
    }
  }
}
