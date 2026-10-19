// M2 transport: length-prefixed frames over TCP (4-byte big-endian length, then the body).
// A Link carries message bodies; it knows nothing about encryption.
import net from "node:net";

const MAX_FRAME = 64 * 1024 * 1024;

/** Wraps a socket: send(body), recv() -> Promise<body>, with counters of bytes each way. */
export class Link {
  constructor(socket) {
    this.socket = socket;
    this.buf = Buffer.alloc(0);
    this.queue = [];
    this.waiters = [];
    this.closed = false;
    this.bytesOut = 0;
    this.bytesIn = 0;
    socket.on("data", (d) => this.onData(d));
    socket.on("close", () => { this.closed = true; this.flushWaiters(new Error("link closed")); });
    socket.on("error", (e) => { this.closed = true; this.flushWaiters(e); });
  }

  onData(d) {
    this.buf = Buffer.concat([this.buf, d]);
    while (this.buf.length >= 4) {
      const n = this.buf.readUInt32BE(0);
      if (n > MAX_FRAME) { this.socket.destroy(new Error("frame too large")); return; }
      if (this.buf.length < 4 + n) break;
      const body = new Uint8Array(this.buf.subarray(4, 4 + n));
      this.buf = this.buf.subarray(4 + n);
      this.bytesIn += 4 + n;
      const w = this.waiters.shift();
      if (w) w.resolve(body); else this.queue.push(body);
    }
  }

  flushWaiters(err) { for (const w of this.waiters.splice(0)) w.reject(err); }

  send(body) {
    const h = Buffer.alloc(4);
    h.writeUInt32BE(body.length);
    this.bytesOut += 4 + body.length;
    return new Promise((resolve, reject) => this.socket.write(Buffer.concat([h, Buffer.from(body)]), (e) => (e ? reject(e) : resolve())));
  }

  /** Next frame body; timeoutMs = 0 waits without a timer (until a frame or the link closes). */
  recv(timeoutMs = 120_000) {
    const q = this.queue.shift();
    if (q) return Promise.resolve(q);
    if (this.closed) return Promise.reject(new Error("link closed"));
    return new Promise((resolve, reject) => {
      const w = { resolve: (b) => { clearTimeout(t); resolve(b); }, reject: (e) => { clearTimeout(t); reject(e); } };
      const t = timeoutMs && setTimeout(() => { this.waiters.splice(this.waiters.indexOf(w), 1); reject(new Error("recv timeout")); }, timeoutMs);
      this.waiters.push(w);
    });
  }

  close() { this.socket.end(); }

  destroy() { this.socket.destroy(); }
}

export function connect(port, host = "127.0.0.1") {
  return new Promise((resolve, reject) => {
    const s = net.connect(port, host, () => resolve(new Link(s)));
    s.once("error", reject);
  });
}

/** Listens and hands every accepted connection to onLink(link). Resolves with the server once listening. */
export function listen(port, onLink, host = "127.0.0.1") {
  return new Promise((resolve, reject) => {
    const server = net.createServer((s) => onLink(new Link(s)));
    server.once("error", reject);
    server.listen(port, host, () => resolve(server));
  });
}
