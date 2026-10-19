// DATA1: loads data/recs/prepared/ (written by data/recs/prep.py) into the row format the M3 code
// uses: rows = { xa[k] (8 int16 at 2^12), y[k] } and xp[i][k] (passive i's encoded row), for every
// position k of I = the cleaned households in DOEID order. Rows are plain arrays: training.mjs uses
// flatMap and flat(), which do not flatten typed arrays. The binaries are checked against the SHA-256
// in the manifest.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const PREPARED = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "..", "data", "recs", "prepared");
export const PASSIVE_NAMES = ["telecom", "retailer", "device"];

export const loadManifest = () => JSON.parse(readFileSync(resolve(PREPARED, "manifest.json"), "utf8"));

function readChecked(file, sha256) {
  const buf = readFileSync(resolve(PREPARED, file));
  const got = createHash("sha256").update(buf).digest("hex");
  if (got !== sha256) throw new Error(`${file}: SHA-256 ${got} != manifest ${sha256}; rerun data/recs/prep.py`);
  return buf;
}

/** The manifest and the two checked binaries as typed arrays: { m, A (n x 8 int16), P (n x W float32), W }. */
export function openPrepared() {
  const m = loadManifest();
  const a = readChecked(m.a.file, m.a.sha256);
  const A = new Int16Array(a.buffer, a.byteOffset, a.byteLength / 2);
  const p = readChecked(m.passive.file, m.passive.sha256);
  const W = m.passive.width;
  const P = new Float32Array(p.buffer, p.byteOffset, p.byteLength / 4);
  if (A.length !== m.rows * 8 || P.length !== m.rows * W) throw new Error("prepared/ binaries do not match the manifest shape");
  return { m, A, P, W };
}

/** partition: optional { telecom: [colIdx], retailer: [...], device: [...] } (E2); default manifest.parties. */
export function loadPrepared({ partition = null } = {}) {
  const { m, A, P, W } = openPrepared();
  const n = m.rows;
  const parts = partition ?? m.parties;
  const cols = PASSIVE_NAMES.map((name) => parts[name]);
  const rows = { xa: Array.from({ length: n }, (_, k) => Array.from(A.subarray(k * 8, k * 8 + 8))), y: m.y.slice() };
  const xp = cols.map((c) => Array.from({ length: n }, (_, k) => c.map((j) => P[k * W + j])));
  return { rows, xp, trainPos: m.split.train_pos, testPos: m.split.test_pos, widths: cols.map((c) => c.length), manifest: m };
}
