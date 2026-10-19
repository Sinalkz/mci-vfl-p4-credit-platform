// MCI loan instance: loads data/mci_loan/prepared/ (written by scenarios/mci_loan/prep.py) with the interface of the
// RECS loader, contracts/offchain/data1/prepared.mjs (decision (f); the RECS loader and its path are not touched):
// rows = { xa[k] (8 int16 at 2^12), y[k] } and xp[i][k] (passive i's encoded row) for every position k of the cohort,
// in SK_ID_CURR order. MCI_LOAN_PREPARED names another folder of the same format (the smoke sample:
// data/mci_loan/prepared-smoke). The binaries are checked against the SHA-256 in the manifest.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "..");
export const PREPARED = process.env.MCI_LOAN_PREPARED ? resolve(process.env.MCI_LOAN_PREPARED)
  : resolve(ROOT, "data", "mci_loan", "prepared");
export const PASSIVE_NAMES = ["mci", "digikala", "bureau"]; // proposal order: slots mci, digikala, maker (slots.json)
/** The manifest's passive keys in proposal order: v2 manifests name them (passiveOrder: bank, ecommerce, bureau; v2/slots.json). */
export const passiveNames = (m) => m.passiveOrder ?? PASSIVE_NAMES;

export const loadManifest = () => JSON.parse(readFileSync(resolve(PREPARED, "manifest.json"), "utf8"));

function readChecked(file, sha256) {
  const buf = readFileSync(resolve(PREPARED, file));
  const got = createHash("sha256").update(buf).digest("hex");
  if (got !== sha256) throw new Error(`${file}: SHA-256 ${got} != manifest ${sha256}; rerun scenarios/mci_loan/prep.py`);
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
  if (A.length !== m.rows * 8 || P.length !== m.rows * W) throw new Error("prepared binaries do not match the manifest shape");
  return { m, A, P, W };
}

/** partition: optional { mci: [colIdx], digikala: [...], bureau: [...] } (the sensitivity variant); default manifest.parties. */
export function loadPrepared({ partition = null } = {}) {
  const { m, A, P, W } = openPrepared();
  const n = m.rows;
  const parts = partition ?? m.parties;
  const cols = passiveNames(m).map((name) => parts[name]);
  const rows = { xa: Array.from({ length: n }, (_, k) => Array.from(A.subarray(k * 8, k * 8 + 8))), y: m.y.slice() };
  const xp = cols.map((c) => Array.from({ length: n }, (_, k) => c.map((j) => P[k * W + j])));
  return { rows, xp, trainPos: m.split.train_pos, testPos: m.split.test_pos, widths: cols.map((c) => c.length), manifest: m };
}
