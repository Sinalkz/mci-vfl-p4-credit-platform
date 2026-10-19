// Download a phase-1 powers-of-tau file (2^k) for the Groth16 setup.
// Usage (from circuits/): node scripts/fetch-ptau.mjs 19 [--source hermez|ppot]
//
// Sources, both public multi-party ceremonies prepared for snarkjs phase 2:
//   hermez: powersOfTau28_hez_final_<k>.ptau (Google Cloud Storage); the
//           blake2b-512 is checked against the table in the snarkjs README.
//   ppot:   PSE Perpetual Powers of Tau, 80 contributions, truncated to 2^k
//           (AWS S3). PSE publishes no per-file hash; the blake2b-512 of the
//           downloaded file is printed and saved so reruns can compare.
// Default: try hermez, fall back to ppot if the download is refused (the GCS
// bucket answers HTTP 403 from some regions).
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { parseArgs } from "node:util";

import { CIRCUITS } from "./lib.mjs";

const { values, positionals } = parseArgs({ allowPositionals: true, options: { source: { type: "string" } } });
const k = Number(positionals[0]);
const dir = resolve(CIRCUITS, "ptau");
mkdirSync(dir, { recursive: true });

function hermez() {
  const name = `powersOfTau28_hez_final_${k}.ptau`;
  const readme = readFileSync(resolve(CIRCUITS, "node_modules", "snarkjs", "README.md"), "utf8");
  const row = readme.split("\n").find((l) => l.includes(`[${name}]`));
  if (!row) throw new Error(`no entry for ${name} in snarkjs README`);
  return { name, url: row.match(/\((https:[^)]+)\)/)[1], expected: row.match(/\|\s*([0-9a-f]{128})\s*\|/)[1] };
}
function ppot() {
  const name = `ppot_0080_${String(k).padStart(2, "0")}.ptau`;
  return { name, url: `https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/${name}`, expected: null };
}

async function blake2b(p) {
  const h = createHash("blake2b512");
  await pipeline(createReadStream(p), h);
  return h.digest("hex");
}

async function fetchOne(src) {
  const dest = resolve(dir, src.name);
  if (!existsSync(dest)) {
    console.log(`downloading ${src.url}`);
    const res = await fetch(src.url);
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${src.url}`);
    await pipeline(Readable.fromWeb(res.body), createWriteStream(dest + ".part"));
    renameSync(dest + ".part", dest);
  }
  const got = await blake2b(dest);
  if (src.expected && got !== src.expected) throw new Error(`blake2b mismatch for ${src.name}`);
  writeFileSync(dest + ".blake2b", `${got}  ${src.name}\n`);
  console.log(`${src.name}: blake2b-512 ${got}` + (src.expected ? " (matches snarkjs README)" : " (no published hash)"));
  return dest;
}

const order = values.source ? [values.source] : ["hermez", "ppot"];
let lastErr;
for (const s of order) {
  try {
    await fetchOne(s === "hermez" ? hermez() : ppot());
    process.exit(0);
  } catch (e) {
    lastErr = e;
    console.log(`${s}: ${e.message}`);
  }
}
throw lastErr;
