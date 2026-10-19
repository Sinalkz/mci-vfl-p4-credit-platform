#!/usr/bin/env node
// check-network: the validator-floor check each member runs before starting its node.
// Plain Node (18+), no dependencies.
//
// Usage:
//   node chain/scripts/check-network.mjs --genesis chain/data/genesis.json \
//     --expected-hash <sha256 hex of the genesis file> --node-address 0x<own validator address> \
//     --rpc http://127.0.0.1:8546 [--only 1,2 | --only 3] [--json]
//
// The first member to start (no network to query yet) runs `--only 1,2` before
// starting its node and `--only 3` right after (design answer to QUESTIONS-M1 Q4);
// every later member runs all three checks before starting its node.
//
// Checks (the selected ones are all run; the exit code is 0 only if every selected check passes):
//   1. SHA-256 of the genesis file's exact bytes equals --expected-hash.
//   2. The QBFT validator list in the genesis extraData has at least 4 distinct
//      validators and includes --node-address.
//   3. On the running network (--rpc, any member's node),
//      qbft_getValidatorsByBlockNumber("latest") returns at least 4 distinct validators.
// Exit codes: 0 all checks pass, 1 a check failed, 2 usage error.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const MIN_VALIDATORS = 4;

function parseArgs(argv) {
  const out = { json: false, only: "1,2,3" };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--json") out.json = true;
    else if (a.startsWith("--")) out[a.slice(2)] = argv[++i];
    else throw new Error(`unexpected argument ${a}`);
  }
  const sel = new Set(String(out.only).split(",").map((x) => x.trim()));
  if (sel.size === 0 || [...sel].some((x) => !["1", "2", "3"].includes(x))) throw new Error(`bad --only ${out.only}`);
  out.checks = sel;
  const need = [...(sel.has("1") || sel.has("2") ? ["genesis", "expected-hash", "node-address"] : []), ...(sel.has("3") ? ["rpc"] : [])];
  for (const k of need) if (!out[k]) throw new Error(`missing --${k}`);
  return out;
}

// Minimal RLP decoder: returns Uint8Array for strings, arrays for lists.
function rlpDecode(buf, pos = 0) {
  const be = (at, len) => { let v = 0; for (let i = 0; i < len; i++) v = v * 256 + buf[at + i]; return v; };
  const b = buf[pos];
  if (b === undefined) throw new Error("RLP: truncated");
  if (b < 0x80) return { item: buf.subarray(pos, pos + 1), end: pos + 1 };
  if (b <= 0xb7) return { item: buf.subarray(pos + 1, pos + 1 + b - 0x80), end: pos + 1 + b - 0x80 };
  if (b <= 0xbf) {
    const ll = b - 0xb7, len = be(pos + 1, ll), start = pos + 1 + ll;
    return { item: buf.subarray(start, start + len), end: start + len };
  }
  let start, len;
  if (b <= 0xf7) { len = b - 0xc0; start = pos + 1; } else { const ll = b - 0xf7; len = be(pos + 1, ll); start = pos + 1 + ll; }
  const items = [];
  let p = start;
  while (p < start + len) { const r = rlpDecode(buf, p); items.push(r.item); p = r.end; }
  if (p !== start + len) throw new Error("RLP: list length mismatch");
  return { item: items, end: p };
}

const hex = (u8) => "0x" + Buffer.from(u8).toString("hex");
const norm = (a) => String(a).toLowerCase();

/** QBFT extraData = RLP([vanity(32), [validators], vote, round, [seals]]). */
export function qbftValidators(extraData) {
  const buf = Buffer.from(extraData.replace(/^0x/, ""), "hex");
  const { item, end } = rlpDecode(buf);
  if (end !== buf.length || !Array.isArray(item) || item.length < 2 || !Array.isArray(item[1])) {
    throw new Error("extraData is not a QBFT RLP list");
  }
  return item[1].map((v) => {
    if (!(v instanceof Uint8Array) || v.length !== 20) throw new Error("validator entry is not a 20-byte address");
    return hex(v);
  });
}

async function liveValidators(url) {
  const res = await fetch(url, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "qbft_getValidatorsByBlockNumber", params: ["latest"] }),
    signal: AbortSignal.timeout(10_000),
  });
  const j = await res.json();
  if (j.error) throw new Error(j.error.message);
  return j.result;
}

async function main() {
  let args;
  try { args = parseArgs(process.argv.slice(2)); } catch (e) {
    console.error(`usage error: ${e.message}`);
    console.error("usage: check-network --genesis <file> --expected-hash <sha256> --node-address <0x..> --rpc <url> [--only 1,2|3] [--json]");
    return 2;
  }
  const checks = [];
  const record = (id, name, pass, detail) => checks.push({ id, name, pass, detail });

  // 1. Genesis file hash (SHA-256 over the exact bytes; confirmed, Q5).
  let bytes = null;
  if (args.checks.has("1") || args.checks.has("2")) try {
    bytes = readFileSync(args.genesis);
    const got = createHash("sha256").update(bytes).digest("hex");
    const want = norm(args["expected-hash"]).replace(/^0x/, "");
    if (args.checks.has("1")) record(1, "genesis hash", got === want, got === want ? `sha256 ${got}` : `sha256 ${got}, expected ${want}`);
  } catch (e) {
    if (args.checks.has("1")) record(1, "genesis hash", false, e.message);
  }

  // 2. Validators in the genesis extraData.
  if (args.checks.has("2")) try {
    if (!bytes) throw new Error("genesis file not readable");
    const extra = JSON.parse(bytes.toString("utf8")).extraData;
    if (!extra) throw new Error("genesis has no extraData");
    const vals = qbftValidators(extra).map(norm);
    const distinct = new Set(vals);
    const own = norm(args["node-address"]);
    const enough = distinct.size >= MIN_VALIDATORS;
    const hasOwn = distinct.has(own);
    record(2, "genesis validators", enough && hasOwn,
      `${distinct.size} distinct of ${vals.length} listed (need >= ${MIN_VALIDATORS}); own node ${own} ${hasOwn ? "included" : "NOT included"}`);
  } catch (e) {
    record(2, "genesis validators", false, e.message);
  }

  // 3. Validators on the running network.
  if (args.checks.has("3")) try {
    const vals = (await liveValidators(args.rpc)).map(norm);
    const distinct = new Set(vals).size;
    record(3, "live validators", distinct >= MIN_VALIDATORS,
      `${distinct} distinct at "latest" via ${args.rpc} (need >= ${MIN_VALIDATORS})`);
  } catch (e) {
    record(3, "live validators", false, `${args.rpc}: ${e.message}`);
  }

  const ok = checks.every((c) => c.pass);
  if (args.json) {
    console.log(JSON.stringify({ ok, only: [...args.checks], checks }));
  } else {
    for (const c of checks) console.log(`check ${c.id} ${c.name.padEnd(18)} ${c.pass ? "PASS" : "FAIL"}  ${c.detail}`);
    console.log(ok ? "network check passed" : "network check FAILED");
  }
  return ok ? 0 : 1;
}

process.exitCode = await main();
