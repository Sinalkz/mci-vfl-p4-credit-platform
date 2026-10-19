// The scoring prover's defaults (scripts/prover-config.mjs): the multi-party key and rapidsnark, with the development key
// and snarkjs selectable; a missing rapidsnark binary fails at once with a message naming --prover snarkjs and RAPIDSNARK,
// in m3-prove-figure.mjs and in contracts/offchain/runner3.mjs (before the runner touches its output folder or the
// chain); contracts/offchain/proposal.mjs's default vkHash is the multi-party key's. full-data: the key folder and the
// circuit follow maxDepth (14: build/members-var-N64-D14-mpc), and runner3's --run full checks its options before
// anything runs. Nothing is proved.
// Usage (from circuits/):  node --test test/prover-defaults.test.mjs
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { after, describe, it } from "node:test";

import { BUILD, CIRCUITS, ROOT } from "../scripts/lib.mjs";
import { DEFAULT_KEY_DIR, DEFAULT_PROVER, KEY_DIRS, checkKeyDir, mpcKeyDir, proverFromEnv } from "../scripts/prover-config.mjs";

const TMP = mkdtempSync(resolve(tmpdir(), "prover-defaults-"));
after(() => rmSync(TMP, { recursive: true, force: true }));
/** The environment without any of the prover's variables. */
const cleanEnv = (extra = {}) => {
  const env = { ...process.env, ...extra };
  for (const k of ["RAPIDSNARK", "VFL_PROVER", "VFL_KEY_DIR", "VFL_VK_FILE"]) if (!(k in extra)) delete env[k];
  return env;
};
const NOT_FOUND = /rapidsnark prover not found.*--prover snarkjs.*RAPIDSNARK/s;

describe("prover-config defaults", () => {
  it("default: the multi-party key and rapidsnark", () => {
    assert.equal(DEFAULT_PROVER, "rapidsnark");
    assert.equal(DEFAULT_KEY_DIR, resolve(BUILD, "members-var-N64-D10-mpc"));
    assert.equal(KEY_DIRS.dev, resolve(BUILD, "members-var-N64-D10"));
    const exe = process.execPath; // any existing file stands in for the binary
    assert.deepEqual(proverFromEnv({ RAPIDSNARK: exe }), { keyDir: KEY_DIRS.mpc, prover: "rapidsnark", rapidsnark: resolve(exe) });
  });

  it("rapidsnark unset or missing: fails, naming --prover snarkjs and RAPIDSNARK", () => {
    assert.throws(() => proverFromEnv({}), NOT_FOUND);
    assert.throws(() => proverFromEnv({ RAPIDSNARK: resolve(TMP, "no-such-prover") }), /no such file/);
  });

  it("snarkjs and the development key stay selectable", () => {
    assert.deepEqual(proverFromEnv({ VFL_PROVER: "snarkjs" }), { keyDir: KEY_DIRS.mpc, prover: "snarkjs", rapidsnark: null });
    assert.equal(proverFromEnv({ VFL_PROVER: "snarkjs", VFL_KEY_DIR: KEY_DIRS.dev }).keyDir, KEY_DIRS.dev);
    assert.throws(() => proverFromEnv({ VFL_PROVER: "groth16" }), /snarkjs \| rapidsnark/);
  });

  it("full-data: without VFL_KEY_DIR the key folder follows the circuit's maxDepth", () => {
    assert.equal(KEY_DIRS["mpc-d14"], resolve(BUILD, "members-var-N64-D14-mpc"));
    assert.equal(mpcKeyDir(10), KEY_DIRS.mpc);
    assert.equal(proverFromEnv({ VFL_PROVER: "snarkjs" }, { maxDepth: 14 }).keyDir, KEY_DIRS["mpc-d14"]);
    assert.equal(proverFromEnv({ VFL_PROVER: "snarkjs", VFL_KEY_DIR: KEY_DIRS.dev }, { maxDepth: 14 }).keyDir, KEY_DIRS.dev);
  });

  it("a key folder without the key fails", () => {
    assert.throws(() => checkKeyDir(TMP), /lacks main_final\.zkey, verification_key\.json/);
  });

  it("this machine's default key folder holds the multi-party key of results/mpc/setup", () => {
    assert.ok(existsSync(resolve(DEFAULT_KEY_DIR, "main_final.zkey")), `${DEFAULT_KEY_DIR}/main_final.zkey (copy it from the ceremony's machine)`);
    assert.deepEqual(JSON.parse(readFileSync(resolve(DEFAULT_KEY_DIR, "verification_key.json"), "utf8")),
      JSON.parse(readFileSync(resolve(ROOT, "results", "mpc", "setup", "verification_key.json"), "utf8")));
  });
});

describe("the defaults where they are used", () => {
  it("m3-prove-figure.mjs without RAPIDSNARK fails before reading its input", () => {
    const r = spawnSync(process.execPath, [resolve(CIRCUITS, "scripts", "m3-prove-figure.mjs"), resolve(TMP, "absent.json"), resolve(TMP, "out.json")],
      { env: cleanEnv(), encoding: "utf8" });
    assert.notEqual(r.status, 0);
    assert.match(r.stderr, NOT_FOUND);
  });

  it("m3-prove-figure.mjs takes the circuit from the input's Merkle depth (none built for maxDepth 7)", () => {
    const f = resolve(TMP, "d7.json");
    writeFileSync(f, JSON.stringify({ sibA: [Array(7).fill("0")] }));
    const r = spawnSync(process.execPath, [resolve(CIRCUITS, "scripts", "m3-prove-figure.mjs"), f, resolve(TMP, "out7.json")],
      { env: cleanEnv({ VFL_PROVER: "snarkjs" }), encoding: "utf8" });
    assert.notEqual(r.status, 0);
    assert.match(r.stderr, /no circuit for maxDepth 7/);
  });

  it("runner3.mjs --run full checks its options before touching its output folder or the chain", () => {
    const out = resolve(TMP, "full");
    mkdirSync(out, { recursive: true });
    writeFileSync(resolve(out, "keep"), "x");
    const r3 = (...a) => spawnSync(process.execPath, [resolve(ROOT, "contracts", "offchain", "runner3.mjs"), "--run", "full", ...a],
      { cwd: resolve(ROOT, "contracts"), env: cleanEnv({ RAPIDSNARK: process.execPath, VFL_MODEL_CONFIG: "x" }), encoding: "utf8", timeout: 60_000 });
    assert.match(r3().stderr, /--run full needs --out-dir/);
    assert.match(r3("--out-dir", out, "--setup", "mpc").stderr, /--run full needs --setup mpc-d14/);
    assert.match(r3("--out-dir", out, "--run-seed", "one").stderr, /--run-seed one: a non-negative integer/);
    assert.match(r3("--out-dir", out, "--alignment-period", "1.5").stderr, /--alignment-period 1.5: a non-negative integer/);
    assert.ok(existsSync(resolve(out, "keep")), "the output folder was not cleared");
  });

  it("runner3.mjs without RAPIDSNARK fails before touching its output folder or the chain", () => {
    const out = resolve(TMP, "run");
    mkdirSync(out, { recursive: true });
    writeFileSync(resolve(out, "keep"), "x");
    const r = spawnSync(process.execPath, [resolve(ROOT, "contracts", "offchain", "runner3.mjs"), "--run", "i6", "--out-dir", out],
      { cwd: resolve(ROOT, "contracts"), env: cleanEnv(), encoding: "utf8", timeout: 60_000 });
    assert.notEqual(r.status, 0);
    assert.match(r.stderr, NOT_FOUND);
    assert.ok(existsSync(resolve(out, "keep")), "the output folder was not cleared");
    const bad = spawnSync(process.execPath, [resolve(ROOT, "contracts", "offchain", "runner3.mjs"), "--setup", "x", "--out-dir", out],
      { cwd: resolve(ROOT, "contracts"), env: cleanEnv(), encoding: "utf8", timeout: 60_000 });
    assert.match(bad.stderr, /--setup dev\|mpc/);
  });

  it("proposal.mjs: the default vkHash is the multi-party key's", async () => {
    const vk = resolve(ROOT, "results", "mpc", "setup", "verification_key.json");
    const script = `const { defaultProposal } = await import(${JSON.stringify(pathToFileURL(resolve(ROOT, "contracts", "offchain", "proposal.mjs")).href)});
const { keccak256 } = await import("ethers"); const { readFileSync } = await import("node:fs");
const p = defaultProposal({ activeCustomers: 1000 });
console.log(JSON.stringify({ got: p.vkHash, want: keccak256(readFileSync(${JSON.stringify(vk)})) }));`;
    const r = spawnSync(process.execPath, ["--input-type=module", "-e", script], { cwd: resolve(ROOT, "contracts"), env: cleanEnv(), encoding: "utf8" });
    assert.equal(r.status, 0, r.stderr);
    const { got, want } = JSON.parse(r.stdout.trim().split("\n").pop());
    assert.equal(got, want);
  });
});
