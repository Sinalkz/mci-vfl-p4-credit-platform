// Review finding (proofcost branch): measurement runs of m1b-build.mjs must not alter a build that already has a
// proving key (the protocol proves from build/members-var-N64-D10). Uses its own tiny builds (N = 2).
// Usage (from circuits/):  node --test test/m1b-build-guards.test.mjs
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { after, describe, it } from "node:test";

import { CIRCUITS } from "../scripts/lib.mjs";
import { varBuildDir } from "../scripts/m1b-lib.mjs";

const build = (args) => spawnSync(process.execPath, [resolve(CIRCUITS, "scripts", "m1b-build.mjs"), ...args],
  { cwd: CIRCUITS, encoding: "utf8" });
const dirs = [varBuildDir(2, 2), varBuildDir(2, 3), varBuildDir(2, 4)];
after(() => dirs.forEach((d) => rmSync(d, { recursive: true, force: true })));

describe("m1b-build guards around an existing proving key", () => {
  it("a --no-verifier setup refuses to replace an existing main_final.zkey", () => {
    const dir = dirs[0];
    mkdirSync(dir, { recursive: true });
    writeFileSync(resolve(dir, "compile.json"), JSON.stringify({ r1csConstraints: 100, publicInputs: 10 }));
    writeFileSync(resolve(dir, "main_final.zkey"), "existing key");
    const r = build(["--n", "2", "--maxdepth", "2", "--setup-only", "--no-verifier"]);
    assert.notEqual(r.status, 0);
    assert.match(r.stderr, /refuses to replace the existing proving key/);
    assert.equal(readFileSync(resolve(dir, "main_final.zkey"), "utf8"), "existing key");
  });

  it("a recompile next to an existing key leaves the build untouched and fails if the r1cs would change", () => {
    const dir = dirs[1];
    mkdirSync(dir, { recursive: true });
    writeFileSync(resolve(dir, "main.r1cs"), "not the circuit's r1cs");
    writeFileSync(resolve(dir, "compile.json"), "{}");
    writeFileSync(resolve(dir, "main_final.zkey"), "existing key");
    const r = build(["--n", "2", "--maxdepth", "3", "--compile-only"]);
    assert.notEqual(r.status, 0);
    assert.match(r.stderr, /would change the r1cs of the existing proving key/);
    assert.equal(readFileSync(resolve(dir, "main.r1cs"), "utf8"), "not the circuit's r1cs");
    assert.equal(readFileSync(resolve(dir, "compile.json"), "utf8"), "{}");
  });

  it("a recompile that reproduces the r1cs byte for byte records that, and still leaves the build untouched", () => {
    const dir = dirs[2];
    const first = build(["--n", "2", "--maxdepth", "4", "--compile-only"]);
    assert.equal(first.status, 0, first.stderr);
    const r1cs = readFileSync(resolve(dir, "main.r1cs"));
    const compileJson = readFileSync(resolve(dir, "compile.json"), "utf8");
    writeFileSync(resolve(dir, "main_final.zkey"), "existing key");
    const again = build(["--n", "2", "--maxdepth", "4", "--compile-only"]);
    assert.equal(again.status, 0, again.stderr);
    const out = JSON.parse(again.stdout.trim().split("\n").filter((l) => l.startsWith("{")).pop());
    assert.equal(out.r1csIdenticalToKeyedBuild, true);
    assert.match(out.r1csSha256, /^[0-9a-f]{64}$/);
    assert.ok(readFileSync(resolve(dir, "main.r1cs")).equals(r1cs));
    assert.equal(readFileSync(resolve(dir, "compile.json"), "utf8"), compileJson);
  });
});
