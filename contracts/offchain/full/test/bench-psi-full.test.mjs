// The full-data alignment benchmark (offchain/full/bench-psi-full.mjs) on a cut of the full-data sets (--limit): four
// processes over TCP, the PSI finds what A and every passive share, every passive receives I, and the declared alignment
// period follows the rule (twice the slowest repetition, in 2 s blocks, multiple of 10, at least 60).
// Usage (from contracts/):  node --test offchain/full/test/*.test.mjs   (spawns 4 processes, each building the scenario)
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { after, describe, it } from "node:test";

const HERE = fileURLToPath(import.meta.url);
const TMP = mkdtempSync(resolve(tmpdir(), "bench-psi-full-"));
after(() => rmSync(TMP, { recursive: true, force: true }));

describe("bench-psi-full", () => {
  it("runs the 3-pair PSI on a cut of the full-data sets and declares the alignment period by its rule", () => {
    const out = resolve(TMP, "psi.json");
    const r = spawnSync(process.execPath, [resolve(HERE, "..", "..", "bench-psi-full.mjs"), "--limit", "300", "--reps", "1", "--out", out],
      { encoding: "utf8", timeout: 300_000 });
    assert.equal(r.status, 0, r.stderr);
    const b = JSON.parse(readFileSync(out, "utf8"));
    const [rep] = b.reps;
    assert.equal(rep.correct, true);
    assert.equal(rep.A.aSetSize, 300);
    assert.ok(rep.A.IEqualsScenario && rep.A.sizeI > 0 && rep.A.sizeI <= 270);
    assert.deepEqual(rep.passives.map((p) => p.received), [rep.A.sizeI, rep.A.sizeI, rep.A.sizeI]);
    assert.equal(rep.A.pairs.length, 3);
    const want = Math.max(60, Math.ceil(Math.ceil((2 * b.slowestAlignmentS) / 2) / 10) * 10);
    assert.equal(b.alignmentPeriod.blocks, want);
    assert.ok(b.alignmentPeriod.seconds >= 2 * b.slowestAlignmentS);
  });
});
