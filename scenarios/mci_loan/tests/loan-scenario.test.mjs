// MCI loan instance, Step 5: the on-chain scenario (scenarios/mci_loan/onchain/loan-scenario.mjs) and the alignment
// benchmark on it (contracts/offchain/full/bench-psi-full.mjs --scenario mci-loan, on a cut of the sets: --limit).
// Needs data/mci_loan/prepared/ (scenarios/mci_loan/prep.py), or the folder MCI_LOAN_PREPARED names (v2: data/mci_loan/v2/prepared).
// Usage (from the repo root):  node --test scenarios/mci_loan/tests/*.test.mjs
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { after, describe, it } from "node:test";

import { LOAN_CLASS_BOUNDARIES, LOAN_SEED, LOAN_SIZES, loanScenario } from "../onchain/loan-scenario.mjs";
import { loadManifest, passiveNames } from "../offchain/prepared.mjs";

const HERE = fileURLToPath(import.meta.url);
const ROOT = resolve(HERE, "..", "..", "..", "..");
const TMP = mkdtempSync(resolve(tmpdir(), "loan-scenario-"));
after(() => rmSync(TMP, { recursive: true, force: true }));

describe("loan scenario", () => {
  const m = loadManifest();
  const sc = loanScenario();
  it("gives the Bank every cohort applicant and each passive every one of them plus its extras", () => {
    assert.equal(sc.A.ids.length, m.rows);
    assert.equal(sc.I.length, m.rows);
    assert.ok(sc.A.ids.length <= 2 ** 14);
    for (const p of sc.passives) assert.equal(p.ids.length, m.rows + LOAN_SIZES.extra);
    assert.deepEqual(sc.pairwise.map((x) => x.length), [m.rows, m.rows, m.rows]);
    const aSet = new Set(sc.A.ids);
    assert.ok(sc.passives.every((p) => p.ids.filter((x) => !aSet.has(x)).length === LOAN_SIZES.extra));
  });
  it("carries each member's columns, the binary label and the label's cut points", () => {
    sc.passives.forEach((p, i) => assert.equal(p.rows.get(p.ids[0]).length, sc.sample.widths[i]));
    assert.deepEqual(sc.sample.widths, passiveNames(m).map((n) => m.parties[n].length)); // v1's order, or v2's passiveOrder
    assert.deepEqual([...new Set([...sc.A.rows.values()].map((r) => r.y))].sort(), [0, 1]);
    assert.deepEqual(sc.classBoundaries, LOAN_CLASS_BOUNDARIES);
    assert.equal(sc.sample.scenario, "mci-loan");
    assert.equal(sc.sample.seed, LOAN_SEED);
  });
  it("draws the same IDs again from the same seed, and other IDs from another", () => {
    const again = loanScenario();
    assert.deepEqual(again.I, sc.I);
    assert.notDeepEqual(loanScenario({ seed: LOAN_SEED + 1 }).I, sc.I);
  });
});

describe("bench-psi-full --scenario mci-loan", () => {
  it("runs the 3-pair PSI on a cut of the loan sets and declares the alignment period by its rule", () => {
    const out = resolve(TMP, "psi.json");
    const r = spawnSync(process.execPath, [resolve(ROOT, "contracts", "offchain", "full", "bench-psi-full.mjs"), "--scenario", "mci-loan",
      "--limit", "300", "--reps", "1", "--out", out], { encoding: "utf8", timeout: 300_000 });
    assert.equal(r.status, 0, r.stderr);
    const b = JSON.parse(readFileSync(out, "utf8"));
    const [rep] = b.reps;
    assert.equal(b.scenario.name, "mci-loan");
    assert.equal(rep.correct, true);
    assert.equal(rep.A.aSetSize, 300);
    // each cut passive holds its own 270 of the Bank's first 300 (bench-psi-full.mjs sets()), so |I| <= 270
    assert.ok(rep.A.IEqualsScenario && rep.A.sizeI > 0 && rep.A.sizeI <= 270);
    assert.deepEqual(rep.passives.map((p) => p.received), [rep.A.sizeI, rep.A.sizeI, rep.A.sizeI]);
    const want = Math.max(60, Math.ceil(Math.ceil((2 * b.slowestAlignmentS) / 2) / 10) * 10);
    assert.equal(b.alignmentPeriod.blocks, want);
  });
});
