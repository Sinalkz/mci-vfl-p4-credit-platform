// scripts/native-prover.mjs: the peak memory read from /usr/bin/time's report on macOS and on GNU/Linux, and the flag
// that asks /usr/bin/time for that report on each.
// Usage (from circuits/):  node --test test/native-prover.test.mjs
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { peakRssMBFromTime, timeFlag } from "../scripts/native-prover.mjs";

describe("peakRssMBFromTime", () => {
  it("reads macOS /usr/bin/time -l (bytes)", () => {
    const mac = "        2.64 real         9.10 user         0.88 sys\n           771751936  maximum resident set size\n";
    assert.equal(peakRssMBFromTime(mac), 771751936 / 2 ** 20);
  });
  it("reads GNU /usr/bin/time -v (kbytes)", () => {
    assert.equal(peakRssMBFromTime("\tMaximum resident set size (kbytes): 753664\n"), 736);
  });
  it("returns null without a report", () => assert.equal(peakRssMBFromTime("prover failed"), null));
});

describe("timeFlag", () => {
  it("is -l on macOS (BSD time: bytes)", () => assert.equal(timeFlag("darwin"), "-l"));
  it("is -v on Linux (GNU time: kbytes)", () => assert.equal(timeFlag("linux"), "-v"));
});
