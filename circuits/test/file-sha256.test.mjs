// scripts/m1b-lib.mjs fileSha256: m1b-build.mjs's r1cs digest, read in chunks. Found on the server (Task 7): the N 512
// r1cs is 2.64 GB and readFileSync refuses files over 2 GiB (ERR_FS_FILE_TOO_LARGE) after circom had finished.
// Usage (from circuits/):  node --test test/file-sha256.test.mjs
import assert from "node:assert/strict";
import { createHash, randomBytes } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, describe, it } from "node:test";

import { fileSha256 } from "../scripts/m1b-lib.mjs";

const dir = mkdtempSync(join(tmpdir(), "sha256-"));
after(() => rmSync(dir, { recursive: true, force: true }));
const whole = (b) => createHash("sha256").update(b).digest("hex");

describe("fileSha256", () => {
  it("equals the digest of the whole file, across chunk boundaries", () => {
    const b = randomBytes(3000);
    const p = join(dir, "a.bin");
    writeFileSync(p, b);
    for (const chunk of [1, 7, 1024, 3000, 4096]) assert.equal(fileSha256(p, chunk), whole(b), `chunk ${chunk}`);
    assert.equal(fileSha256(p), whole(b));
  });
  it("hashes an empty file", () => {
    const p = join(dir, "empty.bin");
    writeFileSync(p, Buffer.alloc(0));
    assert.equal(fileSha256(p), whole(Buffer.alloc(0)));
  });
});
