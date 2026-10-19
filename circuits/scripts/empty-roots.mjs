// M4 (D128): the root of the all-zero-leaf Poseidon tree at depth 1..20 (z_0 = 0, z_{k+1} = Poseidon(z_k, z_k)).
// A passive who left is scored against this root: A opens it with rows of -32768 (they pack to leaf 0), and the
// passive's mask bit 0 keeps those rows out of F. The chain is checked against the scoring circuit's own tree code
// (members-lib buildTree) for depths 1..10.
// Prints the Solidity constant pasted into Federation.sol (EMPTY_ROOTS: 20 x 32 bytes, depth 1 first).
// Usage (from circuits/):  node scripts/empty-roots.mjs
import { buildPoseidon } from "circomlibjs";

import { buildTree } from "./members-lib.mjs";

export async function emptyRoots(maxDepthLimit = 20) {
  const h = await buildPoseidon();
  const out = [];
  let z = 0n;
  for (let d = 1; d <= maxDepthLimit; d++) {
    z = h.F.toObject(h([z, z]));
    out.push(z);
  }
  return out;
}

if (process.argv[1]?.replace(/\\/g, "/").endsWith("scripts/empty-roots.mjs")) {
  const roots = await emptyRoots();
  for (let d = 1; d <= 10; d++) {
    if ((await buildTree([0n], d)).root !== roots[d - 1]) throw new Error(`depth ${d}: chain != buildTree`);
  }
  roots.forEach((r, i) => console.log(`depth ${String(i + 1).padStart(2)}: 0x${r.toString(16).padStart(64, "0")}`));
  console.log(`\nbytes private constant EMPTY_ROOTS =\n  hex"${roots.map((r) => r.toString(16).padStart(64, "0")).join("")}";`);
  process.exit(0);
}
