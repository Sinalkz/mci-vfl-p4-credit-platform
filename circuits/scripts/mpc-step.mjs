// One step of the multi-party phase-2 ceremony for the deployed scoring circuit (mpc-ceremony.mjs runs each step in
// its own process, so the time and peak memory printed are this step's alone).
//   node scripts/mpc-step.mjs setup <main.r1cs> <phase1.ptau> <out.zkey>          (= snarkjs groth16 setup)
//   node scripts/mpc-step.mjs contribute <in.zkey> <out.zkey> <member>             (= snarkjs zkey contribute)
//   node scripts/mpc-step.mjs verify <main.r1cs> <phase1.ptau> <final.zkey>        (= snarkjs zkey verify)
//   node scripts/mpc-step.mjs export <final.zkey> <vkey.json> <Verifier.sol> <ContractName>
// contribute: the member's entropy is 64 bytes from the OS CSPRNG, generated inside this process and never written or
// printed (it is the member's toxic waste); only the contribution hash leaves. Prints one JSON line at the end.
import { randomBytes } from "node:crypto";
import { readFileSync, statSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import * as snarkjs from "snarkjs";

import { CIRCUITS } from "./lib.mjs";

const [step, ...a] = process.argv.slice(2);
const lines = [];
const logger = { debug: () => {}, info: (m) => lines.push(String(m)), warn: (m) => lines.push(`warn: ${m}`),
  error: (m) => lines.push(`error: ${m}`) };
const t0 = performance.now();
let out = { step };
if (step === "setup") {
  const [r1cs, ptau, zkey] = a;
  await snarkjs.zKey.newZKey(r1cs, ptau, zkey, logger);
  out = { ...out, output: zkey, outputBytes: statSync(zkey).size };
} else if (step === "contribute") {
  const [inZkey, outZkey, member] = a;
  const hash = await snarkjs.zKey.contribute(inZkey, outZkey, member, randomBytes(64).toString("hex"), logger);
  out = { ...out, member, output: outZkey, outputBytes: statSync(outZkey).size,
    contributionHash: Buffer.from(hash).toString("hex"), entropy: "64 bytes, OS CSPRNG, not stored" };
} else if (step === "verify") {
  const [r1cs, ptau, zkey] = a;
  const ok = await snarkjs.zKey.verifyFromR1cs(r1cs, ptau, zkey, logger);
  out = { ...out, verified: ok === true, transcript: lines.filter((l) => /contribution|ZKey Ok|Circuit hash|warn|error/i.test(l)) };
} else if (step === "export") {
  const [zkey, vkeyFile, solFile, name] = a;
  writeFileSync(vkeyFile, JSON.stringify(await snarkjs.zKey.exportVerificationKey(zkey), null, 1));
  const templates = { groth16: readFileSync(resolve(CIRCUITS, "node_modules", "snarkjs", "templates", "verifier_groth16.sol.ejs"), "utf8") };
  const sol = await snarkjs.zKey.exportSolidityVerifier(zkey, templates);
  writeFileSync(solFile, sol.replace(/contract Groth16Verifier/, `contract ${name}`));
  out = { ...out, verificationKey: vkeyFile, verifier: solFile, contract: name };
} else {
  throw new Error("step: setup | contribute | verify | export");
}
console.log(JSON.stringify({ ...out, ms: performance.now() - t0, peakRssMB: process.resourceUsage().maxRSS / 1024 }));
await globalThis.curve_bn128?.terminate();
process.exit(step === "verify" && !out.verified ? 1 : 0);
