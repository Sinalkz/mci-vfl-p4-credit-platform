// M2 runner: minimal chain client for a party process (plain Node + ethers, no Hardhat).
// Each process signs with its own account through its own node, tracks its nonce locally,
// pays gas price 0 (the S1 chain runs zeroBaseFee with --min-gas-price=0), and records
// every transaction (step, hash, block, gas, wall-clock).
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Contract, ContractFactory, HDNodeWallet, JsonRpcProvider, Mnemonic } from "ethers";

const CONTRACTS = resolve(fileURLToPath(import.meta.url), "..", "..");
const MNEMONIC = "test test test test test test test test test test test junk"; // Hardhat dev keys, local chain only

export function artifact(name) {
  const sub = name === "MockERC20" ? "test/" : name.startsWith("Members") ? "verifiers/" : "";
  return JSON.parse(readFileSync(resolve(CONTRACTS, "artifacts", "contracts", `${sub}${name}.sol`, `${name}.json`), "utf8"));
}

export const devWallet = (i) => HDNodeWallet.fromMnemonic(Mnemonic.fromPhrase(MNEMONIC), `m/44'/60'/0'/0/${i}`);

/**
 * M4 (E3 on the Mac): which chain a run ran on. Every node's client version (the Besu version) and genesis block
 * hash, and the SHA-256 of the genesis file's bytes. `nodes` are providers (anything with send); throws if they disagree.
 */
export async function chainIdentity(nodes, genesisFile) {
  const each = await Promise.all(nodes.map(async (p) => ({ clientVersion: await p.send("web3_clientVersion", []),
    genesisBlockHash: (await p.send("eth_getBlockByNumber", ["0x0", false])).hash })));
  for (const k of ["clientVersion", "genesisBlockHash"]) if (new Set(each.map((n) => n[k])).size !== 1) throw new Error(`nodes disagree on ${k}`);
  return { clientVersion: each[0].clientVersion, genesisBlockHash: each[0].genesisBlockHash,
    genesisFileSha256: createHash("sha256").update(readFileSync(genesisFile)).digest("hex"), nodes: each };
}

export class ChainClient {
  constructor(rpcUrl, accountIndex, label) {
    this.provider = new JsonRpcProvider(rpcUrl, undefined, { staticNetwork: true });
    this.provider.pollingInterval = 250;
    this.wallet = devWallet(accountIndex).connect(this.provider);
    this.address = this.wallet.address;
    this.label = label;
    this.nonce = null;
    this.txs = [];
  }

  contract(name, address) { return new Contract(address, artifact(name).abi, this.wallet); }

  async waitReceipt(hash) {
    for (let i = 0; i < 2400; i++) {
      const r = await this.provider.getTransactionReceipt(hash);
      if (r) return r;
      await new Promise((ok) => setTimeout(ok, 250));
    }
    throw new Error(`no receipt for ${hash}`);
  }

  /**
   * Sends one transaction and waits for its receipt. Gas limit: estimate + 100k (state-dependent branches, see M1), at
   * least opts.minGas (M4: an acknowledge that may settle, see party-m3's ACK_GAS_FLOOR).
   */
  async send(step, to, data, { minGas = 0 } = {}) {
    this.nonce ??= await this.provider.getTransactionCount(this.address, "latest");
    const t0 = Date.now();
    const req = { to, data, gasPrice: 0n, type: 0, nonce: this.nonce, chainId: (await this.provider.getNetwork()).chainId };
    const est = (await this.provider.estimateGas({ ...req, from: this.address })) + 100_000n;
    req.gasLimit = est > BigInt(minGas) ? est : BigInt(minGas);
    const tx = await this.wallet.sendTransaction(req);
    this.nonce++;
    const rc = await this.waitReceipt(tx.hash);
    const rec = { step, from: this.label, hash: tx.hash, block: rc.blockNumber, gasUsed: Number(rc.gasUsed), status: rc.status,
      wallMs: Date.now() - t0 };
    this.txs.push(rec);
    if (rc.status !== 1) throw new Error(`${step} reverted (${tx.hash})`);
    return rc;
  }

  call(step, c, method, args = [], opts = {}) { return this.send(step, c.target, c.interface.encodeFunctionData(method, args), opts); }

  async deploy(step, name, args = []) {
    const art = artifact(name);
    const { data } = await new ContractFactory(art.abi, art.bytecode).getDeployTransaction(...args);
    const rc = await this.send(step, null, data);
    return this.contract(name, rc.contractAddress);
  }

  /**
   * Polls until pred() is true (reads through this process's own node). A failed read (e.g.
   * Besu's JSON-RPC "Internal error" under heavy load, seen in the first M3 I6 run) is retried;
   * 20 failures in a row end the wait. Failures are counted in rpcErrors.
   */
  async until(pred, what, timeoutMs = 600_000) {
    const t0 = Date.now();
    let failures = 0;
    for (;;) {
      try {
        if (await pred()) return;
        failures = 0;
      } catch (e) {
        this.rpcErrors = (this.rpcErrors ?? 0) + 1;
        if (++failures >= 20) throw e;
      }
      if (Date.now() - t0 > timeoutMs) throw new Error(`timeout waiting for ${what}`);
      await new Promise((ok) => setTimeout(ok, 500));
    }
  }
}
