// M4 (E3 on the Mac): the runner records which chain a run ran on: every node's client version and genesis block
// hash, and the SHA-256 of the genesis file. Two handles on the in-process chain stand in for two nodes.
import { expect } from "chai";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { network } from "hardhat";

// @ts-ignore -- plain ESM module
import { chainIdentity } from "../offchain/chain.mjs";

const { ethers } = await network.create();
// a committed file stands in for chain/data/genesis.json, which is generated and git-ignored
const FILE = new URL("../../chain/qbftConfigFile.json", import.meta.url);
const node = { send: (m: string, p: unknown[]) => ethers.provider.send(m, p) };

describe("chainIdentity (M4, the E3 run's chain record)", function () {
  it("C01 records each node's client version and genesis block hash, and the genesis file's SHA-256", async function () {
    const id = await chainIdentity([node, node], FILE);
    const block0 = await ethers.provider.send("eth_getBlockByNumber", ["0x0", false]);
    const version = await ethers.provider.send("web3_clientVersion", []);
    expect(id.genesisBlockHash).to.equal(block0.hash);
    expect(id.clientVersion).to.equal(version);
    expect(id.genesisFileSha256).to.equal(createHash("sha256").update(readFileSync(FILE)).digest("hex"));
    expect(id.nodes).to.deep.equal([{ clientVersion: version, genesisBlockHash: block0.hash }, { clientVersion: version, genesisBlockHash: block0.hash }]);
  });

  it("C02 throws when the nodes report different genesis blocks", async function () {
    const other = { send: async (m: string, p: unknown[]) => (m === "eth_getBlockByNumber" ? { ...(await node.send(m, p)), hash: "0x" + "11".repeat(32) } : node.send(m, p)) };
    let err: any;
    try { await chainIdentity([node, other], FILE); } catch (e) { err = e; }
    expect(String(err)).to.contain("nodes disagree on genesisBlockHash");
  });

  it("C03 throws when the nodes run different client versions", async function () {
    const other = { send: async (m: string, p: unknown[]) => (m === "web3_clientVersion" ? "besu/v0.0.0" : node.send(m, p)) };
    let err: any;
    try { await chainIdentity([node, other], FILE); } catch (e) { err = e; }
    expect(String(err)).to.contain("nodes disagree on clientVersion");
  });
});
