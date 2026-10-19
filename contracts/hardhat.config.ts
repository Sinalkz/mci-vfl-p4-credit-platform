import { defineConfig } from "hardhat/config";
import hardhatEthers from "@nomicfoundation/hardhat-ethers";
import hardhatEthersChaiMatchers from "@nomicfoundation/hardhat-ethers-chai-matchers";
import hardhatMocha from "@nomicfoundation/hardhat-mocha";
import hardhatNetworkHelpers from "@nomicfoundation/hardhat-network-helpers";

// EVM target is pinned to Cancun for both the compiler and the in-process
// network, so gas figures measured here match the Besu chain of S1.
export default defineConfig({
  plugins: [hardhatEthers, hardhatEthersChaiMatchers, hardhatMocha, hardhatNetworkHelpers],
  solidity: {
    version: "0.8.28",
    settings: {
      optimizer: { enabled: true, runs: 200 },
      evmVersion: "cancun",
    },
  },
  networks: {
    default: {
      type: "edr-simulated",
      chainType: "l1",
      hardfork: "cancun",
    },
    // S1: the 4-validator Besu QBFT chain in chain/docker-compose.yml.
    // Accounts are Hardhat's public test mnemonic, prefunded in the genesis: dev only.
    besu: {
      type: "http",
      chainType: "l1",
      url: process.env.BESU_RPC_URL ?? "http://127.0.0.1:8545",
      chainId: 2026,
      accounts: { mnemonic: "test test test test test test test test test test test junk", count: 4 },
    },
  },
  test: {
    mocha: { timeout: 600_000 },
  },
});
