# Delivery check

Checks on this repository (495 files), 2026-10-20: each check ran on a fresh clone, set up as the README's "Setup" says.

**Commits checked.** This repository has one commit; the checks ran on earlier versions of it.
- **Laptop:** the clone had the same files as this commit, with these exceptions:
  - this file;
  - `results/delivery-check/`, the checks' records;
  - one row of the README's results table, which points to those records;
  - `results/mci_loan/data/raw-manifest.json` (the downloaded files' SHA-256 values), with the `.gitignore` line that had kept it out of the commit. The line `data/` also matched `results/mci_loan/data/`; it is now `/data/`. No check reads that manifest.
- **Server:** the clone was one version earlier. It lacked `results/mpc/setup/verification_key.json` and some README and DELIVERY-NOTES text (section 1, "The first run"). The server ran only M3Scoring, which passed without reading that key file.

| Machine | Hardware | Software |
|---|---|---|
| laptop | Intel i7-9750H (6 cores, 12 threads), 15.9 GB | Windows 11, Node.js 24.16.0, Python 3.14.5 |
| server (shared) | Intel Xeon E7-8880 v3 (20 vCPU), 58.9 GB | Linux 5.15, Node.js 24.16.0, Python 3.12.12, circom 2.2.3 |

Hardhat 3.17.0, mocha 11.8.0, snarkjs 0.7.6, solc 0.8.28 on both.

## 1. Contract suite: 195 passing, 0 failing, 4 skipped

| Part | Machine | Passing | Failing | Skipped | Time |
|---|---|---:|---:|---:|---:|
| light (16 files) | laptop | 179 | 0 | 4 | 644.4 s |
| heavy (`M3Scoring.test.ts`) | server | 16 | 0 | 0 | 571.1 s |
| **all 17 files** | | **195** | **0** | **4** | |

Both parts ran with `py/mci_baseline.py hardhat --part light|heavy --out results/delivery-check`. Their output and records are in `results/delivery-check/hardhat-{light,heavy}.{txt,json}`.

**The 4 skipped cases** need the thesis's electricity (RECS) data, which is not delivered (DELIVERY-NOTES.md, sections 1 and 2). Each one's title gives the reason, for example:

```
- D04b loadPrepared: 15,965 rows in DOEID order, ... disjoint split [skipped: the thesis's RECS data is absent; missing data/recs/prepared/manifest.json]
```

The four are `D04b` and `D08` (`Data1.test.ts`), and `L01` and `L02` (`LdpConfigs.test.ts`).
- **With the data present they pass.** The thesis's `data/recs/prepared/` and the E1 configs were copied into the laptop clone, and the four cases ran: 4 passing. The copies were then deleted.
- **Their sum with the rest gives the full suite:** 195 passing plus 4 skipped equals the 199 cases that pass on the source code with all the thesis's data (`results/mci_loan/suite-recheck/`).

**The first run** found two more failing cases: `D09` (`Data1.test.ts`) and `Q5` (`M2Alignment.test.ts`).
- **Cause:** `contracts/offchain/proposal.mjs` hashes `results/mpc/setup/verification_key.json` by default, and that file had been left out with the rest of `results/mpc/`.
- **Fix:** it is now delivered (4.6 KB, a public verification key; DELIVERY-NOTES.md, section 1). The second run, recorded above, passed both.
- **The README** also now says which `py/tests` files cannot run on this delivery; see section 3.

**The development key, regenerated on the server with the README's commands** (`README.md`, "Proving keys"):

| Step | Time |
|---|---:|
| `fetch-ptau.mjs 20 --source ppot` | 10 s |
| `m1b-build.mjs --n 64 --maxdepth 10`: compile 116 s (773,221 constraints), setup 548 s | 709 s |
| `npx hardhat compile` (the rewritten verifier) | 4 s |

- **The phase-1 file (1.2 GB)** was copied from a local copy rather than downloaded. `fetch-ptau.mjs` hashed it and rewrote `circuits/ptau/ppot_0080_20.ptau.blake2b`, which stayed unchanged.
- **The new key differs from `results/m1b/verification_key.json` only in `vk_delta_2`.**
  - snarkjs mixes system randomness into the fixed entropy string, so each build gives a new key.
  - The build rewrote `MembersVarVerifierN64D10.sol` to match the new key.
  - M3Scoring passed with the new key and verifier. The README now says this.

**Setup times on the fresh clones:**

| Step | Laptop | Server |
|---|---:|---:|
| `git clone` | 6 s | 2 s |
| `npm ci` in `contracts/` and `circuits/` | 15 s, 7 s | 9 s, 9 s |
| `npx hardhat compile` | 6 s | 9 s |
| `.venv` and `pip install -r py/requirements.txt` | 8 s, 121 s | 5 s, 114 s |

## 2. Dashboard headless check: passed

`dashboard/check_pages.py` ran on the laptop clone:
- the app and its eight pages rendered without an error;
- the naming check passed;
- the rewritten `results/mci_loan/v2/dashboard-check.json` is byte-identical to the delivered one.

## 3. Python checks

| Check | Result |
|---|---|
| `py_compile` on every Python file; `node --check` on every `.mjs` under `scenarios/` and `dashboard/` | passed (on the delivery tree, 2026-10-19) |
| Import of each of the 13 v2 modules (`scenarios/mci_loan/v2/*.py`) | 13 imported |
| The v2 tests, which need no data: `pytest -q scenarios/mci_loan/v2/tests` | 7 passed, 1 skipped (the one that reads the Home Credit data) |
| The thesis's reference tests, as the README runs them (`py/tests`, with two files left out and one case deselected) | 54 passed, 1 skipped |

The two `py/tests` files and the one case read thesis records that are not delivered: `py/ch5_stats.py`, `results/tests/` and `results/m4/` (README, "Tests").

No check left a tracked file changed on the laptop clone.

## 4. Secret and personal-data scan

gitleaks was not available, so the scan used grep over the tracked files:

| Pattern | Result |
|---|---|
| `KGAT_` (Kaggle tokens) | none; it appears only in this table |
| `BEGIN … PRIVATE KEY` | none |
| 64-hex strings outside `results/` and `docs/` | 22 files; none holds a key, see the breakdown below |
| 64-hex strings in `results/` and `docs/` | 2,854: transaction hashes, commitments, SHA-256 values of files, proof and verification-key elements (long decimal numbers also match the pattern) |
| IP addresses | `127.0.0.1` (local JSON-RPC); `0.0.0.0` (listen on all interfaces: `chain/docker-compose.yml`, the Streamlit notes); `172.28.0.11`–`.14` and `172.28.0.0/16` (Docker's internal bridge network in `chain/docker-compose.yml` and `generate.sh`); `25.0.4.1` (a version string, not an address) |
| User paths (`C:\Users\…`, `/Users/…`, `/home/…`, the source machines' tool folders), host names | none (replaced; DELIVERY-NOTES.md, section 2). The new check records carry only the machine labels `laptop` and `server` |
| The old real company name | in 27 files, only as the E-commerce company's slot identifier in code and records, or as the pattern the naming check and `comparison2.py` look for; no prose use (DELIVERY-NOTES.md, section 3) |

The 64-hex strings outside `results/` and `docs/` are:
- verifier constants (`contracts/contracts/verifiers/*.sol`);
- EIP-712 type hashes and constants (`Federation.sol`, `Reconciliation.sol`, test helpers);
- curve constants: the BN254 field prime (`circuits/scripts/members-lib.mjs`, `proofcost.mjs`) and the secp256k1 order (`Reconciliation.test.ts`);
- a model config's expected keccak256 (`LdpConfigs.test.ts`);
- the phase-1 files' checksums (`circuits/ptau/*.blake2b`);
- the recorded vkHash and SHA-256 values (`README.md`);
- the delivered runs' genesis hash (`mac-runs-v2.sh`, a comment);
- the genesis header constant (`chain/qbftConfigFile.json`).

The chain's account keys are derived from Hardhat's public test mnemonic. They are not stored as literals; README, "Local development keys".

## 5. The 20 largest files

| Bytes | File |
|---:|---|
| 1,180,656 | `circuits/inputs/s2b_data.json` (synthetic data of the scoring oracle) |
| 447,514 | `docs/mci/v2/figures/report/dashboard-1-mci-alone-vs-partners.png` |
| 266,083 | `docs/mci/v2/figures/report/dashboard-2-credit-tiers.png` |
| 254,609 | `docs/mci/v2/figures/report/dashboard-3b-audit-trail.png` |
| 228,169 | `docs/mci/v2/figures/01-architecture.png` |
| 187,115 | `docs/mci/v2/figures/07-tiers.png` |
| 178,837 | `results/mci_loan/v2/offchain/summary.json` |
| 172,546 | `docs/mci/v2/figures/05-business.png` |
| 170,143 | `docs/mci/v2/figures/12-privacy.png` |
| 168,954 | `docs/mci/v2/figures/report/dashboard-3a-settlement.png` |
| 153,736 | `docs/mci/v2/figures/06-partner-value.png` |
| 150,102 | `docs/mci/v2/figures/13-settlement-run-5.png` |
| 149,842 | `docs/mci/v2/figures/04-auc.png` |
| 145,762 | `docs/mci/v2/figures/13-settlement-run-2.png` |
| 143,649 | `docs/mci/v2/figures/13-settlement-run-4.png` |
| 142,766 | `docs/mci/v2/figures/03-quality.png` |
| 139,014 | `docs/mci/v2/figures/13-settlement-run-1.png` |
| 138,032 | `docs/mci/v2/figures/13-settlement-run-3.png` |
| 135,112 | `docs/mci/v2/figures/report/01-architecture.png` |
| 130,870 | `contracts/test/vectors/talmud_vectors.json` |

No file is over 50 MB. The largest is 1.2 MB.

## Note for Windows

A clone placed in a deep folder can exceed Windows' 260-character path limit. One placed under a very long path failed
to check out. Clone near the drive root, or set `git config --global core.longpaths true`.
