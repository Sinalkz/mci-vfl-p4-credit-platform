# MCI credit instance: vertical federated learning with verifiable reward settlement

| MCI-VFL series | Repository | Scope |
|:--|:--|:--|
| Phase 1/4 | [mci-vfl-p1-bi-platform](https://github.com/Sinalkz/mci-vfl-p1-bi-platform) | Blockchain-based VFL platform for business intelligence |
| Phase 2/4 | [mci-vfl-p2-loan-simulation](https://github.com/Sinalkz/mci-vfl-p2-loan-simulation) | VFL loan-eligibility simulation (telecom + bank) |
| Phase 3/4 | [mci-vfl-p3-fabric-chaincode](https://github.com/Sinalkz/mci-vfl-p3-fabric-chaincode) | Hyperledger Fabric audit chaincode |
| **Phase 4/4** | **[mci-vfl-p4-credit-platform](https://github.com/Sinalkz/mci-vfl-p4-credit-platform)** | **Blockchain-coordinated VFL credit platform** |

This repository is the **MCI credit instance** of an MSc thesis architecture: blockchain-based vertical federated
learning (VFL) whose reward settlement is verified on-chain. It is the code, the aggregated results and the documents of
the instance's second version (v2), in which **MCI is the lender**:
- **The question.** MCI grants credit to its subscribers and must estimate whether they will repay.
- **The partners.** MCI trains a repayment model jointly with three partners: a **Bank**, an **E-commerce company** and a **Credit bureau**. Each partner's data stays with it.
- **The settlement.** A permissioned blockchain records what each partner committed to, accepts each model-quality figure only with a zero-knowledge proof, and divides MCI's reward pool among the partners by the Talmud rule.

**The organisations are simulated.** The three partners are generic roles defined over one public dataset, Kaggle's
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition data. No real
company's data is used, MCI's included.
- **MCI's side.** The public data has no telecom behaviour (usage, top-ups, bills), so "MCI alone" here is a lender's application model.
- **In a pilot,** MCI's own telecom data would join MCI's side. The federation, the proofs and the settlement would not change.

The repository contains no row-level data (see [Data](#data-download-and-preparation)). [DELIVERY-NOTES.md](DELIVERY-NOTES.md)
lists what was left out and why. [DELIVERY-CHECK.md](DELIVERY-CHECK.md) records the checks run on this delivery, and the ones still to run.

## Contents

| Folder | What |
|---|---|
| `contracts/` | Solidity contracts (Hardhat 3): `Federation` (the protocol), `SettlementMath` and `TalmudLib` (the division, in `TalmudDivision.sol`), the Groth16 verifiers in `contracts/verifiers/`. `offchain/` is the off-chain protocol and model code (each member's process, training, scoring, the runner); `test/` is the Hardhat test suite. |
| `circuits/` | The Circom 2 scoring circuit (`src/macro_f1_members_var.circom`: macro-F1 over the members' commitments) and the scripts that compile it and make its keys (`scripts/m1b-build.mjs`, `scripts/mpc-ceremony.mjs`). |
| `chain/` | The 4-node Besu QBFT network: `docker-compose.yml`, `scripts/generate.sh` (node keys and genesis), `scripts/native-besu.sh` (the same network without Docker), `scripts/check-network.mjs`. |
| `scenarios/mci_loan/` | The credit instance. `v2/` is this version (MCI active). The modules beside it are shared code that v2 imports (data layer, features, metrics, the loan scenario for the on-chain runs). |
| `py/` | Python reference implementations the tests and reports call: the scoring oracle (`s2b_oracle.py`), the settlement and Talmud references, the on-chain run checks (`m4_e3.py`, `m4_e3_ch5.py`), shared analysis helpers. |
| `dashboard/` | The Persian Streamlit dashboard, its headless page check, and the figure exporters. |
| `docs/mci/v2/` | The documents: pre-registration, results, Phase 3 mapping, scenario comparison, how-to (`README-mci-v2.md`), figures. |
| `results/` | Aggregated results: `mci_loan/v2/` (off-chain and on-chain), `mci_loan/suite-recheck/` (the test-suite run), and the small inputs the code reads (see [Results](#where-the-results-and-figures-are)). |

## The architecture in brief

There are two planes:
- **The data plane, off-chain.** Each member keeps its own columns. Training is split: MCI holds the labels, its own bottom model and the fusion head, and each partner sends embeddings of its own columns.
- **The control plane, on-chain.** A Hyperledger Besu network with QBFT consensus runs one validator per member. The `Federation` contract runs one federation from proposal to settlement.

One run goes through these steps:
1. **Proposal and deposits.** MCI creates the run and escrows the reward pool B. Each partner joins with a participation deposit D_p.
2. **Alignment.** A private set intersection finds the applicants all members hold. MCI signs the aligned set.
3. **Training,** off-chain. Only the final round's embedding matrix of each partner is committed (γ_i = hash(h_i)) and signed.
4. **Scoring.** Six macro-F1 figures, each on the evaluation rows and each accepted only with a Groth16 proof against the registered commitments:
   - F_r, the federation;
   - F_j, the jointly trained federation;
   - LOO_1, LOO_2 and LOO_3, the federation without partner 1, 2 or 3;
   - U_A, MCI alone (BASELINE).
5. **Settlement.** The contract computes U_fed = max(F_r, F_j), the estate E = U_fed − U_A, each partner's claim c_i = max(0, U_fed − LOO_i) and the capped estate E′ = min(E, ⌊B/ρ⌋). It divides E′ by the Talmud rule (TalmudLib) and pays partner i ρ·a_i tokens. MCI keeps the rest of the pool.

Off-chain, a **credit-policy layer** turns any model's score into a calibrated probability of default (PD):
- **Calibration:** 2-fold cross-fitted Platt scaling on the test applicants, prior-corrected to the population default rate of 8.07%.
- **Tiers:** 4-month credit (PD < 4%), 1-month credit (PD < 12%), or no credit (`scenarios/mci_loan/v2/policy.py`).

The AUC, business metrics, PDs and tiers are off-chain and not ZK-verified. The macro-F1 figures are the ones the circuit proves.

### Protocol rules

- **The chain.** A permissioned EVM chain with an IBFT-family consensus (Besu, QBFT): at least 4 validators, one per member.
- **What is recorded.** Only the final-round embedding matrix of each passive member is committed and signed; nothing is recorded during training rounds.
- **The commitment** is γ_i = hash(h_i) of member i's final-round matrix, with no salt.
- **The signature** is σ_i = Sig_i(ctx_i, γ_i), made with the member's own chain account key, where ctx_i = (contract address, member address, final-round marker). It is EIP-712 typed data, so the signed message can never be a valid transaction and cannot be replayed in another run.
- **Equivocation.** If a member's registered γ_i differs from the digest it signed, MCI (the active member) submits the signed digest. The contract then:
  - verifies the signature;
  - records the equivocation;
  - forfeits the member's deposit D_p;
  - makes the signed digest the binding commitment.
- **Figures** are macro-F1 on the evaluation set, as integers in units of 10⁻⁴: F1 = 0.8123 is recorded as 8123. Nothing on-chain is a decimal.
- **Division** is in integer token units, and each award is floored.
- **Termination.** Settlement always terminates, even if a member stays silent.
- **Cost.** On-chain cost does not depend on the data size, and grows linearly with the number of members.

## Setup

Requirements:
- **Node.js 24.** The delivered runs used 24.16.0.
- **Python 3.12 or later.** Used: 3.14.5 and 3.12.12.
- **Git Bash** on Windows, for the shell scripts.
- **For the network:** Docker, or Besu 26.8.1 with Java 21 or later.
- **For the on-chain runs:** [rapidsnark](https://github.com/iden3/rapidsnark), or snarkjs, which is slower.
- **To regenerate the proving keys:** [circom](https://docs.circom.io/) 2.1.9 or later.

```bash
cd contracts && npm ci && npx hardhat compile && cd ..
cd circuits && npm ci && cd ..
python -m venv .venv                      # the code calls <repo>/.venv/bin/python (Windows: .venv/Scripts/python.exe)
.venv/bin/python -m pip install -r py/requirements.txt
```

**Tests:**
- **The contract suite:** `cd contracts && npx hardhat test mocha` (17 files). `M3Scoring.test.ts` needs:
  - the development proving key (see [Keys](#proving-keys-regeneration));
  - about 7 GB of free memory, because it proves with snarkjs.

  Four cases read the thesis's electricity (RECS) data, which is not delivered: `D04b` and `D08` in `Data1.test.ts`, `L01` and `L02` in `LdpConfigs.test.ts`. Without that data they skip, and each one's title names the missing files.
  
  Each part can also run alone, with its output recorded: `python py/mci_baseline.py hardhat --part light|heavy --out results/<folder>`. DELIVERY-CHECK.md has the counts on this delivery.
- **The instance's Python tests:** `.venv/bin/python -m pytest -q scenarios/mci_loan/v2/tests`. They need no data; the one that reads the Home Credit data skips without it.
- **The thesis's Python reference tests** (`py/tests`): three of them read thesis records that are not delivered:
  - `test_ldp_analysis.py` imports `py/ch5_stats.py`;
  - `test_tests_by_requirement.py` reads `results/tests/`;
  - one case of `test_m4_e3.py` reads `results/m4/`.

  The rest run with `.venv/bin/python -m pytest -q py/tests --ignore=py/tests/test_ldp_analysis.py --ignore=py/tests/test_tests_by_requirement.py --deselect=py/tests/test_m4_e3.py::test_e3_run1_is_rechecked_under_the_backup_rule_it_settled_by`.

## Data: download and preparation

The data is Kaggle's Home Credit Default Risk competition data. Its rules do not allow redistribution, so this
repository holds **no row-level data**:
- no CSVs;
- no prepared feature tables;
- no embeddings;
- no per-applicant scores or PDs;
- no applicant-ID lists.

Only aggregated results are included.

1. Sign in to Kaggle and **accept the competition rules** on the competition page.
2. Create a Kaggle API token and store it as the Kaggle CLI expects (`~/.kaggle/`, or the `KAGGLE_API_TOKEN` environment variable). Never commit it; `.gitignore` excludes `kaggle.json`.
3. Download and unpack:
   ```bash
   .venv/bin/kaggle competitions download -c home-credit-default-risk -p data/raw
   .venv/bin/python -I scenarios/mci_loan/data_manifest.py
   ```
   - `data_manifest.py` checks every member name in the archive and extracts the CSVs to `data/raw/home-credit/`.
   - It rewrites `results/mci_loan/data/raw-manifest.json` with each file's size, SHA-256, column count and row count. `git diff` on that file should show no change in the hashes.
   - It also writes `docs/mci/01-data.md`.

`data/` is git-ignored, and so are the row-level outputs the pipeline writes under `results/`.

## Proving keys: regeneration

The proving keys are not included: each is hundreds of MB, and the phase-1 file is over 1 GB.

**What is included:**
- the verifier contracts (`contracts/contracts/verifiers/`);
- the thesis ceremony's verification key and its record (`results/full/setup/verification_key.json`, `ceremony.json`);
- the development key's verification key (`results/m1b/verification_key.json`);
- the verification key of the thesis's maxDepth 10 multi-party key (`results/mpc/setup/verification_key.json`). `contracts/offchain/proposal.mjs` hashes it by default; tests `D09` and `Q5` read it.

**The delivered on-chain runs used the thesis ceremony's key.** That key comes from a multi-party Groth16 setup for the maxDepth 14 circuit, with four contributors, one per member slot, labelled with the slot identifiers (DELIVERY-NOTES.md, section 3, maps each slot to its v2 member). Its records:
- **vkHash** (keccak256 of the verification key file), recorded by every run: `0x43a9e64ae50cc7ca36385e354ca8406a662cb96f90f8fa90c5ee87299f8566e1` (`results/mci_loan/v2/onchain/run-<k>/run.json`, `keys`);
- **the key file's SHA-256:** `ff0e301209884cf214d65f6d7886239ebbf85e7fa78d2ca8934710d667109de1` (`ceremony.json`).

To make your own keys, run from `circuits/`, with circom on the PATH (or `CIRCOM=<path>`):

```bash
node scripts/fetch-ptau.mjs 20 --source ppot                          # the public phase-1 file (PSE Perpetual Powers of Tau, 2^20); checked against ptau/*.blake2b
node scripts/m1b-build.mjs --n 64 --maxdepth 10                       # the development key (tests, --setup dev): one local contribution, fixed entropy
node scripts/m1b-build.mjs --n 64 --maxdepth 14 --compile-only        # the circuit the runs use
node scripts/mpc-ceremony.mjs --maxdepth 14 --out ../results/full/setup   # the multi-party key (--setup mpc-d14)
```

- **Time and memory.** The thesis ceremony took **about 21 minutes and 7.6 GB** of peak memory on an Apple M4 with 16 GB: 1,242 s over its seven steps, plus 35 s to compile, with a peak of 7,636 MB in the verify step (`results/full/setup/ceremony.json`). Plan for at least 10 GB of free memory; `--verify-guard` stops early when there is too little.
- **The development key changes on each build.** snarkjs mixes system randomness into the fixed entropy string, so a new build gives a new `vk_delta_2`; the rest of the key is the same. The build rewrites `contracts/contracts/verifiers/MembersVarVerifierN64D10.sol`, so recompile the contracts before the suite. The delivered verifier and `results/m1b/verification_key.json` are from the thesis's build.
- **What a new ceremony overwrites:** `results/full/setup/verification_key.json` and `contracts/contracts/verifiers/MembersVarVerifierN64D14Mpc.sol`. Recompile the contracts (`npx hardhat compile`). New runs then record the new key's vkHash.

## Running

The v2 how-to with every command and its output is [docs/mci/v2/README-mci-v2.md](docs/mci/v2/README-mci-v2.md). In short:

**Off-chain pipeline** (Steps 3 and 4: a laptop is enough; about an hour for the three sets):
```bash
.venv/bin/python -I scenarios/mci_loan/v2/select_mci.py        # MCI's 8 columns, forward selection on the training split
.venv/bin/python -I scenarios/mci_loan/v2/prep2.py             # data/mci_loan/v2/prepared/ (row-level, git-ignored); redacted copies in results/
.venv/bin/python -I scenarios/mci_loan/v2/smoke2.py            # model configs and a 1,000-row smoke run
node scenarios/mci_loan/v2/offchain/step4v2.mjs --set main      # 10 seeds; then --set scales and --set eps (the privacy sweeps)
.venv/bin/python -I scenarios/mci_loan/v2/analysis2.py         # results/mci_loan/v2/offchain/*.csv, summary.json
.venv/bin/python -I scenarios/mci_loan/v2/results_doc2.py      # docs/mci/v2/03-results.md
```

**On-chain runs** (Step 5):
1. **Network.** `bash chain/scripts/generate.sh` makes the four node keys and the genesis with Besu's operator tool, in Docker. Then either:
   - `docker compose -f chain/docker-compose.yml up -d`, or
   - without Docker, `chain/scripts/native-besu.sh up`, as the delivered runs did.
2. **Runs.** `scenarios/mci_loan/v2/onchain/mac-runs-v2.sh` drives the benchmark, the five runs and their replays on macOS.
   - It takes the machine's tool locations from the environment: `BENCH` (required), and optionally `NODE_BIN`, `BESU_HOME`, `JAVA_HOME`, `RAPIDSNARK`, `GENESIS_SHA256`. The script's header explains each.
   - Each run calls `contracts/offchain/runner3.mjs --run mci-loan --setup mpc-d14`.
3. **Report.** `.venv/bin/python -I scenarios/mci_loan/v2/onchain_report2.py` writes the run tables and `summary.json` from complete run folders, which is to say after new runs.

**Dashboard** (Persian, right to left; it reads only `results/mci_loan/v2/`):
```bash
.venv/bin/streamlit run dashboard/app.py                         # http://localhost:8501
.venv/bin/python -I dashboard/check_pages.py                     # every page headless, plus the naming check
.venv/bin/python -I dashboard/export_figures.py                  # the dashboard's charts, docs/mci/v2/figures/
node dashboard/report_screens.mjs                                # page screenshots (headless Edge or Chrome: --browser <path>)
.venv/bin/python -I dashboard/report_figures.py                  # the report's figure pack, docs/mci/v2/figures/report/
```

## Local development keys

These keys are **for local development only**. Never use them on a shared or public network.
- **Account keys.** The chain's accounts are Hardhat's public test mnemonic (`test test … junk`), prefunded in the genesis (`chain/qbftConfigFile.json`, `contracts/hardhat.config.ts`, `contracts/offchain/chain.mjs`). Anyone can derive their keys.
- **Validator node keys.** These are not included: `chain/scripts/generate.sh` creates them at setup, under `chain/data/` (git-ignored).

## Where the results and figures are

| What | Where |
|---|---|
| Pre-registration (fixed before the runs), results, Phase 3 mapping, scenario comparison | `docs/mci/v2/02-preregistration.md`, `03-results.md`, `04-phase3-mapping.md`, `05-scenario-comparison.md` |
| Off-chain results, 10 seeds: quality, contributions, Talmud payments, business metrics, tiers, calibration, privacy sweeps | `results/mci_loan/v2/offchain/` (`summary.json` and one CSV per table; schemas in `03-results.md`) |
| On-chain runs, 5: states, gas, proofs, payouts, every transaction and event | `results/mci_loan/v2/onchain/` (`runs.csv`, `gas.csv`, `proofs.csv`, `payouts.csv`, `transactions.csv`, `events.csv`, `summary.json`, `run-<k>/`) |
| The contract suite on this code | `results/mci_loan/suite-recheck/` (on the source code with the thesis's data: 199 passing, 0 failing); `results/delivery-check/` (fresh clones of this delivery: 195 passing, 0 failing, 4 skipped without the RECS data); DELIVERY-CHECK.md |
| Figures: dashboard charts (17), report figures (charts at 16 cm, policy flow, 4 screenshots, Persian captions) | `docs/mci/v2/figures/`, `docs/mci/v2/figures/report/` (`captions.md`) |

The results in brief (10 seeds; macro-F1 × 10⁴; sources in brackets):
- **Quality.** MCI alone scores U_A = 5,940; MCI with all three partners scores U_fed = 6,791 [`offchain/summary.json`, `loo`].
- **Each partner's contribution** l_i: Credit bureau 472.4, Bank 76.3, E-commerce company 55.6 [same].
- **Payments.** At the pool B = 1,000,003 tokens, MCI pays on average: Bank 189,500, E-commerce company 142,500, Credit bureau 665,000 [`payments`].
- **Defaults.** At MCI alone's approval rate of 57.2%, the default rate among approved applicants falls from 5.36% (MCI alone) to 3.44% (with partners) [`business`, off-chain].
- **Credit tiers.** With partners, 38.4% of subscribers get 4-month credit, against 8.6% for MCI alone; all credit granted falls from 84.7% to 78.0%, at a 4.8% default rate instead of 6.8% [`tiers`, off-chain].
- **On-chain.** Five runs settled by the normal path; every figure was accepted with its proof, every replay equalled the chain, and tokens were conserved. Gas per run averaged 10,126,305 [`onchain/summary.json`, `cost_comparison.csv`].
