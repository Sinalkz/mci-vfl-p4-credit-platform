# Delivery notes

**Source.** This repository was built from the tracked files at commit `11decb2` of the thesis repository's
`mci-loan-demo` branch, taken with `git archive`. It was committed once, with no history, and has no remote. Of the
source's 3,061 tracked files, 485 were taken. This file records:
1. what was left out, and why;
2. what was changed for the delivery;
3. where the code's slot names remain.

## 1. What was left out, and why

### Row-level data (Kaggle competition rules)

The Home Credit data may not be redistributed, so no row-level data is included: no raw CSVs, no prepared feature
tables, no embeddings, no per-applicant scores or PDs, no evaluation rows or applicant-ID lists. The
README explains how to download the data and regenerate everything.

Removed from `results/mci_loan/v2/` (268 files; `.gitignore` keeps them out of future commits):

| Files | What they hold |
|---|---|
| `offchain/runs/*.json` (200) | every off-chain run's per-applicant margins and predictions on the 3,193 test applicants |
| `offchain/check-c2-s1.json`, `offchain/check-eps-x0-s1.json` | the determinism checks' runs (the same per-applicant arrays) |
| `smoke/c2-s1.json` | the smoke run (per-applicant arrays for 200 test rows); `smoke/smoke-check.json` stays |
| `onchain/run-<k>/{A,p1,p2,p3}/I.json`, `I_tr.json`, `I_ev.json` (60) | each process's aligned applicant IDs (all, training, evaluation) |
| `onchain/run-<k>/sample.json` (5) | the run's scenario record, which lists the extra applicants' IDs |

**Redacted, not removed:** `results/mci_loan/v2/prepared/manifest.json`, `prepared/extra_manifest.json` and
`prepared-smoke/manifest.json`.
- **What was replaced.** Each list of applicant IDs, labels, split positions, thin-file positions and smoke positions is replaced by `{"redacted": …, "count": n}`.
- **What stays.** The column lists, the SHA-256 values of the prepared files, the seeds, the population default rate and the cohort sizes. The dashboard and the report figures read these.
- **The full manifests** are written under `data/` (git-ignored) when `prep2.py` runs.

### Thesis-only material

These were left out unless v2 reads them:
- the electricity (RECS) scenario's data (`data/recs/`); the four contract-suite cases that read it skip (section 2);
- its experiments and results (`results/data1`, `ldp`, `full`, `m1`–`m4`, `mpc`, `overhead`, `proofcost`, `s1`–`s4`, `stats`, `talmud`, `tests`, `zcite`);
- benchmark scripts (`contracts/scripts/`) and their outputs;
- the thesis's reports and writing (the root `RESULTS-*.md`, `QUESTIONS-*.md`, `CH5-*.md`, `DATA-STATUS.md`, `NAMES-MAP.md`, `docs/ch5.txt`, `docs/decision-log.md`, `docs/SPIKE.md`, `figures/`, `refs/`);
- the Python modules only the thesis's reports use. Three files of the thesis's reference tests (`py/tests`) read such modules or records: `test_ldp_analysis.py`, `test_tests_by_requirement.py`, and one case of `test_m4_e3.py`. The README says how to run the rest.

**Thesis files kept because v2, the tests or the setup read them:**

| File | Read by |
|---|---|
| `results/full/setup/verification_key.json`, `ceremony.json` | `runner3.mjs` (`--setup mpc-d14`: the vkHash each run records); the README's key-regeneration figures |
| `results/m1b/verification_key.json` | `runner3.mjs` (`--setup dev`) |
| `results/mpc/setup/verification_key.json` (the maxDepth 10 multi-party key's) | `contracts/offchain/proposal.mjs`, which hashes it by default; the Hardhat tests `D09` (`Data1`) and `Q5` (`M2Alignment`). Added after the first fresh-clone check, where those two failed without it |
| `results/m1b/proofs/N64-D10-I*.json`, `results/s2b/proofs/N64-I1000/` | the Hardhat tests `MembersVar` and `Scoring` (real proofs checked against the verifiers) |
| `results/ldp/eps-levels.json` | `step4v2.mjs`, `analysis2.py`, `results_doc2.py`, `prereg2.py` (the privacy sweep's ε levels) |
| `circuits/inputs/*.json` | `py/s2b_model.py` and `py/s2_model.py` (synthetic data of the scoring oracle) |

### v1 material (the instance's first version, with the Bank active)

v1's documents, results and the scripts only v1 uses were left out:
- documents: `docs/mci/*.md`;
- results: `results/mci_loan/` outside `v2/`;
- scripts: `prereg.py`, `results_doc.py`, `comparison.py`, `offchain/step4.mjs`, `offchain/configs.mjs`, `offchain/compare-runs.mjs`, `onchain/mac-runs.sh`, `tests/test_scenario.py`, `slots.json`.

**Kept, because v2 imports them:**
- `common.py`, `bank.py`, `features.py`, `party_rule.py`, `prep.py`, `select_bank.py`, `metrics.py`, `analysis.py`, `age_check.py`, `smoke.py`, `onchain_plan.py`, `onchain_report.py`, `data_manifest.py`;
- `offchain/prepared.mjs`, `vfl-run.mjs`, `central-run.mjs`, `n64.mjs`, `score.mjs`;
- `onchain/loan-scenario.mjs`, `tests/loan-scenario.test.mjs`;
- `results/mci_loan/data/raw-manifest.json` (the downloaded files' SHA-256 values, to check a new download against);
- `results/mci_loan/age-check.json` (the default rate per age band, aggregated; `prereg2.py` reads it).

**Not delivered, although v2 code reads them.** The code runs without them, as described in section 2.

| File | Why not | Effect |
|---|---|---|
| v1's prepared manifests (`results/mci_loan/prepared*/manifest.json`) | row-level | `prep2.py` and `prereg2.py` check v2's cohort and split against v1's only when they are present |
| v1's C1b runs (`results/mci_loan/offchain/runs/c1b-s*.json`) | row-level | `analysis2.py` records `c1bEqualsV1` as not checked |
| the electricity and v1 on-chain `run.json` files | thesis and v1 run records | `onchain_report2.py` leaves those two series of `cost_comparison.csv` blank; the delivered file keeps them |
| v1's `results/mci_loan/comparison/scenario-comparison.json` | v1 record, which names the e-commerce role by a real company | `comparison2.py` (electricity, v1 and v2 side by side) cannot run; its delivered outputs are `results/mci_loan/v2/comparison/` and `docs/mci/v2/05-scenario-comparison.md` |

**The delivered v2 documents also cite files that are not here:**
- v1's documents `docs/mci/00-recon.md`, `01-data.md` (which `data_manifest.py` regenerates), `02-preregistration.md` and `04-phase3-mapping.md`;
- v1's `results/mci_loan/select_bank.json`;
- the row-level files listed above.

The documents are kept as they were written, apart from the changes in section 2.

### Other material left out

| What | Why |
|---|---|
| `node_modules/`, `.venv/`, `contracts/artifacts/` and caches | regenerated by `npm ci`, `pip install`, `npx hardhat compile` |
| `circuits/build/` (keys, wasm, r1cs) and `circuits/ptau/*.ptau` | hundreds of MB to several GB each; the README gives the regeneration commands. The verifier contracts are kept. |
| `chain/data/` (node keys, genesis) | generated at setup by `chain/scripts/generate.sh` |
| The agent instruction and prompt files | agent instructions; the two citations of the instruction file now point to the README's "Protocol rules" |
| `py/mci_baseline.py smoke` | its code is kept, but it reruns the electricity scenario and needs that scenario's data |

## 2. Changes made for the delivery

Compared with `11decb2`, the following files differ. Everything else is byte-identical to the source.

**Code: v1 and thesis inputs are checked when present.** Each change is marked `# delivery:` in the code.
- `scenarios/mci_loan/v2/common2.py`: adds `redact_manifest()`.
- `scenarios/mci_loan/v2/prep2.py`: the v1 cohort check runs when v1's manifests are present. The `results/` copies of the manifests are redacted after they are written.
- `scenarios/mci_loan/v2/analysis2.py`: reads the full manifest from `data/mci_loan/v2/prepared/`; the C1b comparison with v1 runs when v1's runs are present.
- `scenarios/mci_loan/v2/results_doc2.py`: reports the C1b comparison as not checked when it was not.
- `scenarios/mci_loan/v2/onchain_report2.py`: the electricity and v1 cost series are used when their run files are present.
- `scenarios/mci_loan/v2/prereg2.py` and `comparison2.py`: read the full manifest from `data/mci_loan/v2/prepared/`; `prereg2.py`'s v1 check runs when present.

**Tests: four cases skip without the thesis's RECS data.**
- `contracts/test/helpers/recs.ts` (new): skips a case when a file it reads is absent, and adds the missing paths to the case's title.
- `contracts/test/Data1.test.ts` (`D04b`, `D08`) and `contracts/test/LdpConfigs.test.ts` (`L01`, `L02`) call it first. `D04b`, `D08` and `L02` need `data/recs/prepared/`; `L01` needs `results/data1/configs/e1-s1.json` and `e1-s5.json`.
- With those files present, the four cases run as before.

**Machine details.**
- `scenarios/mci_loan/v2/onchain/mac-runs-v2.sh`:
  - the Mac's tool and checkout paths became environment settings (`BENCH`, `NODE_BIN`, `BESU_HOME`, `JAVA_HOME`, `RAPIDSNARK`);
  - the expected genesis hash defaults to the local genesis's (`GENESIS_SHA256` overrides it);
  - a helper script that was never in the repository (`mem.sh`) became an inline `vm_stat` reading;
  - the checkout is found from the script's own location.
- `results/mci_loan/v2/offchain/step4.log`: the laptop's absolute repository path became `<repo>`.
- `step4.log`, `offchain/summary.json` and the five `onchain/run-<k>/replay.json` files: the two machines' host names became `laptop` and `macbook-air`.
- `docs/mci/v2/README-mci-v2.md`: the Mac checkout's path, and a pointer to v1's README, now point to this README.

**Names, in prose only.** The company name the e-commerce role was once given became "E-commerce company" in these files:
- comments: `chain/qbftConfigFile.json`, `contracts/offchain/data.mjs`, `py/s2b_model.py`;
- v1's display names and docstrings: `scenarios/mci_loan/analysis.py`, `onchain_report.py`, `party_rule.py`, `smoke.py`.

v2 sets its own names from `scenarios/mci_loan/v2/slots.json`, so v2's output does not change.

**Citations.** `contracts/test/helpers/reconciliation.ts` and `docs/mci/v2/04-phase3-mapping.md` cited the agent instruction file; they now cite the README's "Protocol rules".

**New or replaced files:** `README.md` and `.gitignore` (replacing the thesis's), `DELIVERY-NOTES.md`, `DELIVERY-CHECK.md`.

## 3. Where the code's slot names remain

The protocol code names its four member slots after the thesis's electricity
scenario. In v2 they map to these members (`scenarios/mci_loan/v2/slots.json`, the one mapping config):

| Slot | v2 member | Role |
|---|---|---|
| `power` | MCI | active: the lender, label owner, pool funder |
| `mci` | Bank | passive |
| E-commerce slot | E-commerce company | passive |
| `maker` | Credit bureau | passive |

User-facing files (docs, dashboard, figures, captions) use only the member names. The E-commerce company's slot
identifier remains in code and records only, here:
- **Network:**
  - `chain/docker-compose.yml`: a service, its data folder, a port comment;
  - `chain/scripts/generate.sh`, `chain/scripts/native-besu.sh`: the member list;
  - `chain/qbftConfigFile.json`: an alloc comment naming the slot.
- **Protocol runner:** `contracts/offchain/runner.mjs`, `runner3.mjs` (the node map).
- **Tests:** `contracts/test/Reconciliation.test.ts`, `Scoring.test.ts` (signer variable names).
- **Ceremony:** `circuits/scripts/mpc-ceremony.mjs` (`MEMBERS`, the contributor labels) and `results/full/setup/ceremony.json` (the thesis ceremony's record of them).
- **Scoring oracle's synthetic data:** `py/s2_model.py` (`MEMBERS`) and `circuits/inputs/model_eval.json` (member labels).
- **v1's owner key for the e-commerce role** (v2 uses `ecommerce`): `scenarios/mci_loan/party_rule.py`, `scenarios/mci_loan/offchain/prepared.mjs`.
- **v2 records:** the `slot` field in `results/mci_loan/v2/onchain/plan.json`, `onchain/summary.json` and the two prepared manifests; `arbiterNode` in `onchain/run-4/run.json`.

**The name as a pattern,** so code can detect or remove it:
- `dashboard/check_pages.py` and `dashboard/report_figures.py`: the naming check's regular expression;
- `scenarios/mci_loan/v2/comparison2.py`: `RENAME`, applied to v1's file.

**Other names.** "Home Credit" appears as the name of the public dataset (Kaggle's Home Credit Default Risk
competition), which the data download needs. The dashboard banner names it as the data source.
