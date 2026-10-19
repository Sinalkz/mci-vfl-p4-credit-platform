"""MCI loan instance, Step 5: the plan of the five on-chain runs, fixed before run 1 (as py/full_wait10s_plan.py fixed the
thesis's series), from the pre-registration (docs/mci/02-preregistration.md, section 9), the model configurations
(results/mci_loan/configs/configs.json) and the alignment benchmark on the run machine (results/mci_loan/onchain/
psi-bench.json: the declared period is at least twice the slowest measured alignment, bench-psi-full.mjs's rule).

Output: results/mci_loan/onchain/plan.json.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/onchain_plan.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

OUT = cm.RESULTS / "onchain"
SCENARIO_SEED = 20267005


def main() -> None:
    cfg = {c["name"]: c for c in json.loads((cm.RESULTS / "configs" / "configs.json").read_text(encoding="utf-8"))["configs"]}
    bench = json.loads((OUT / "psi-bench.json").read_text(encoding="utf-8"))
    if not all(x["correct"] for x in bench["reps"]) or bench["scenario"]["name"] != "mci-loan" or bench["scenario"]["limit"]:
        sys.exit("psi-bench.json is not a correct full-size benchmark of the mci-loan scenario")
    blocks = bench["alignmentPeriod"]["blocks"]
    runs = []
    for k in (1, 2, 3, 4, 5):
        c = cfg[f"loan-s{k}-wait10s"]
        runs.append({"run": k, "modelConfig": c["file"], "modelConfigSha256": c["sha256"], "modelConfigHash": c["keccak256"],
                     "offchainConfig": cfg[f"loan-s{k}"]["file"], "change": c["change"], "runSeed": k, "scenarioSeed": SCENARIO_SEED,
                     "setup": "mpc-d14", "prover": "rapidsnark", "alignmentPeriodBlocks": blocks,
                     "compareWith": f"results/mci_loan/offchain/runs/c2-s{k}.json (off-chain seed {k}, the same model seeds)",
                     "outDir": f"results/mci_loan/onchain/run-{k}",
                     "command": f"caffeinate -ims bash scenarios/mci_loan/onchain/mac-runs.sh run {k} {blocks}"})
    plan = {"what": "the MCI loan instance's five on-chain runs, fixed before run 1 (scenarios/mci_loan/onchain_plan.py)",
            "preregistration": "docs/mci/02-preregistration.md section 9 (tag mci-prereg)",
            "alignmentBenchmark": {"file": "results/mci_loan/onchain/psi-bench.json", "slowestAlignmentS": bench["slowestAlignmentS"],
                                   "rule": bench["alignmentPeriod"]["rule"], "machine": bench["machine"]},
            "runs": runs}
    cm.write_json(OUT / "plan.json", plan, indent=1)
    print(json.dumps({"alignmentPeriodBlocks": blocks, "slowestAlignmentS": bench["slowestAlignmentS"], "runs": len(runs)}))


if __name__ == "__main__":
    main()
