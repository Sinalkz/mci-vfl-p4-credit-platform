"""MCI loan instance v2, Step 5: the plan of the five on-chain runs, fixed before run 1, as v1's onchain_plan.py, from the
v2 pre-registration (docs/mci/v2/02-preregistration.md, section 10; tag mci-prereg-v2), the v2 model configurations
(results/mci_loan/v2/configs/configs.json) and the alignment benchmark measured again on the run machine
(results/mci_loan/v2/onchain/psi-bench.json: the declared period is at least twice the slowest measured alignment).

Output: results/mci_loan/v2/onchain/plan.json.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/v2/onchain_plan2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as cm  # noqa: E402
import common2 as c2  # noqa: E402
from onchain_plan import SCENARIO_SEED  # noqa: E402

OUT = c2.RESULTS2 / "onchain"


def main() -> None:
    cfg = {c["name"]: c for c in json.loads((c2.RESULTS2 / "configs" / "configs.json").read_text(encoding="utf-8"))["configs"]}
    bench = json.loads((OUT / "psi-bench.json").read_text(encoding="utf-8"))
    if not all(x["correct"] for x in bench["reps"]) or bench["scenario"]["name"] != "mci-loan" or bench["scenario"]["limit"]:
        sys.exit("psi-bench.json is not a correct full-size benchmark of the mci-loan scenario")
    blocks = bench["alignmentPeriod"]["blocks"]
    runs = []
    for k in (1, 2, 3, 4, 5):
        c = cfg[f"loan2-s{k}-wait10s"]
        runs.append({"run": k, "modelConfig": c["file"], "modelConfigSha256": c["sha256"], "modelConfigHash": c["keccak256"],
                     "offchainConfig": cfg[f"loan2-s{k}"]["file"], "change": c["change"], "runSeed": k, "scenarioSeed": SCENARIO_SEED,
                     "prepared": "data/mci_loan/v2/prepared", "setup": "mpc-d14", "prover": "rapidsnark", "alignmentPeriodBlocks": blocks,
                     "compareWith": f"results/mci_loan/v2/offchain/runs/c2-s{k}.json (off-chain seed {k}, the same model seeds)",
                     "outDir": f"results/mci_loan/v2/onchain/run-{k}",
                     "command": f"caffeinate -ims bash scenarios/mci_loan/v2/onchain/mac-runs-v2.sh run {k} {blocks}"})
    plan = {"what": "the MCI loan instance v2's five on-chain runs, fixed before run 1 (scenarios/mci_loan/v2/onchain_plan2.py)",
            "preregistration": "docs/mci/v2/02-preregistration.md section 10 (tag mci-prereg-v2)",
            "members": c2.slots()["members"],
            "alignmentBenchmark": {"file": "results/mci_loan/v2/onchain/psi-bench.json", "slowestAlignmentS": bench["slowestAlignmentS"],
                                   "rule": bench["alignmentPeriod"]["rule"], "machine": bench["machine"]},
            "runs": runs}
    cm.write_json(OUT / "plan.json", plan, indent=1)
    print(json.dumps({"alignmentPeriodBlocks": blocks, "slowestAlignmentS": bench["slowestAlignmentS"], "runs": len(runs)}))


if __name__ == "__main__":
    main()
