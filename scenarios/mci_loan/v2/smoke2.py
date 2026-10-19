"""MCI loan instance v2, Step 3: the end-to-end smoke run on the 1,000-row sample (data/mci_loan/v2/prepared-smoke/).

As v1's smoke.py, with MCI as the active member:
1. scenarios/mci_loan/v2/offchain/configs2.mjs writes the pre-registered model configurations.
2. One off-chain VFL run, seed 1 (scenarios/mci_loan/offchain/vfl-run.mjs --mode c2, the thesis's training and scoring
   code, reading the v2 folder through MCI_LOAN_PREPARED).
3. The checks:
   - the run completes with the thesis's own checks: no zero-filled batch, every final message received and signed;
   - every figure's head puts no row, training or test, in classes 2-4 (the binary label inside K = 5); a single such
     row fails the smoke run;
   - the off-chain layer works on the test rows and their thin_file segment (off-chain, not ZK-verified): AUC per
     figure; the business metrics with MCI alone (BASELINE) as the baseline (metrics2.business); the credit policy
     (policy.policy: calibrated PD and the three tiers).
Writes results/mci_loan/v2/smoke/c2-s1.json and smoke-check.json.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/v2/smoke2.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import numpy as np  # noqa: E402

import common as cm  # noqa: E402
import common2 as c2  # noqa: E402
import metrics2  # noqa: E402
import policy  # noqa: E402
from smoke import VFL, node  # noqa: E402

OUT = c2.RESULTS2 / "smoke"
CONFIGS = c2.RESULTS2 / "configs"
_n = c2.names()
NAMES = {"FULL": "all members (F_r)", "FULL_J": "all members, joint head (F_j)", "LOO_1": f"all but the {_n['bank']}",
         "LOO_2": f"all but the {_n['ecommerce']}", "LOO_3": f"all but the {_n['bureau']}", "BASELINE": "MCI alone (U_A)"}
BUSINESS_MODELS = ("BASELINE", "FULL", "FULL_J", "LOO_1", "LOO_2", "LOO_3")


def offchain(models: dict, m: dict) -> dict:
    """models: name -> (margins, preds) on the test rows of manifest m. The off-chain layer, overall and on thin_file."""
    y = np.asarray(m["y"])
    test = np.asarray(m["split"]["test_pos"])
    yt = y[test]
    w = np.vectorize(metrics2.class_weights(m["population"]["default_rate"]).get)(yt)
    thin = np.isin(test, np.asarray(m["thin_file_pos"], dtype=int))
    ids = np.asarray(m["ids"])[test]
    seg = {k: (mg[thin], pr[thin]) for k, (mg, pr) in models.items()}
    sub = {k: models[k] for k in BUSINESS_MODELS}
    return {
        "auc": {k: metrics2.auc(models[k][0], yt) for k in models},
        "business": metrics2.business(sub, yt, w, "BASELINE"),
        "policy": policy.policy({k: models[k][0] for k in BUSINESS_MODELS}, ids, yt, w, m["population"]["default_rate"], thin),
        "thinFile": {"rows": int(thin.sum()), "defaults": int(yt[thin].sum()),
                     "auc": {k: metrics2.auc(seg[k][0], yt[thin]) for k in models},
                     "business": metrics2.business({k: seg[k] for k in BUSINESS_MODELS}, yt[thin], w[thin], "BASELINE")
                     if thin.sum() > 0 else None},
    }


def check(run_file: Path, m: dict) -> dict:
    run = json.loads(run_file.read_text(encoding="utf-8"))
    figs = {f["name"]: f for f in run["figures"]}
    models = {k: (np.asarray(f["margins"]), np.asarray(f["preds"])) for k, f in figs.items()}
    unused = {k: f["unusedClass"] for k, f in figs.items()}
    return {
        "run": str(run_file.relative_to(cm.ROOT)).replace("\\", "/"), "configHash": run["configHash"], "widths": run["widths"],
        "rows": run["rows"], "wallMs": run["wallMs"], "peakRssMB": run["peakRssMB"],
        "zeroFilledBatches": len(run["training"]["zeroFilled"]),
        "finalsReceivedAndSigned": all(f["received"] and f["signatureValid"] for f in run["training"]["finals"]),
        "F": {k: f["F"] for k, f in figs.items()},
        "unusedClassRows": unused, "unusedClassTotal": int(sum(u["train"] + u["test"] for u in unused.values())),
        "offchainNotZkVerified": offchain(models, m),
    }


def main() -> None:
    t0 = time.time()
    node([str(HERE / "offchain" / "configs2.mjs")])
    m = json.loads((c2.PREPARED2_SMOKE / "manifest.json").read_text(encoding="utf-8"))
    dst = OUT / "c2-s1.json"
    node([str(VFL), "--mode", "c2", "--seed", "1", "--out", str(dst)],
         {"MCI_LOAN_PREPARED": str(c2.PREPARED2_SMOKE), "VFL_MODEL_CONFIG": str(CONFIGS / "loan2-s1.json")})
    out = {"what": "the v2 Step 3 smoke run on the 1,000-row sample (scenarios/mci_loan/v2/smoke2.py)",
           "sample": {"rows": m["rows"], "train": len(m["split"]["train_pos"]), "test": len(m["split"]["test_pos"]),
                      "classCounts": m["class_counts"], "thinFile": len(m["thin_file_pos"]), "rule": m["smoke"]["rule"]},
           "labels": NAMES, "runs": {"main": check(dst, m)}}
    r = out["runs"]["main"]
    out["pass"] = {"noPredictionInClasses2to4": r["unusedClassTotal"] == 0,
                   "completedWithThesisChecks": r["zeroFilledBatches"] == 0 and r["finalsReceivedAndSigned"]}
    out["seconds"] = round(time.time() - t0, 1)
    out["machine"] = cm.machine()
    cm.write_json(OUT / "smoke-check.json", out, indent=1)
    print(json.dumps({"pass": out["pass"], "F": r["F"], "unusedClassTotal": r["unusedClassTotal"]}))
    if not all(out["pass"].values()):
        sys.exit("smoke check failed")


if __name__ == "__main__":
    main()
