"""MCI loan instance, Step 3: the end-to-end smoke run on the 1,000-row sample (data/mci_loan/prepared-smoke/).

1. scenarios/mci_loan/offchain/configs.mjs writes the pre-registered model configurations.
2. Two off-chain VFL runs, seed 1 (scenarios/mci_loan/offchain/vfl-run.mjs --mode c2, the thesis's training and
   scoring code): the main split (loan-s1.json) and the sensitivity variant (loan-mcisens-s1.json with
   partition-mcisens.json: MCI with its phone and contact columns only).
3. The checks:
   - the run completes with the thesis's own checks: no zero-filled batch, every final message received and signed;
   - every figure's head puts no row, training or test, in classes 2-4 (the binary label inside K = 5: decision
     point 3 of docs/mci/00-recon.md); a single such row fails the smoke run;
   - the off-chain score works: AUC per figure, and the business metric (bank alone = BASELINE, all members = FULL,
     all but MCI = LOO_1; FULL_J alongside), weighted to the population default rate, on the test rows and on their
     thin_file segment (off-chain, not ZK-verified).
Writes results/mci_loan/smoke/c2-s1.json, c2-mcisens-s1.json and smoke-check.json.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/smoke.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import common as cm  # noqa: E402
import metrics  # noqa: E402

OUT = cm.RESULTS / "smoke"
CONFIGS = cm.RESULTS / "configs"
VFL = cm.HERE / "offchain" / "vfl-run.mjs"
NAMES = {"FULL": "all members (F_r)", "FULL_J": "all members, joint head (F_j)", "LOO_1": "all but MCI",
         "LOO_2": "all but the E-commerce company", "LOO_3": "all but the credit bureau", "BASELINE": "Bank alone (U_A)"}


def node(args: list[str], env: dict | None = None) -> None:
    exe = shutil.which("node") or sys.exit("node not found on PATH")
    r = subprocess.run([exe, *args], cwd=cm.ROOT, env={**os.environ, **(env or {})})
    if r.returncode != 0:
        sys.exit(f"node {' '.join(args)} exited {r.returncode}")


def check(run_file: Path, m: dict) -> dict:
    run = json.loads(run_file.read_text(encoding="utf-8"))
    y = np.asarray(m["y"])
    test = np.asarray(m["split"]["test_pos"])
    yt = y[test]
    w = np.vectorize(metrics.class_weights(m["population"]["default_rate"]).get)(yt)
    thin = np.isin(test, np.asarray(m["thin_file_pos"], dtype=int))
    figs = {f["name"]: f for f in run["figures"]}
    models = {n: (np.asarray(f["margins"]), np.asarray(f["preds"])) for n, f in figs.items()}
    unused = {n: f["unusedClass"] for n, f in figs.items()}
    seg = {n: (mg[thin], pr[thin]) for n, (mg, pr) in models.items()}
    return {
        "run": str(run_file.relative_to(cm.ROOT)).replace("\\", "/"), "configHash": run["configHash"], "widths": run["widths"],
        "rows": run["rows"], "wallMs": run["wallMs"], "peakRssMB": run["peakRssMB"],
        "zeroFilledBatches": len(run["training"]["zeroFilled"]),
        "finalsReceivedAndSigned": all(f["received"] and f["signatureValid"] for f in run["training"]["finals"]),
        "F": {n: f["F"] for n, f in figs.items()},
        "unusedClassRows": unused, "unusedClassTotal": int(sum(u["train"] + u["test"] for u in unused.values())),
        "offchainNotZkVerified": {
            "auc": {n: metrics.auc(models[n][0], yt) for n in models},
            "business": metrics.business({k: models[k] for k in ("BASELINE", "FULL", "LOO_1", "FULL_J")}, yt, w, "BASELINE"),
            "thinFile": {"rows": int(thin.sum()), "defaults": int(yt[thin].sum()),
                         "auc": {n: metrics.auc(seg[n][0], yt[thin]) for n in models},
                         "business": metrics.business({k: seg[k] for k in ("BASELINE", "FULL", "LOO_1", "FULL_J")}, yt[thin], w[thin], "BASELINE")
                         if thin.sum() > 0 else None},
        },
    }


def main() -> None:
    t0 = time.time()
    node([str(cm.HERE / "offchain" / "configs.mjs")])
    env = {"MCI_LOAN_PREPARED": str(cm.PREPARED_SMOKE)}
    runs = {"main": (OUT / "c2-s1.json", CONFIGS / "loan-s1.json", []),
            "mciPhoneOnly": (OUT / "c2-mcisens-s1.json", CONFIGS / "loan-mcisens-s1.json",
                             ["--partition", str(CONFIGS / "partition-mcisens.json")])}
    m = json.loads((cm.PREPARED_SMOKE / "manifest.json").read_text(encoding="utf-8"))
    out = {"what": "the Step 3 smoke run on the 1,000-row sample (scenarios/mci_loan/smoke.py)",
           "sample": {"rows": m["rows"], "train": len(m["split"]["train_pos"]), "test": len(m["split"]["test_pos"]),
                      "classCounts": m["class_counts"], "thinFile": len(m["thin_file_pos"]), "rule": m["smoke"]["rule"]},
           "labels": NAMES, "runs": {}}
    for name, (dst, cfg, extra) in runs.items():
        node([str(VFL), "--mode", "c2", "--seed", "1", *extra, "--out", str(dst)], {**env, "VFL_MODEL_CONFIG": str(cfg)})
        out["runs"][name] = check(dst, m)
    out["pass"] = {"noPredictionInClasses2to4": all(r["unusedClassTotal"] == 0 for r in out["runs"].values()),
                   "completedWithThesisChecks": all(r["zeroFilledBatches"] == 0 and r["finalsReceivedAndSigned"] for r in out["runs"].values())}
    out["seconds"] = round(time.time() - t0, 1)
    out["machine"] = cm.machine()
    cm.write_json(OUT / "smoke-check.json", out, indent=1)
    print(json.dumps({"pass": out["pass"], **{k: {"F": r["F"], "unusedClassTotal": r["unusedClassTotal"]} for k, r in out["runs"].items()}}))
    if not all(out["pass"].values()):
        sys.exit("smoke check failed")


if __name__ == "__main__":
    main()
