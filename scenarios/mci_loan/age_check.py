"""MCI loan instance, Step 3: the default rate per age band, the evidence for keeping DAYS_BIRTH (docs/mci/02-preregistration.md,
section 5.5): the variable's direction should favour older applicants.

- Age = -DAYS_BIRTH / 365.25 years, in the bands of AGE_BANDS (lower bound included).
- Population: every row of application_train, unweighted.
- Cohort training rows: the default rate reweighted to the population (metrics.class_weights), as the business metrics are.

Output: results/mci_loan/age-check.json.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/age_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as cm  # noqa: E402
import metrics  # noqa: E402

AGE_BANDS = [20, 25, 35, 45, 55, 65, 70]  # years; the last band is 65 up to 70 (application_train's oldest is under 70)
OUT = cm.RESULTS / "age-check.json"


def bands(age: np.ndarray) -> np.ndarray:
    return np.digitize(age, AGE_BANDS[1:-1])


def main() -> None:
    app = pd.read_csv(cm.RAW / "application_train.csv", usecols=["SK_ID_CURR", "TARGET", "DAYS_BIRTH"]).set_index("SK_ID_CURR")
    age = -app["DAYS_BIRTH"].to_numpy() / 365.25
    assert age.min() >= AGE_BANDS[0] and age.max() < AGE_BANDS[-1], (age.min(), age.max())
    y = app["TARGET"].to_numpy()
    m = json.loads((cm.RESULTS / "prepared" / "manifest.json").read_text(encoding="utf-8"))
    tr = np.asarray(m["split"]["train_pos"])
    ids = np.asarray(m["ids"])[tr]
    a_tr, y_tr = -app.loc[ids, "DAYS_BIRTH"].to_numpy() / 365.25, app.loc[ids, "TARGET"].to_numpy()
    w = np.vectorize(metrics.class_weights(m["population"]["default_rate"]).get)(y_tr)
    rows = []
    for k in range(len(AGE_BANDS) - 1):
        p, c = bands(age) == k, bands(a_tr) == k
        rows.append({"band": f"{AGE_BANDS[k]}-{AGE_BANDS[k + 1]}", "populationRows": int(p.sum()),
                     "populationDefaultRate": float(y[p].mean()),
                     "cohortTrainRows": int(c.sum()),
                     "cohortTrainDefaultRateReweighted": float((w[c] * y_tr[c]).sum() / w[c].sum())})
    rates = [r["populationDefaultRate"] for r in rows]
    out = {"what": "default rate per age band (scenarios/mci_loan/age_check.py)", "ageYears": "-DAYS_BIRTH / 365.25",
           "bands": rows, "populationRateFallsWithEveryBand": all(a > b for a, b in zip(rates, rates[1:]))}
    cm.write_json(OUT, out, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
