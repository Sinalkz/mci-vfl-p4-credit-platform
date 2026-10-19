"""MCI loan instance: choose the Bank's 8 columns from its candidates, on the TRAINING split only (as data/recs/select_a.py
chooses the power company's 8; the circuit takes exactly 8 columns for the active party).

Sequential forward selection: starting from no columns, each step adds the candidate whose addition gives the highest
mean 5-fold cross-validated macro-F1; ties go to the earlier candidate in party_rule.BANK_CANDIDATES. Model:
HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08), as select_a.py; it handles missing values itself.
Folds: bank.folds over the training rows (seed FOLD_SEED, stratified on TARGET). WoE candidates are fitted inside each
fold: out of fold over the inner folds for the fold's training rows, the fold's full fit for its held-out rows.
Test rows are never read.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/select_bank.py
  -> results/mci_loan/select_bank.json, select_bank.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import bank  # noqa: E402
import common as cm  # noqa: E402
import party_rule as pr  # noqa: E402

OUT = cm.RESULTS / "select_bank.json"
PARAMS = {"max_iter": 300, "learning_rate": 0.08, "random_state": cm.FOLD_SEED}


def training_frame():
    app = pd.read_csv(cm.RAW / "application_train.csv").set_index("SK_ID_CURR", drop=False)
    cohort = cm.draw_cohort(app.index.to_numpy(), app["TARGET"].to_numpy())
    a = app.loc[cohort]
    y = a["TARGET"].to_numpy()
    train_pos, _ = cm.split_positions(cohort, y)
    return a.iloc[train_pos], y[train_pos]


def main() -> None:
    t0 = time.time()
    a, y = training_frame()
    ids = a.index.to_numpy()
    raw = bank.raw_candidates(a)
    fold = bank.folds(ids, y, cm.FOLD_SEED)
    cands = pr.BANK_CANDIDATES
    X = {}
    for k in range(bank.N_FOLDS):
        fit_rows = np.flatnonzero(fold != k)
        X[k], _ = bank.numeric_matrix(raw, y, fit_rows, fold, cands)
    col = {c: j for j, c in enumerate(cands)}

    def cv(cols):
        scores = []
        for k in range(bank.N_FOLDS):
            tr, va = fold != k, fold == k
            m = HistGradientBoostingClassifier(**PARAMS).fit(X[k][tr][:, [col[c] for c in cols]], y[tr])
            scores.append(float(f1_score(y[va], m.predict(X[k][va][:, [col[c] for c in cols]]), average="macro")))
        return float(np.mean(scores)), scores

    chosen, steps, prev = [], [], 0.0
    for step in range(1, pr.BANK_N_COLUMNS + 1):
        ts = time.time()
        results = {c: cv(chosen + [c]) for c in cands if c not in chosen}
        best = max(results, key=lambda c: (results[c][0], -cands.index(c)))
        chosen.append(best)
        mean, folds_ = results[best]
        steps.append({"step": step, "added": best, "cvMacroF1": mean, "foldScores": folds_, "gain": mean - prev,
                      "seconds": round(time.time() - ts, 1), "all": {c: r[0] for c, r in results.items()}})
        prev = mean
        print(json.dumps({"step": step, "added": best, "cvMacroF1": round(mean, 4), "seconds": steps[-1]["seconds"]}), flush=True)
    out = {
        "what": "the Bank's 8 columns: sequential forward selection on the training split (scenarios/mci_loan/select_bank.py)",
        "rows": {"train": int(len(y)), "train_default": int(y.sum())}, "folds": {"k": bank.N_FOLDS, "seed": cm.FOLD_SEED},
        "model": {"class": "sklearn HistGradientBoostingClassifier", **PARAMS}, "metric": "macro-F1, mean over the folds",
        "candidates": cands, "selected": chosen, "steps": steps,
        "seconds": round(time.time() - t0, 1), "freeRamMBAtEnd": cm.free_ram_mb(), "machine": cm.machine(),
    }
    cm.write_json(OUT, out, indent=1)
    with open(OUT.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "added", "cv_macro_f1", "gain", "seconds"])
        for s in steps:
            w.writerow([s["step"], s["added"], f"{s['cvMacroF1']:.6f}", f"{s['gain']:.6f}", s["seconds"]])
    print(json.dumps({"selected": chosen, "out": str(OUT.relative_to(cm.ROOT)), "seconds": out["seconds"]}))


if __name__ == "__main__":
    main()
