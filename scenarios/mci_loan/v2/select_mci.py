"""MCI loan instance v2: choose MCI's 8 columns from its candidates (rule2.MCI_CANDIDATES), on the TRAINING split only.

v1's procedure (scenarios/mci_loan/select_bank.py), unchanged except for the candidate list: sequential forward
selection, each step adding the candidate whose addition gives the highest mean 5-fold cross-validated macro-F1, ties
to the earlier candidate; HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08); folds bank.folds over the
training rows (seed FOLD_SEED, stratified on TARGET); WoE fitted inside each fold. Test rows are never read.

One rule is added (proposed in docs/mci/v2/02-preregistration.md, section 5.1, for the student's decision): from step 2
on, a candidate whose addition changes no prediction in any fold is skipped. Such a column gives exactly the previous
step's score, so v1's rule would pick it at any step where every informative candidate lowers the score. In v2 two
phone columns are such no-ops: FLAG_MOBIL is 1 on every training row, and FLAG_CONT_MOBILE is 0 on 22, fewer than a
fold's leaf minimum (min_samples_leaf 20). --literal runs v1's rule as it is, for the record.

Usage (from the repo root):  .venv/Scripts/python -I scenarios/mci_loan/v2/select_mci.py [--literal]
  -> results/mci_loan/v2/select_mci.json, select_mci.csv (--literal: select_mci-literal.json, .csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import bank  # noqa: E402
import common as cm  # noqa: E402
import common2 as c2  # noqa: E402
import rule2  # noqa: E402
from select_bank import PARAMS, training_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--literal", action="store_true", help="v1's rule as it is: no-op candidates are not skipped")
    literal = ap.parse_args().literal
    out_path = c2.RESULTS2 / ("select_mci-literal.json" if literal else "select_mci.json")
    t0 = time.time()
    a, y = training_frame()
    ids = a.index.to_numpy()
    raw = rule2.raw_candidates(a)
    fold = bank.folds(ids, y, cm.FOLD_SEED)
    cands = rule2.MCI_CANDIDATES
    X = {}
    for k in range(bank.N_FOLDS):
        X[k], _ = bank.numeric_matrix(raw, y, np.flatnonzero(fold != k), fold, cands)
    col = {c: j for j, c in enumerate(cands)}
    # the least common value's rows on the training split, for candidates with at most two values (the no-op rule's evidence)
    few = {c: {"values": int(v.nunique(dropna=False)), "leastCommonRows": int(v.value_counts(dropna=False).min()) if v.nunique(dropna=False) > 1 else 0}
           for c in cands if c not in bank.pr.BANK_RAW["woe"] for v in [raw[c]] if v.nunique(dropna=False) <= 2}

    def cv(cols):
        scores, preds = [], []
        for k in range(bank.N_FOLDS):
            tr, va = fold != k, fold == k
            m = HistGradientBoostingClassifier(**PARAMS).fit(X[k][tr][:, [col[c] for c in cols]], y[tr])
            preds.append(m.predict(X[k][va][:, [col[c] for c in cols]]))
            scores.append(float(f1_score(y[va], preds[-1], average="macro")))
        return float(np.mean(scores)), scores, preds

    chosen, steps, prev, prev_preds = [], [], 0.0, None
    for step in range(1, rule2.MCI_N_COLUMNS + 1):
        ts = time.time()
        results = {c: cv(chosen + [c]) for c in cands if c not in chosen}
        noop = [] if prev_preds is None else [c for c, r in results.items()
                                              if all(np.array_equal(p, q) for p, q in zip(r[2], prev_preds))]
        eligible = list(results) if literal else [c for c in results if c not in noop]
        best = max(eligible, key=lambda c: (results[c][0], -cands.index(c)))
        chosen.append(best)
        mean, folds_, prev_preds = results[best]
        steps.append({"step": step, "added": best, "cvMacroF1": mean, "foldScores": folds_, "gain": mean - prev,
                      "noOp": noop, "addedIsNoOp": best in noop,
                      "seconds": round(time.time() - ts, 1), "all": {c: r[0] for c, r in results.items()}})
        prev = mean
        print(json.dumps({"step": step, "added": best, "cvMacroF1": round(mean, 4), "noOp": noop,
                          "seconds": steps[-1]["seconds"]}), flush=True)
    out = {
        "what": "MCI's 8 columns: sequential forward selection on the training split (scenarios/mci_loan/v2/select_mci.py"
                + (" --literal: v1's rule, no-op candidates not skipped)" if literal else "; no-op candidates skipped from step 2)"),
        "skipNoOp": not literal,
        "rows": {"train": int(len(y)), "train_default": int(y.sum())}, "folds": {"k": bank.N_FOLDS, "seed": cm.FOLD_SEED},
        "model": {"class": "sklearn HistGradientBoostingClassifier", **PARAMS,
                  "min_samples_leaf": HistGradientBoostingClassifier(**PARAMS).get_params()["min_samples_leaf"]},
        "metric": "macro-F1, mean over the folds", "twoValuedCandidates": few,
        "candidates": cands, "selected": chosen, "steps": steps,
        "seconds": round(time.time() - t0, 1), "freeRamMBAtEnd": cm.free_ram_mb(), "machine": cm.machine(),
    }
    cm.write_json(out_path, out, indent=1)
    with open(out_path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "added", "cv_macro_f1", "gain", "added_is_noop", "noop_candidates", "seconds"])
        for s in steps:
            w.writerow([s["step"], s["added"], f"{s['cvMacroF1']:.6f}", f"{s['gain']:.6f}", int(s["addedIsNoOp"]),
                        " ".join(s["noOp"]), s["seconds"]])
    print(json.dumps({"selected": chosen, "out": str(out_path.relative_to(cm.ROOT)), "seconds": out["seconds"]}))


if __name__ == "__main__":
    main()
