"""MCI loan instance: off-chain metrics on the score helper's margins (decision (a): off-chain, not ZK-verified).

- AUC: Mann-Whitney over the margin z_1 - z_0 (ties at their average rank).
- Reweighting (decision (b)): the cohort samples defaults and repaid applicants at fixed, different rates; the weight
  w_c = (population share of class c) / (cohort share of class c) undoes that, in any subset of rows. Population: the
  default rate of application_train; cohort: 7,982 / 15,965 defaults.
- Business metric (the task's Step 3 rule): the bank-alone model's approval rate r is fixed: the weighted share of rows
  it predicts as class 0 (repaid). Every model then approves the same weighted share r, taking its rows from the lowest
  margin up (ties in row order), and the result is the weighted default rate among the approved. The same is done inside
  the thin_file segment, with r taken from the bank-alone model inside the segment.
- Secondary business metric (the student's review, 2026-10-19): the bank-alone model's default rate among approved d*
  (the primary metric's value for it) is fixed instead. Every model approves from the lowest margin up as far as the
  weighted default rate among the approved stays at or below d*; the result is that weighted share, and the extra
  approvals are its difference from the bank-alone model's share under the same rule. The same inside thin_file.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

import common as cm


def auc(margin, y) -> float | None:
    margin, y = np.asarray(margin, dtype=float), np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(margin)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def class_weights(population_default_rate: float) -> dict[int, float]:
    s1 = cm.N_DEFAULT / cm.N_COHORT
    return {1: population_default_rate / s1, 0: (1 - population_default_rate) / (1 - s1)}


def approve_share(margin, w, rate: float) -> np.ndarray:
    """A boolean mask: rows taken from the lowest margin up until their weight reaches rate times the total."""
    margin, w = np.asarray(margin, dtype=float), np.asarray(w, dtype=float)
    order = np.lexsort((np.arange(len(margin)), margin))
    cum = np.cumsum(w[order])
    n = int(np.searchsorted(cum, rate * w.sum() - 1e-9) + 1) if rate > 0 else 0
    mask = np.zeros(len(margin), dtype=bool)
    mask[order[:min(n, len(margin))]] = True
    return mask


def approve_at_default_rate(margin, y, w, target: float) -> np.ndarray:
    """A boolean mask: the longest run of rows from the lowest margin up whose weighted default rate is at most target."""
    margin, y, w = np.asarray(margin, dtype=float), np.asarray(y), np.asarray(w, dtype=float)
    order = np.lexsort((np.arange(len(margin)), margin))
    cw = np.cumsum(w[order])
    cb = np.cumsum((w * (y == 1))[order])
    within = np.flatnonzero(cb <= target * cw * (1 + 1e-9) + 1e-12)
    mask = np.zeros(len(margin), dtype=bool)
    if len(within):
        mask[order[:within[-1] + 1]] = True
    return mask


def business(models: dict, y, w, bank: str) -> dict:
    """models: name -> (margins, preds) on the same rows; bank: the bank-alone model's name. Weighted figures: the
    primary metric (approval rate fixed) in models, the secondary one (default rate fixed) in atBankAloneDefaultRate."""
    y, w = np.asarray(y), np.asarray(w, dtype=float)
    rate = float(w[np.asarray(models[bank][1]) == 0].sum() / w.sum())
    out = {"bankAloneModel": bank, "approvalRate": rate, "rows": int(len(y)), "models": {}}
    for name, (margin, _) in models.items():
        ok = approve_share(margin, w, rate)
        wa = w[ok].sum()
        out["models"][name] = {"approvedRows": int(ok.sum()), "approvedWeightShare": float(wa / w.sum()),
                               "defaultRateAmongApproved": float(w[ok & (y == 1)].sum() / wa) if wa > 0 else None}
    base = out["models"][bank]["defaultRateAmongApproved"]
    for v in out["models"].values():
        d = v["defaultRateAmongApproved"]
        v["relativeToBankAlone"] = (d - base) / base if (d is not None and base) else None
    sec = {"defaultRate": base, "models": {}}
    for name, (margin, _) in models.items():
        ok = approve_at_default_rate(margin, y, w, base) if base is not None else np.zeros(len(y), dtype=bool)
        wa = w[ok].sum()
        sec["models"][name] = {"approvedRows": int(ok.sum()), "approvedWeightShare": float(wa / w.sum()),
                               "defaultRateAmongApproved": float(w[ok & (y == 1)].sum() / wa) if wa > 0 else None}
    a_bank = sec["models"][bank]["approvedWeightShare"]
    sec["bankAloneShareMinusApprovalRate"] = a_bank - rate  # the rule's own slack for the bank-alone model (>= 0)
    for v in sec["models"].values():
        v["extraApprovalShare"] = v["approvedWeightShare"] - a_bank
        v["extraPer10000"] = 1e4 * v["extraApprovalShare"]
        v["relativeToBankAlone"] = v["approvedWeightShare"] / a_bank - 1 if a_bank > 0 else None
    out["atBankAloneDefaultRate"] = sec
    return out
