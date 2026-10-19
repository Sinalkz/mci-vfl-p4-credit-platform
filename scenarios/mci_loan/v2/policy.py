"""MCI loan instance v2: the off-chain credit policy on the score helper (the student's request of 2026-10-19).
Off-chain, not ZK-verified (decision (a)): the circuit proves macro-F1 and nothing else.

For each model (figure), on the test rows:
1. Calibration folds: within each class, the test rows in SHA-256("<CALIB_SEED>:<SK_ID_CURR>") order take folds 0 and 1
   in turn (bank.folds with k = 2). No training row is read: the model was fitted on those, so their margins are more
   separated than a new applicant's.
2. Cross-fitted calibration (the student's review, 2026-10-19: never calibrate and report on the same rows): calibrate
   on one half, report on the other, swap, pool. For the rows of fold f, a logistic regression of TARGET on the margin
   z_1 - z_0 (standardized with the other fold's mean and SD) is fitted by maximum likelihood on the other fold only,
   unweighted. It gives p_bal, a probability of default in the balanced cohort. Every model gets the same folds and
   the same procedure. No row's PD depends on its own label (tests/test_v2.py).
3. Prior correction to application_train's default rate pi:
   logit PD = logit p_bal + logit pi - logit s, with s the fit fold's default share (about 1/2).
4. Tiers on PD, with the cut-offs CUTS: PD < 4%: 4-month credit; 4% <= PD < 12%: 1-month credit; PD >= 12%: no credit.
5. Reported per tier: rows, the weighted share of subscribers, the weighted default rate and the weighted mean PD; also
   for "granted" (both credit tiers), and overall (the weighted mean PD against the weighted default rate, a check of
   the prior correction). The weights are metrics.class_weights, as the business metrics'. The same inside thin_file,
   with the same calibrated PDs.
"""
from __future__ import annotations

import numpy as np

import bank
import common2 as c2

CUTS = (0.04, 0.12)  # PD cut-offs: 1/2 and 3/2 of application_train's default rate (0.0807), rounded
TIERS = [{"key": "credit4m", "name": "4-month credit", "nameFa": "اعتبار چهارماهه", "rule": "PD < 4%"},
         {"key": "credit1m", "name": "1-month credit", "nameFa": "اعتبار یک‌ماهه", "rule": "4% <= PD < 12%"},
         {"key": "none", "name": "no credit", "nameFa": "بدون اعتبار", "rule": "PD >= 12%"}]


def logit(p):
    return np.log(p / (1 - p))


def fit_logistic(x: np.ndarray, y: np.ndarray, tol: float = 1e-12, max_iter: int = 100) -> tuple[float, float]:
    """(a, b) maximizing the likelihood of P(y = 1) = 1 / (1 + exp(-(a + b x))): Newton's method."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-(X @ beta)))
        step = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < tol:
            return float(beta[0]), float(beta[1])
    raise RuntimeError("logistic fit did not converge (separable fold?)")


def calibration_folds(ids, y) -> np.ndarray:
    return bank.folds(np.asarray(ids), np.asarray(y), c2.CALIB_SEED, k=2)


def calibrated_pd(margin, y, fold, pi: float) -> tuple[np.ndarray, list[dict]]:
    """The cross-fitted, prior-corrected PD of every row, and each fold's fit."""
    margin, y = np.asarray(margin, dtype=float), np.asarray(y)
    out = np.empty(len(margin))
    fits = []
    for f in (0, 1):
        fit, app = fold != f, fold == f
        mu, sd = float(margin[fit].mean()), float(margin[fit].std())
        a, b = fit_logistic((margin[fit] - mu) / sd, y[fit].astype(float))
        s = float(y[fit].mean())
        shift = float(logit(pi) - logit(s))
        out[app] = 1 / (1 + np.exp(-(a + b * (margin[app] - mu) / sd + shift)))
        fits.append({"fittedOnFold": 1 - f, "appliedToFold": f, "fitRows": int(fit.sum()), "appliedRows": int(app.sum()),
                     "marginMean": mu, "marginSd": sd, "a": a, "b": b, "fitDefaultShare": s, "priorShift": shift})
    return out, fits


def tier_of(pd_) -> np.ndarray:
    """0: 4-month credit, 1: 1-month credit, 2: no credit."""
    return np.digitize(np.asarray(pd_), CUTS)


def tier_table(pd_, y, w) -> dict:
    pd_, y, w = np.asarray(pd_), np.asarray(y), np.asarray(w, dtype=float)
    t = tier_of(pd_)

    def stats(mask):
        wm = w[mask].sum()
        return {"rows": int(mask.sum()), "defaults": int(y[mask].sum()), "share": float(wm / w.sum()),
                "defaultRate": float((w * y)[mask].sum() / wm) if wm > 0 else None,
                "meanPD": float((w * pd_)[mask].sum() / wm) if wm > 0 else None}

    return {"rows": int(len(y)), "tiers": {T["key"]: stats(t == k) for k, T in enumerate(TIERS)},
            "granted": stats(t < 2), "all": stats(np.ones(len(y), dtype=bool))}


def policy(models: dict, ids, y, w, pi: float, thin) -> dict:
    """models: name -> margins on the test rows (ids, y, w, the thin_file mask on the same rows). Per model: each fold's
    calibration, and the tier table overall and on thin_file."""
    y, w, thin = np.asarray(y), np.asarray(w, dtype=float), np.asarray(thin, dtype=bool)
    fold = calibration_folds(ids, y)
    out = {"cuts": list(CUTS), "tiers": TIERS, "prior": pi, "calibrationSeed": c2.CALIB_SEED,
           "crossFit": "2 folds of the test rows: calibrate on one, report on the other, swap, pool; every model alike",
           "foldRows": [int((fold == f).sum()) for f in (0, 1)], "models": {}}
    for name, margin in models.items():
        pd_, fits = calibrated_pd(margin, y, fold, pi)
        out["models"][name] = {"calibration": fits, "overall": tier_table(pd_, y, w),
                               "thinFile": tier_table(pd_[thin], y[thin], w[thin]) if thin.any() else None}
    return out
