"""MCI loan instance: the Bank's candidate columns, one number each (party_rule.BANK_RAW, BANK_DERIVED).

- log, num, flag, ordinal and derived candidates come from the applicant's own row (NaN where the source is empty).
- woe candidates (nominal columns) are weight of evidence fitted on training rows only:
  woe(c) = ln((n1_c + 0.5) / (N1 + 0.5 K)) - ln((n0_c + 0.5) / (N0 + 0.5 K)), K = levels seen in the fit, a blank is its
  own level, an unseen level is 0. Rows of the fit set get out-of-fold values (5 folds), other rows the full fit.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

import party_rule as pr

WOE_ALPHA = 0.5
N_FOLDS = 5


def raw_candidates(app: pd.DataFrame) -> pd.DataFrame:
    """Every non-woe candidate as a float column (NaN = missing), and the woe columns as strings ('MISSING' if blank)."""
    out = pd.DataFrame(index=app.index)
    emp = app["DAYS_EMPLOYED"].where(app["DAYS_EMPLOYED"] != pr.DAYS_EMPLOYED_NONE)
    for c in pr.BANK_RAW["log"]:
        out[c] = np.log1p(app[c].clip(lower=0))
    for c in pr.BANK_RAW["num"]:
        out[c] = (emp if c == "DAYS_EMPLOYED" else app[c]).astype(float)
    for c in pr.BANK_RAW["flag"]:
        out[c] = (app[c] == pr.FLAG_TRUE[c]).astype(float) if c in pr.FLAG_TRUE else app[c].astype(float)
    out["NAME_EDUCATION_TYPE"] = app["NAME_EDUCATION_TYPE"].map(pr.EDUCATION_ORDER).astype(float)
    for c in pr.BANK_RAW["woe"]:
        out[c] = app[c].astype("string").fillna("MISSING")
    inc, cred = app["AMT_INCOME_TOTAL"].where(app["AMT_INCOME_TOTAL"] > 0), app["AMT_CREDIT"].where(app["AMT_CREDIT"] > 0)
    out["CREDIT_TO_INCOME"] = np.log1p(app["AMT_CREDIT"] / inc)
    out["ANNUITY_TO_INCOME"] = app["AMT_ANNUITY"] / inc
    out["ANNUITY_TO_CREDIT"] = app["AMT_ANNUITY"] / cred
    out["GOODS_TO_CREDIT"] = app["AMT_GOODS_PRICE"] / cred
    out["EMPLOYED_TO_AGE"] = emp / app["DAYS_BIRTH"]
    out["DOC_COUNT"] = app[pr.DOCUMENT_FLAGS].sum(axis=1).astype(float)
    return out[pr.BANK_CANDIDATES]


def folds(ids: np.ndarray, y: np.ndarray, seed: int, k: int = N_FOLDS) -> np.ndarray:
    """Fold number per row: within each class, rows in SHA-256("<seed>:<id>") order take folds 0, 1, ..., k-1 in turn."""
    out = np.empty(len(ids), dtype=int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        keys = [hashlib.sha256(f"{seed}:{int(i)}".encode()).hexdigest() for i in ids[idx]]
        order = idx[np.argsort(np.asarray(keys), kind="stable")]
        out[order] = np.arange(len(order)) % k
    return out


def woe_fit(cat: pd.Series, y: np.ndarray) -> dict[str, float]:
    t = pd.crosstab(cat.to_numpy(), y)
    n0 = t.get(0, pd.Series(0, index=t.index)).astype(float)
    n1 = t.get(1, pd.Series(0, index=t.index)).astype(float)
    k = len(t.index)
    w = np.log((n1 + WOE_ALPHA) / (n1.sum() + WOE_ALPHA * k)) - np.log((n0 + WOE_ALPHA) / (n0.sum() + WOE_ALPHA * k))
    return {str(lv): float(v) for lv, v in w.items()}


def woe_apply(cat: pd.Series, table: dict[str, float]) -> np.ndarray:
    return cat.map(lambda v: table.get(str(v), 0.0)).to_numpy(dtype=float)


def woe_encode(cat: pd.Series, y: np.ndarray, fit_rows: np.ndarray, fold_of: np.ndarray) -> tuple[np.ndarray, dict]:
    """WoE for every row: out of fold for fit_rows (fold_of[fit_rows] gives their folds), the full fit for the rest."""
    out = np.empty(len(cat))
    full = woe_fit(cat.iloc[fit_rows], y[fit_rows])
    out[:] = woe_apply(cat, full)
    for f in np.unique(fold_of[fit_rows]):
        inner = fit_rows[fold_of[fit_rows] != f]
        target = fit_rows[fold_of[fit_rows] == f]
        out[target] = woe_apply(cat.iloc[target], woe_fit(cat.iloc[inner], y[inner]))
    return out, full


def numeric_matrix(raw: pd.DataFrame, y: np.ndarray, fit_rows: np.ndarray, fold_of: np.ndarray,
                   columns: list[str]) -> tuple[np.ndarray, dict]:
    """The chosen candidate columns as floats (NaN kept), woe columns encoded as woe_encode does; also the woe tables."""
    cols, tables = [], {}
    for c in columns:
        if c in pr.BANK_RAW["woe"]:
            v, tables[c] = woe_encode(raw[c], y, fit_rows, fold_of)
            cols.append(v)
        else:
            cols.append(raw[c].to_numpy(dtype=float))
    return np.column_stack(cols), tables
