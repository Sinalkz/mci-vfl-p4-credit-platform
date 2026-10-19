"""MCI loan instance: the passives' per-applicant aggregates (party_rule.PASSIVE_COLUMNS), raw values before transform.

For a set of applicants (SK_ID_CURR), each source table is read in chunks and filtered to them, then aggregated per
applicant. An applicant with no row in a table gets NaN in that table's aggregates and 0 in its has_record indicator
(party_rule.HAS_RECORD); prep.py applies the transforms, the standardization and the neutral fill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import party_rule as pr
from common import RAW

CHUNK = 2_000_000


def read_filtered(table: str, usecols: list[str], key: str, keep) -> pd.DataFrame:
    keep = pd.Index(keep)
    parts = [c[c[key].isin(keep)] for c in pd.read_csv(RAW / f"{table}.csv", usecols=usecols, chunksize=CHUNK)]
    return pd.concat(parts, ignore_index=True)


def share(mask: pd.Series, by: pd.Series) -> pd.Series:
    return mask.astype(float).groupby(by).mean()


def installments(ids) -> pd.DataFrame:
    d = read_filtered("installments_payments", ["SK_ID_CURR", *pr.INS], "SK_ID_CURR", ids)
    g = d["SK_ID_CURR"]
    paid = d["DAYS_ENTRY_PAYMENT"].notna()
    late_days = d["DAYS_ENTRY_PAYMENT"] - d["DAYS_INSTALMENT"]
    due = d["AMT_INSTALMENT"] > 0
    ratio = (d["AMT_PAYMENT"] / d["AMT_INSTALMENT"]).where(due).clip(0, 2)
    with_pay = d["AMT_PAYMENT"].notna()
    return pd.DataFrame({
        "INS_N": g.groupby(g).size().astype(float),
        "INS_N_PREV": d.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique().astype(float),
        "INS_LATE_SHARE": (late_days > 0).where(paid).astype(float).groupby(g).mean(),
        "INS_DPD_MEAN": late_days.clip(lower=0).groupby(g).mean(),
        "INS_DPD_MAX": late_days.clip(lower=0).groupby(g).max(),
        "INS_EARLY_MEAN": (-late_days).clip(lower=0).groupby(g).mean(),
        "INS_UNPAID_SHARE": share(~paid, g),
        "INS_UNDERPAID_SHARE": (d["AMT_PAYMENT"] < d["AMT_INSTALMENT"]).where(with_pay).astype(float).groupby(g).mean(),
        "INS_PAY_RATIO_MEAN": ratio.groupby(g).mean(),
        "INS_AMT_MEAN": d["AMT_INSTALMENT"].groupby(g).mean(),
        "INS_LAST_DAYS": d["DAYS_INSTALMENT"].groupby(g).max(),
    })


def previous(ids) -> pd.DataFrame:
    cols = ["SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "NAME_PORTFOLIO", "AMT_APPLICATION",
            "AMT_CREDIT", "RATE_DOWN_PAYMENT", "CNT_PAYMENT", "DAYS_DECISION"]
    d = read_filtered("previous_application", cols, "SK_ID_CURR", ids)
    g = d["SK_ID_CURR"]
    asked = d["AMT_APPLICATION"] > 0
    out = {"PREV_N": g.groupby(g).size().astype(float)}
    for s, v in [("APPROVED", "Approved"), ("REFUSED", "Refused"), ("CANCELED", "Canceled")]:
        out[f"PREV_{s}_SHARE"] = share(d["NAME_CONTRACT_STATUS"] == v, g)
    for s, v in [("CONSUMER", "Consumer loans"), ("CASH", "Cash loans"), ("REVOLVING", "Revolving loans")]:
        out[f"PREV_{s}_SHARE"] = share(d["NAME_CONTRACT_TYPE"] == v, g)
    out["PREV_POS_SHARE"] = share(d["NAME_PORTFOLIO"] == "POS", g)
    out["PREV_AMT_APPLICATION_MEAN"] = d["AMT_APPLICATION"].groupby(g).mean()
    out["PREV_CREDIT_TO_APP_MEAN"] = (d["AMT_CREDIT"] / d["AMT_APPLICATION"]).where(asked).clip(0, 3).groupby(g).mean()
    out["PREV_DOWN_RATE_MEAN"] = d["RATE_DOWN_PAYMENT"].groupby(g).mean()
    out["PREV_CNT_PAYMENT_MEAN"] = d["CNT_PAYMENT"].groupby(g).mean()
    out["PREV_DAYS_DECISION_MAX"] = d["DAYS_DECISION"].groupby(g).max()
    out["PREV_DAYS_DECISION_MEAN"] = d["DAYS_DECISION"].groupby(g).mean()
    return pd.DataFrame(out)


def pos_cash(ids) -> pd.DataFrame:
    cols = ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"]
    d = read_filtered("POS_CASH_balance", cols, "SK_ID_CURR", ids)
    g = d["SK_ID_CURR"]
    return pd.DataFrame({
        "POS_N_MONTHS": g.groupby(g).size().astype(float),
        "POS_N_PREV": d.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique().astype(float),
        "POS_DPD_SHARE": share(d["SK_DPD"] > 0, g),
        "POS_DPD_DEF_SHARE": share(d["SK_DPD_DEF"] > 0, g),
        "POS_DPD_MAX": d["SK_DPD"].groupby(g).max(),
        "POS_FUTURE_MEAN": d["CNT_INSTALMENT_FUTURE"].groupby(g).mean(),
        "POS_COMPLETED_SHARE": share(d["NAME_CONTRACT_STATUS"] == "Completed", g),
    })


def credit_card(ids) -> pd.DataFrame:
    cols = ["SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT",
            "AMT_DRAWINGS_POS_CURRENT", "AMT_DRAWINGS_ATM_CURRENT", "AMT_PAYMENT_CURRENT", "AMT_INST_MIN_REGULARITY", "SK_DPD"]
    d = read_filtered("credit_card_balance", cols, "SK_ID_CURR", ids)
    g = d["SK_ID_CURR"]
    util = (d["AMT_BALANCE"] / d["AMT_CREDIT_LIMIT_ACTUAL"]).where(d["AMT_CREDIT_LIMIT_ACTUAL"] > 0).clip(0, 2)
    to_min = (d["AMT_PAYMENT_CURRENT"] / d["AMT_INST_MIN_REGULARITY"]).where(d["AMT_INST_MIN_REGULARITY"] > 0).clip(0, 5)
    return pd.DataFrame({
        "CC_N_MONTHS": g.groupby(g).size().astype(float),
        "CC_BALANCE_MEAN": d["AMT_BALANCE"].groupby(g).mean(),
        "CC_UTIL_MEAN": util.groupby(g).mean(),
        "CC_UTIL_MAX": util.groupby(g).max(),
        "CC_DRAWINGS_MEAN": d["AMT_DRAWINGS_CURRENT"].clip(lower=0).groupby(g).mean(),
        "CC_POS_DRAWINGS_MEAN": d["AMT_DRAWINGS_POS_CURRENT"].clip(lower=0).groupby(g).mean(),
        "CC_ATM_DRAWINGS_MEAN": d["AMT_DRAWINGS_ATM_CURRENT"].clip(lower=0).groupby(g).mean(),
        "CC_PAY_TO_MIN_MEAN": to_min.groupby(g).mean(),
        "CC_DPD_SHARE": share(d["SK_DPD"] > 0, g),
        "CC_DPD_MAX": d["SK_DPD"].groupby(g).max(),
    })


def bureau(ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE",
            "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
            "CREDIT_TYPE"]
    d = read_filtered("bureau", cols, "SK_ID_CURR", ids)
    g = d["SK_ID_CURR"]
    credit = d["AMT_CREDIT_SUM"].groupby(g).sum()
    debt = d["AMT_CREDIT_SUM_DEBT"].groupby(g).sum()
    out = {
        "BUR_N": g.groupby(g).size().astype(float),
        "BUR_ACTIVE_N": (d["CREDIT_ACTIVE"] == "Active").astype(float).groupby(g).sum(),
        "BUR_CLOSED_SHARE": share(d["CREDIT_ACTIVE"] == "Closed", g),
        "BUR_DAYS_CREDIT_MEAN": d["DAYS_CREDIT"].groupby(g).mean(),
        "BUR_DAYS_CREDIT_MAX": d["DAYS_CREDIT"].groupby(g).max(),
        "BUR_ENDDATE_MEAN": d["DAYS_CREDIT_ENDDATE"].groupby(g).mean(),
        "BUR_OVERDUE_DAYS_MAX": d["CREDIT_DAY_OVERDUE"].groupby(g).max(),
        "BUR_MAX_OVERDUE_AMT_MAX": d["AMT_CREDIT_MAX_OVERDUE"].groupby(g).max(),
        "BUR_PROLONG_SUM": d["CNT_CREDIT_PROLONG"].groupby(g).sum(),
        "BUR_CREDIT_SUM": credit,
        "BUR_DEBT_SUM": debt,
        "BUR_DEBT_RATIO": (debt / credit).where(credit > 0).clip(-1, 2),
        "BUR_OVERDUE_SUM": d["AMT_CREDIT_SUM_OVERDUE"].groupby(g).sum(),
    }
    for s, v in [("CARD", "Credit card"), ("CONSUMER", "Consumer credit"), ("CAR", "Car loan"), ("MORTGAGE", "Mortgage")]:
        out[f"BUR_{s}_SHARE"] = share(d["CREDIT_TYPE"] == v, g)
    bb = read_filtered("bureau_balance", ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"], "SK_ID_BUREAU", d["SK_ID_BUREAU"].unique())
    bb = bb.merge(d[["SK_ID_BUREAU", "SK_ID_CURR"]], on="SK_ID_BUREAU", how="inner")
    h = bb["SK_ID_CURR"]
    st = bb["STATUS"].astype(str)
    bal = pd.DataFrame({
        "BB_N_MONTHS": h.groupby(h).size().astype(float),
        "BB_DPD_SHARE": share(st.isin(list("12345")), h),
        "BB_DPD_SEVERE_SHARE": share(st.isin(list("2345")), h),
        "BB_CLOSED_SHARE": share(st == "C", h),
        "BB_UNKNOWN_SHARE": share(st == "X", h),
    })
    return pd.DataFrame(out), bal


def passive_frame(app: pd.DataFrame) -> pd.DataFrame:
    """Raw passive columns for the applicants in app (application_train rows indexed by SK_ID_CURR), in
    party_rule.PASSIVE_COLUMNS order. NaN = no value; has_record indicators are 0/1."""
    ids = app.index
    bur, bal = bureau(ids)
    tables = {"installments_payments": installments(ids), "previous_application": previous(ids),
              "POS_CASH_balance": pos_cash(ids), "credit_card_balance": credit_card(ids), "bureau": bur, "bureau_balance": bal}
    out = pd.DataFrame(index=ids)
    for table, frame in tables.items():
        frame = frame.reindex(ids)
        out[pr.HAS_RECORD[table]] = ids.isin(tables[table].index).astype(float)
        for c in frame.columns:
            out[c] = frame[c]
    for c in pr.MCI_PHONE:
        out[c] = app[c].astype(float)
    for c in pr.EXT_SOURCES:
        out[f"{c}_HAS_VALUE"] = app[c].notna().astype(float)
        out[c] = app[c].astype(float)
    out["AMT_REQ_HAS_VALUE"] = app[pr.BUREAU_ENQUIRIES].notna().all(axis=1).astype(float)
    for c in pr.BUREAU_ENQUIRIES:
        out[c] = app[c].astype(float)
    names = [c["name"] for c in pr.PASSIVE_COLUMNS]
    missing = [n for n in names if n not in out.columns]
    if missing:
        raise RuntimeError(f"aggregates not computed: {missing}")
    return out[names]
