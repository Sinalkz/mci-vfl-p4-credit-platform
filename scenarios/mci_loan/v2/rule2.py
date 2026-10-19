"""MCI loan instance v2: the party rule as code, fixed before any v2 model result.

MCI is the lender: it grants credit to its subscribers and must estimate repayment, so it is the active member (label
owner, creates the federation, funds the pool). The student's correction of 2026-10-19:

  MCI (active)        application_train's application, income, employment and phone/contact columns, and TARGET read
                      as repayment of MCI-issued credit; its 8 model columns are chosen from MCI_CANDIDATES by forward
                      selection on the training split (select_mci.py), as v1 chose the Bank's
  Bank                per-applicant aggregates of installments_payments and credit_card_balance
  E-commerce company  per-applicant aggregates of POS_CASH_balance and previous_application
  Credit bureau       v1's, unchanged: bureau, bureau_balance, EXT_SOURCE_1/2/3 and AMT_REQ_CREDIT_BUREAU_*

Owner keys: mci, bank, ecommerce, bureau; scenarios/mci_loan/v2/slots.json maps them to the thesis code's slots
(power, mci, digikala, maker) and to the user-facing names. Every aggregate, transform, exclusion and the neutral fill
for applicants with no record are v1's (scenarios/mci_loan/party_rule.py, features.py); only the owners change.
"""
from __future__ import annotations

import pandas as pd

import bank
import party_rule as pr

# ---- MCI (active) ----------------------------------------------------------------------------------------------

# v1's Bank candidates (application, income, employment, family, region, documents and the derived ratios) plus v1
# MCI's phone and contact columns: every application_train column v1 used outside the Credit bureau's.
MCI_PHONE = pr.MCI_PHONE
PHONE_KIND = {c: "num" if c == "DAYS_LAST_PHONE_CHANGE" else "flag" for c in MCI_PHONE}
MCI_CANDIDATES = pr.BANK_CANDIDATES + MCI_PHONE
MCI_N_COLUMNS = 8  # the circuit's FA (circuits/src/macro_f1_members_var.circom:57)


def kind_of(c: str) -> str:
    return next((k for k, v in pr.BANK_RAW.items() if c in v), None) or PHONE_KIND.get(c) or "derived"


def raw_candidates(app: pd.DataFrame) -> pd.DataFrame:
    """MCI's candidates, one number each (NaN = missing): v1's Bank candidates (bank.raw_candidates; WoE columns as
    strings) and the phone and contact columns as recorded."""
    out = bank.raw_candidates(app)
    for c in MCI_PHONE:
        out[c] = app[c].astype(float)
    return out[MCI_CANDIDATES]


# ---- the passives ----------------------------------------------------------------------------------------------

PASSIVES = ["bank", "ecommerce", "bureau"]  # proposal order: slots mci, digikala, maker
TABLES = {"installments_payments": "bank", "credit_card_balance": "bank", "POS_CASH_balance": "ecommerce",
          "previous_application": "ecommerce", "bureau": "bureau", "bureau_balance": "bureau"}
# v1's passive columns without the phone and contact ones (now MCI's candidates); the rest of application_train's
# passive columns (EXT_SOURCE_*, AMT_REQ_CREDIT_BUREAU_*) stay the Credit bureau's
OWNER_OF_SOURCE = {**TABLES, "application_train": "bureau"}
PASSIVE_COLUMNS = [{**c, "owner": OWNER_OF_SOURCE[c["source"]]} for c in pr.PASSIVE_COLUMNS if c["name"] not in MCI_PHONE]
HAS_RECORD = pr.HAS_RECORD

# ---- application_train columns no one uses: v1's exclusions, unchanged ----------------------------------------

EXCLUDED = pr.EXCLUDED
KEYS = {"applicant ID (the join key; replaced by drawn national IDs on-chain)": ["SK_ID_CURR"],
        "label (MCI's: repayment of MCI-issued credit)": ["TARGET"]}


def application_groups() -> dict[str, list[str]]:
    """Every application_train column's group: mci, bureau, or an excluded / key group."""
    mci_raw = [c for cols in pr.BANK_RAW.values() for c in cols] + pr.DOCUMENT_FLAGS + MCI_PHONE
    return {"mci": mci_raw, "bureau": pr.EXT_SOURCES + pr.BUREAU_ENQUIRIES,
            **{f"excluded: {k}": v for k, v in EXCLUDED.items()}, **{f"key: {k}": v for k, v in KEYS.items()}}
