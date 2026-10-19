"""MCI loan instance: the party rule as code (the counterpart of data/recs/party_rule.py), fixed before any model result.

Roles over one public dataset (the organisations are simulated). Slots as in the thesis code (decision (e)): the active
party is slot `power` (user-facing: Bank); the passives in proposal order are `mci` (MCI), `digikala` (E-commerce company) and
`maker` (Credit bureau); scenarios/mci_loan/slots.json maps them.

  Bank (active)    financial and application columns of application_train, and TARGET; its 8 model columns are chosen
                   from BANK_CANDIDATES by forward selection on the training split (select_bank.py)
  MCI              the phone and contact columns of application_train, and per-applicant aggregates of
                   installments_payments (framed as bill and top-up payment punctuality)
  E-commerce company per-applicant aggregates of previous_application, POS_CASH_balance and credit_card_balance
                   (purchase and instalment-purchase behaviour)
  Credit bureau    per-applicant aggregates of bureau and bureau_balance, EXT_SOURCE_1/2/3, and the enquiry counts
                   AMT_REQ_CREDIT_BUREAU_* (enquiries to the bureau are its own data: the student's review, 2026-10-19)

Every one of application_train's 122 columns is in exactly one group below (tests/test_party_rule.py). A passive emits
a row for every applicant (decision (c)): where its source table has no record, the aggregates take the neutral fill
(0 after standardization) and the table's has_record indicator is 0.
"""
from __future__ import annotations

# ---- the Bank (active) ---------------------------------------------------------------------------------------

# Candidates, one number each: kind -> columns. log: log1p of a non-negative amount; num: as is; flag: 0/1;
# ordinal: NAME_EDUCATION_TYPE's order; woe: weight of evidence on the training split (out of fold for training
# rows), the usual credit-scoring encoding of a nominal column; derived: from the Bank's own columns only.
BANK_RAW = {
    "log": ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"],
    "num": ["CNT_CHILDREN", "CNT_FAM_MEMBERS", "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED",
            "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
            "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"],
    "flag": ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
             "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
             "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"],
    "ordinal": ["NAME_EDUCATION_TYPE"],
    "woe": ["NAME_INCOME_TYPE", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_TYPE_SUITE",
            "NAME_FAMILY_STATUS"],
}
FLAG_TRUE = {"NAME_CONTRACT_TYPE": "Revolving loans", "FLAG_OWN_CAR": "Y", "FLAG_OWN_REALTY": "Y"}
EDUCATION_ORDER = {"Lower secondary": 0, "Secondary / secondary special": 1, "Incomplete higher": 2,
                   "Higher education": 3, "Academic degree": 4}
DAYS_EMPLOYED_NONE = 365243  # the source's code for "not employed" (pensioners, unemployed): treated as missing
DOCUMENT_FLAGS = [f"FLAG_DOCUMENT_{k}" for k in range(2, 22)]
# derived candidate -> (formula, the raw columns it reads)
BANK_DERIVED = {
    "CREDIT_TO_INCOME": ("log1p(AMT_CREDIT / AMT_INCOME_TOTAL)", ["AMT_CREDIT", "AMT_INCOME_TOTAL"]),
    "ANNUITY_TO_INCOME": ("AMT_ANNUITY / AMT_INCOME_TOTAL", ["AMT_ANNUITY", "AMT_INCOME_TOTAL"]),
    "ANNUITY_TO_CREDIT": ("AMT_ANNUITY / AMT_CREDIT", ["AMT_ANNUITY", "AMT_CREDIT"]),
    "GOODS_TO_CREDIT": ("AMT_GOODS_PRICE / AMT_CREDIT", ["AMT_GOODS_PRICE", "AMT_CREDIT"]),
    "EMPLOYED_TO_AGE": ("DAYS_EMPLOYED / DAYS_BIRTH (DAYS_EMPLOYED 365243 as missing)", ["DAYS_EMPLOYED", "DAYS_BIRTH"]),
    "DOC_COUNT": ("number of FLAG_DOCUMENT_2..21 set", DOCUMENT_FLAGS),
}
BANK_CANDIDATES = [c for cols in BANK_RAW.values() for c in cols] + list(BANK_DERIVED)
BANK_N_COLUMNS = 8  # the circuit's FA (circuits/src/macro_f1_members_var.circom:57)

# ---- the passives ----------------------------------------------------------------------------------------------

PASSIVES = ["mci", "digikala", "bureau"]  # proposal order: slots mci, digikala, maker
MCI_PHONE = ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
             "DAYS_LAST_PHONE_CHANGE"]
EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
BUREAU_ENQUIRIES = [f"AMT_REQ_CREDIT_BUREAU_{p}" for p in ("HOUR", "DAY", "WEEK", "MON", "QRT", "YEAR")]
TABLES = {"installments_payments": "mci", "previous_application": "digikala", "POS_CASH_balance": "digikala",
          "credit_card_balance": "digikala", "bureau": "bureau", "bureau_balance": "bureau"}

# Passive columns: (name, owner, source, transform, source columns, meaning). transform: log = log1p(x >= 0),
# slog = sign(x) log1p(|x|), num = as is, ind = a 0/1 indicator (never standardized). Every non-indicator column is
# standardized on the cohort's training rows that have a value, then 0 where there is none, then clipped to +-5.
P = []


def col(name, owner, source, transform, cols, meaning):
    P.append({"name": name, "owner": owner, "source": source, "transform": transform, "cols": cols, "meaning": meaning})


# MCI: phone and contact (application_train), every applicant has them
for c in MCI_PHONE[:-1]:
    col(c, "mci", "application_train", "ind", [c], "phone/contact flag as recorded")
col("DAYS_LAST_PHONE_CHANGE", "mci", "application_train", "num", ["DAYS_LAST_PHONE_CHANGE"], "days since the last phone change")
# MCI: installments_payments as bill and top-up payment punctuality
INS = ["SK_ID_PREV", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
col("INS_HAS_RECORD", "mci", "installments_payments", "ind", ["SK_ID_CURR"], "any instalment row")
col("INS_N", "mci", "installments_payments", "log", ["SK_ID_CURR"], "instalment rows")
col("INS_N_PREV", "mci", "installments_payments", "log", ["SK_ID_PREV"], "distinct plans paid")
col("INS_LATE_SHARE", "mci", "installments_payments", "num", ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"], "share paid after the due day")
col("INS_DPD_MEAN", "mci", "installments_payments", "log", ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"], "mean days late (0 if on time)")
col("INS_DPD_MAX", "mci", "installments_payments", "log", ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"], "max days late")
col("INS_EARLY_MEAN", "mci", "installments_payments", "log", ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"], "mean days early")
col("INS_UNPAID_SHARE", "mci", "installments_payments", "num", ["DAYS_ENTRY_PAYMENT"], "share with no payment day")
col("INS_UNDERPAID_SHARE", "mci", "installments_payments", "num", ["AMT_INSTALMENT", "AMT_PAYMENT"], "share paid below the amount due")
col("INS_PAY_RATIO_MEAN", "mci", "installments_payments", "num", ["AMT_INSTALMENT", "AMT_PAYMENT"], "mean paid / due, clipped to [0, 2]")
col("INS_AMT_MEAN", "mci", "installments_payments", "log", ["AMT_INSTALMENT"], "mean amount due")
col("INS_LAST_DAYS", "mci", "installments_payments", "num", ["DAYS_INSTALMENT"], "most recent due day (days before the application)")

# E-commerce company (slot digikala): previous_application
col("PREV_HAS_RECORD", "digikala", "previous_application", "ind", ["SK_ID_CURR"], "any previous application")
col("PREV_N", "digikala", "previous_application", "log", ["SK_ID_PREV"], "previous applications")
for s, v in [("APPROVED", "Approved"), ("REFUSED", "Refused"), ("CANCELED", "Canceled")]:
    col(f"PREV_{s}_SHARE", "digikala", "previous_application", "num", ["NAME_CONTRACT_STATUS"], f"share {v}")
for s, v in [("CONSUMER", "Consumer loans"), ("CASH", "Cash loans"), ("REVOLVING", "Revolving loans")]:
    col(f"PREV_{s}_SHARE", "digikala", "previous_application", "num", ["NAME_CONTRACT_TYPE"], f"share {v}")
col("PREV_POS_SHARE", "digikala", "previous_application", "num", ["NAME_PORTFOLIO"], "share in the POS portfolio")
col("PREV_AMT_APPLICATION_MEAN", "digikala", "previous_application", "log", ["AMT_APPLICATION"], "mean amount asked")
col("PREV_CREDIT_TO_APP_MEAN", "digikala", "previous_application", "num", ["AMT_CREDIT", "AMT_APPLICATION"], "mean granted / asked, clipped to [0, 3]")
col("PREV_DOWN_RATE_MEAN", "digikala", "previous_application", "num", ["RATE_DOWN_PAYMENT"], "mean down-payment rate")
col("PREV_CNT_PAYMENT_MEAN", "digikala", "previous_application", "num", ["CNT_PAYMENT"], "mean number of instalments")
col("PREV_DAYS_DECISION_MAX", "digikala", "previous_application", "num", ["DAYS_DECISION"], "most recent decision day")
col("PREV_DAYS_DECISION_MEAN", "digikala", "previous_application", "num", ["DAYS_DECISION"], "mean decision day")
# E-commerce company (slot digikala): POS_CASH_balance
col("POS_HAS_RECORD", "digikala", "POS_CASH_balance", "ind", ["SK_ID_CURR"], "any POS/cash month")
col("POS_N_MONTHS", "digikala", "POS_CASH_balance", "log", ["MONTHS_BALANCE"], "monthly rows")
col("POS_N_PREV", "digikala", "POS_CASH_balance", "log", ["SK_ID_PREV"], "distinct plans")
col("POS_DPD_SHARE", "digikala", "POS_CASH_balance", "num", ["SK_DPD"], "share of months past due")
col("POS_DPD_DEF_SHARE", "digikala", "POS_CASH_balance", "num", ["SK_DPD_DEF"], "share of months past due with tolerance")
col("POS_DPD_MAX", "digikala", "POS_CASH_balance", "log", ["SK_DPD"], "max days past due")
col("POS_FUTURE_MEAN", "digikala", "POS_CASH_balance", "num", ["CNT_INSTALMENT_FUTURE"], "mean instalments left")
col("POS_COMPLETED_SHARE", "digikala", "POS_CASH_balance", "num", ["NAME_CONTRACT_STATUS"], "share of months marked Completed")
# E-commerce company (slot digikala): credit_card_balance
col("CC_HAS_RECORD", "digikala", "credit_card_balance", "ind", ["SK_ID_CURR"], "any card month")
col("CC_N_MONTHS", "digikala", "credit_card_balance", "log", ["MONTHS_BALANCE"], "monthly rows")
col("CC_BALANCE_MEAN", "digikala", "credit_card_balance", "slog", ["AMT_BALANCE"], "mean balance")
col("CC_UTIL_MEAN", "digikala", "credit_card_balance", "num", ["AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"], "mean balance / limit, clipped to [0, 2]")
col("CC_UTIL_MAX", "digikala", "credit_card_balance", "num", ["AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"], "max balance / limit, clipped to [0, 2]")
col("CC_DRAWINGS_MEAN", "digikala", "credit_card_balance", "log", ["AMT_DRAWINGS_CURRENT"], "mean drawings")
col("CC_POS_DRAWINGS_MEAN", "digikala", "credit_card_balance", "log", ["AMT_DRAWINGS_POS_CURRENT"], "mean purchase drawings")
col("CC_ATM_DRAWINGS_MEAN", "digikala", "credit_card_balance", "log", ["AMT_DRAWINGS_ATM_CURRENT"], "mean cash drawings")
col("CC_PAY_TO_MIN_MEAN", "digikala", "credit_card_balance", "num", ["AMT_PAYMENT_CURRENT", "AMT_INST_MIN_REGULARITY"], "mean paid / minimum due, clipped to [0, 5]")
col("CC_DPD_SHARE", "digikala", "credit_card_balance", "num", ["SK_DPD"], "share of months past due")
col("CC_DPD_MAX", "digikala", "credit_card_balance", "log", ["SK_DPD"], "max days past due")

# Credit bureau: bureau
col("BUR_HAS_RECORD", "bureau", "bureau", "ind", ["SK_ID_CURR"], "any bureau credit")
col("BUR_N", "bureau", "bureau", "log", ["SK_ID_BUREAU"], "bureau credits")
col("BUR_ACTIVE_N", "bureau", "bureau", "log", ["CREDIT_ACTIVE"], "active credits")
col("BUR_CLOSED_SHARE", "bureau", "bureau", "num", ["CREDIT_ACTIVE"], "share closed")
col("BUR_DAYS_CREDIT_MEAN", "bureau", "bureau", "num", ["DAYS_CREDIT"], "mean days since the credit was granted")
col("BUR_DAYS_CREDIT_MAX", "bureau", "bureau", "num", ["DAYS_CREDIT"], "most recent credit")
col("BUR_ENDDATE_MEAN", "bureau", "bureau", "num", ["DAYS_CREDIT_ENDDATE"], "mean planned end day")
col("BUR_OVERDUE_DAYS_MAX", "bureau", "bureau", "log", ["CREDIT_DAY_OVERDUE"], "max days overdue now")
col("BUR_MAX_OVERDUE_AMT_MAX", "bureau", "bureau", "log", ["AMT_CREDIT_MAX_OVERDUE"], "largest overdue amount so far")
col("BUR_PROLONG_SUM", "bureau", "bureau", "log", ["CNT_CREDIT_PROLONG"], "prolongations")
col("BUR_CREDIT_SUM", "bureau", "bureau", "log", ["AMT_CREDIT_SUM"], "total credit")
col("BUR_DEBT_SUM", "bureau", "bureau", "slog", ["AMT_CREDIT_SUM_DEBT"], "total debt")
col("BUR_DEBT_RATIO", "bureau", "bureau", "num", ["AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"], "total debt / total credit, clipped to [-1, 2]")
col("BUR_OVERDUE_SUM", "bureau", "bureau", "log", ["AMT_CREDIT_SUM_OVERDUE"], "total overdue now")
for s, v in [("CARD", "Credit card"), ("CONSUMER", "Consumer credit"), ("CAR", "Car loan"), ("MORTGAGE", "Mortgage")]:
    col(f"BUR_{s}_SHARE", "bureau", "bureau", "num", ["CREDIT_TYPE"], f"share {v}")
# Credit bureau: bureau_balance (through SK_ID_BUREAU)
col("BB_HAS_RECORD", "bureau", "bureau_balance", "ind", ["SK_ID_BUREAU"], "any monthly bureau status")
col("BB_N_MONTHS", "bureau", "bureau_balance", "log", ["MONTHS_BALANCE"], "monthly statuses")
col("BB_DPD_SHARE", "bureau", "bureau_balance", "num", ["STATUS"], "share of months with status 1-5 (past due)")
col("BB_DPD_SEVERE_SHARE", "bureau", "bureau_balance", "num", ["STATUS"], "share of months with status 2-5 (over 30 days)")
col("BB_CLOSED_SHARE", "bureau", "bureau_balance", "num", ["STATUS"], "share of months with status C")
col("BB_UNKNOWN_SHARE", "bureau", "bureau_balance", "num", ["STATUS"], "share of months with status X")
# Credit bureau: the external scores (application_train)
for c in EXT_SOURCES:
    col(f"{c}_HAS_VALUE", "bureau", "application_train", "ind", [c], f"{c} present")
    col(c, "bureau", "application_train", "num", [c], "external credit score")
# Credit bureau: the enquiry counts (application_train); in the cohort the six are missing together, so one indicator
col("AMT_REQ_HAS_VALUE", "bureau", "application_train", "ind", BUREAU_ENQUIRIES, "enquiry counts present")
for c, span in zip(BUREAU_ENQUIRIES, ("hour", "day", "week", "month", "quarter", "year")):
    col(c, "bureau", "application_train", "log", [c], f"enquiries to the bureau in the {span} before the application")

PASSIVE_COLUMNS = P
HAS_RECORD = {"installments_payments": "INS_HAS_RECORD", "previous_application": "PREV_HAS_RECORD",
              "POS_CASH_balance": "POS_HAS_RECORD", "credit_card_balance": "CC_HAS_RECORD",
              "bureau": "BUR_HAS_RECORD", "bureau_balance": "BB_HAS_RECORD"}

# Sensitivity variant (Step 3 rule): MCI holds only its phone and contact columns; its installment aggregates are
# held by no one; the E-commerce company and the bureau are unchanged.
MCI_SENSITIVITY_KEEP = MCI_PHONE

# ---- application_train columns no one uses ----------------------------------------------------------------------

BUILDING = [f"{m}_{s}" for s in ("AVG", "MODE", "MEDI") for m in (
    "APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD", "COMMONAREA", "ELEVATORS", "ENTRANCES",
    "FLOORSMAX", "FLOORSMIN", "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA", "NONLIVINGAPARTMENTS", "NONLIVINGAREA")] + [
    "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "TOTALAREA_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"]
EXCLUDED = {
    "protected attribute (sex): not a legitimate credit factor (the pre-registration's section 5.5)": ["CODE_GENDER"],
    "normalised building statistics (AVG / MODE / MEDI versions of the same 14 measures, and 5 building descriptors): "
    "left out to keep the Bank's selection to one column per concept": BUILDING,
    "when the application was filed, not about the applicant": ["WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START"],
}
KEYS = {"applicant ID (the join key; replaced by drawn national IDs on-chain)": ["SK_ID_CURR"],
        "label (the Bank's)": ["TARGET"]}


def application_groups() -> dict[str, list[str]]:
    """Every application_train column's group: bank, mci, bureau, or an excluded / key group."""
    bank_raw = [c for cols in BANK_RAW.values() for c in cols] + DOCUMENT_FLAGS
    return {"bank": bank_raw, "mci": MCI_PHONE, "bureau": EXT_SOURCES + BUREAU_ENQUIRIES,
            **{f"excluded: {k}": v for k, v in EXCLUDED.items()}, **{f"key: {k}": v for k, v in KEYS.items()}}
