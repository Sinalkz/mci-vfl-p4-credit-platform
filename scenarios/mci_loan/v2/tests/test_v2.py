"""MCI loan instance v2: the party rule covers every column once, the slot config follows the naming rule, and the
credit policy and the renamed business metrics keep their rules.

Run (from the repo root):  .venv/Scripts/python -m pytest -q scenarios/mci_loan/v2/tests
The data-dependent test skips when data/raw/home-credit/ is absent.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as cm  # noqa: E402
import common2 as c2  # noqa: E402
import metrics2  # noqa: E402
import party_rule as pr  # noqa: E402
import policy  # noqa: E402
import rule2  # noqa: E402

HAVE_DATA = (cm.RAW / "application_train.csv").exists()


@pytest.mark.skipif(not HAVE_DATA, reason="no data/raw/home-credit")
def test_every_application_column_in_exactly_one_group():
    import pandas as pd
    header = list(pd.read_csv(cm.RAW / "application_train.csv", nrows=0).columns)
    seen = Counter(c for cols in rule2.application_groups().values() for c in cols)
    assert sorted(seen) == sorted(header), set(seen) ^ set(header)
    assert all(n == 1 for n in seen.values()), [c for c, n in seen.items() if n > 1]


def test_passive_columns_owned_by_table():
    names = [c["name"] for c in rule2.PASSIVE_COLUMNS]
    assert len(names) == len(set(names))
    assert {c["owner"] for c in rule2.PASSIVE_COLUMNS} == set(rule2.PASSIVES) == {"bank", "ecommerce", "bureau"}
    for table, owner in rule2.TABLES.items():
        assert all(c["owner"] == owner for c in rule2.PASSIVE_COLUMNS if c["source"] == table)
        assert rule2.HAS_RECORD[table] in names
    assert {t for t, o in rule2.TABLES.items() if o == "bank"} == {"installments_payments", "credit_card_balance"}
    assert {t for t, o in rule2.TABLES.items() if o == "ecommerce"} == {"POS_CASH_balance", "previous_application"}
    # the Credit bureau's columns are v1's, unchanged
    v1_bureau = [c for c in pr.PASSIVE_COLUMNS if c["owner"] == "bureau"]
    assert [c for c in rule2.PASSIVE_COLUMNS if c["owner"] == "bureau"] == v1_bureau


def test_mci_candidates():
    assert len(rule2.MCI_CANDIDATES) == len(set(rule2.MCI_CANDIDATES)) and rule2.MCI_N_COLUMNS == 8
    assert set(rule2.MCI_PHONE) <= set(rule2.MCI_CANDIDATES)
    assert not set(rule2.MCI_CANDIDATES) & {c["name"] for c in rule2.PASSIVE_COLUMNS}
    assert all(rule2.kind_of(c) in ("log", "num", "flag", "ordinal", "woe", "derived") for c in rule2.MCI_CANDIDATES)


def test_slots_follow_the_naming_rule():
    members = c2.slots()["members"]
    assert [s["slot"] for s in members] == ["power", "mci", "digikala", "maker"]   # the thesis code's slots, unchanged
    assert [s["owner"] for s in members] == ["mci", *rule2.PASSIVES]
    assert members[0]["role"] == "active" and all(s["role"] == "passive" for s in members[1:])
    assert c2.names() == {"mci": "MCI", "bank": "Bank", "ecommerce": "E-commerce company", "bureau": "Credit bureau"}
    assert c2.names("fa") == {"mci": "همراه اول", "bank": "بانک", "ecommerce": "شرکت تجارت الکترونیک", "bureau": "شرکت اعتبارسنجی"}


def test_logistic_fit_and_prior_correction():
    rng = np.random.default_rng(11)
    x = rng.normal(size=20_000)
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-(0.3 + 1.2 * x)))).astype(float)
    a, b = policy.fit_logistic(x, y)
    assert abs(a - 0.3) < 0.06 and abs(b - 1.2) < 0.06
    # a balanced sample, prior-corrected to pi: the weighted mean PD comes back near pi
    yb = np.repeat([0, 1], 2_000)
    margin = yb + rng.normal(0, 1.0, len(yb))
    ids = np.arange(500_000, 500_000 + len(yb))
    fold = policy.calibration_folds(ids, yb)
    assert [int((fold == f).sum()) for f in (0, 1)] == [2_000, 2_000]
    pd_, fits = policy.calibrated_pd(margin, yb, fold, 0.08)
    w = np.where(yb == 1, 0.08 / 0.5, 0.92 / 0.5)
    assert abs((w * pd_).sum() / w.sum() - 0.08) < 0.01
    assert all(abs(f["fitDefaultShare"] - 0.5) < 1e-12 for f in fits)


def test_calibration_never_reports_on_its_own_rows():
    rng = np.random.default_rng(5)
    y = np.repeat([0, 1], 400)
    margin = y + rng.normal(0, 1.0, len(y))
    ids = np.arange(700_000, 700_000 + len(y))
    fold = policy.calibration_folds(ids, y)
    pd0, fits = policy.calibrated_pd(margin, y, fold, 0.08)
    assert [(f["fittedOnFold"], f["appliedToFold"]) for f in fits] == [(1, 0), (0, 1)]
    for f in (0, 1):
        rows = np.flatnonzero(fold == f)
        # flip the labels and move the margins of fold f's other rows: fold f's own PDs do not change
        y2, m2 = y.copy(), margin.copy()
        y2[rows[1:]] = 1 - y2[rows[1:]]
        m2[rows[1:]] += rng.normal(0, 3.0, len(rows) - 1)
        pd2, _ = policy.calibrated_pd(m2, y2, fold, 0.08)
        assert pd2[rows[0]] == pd0[rows[0]]
        # and a row's own label never enters its PD
        y3 = y.copy()
        y3[rows] = 1 - y3[rows]
        pd3, _ = policy.calibrated_pd(margin, y3, fold, 0.08)
        assert np.array_equal(pd3[rows], pd0[rows])


def test_tiers():
    assert policy.tier_of([0.0, 0.0399, 0.04, 0.1199, 0.12, 0.9]).tolist() == [0, 0, 1, 1, 2, 2]
    pd_ = np.array([0.01, 0.05, 0.2, 0.3])
    y = np.array([0, 0, 1, 1])
    t = policy.tier_table(pd_, y, np.ones(4))
    assert t["tiers"]["credit4m"]["share"] == 0.25 and t["tiers"]["none"]["defaultRate"] == 1.0
    assert t["granted"]["rows"] == 2 and t["granted"]["defaultRate"] == 0.0
    assert sum(v["share"] for v in t["tiers"].values()) == pytest.approx(1.0)


def test_business_keys_renamed():
    rng = np.random.default_rng(7)
    y = np.repeat([0, 1], 300)
    w = np.vectorize(metrics2.class_weights(0.08).get)(y)
    s = y + rng.normal(0, 0.8, len(y))
    b = metrics2.business({"BASELINE": (s, (s > 0.5).astype(int)), "FULL": (s, (s > 0.5).astype(int))}, y, w)
    flat = str(b)
    assert "bankAlone" not in flat and "BankAlone" not in flat
    assert b["baselineModel"] == "BASELINE" and "baselineShareMinusApprovalRate" in b["atBaselineDefaultRate"]
    assert b["models"]["FULL"]["relativeToBaseline"] == 0.0
