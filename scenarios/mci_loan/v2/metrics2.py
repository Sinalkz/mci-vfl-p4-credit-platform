"""MCI loan instance v2: v1's off-chain metrics (scenarios/mci_loan/metrics.py), with MCI alone (U_A, the BASELINE figure)
as the baseline. The computation is v1's, unchanged; only the output keys name the baseline neutrally, since it is no
longer the Bank: bankAloneModel -> baselineModel, atBankAloneDefaultRate -> atBaselineDefaultRate,
bankAloneShareMinusApprovalRate -> baselineShareMinusApprovalRate, relativeToBankAlone -> relativeToBaseline.
Off-chain, not ZK-verified (decision (a)).
"""
from __future__ import annotations

import metrics
from metrics import auc, class_weights  # noqa: F401  (re-exported for the v2 scripts)

RENAME = {"bankAloneModel": "baselineModel", "atBankAloneDefaultRate": "atBaselineDefaultRate",
          "bankAloneShareMinusApprovalRate": "baselineShareMinusApprovalRate", "relativeToBankAlone": "relativeToBaseline"}


def _rename(o):
    if isinstance(o, dict):
        return {RENAME.get(k, k): _rename(v) for k, v in o.items()}
    return o


def business(models: dict, y, w, baseline: str = "BASELINE") -> dict:
    """metrics.business with the baseline model named `baseline` (MCI alone); keys renamed as above."""
    return _rename(metrics.business(models, y, w, baseline))
