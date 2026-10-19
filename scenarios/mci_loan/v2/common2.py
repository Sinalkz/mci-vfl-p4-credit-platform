"""MCI loan instance v2 (MCI as the active member): where v2 reads and writes, and its member names.

The cohort, split, extra pool, smoke sample and their seeds are v1's (scenarios/mci_loan/common.py), unchanged: v2
changes who holds which columns, not which applicants are used. v1's results stay where they are (results/mci_loan/,
docs/mci/; tag mci-v1-bank-active); v2 writes only under results/mci_loan/v2/, docs/mci/v2/ and data/mci_loan/v2/.

Usage: imported by the v2 scripts, which put scenarios/mci_loan/ and scenarios/mci_loan/v2/ on the import path.
"""
from __future__ import annotations

import json
from pathlib import Path

import common as cm

HERE2 = Path(__file__).resolve().parent
RESULTS2 = cm.RESULTS / "v2"
PREPARED2 = cm.ROOT / "data" / "mci_loan" / "v2" / "prepared"            # git-ignored, as v1's
PREPARED2_SMOKE = cm.ROOT / "data" / "mci_loan" / "v2" / "prepared-smoke"
DOCS2 = cm.ROOT / "docs" / "mci" / "v2"
SLOTS2 = HERE2 / "slots.json"

CALIB_SEED = 20267007                    # policy.py's calibration folds over the test rows (v1 used 20267001-20267006)


def slots() -> dict:
    return json.loads(SLOTS2.read_text(encoding="utf-8"))


def names(lang: str = "en") -> dict[str, str]:
    """Data-owner key (mci, bank, ecommerce, bureau) -> the user-facing name, English or Persian ("fa")."""
    return {s["owner"]: s["nameFa" if lang == "fa" else "name"] for s in slots()["members"]}


# delivery: results/ keeps a redacted copy of each prepared manifest; the full one (applicant IDs, labels, split
# positions) stays with the data under data/, which git ignores. The Kaggle competition rules allow no row-level data.
ROW_KEYS = {"ids": "applicant IDs (SK_ID_CURR)", "y": "labels", "thin_file_pos": "thin-file positions"}


def redact_manifest(p: Path) -> None:
    m = json.loads(p.read_text(encoding="utf-8"))
    for k, what in ROW_KEYS.items():
        if isinstance(m.get(k), list):
            m[k] = {"redacted": f"row-level {what}, not delivered (Kaggle competition rules)", "count": len(m[k])}
    for k in ("train_pos", "test_pos"):
        if isinstance(m.get("split", {}).get(k), list):
            m["split"][k] = {"redacted": "row-level split positions, not delivered", "count": len(m["split"][k])}
    if isinstance(m.get("smoke", {}).get("cohort_pos"), list):
        m["smoke"]["cohort_pos"] = {"redacted": "row-level smoke positions, not delivered", "count": len(m["smoke"]["cohort_pos"])}
    p.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
