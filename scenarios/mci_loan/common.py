"""MCI loan instance: the shared data layer (the counterpart of data/recs/data1_common.py).

- Source rows: application_train.csv of the Kaggle Home Credit Default Risk data (data/raw/home-credit/, Step 2).
- Cohort (the student's decision (b), 2026-10-18): 15,965 applicants, the thesis's household count (so |X_A| <= 2^14),
  balanced on TARGET: 7,982 defaults (TARGET = 1) and 7,983 repaid (TARGET = 0).
- Split: 80/20 within each class, so 12,772 training and 3,193 test rows, as the thesis.
- Every draw is a seeded hash order, not a random-number stream, so it does not depend on a library version:
  applicants are ordered by SHA-256("<seed>:<SK_ID_CURR>") and the first n are taken.
- thin_file (evaluator-side, segment analysis only): the applicant has no row in bureau.csv.

Usage: imported by the other scenario scripts; run with `python -I`, which keeps data/ off the import path.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "data" / "raw" / "home-credit"
PREPARED = ROOT / "data" / "mci_loan" / "prepared"            # git-ignored: binaries and their manifest
PREPARED_SMOKE = ROOT / "data" / "mci_loan" / "prepared-smoke"
RESULTS = ROOT / "results" / "mci_loan"
SLOTS = HERE / "slots.json"

N_COHORT = 15_965
N_DEFAULT = N_COHORT // 2                # 7,982
N_REPAID = N_COHORT - N_DEFAULT          # 7,983
TEST_SHARE = 0.2
N_EXTRA_POOL = 2_531                     # applicants outside the cohort that the passives may also hold (as the thesis's pool)
N_SMOKE = 1_000                          # the Step 3 smoke sample: 400 + 400 training, 100 + 100 test rows

COHORT_SEED = 20267001
SPLIT_SEED = 20267002
EXTRA_SEED = 20267003
SMOKE_SEED = 20267004
FOLD_SEED = 20267006                     # select_bank.py's cross-validation folds


def hash_order(seed: int, ids) -> np.ndarray:
    """The ids sorted by SHA-256("<seed>:<id>"): the seeded order every draw takes its first n from."""
    keys = [hashlib.sha256(f"{seed}:{int(i)}".encode()).hexdigest() for i in ids]
    return np.asarray(ids)[np.argsort(np.asarray(keys), kind="stable")]


def draw_cohort(ids: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The cohort's SK_ID_CURR, sorted ascending: the first N_DEFAULT defaults and N_REPAID repaid in hash order."""
    d = hash_order(COHORT_SEED, ids[target == 1])[:N_DEFAULT]
    r = hash_order(COHORT_SEED, ids[target == 0])[:N_REPAID]
    return np.sort(np.concatenate([d, r]))


def split_positions(cohort_ids: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(train_pos, test_pos): positions in the cohort; within each class the first round(0.2 n_c) in hash order are test."""
    pos = np.arange(len(cohort_ids))
    test = []
    for c in (0, 1):
        pc = pos[y == c]
        n_test = int(np.floor(TEST_SHARE * len(pc) + 0.5))
        order = hash_order(SPLIT_SEED, cohort_ids[pc])
        test_ids = set(order[:n_test].tolist())
        test += [p for p in pc if int(cohort_ids[p]) in test_ids]
    test = np.sort(np.asarray(test))
    train = np.setdiff1d(pos, test)
    return train, test


def draw_extra_pool(all_ids: np.ndarray, cohort_ids: np.ndarray) -> np.ndarray:
    """N_EXTRA_POOL applicants of application_train outside the cohort, sorted ascending."""
    outside = np.setdiff1d(all_ids, cohort_ids)
    return np.sort(hash_order(EXTRA_SEED, outside)[:N_EXTRA_POOL])


def smoke_positions(cohort_ids: np.ndarray, y: np.ndarray, train_pos: np.ndarray, test_pos: np.ndarray) -> np.ndarray:
    """The smoke sample's cohort positions, sorted: 400 + 400 from the training split, 100 + 100 from the test split."""
    out = []
    for part, per_class in ((train_pos, 400), (test_pos, 100)):
        for c in (0, 1):
            pc = part[y[part] == c]
            ids = set(hash_order(SMOKE_SEED, cohort_ids[pc])[:per_class].tolist())
            out += [p for p in pc if int(cohort_ids[p]) in ids]
    return np.sort(np.asarray(out))


def free_ram_mb() -> int:
    """Available physical memory in MB, as data/recs/data1_common.py measures it (GlobalMemoryStatusEx / sysconf)."""
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MEMORYSTATUSEX()
        s.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return int(s.ullAvailPhys // 2**20)
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 2**20)


def machine() -> dict:
    import pandas
    import sklearn
    return {"cpu": platform.processor(), "cores": os.cpu_count(), "platform": platform.platform(),
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pandas.__version__,
            "sklearn": sklearn.__version__}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj, indent=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")  # *.json eol=lf
