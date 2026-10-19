"""MCI loan dashboard (v2: MCI as the active member): everything it shows, read from results/mci_loan/v2/ only (the
task's Step 6 rule). v1's dashboard is kept at the tag mci-v1-bank-active.

- members: the four members with their roles and names, from results/mci_loan/v2/prepared/manifest.json (`slots`, copied
  there by scenarios/mci_loan/v2/prep2.py from scenarios/mci_loan/v2/slots.json), and each member's columns from
  results/mci_loan/v2/prepared/columns.json (the partners) and manifest.json `a.vars` (MCI).
- offchain: results/mci_loan/v2/offchain/summary.json and its CSVs (scenarios/mci_loan/v2/analysis2.py).
- onchain: results/mci_loan/v2/onchain/summary.json and its CSVs (scenarios/mci_loan/v2/onchain_report2.py); None
  while Step 5 has not been run.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "mci_loan" / "v2"
OFF = R / "offchain"
ON = R / "onchain"


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def manifest() -> dict:
    return _json(R / "prepared" / "manifest.json")


@lru_cache(maxsize=None)
def columns() -> dict:
    return _json(R / "prepared" / "columns.json")


@lru_cache(maxsize=None)
def members() -> list[dict]:
    """MCI, Bank, E-commerce company, Credit bureau in proposal order, with their model columns and source tables."""
    m = manifest()
    out = []
    for s in m["slots"]["members"]:
        key = s["owner"]
        active = s["role"] == "active"
        cols = m["a"]["vars"] if active else [c["name"] for c in columns()["kept"] if c["owner"] == key]
        sources = ["application_train"] if active else sorted({c["source"] for c in columns()["kept"] if c["owner"] == key})
        out.append({**s, "key": key, "columns": cols, "sources": sources})
    return out


def name_fa(key: str) -> str:
    """Persian member name by owner key (mci, bank, ecommerce, bureau)."""
    return next(m["nameFa"] for m in members() if m["key"] == key)


def passive_keys() -> list[str]:
    """Passive i (1-based, proposal order) is passive_keys()[i - 1]."""
    return manifest()["passiveOrder"]


@lru_cache(maxsize=None)
def off_summary() -> dict:
    return _json(OFF / "summary.json")


@lru_cache(maxsize=None)
def off_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OFF / name)


def on_available() -> bool:
    return (ON / "summary.json").exists()


@lru_cache(maxsize=None)
def on_summary() -> dict | None:
    return _json(ON / "summary.json") if on_available() else None


@lru_cache(maxsize=None)
def on_csv(name: str) -> pd.DataFrame | None:
    p = ON / name
    return pd.read_csv(p) if p.exists() else None
