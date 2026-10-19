"""MCI loan dashboard (Step 6, v2: MCI as the active member, MCI-centric): a Persian, right-to-left Streamlit app over
the committed results in results/mci_loan/v2/. v1's dashboard (the Bank active) is kept at the tag mci-v1-bank-active.

Eight pages (dashboard/views.py): consortium; MCI alone and with partners; credit tiers; what MCI pays each partner;
settlement; audit trail; cost; privacy. Every chart is also exported at 300 dpi to docs/mci/v2/figures/ by
dashboard/export_figures.py.

Usage (from the repo root):  .venv/Scripts/streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st  # noqa: E402

import views  # noqa: E402

st.set_page_config(page_title="داشبورد نمونهٔ وام همراه اول", page_icon="🏦", layout="wide")
views.style()
views.banner()
nav = st.navigation([st.Page(fn, title=title, url_path=slug, default=(slug == "consortium")) for slug, title, fn in views.PAGES])
nav.run()
