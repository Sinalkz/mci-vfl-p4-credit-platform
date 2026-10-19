"""MCI loan dashboard (v2): the Step 6 done-criterion as a check. Renders the app (dashboard/app.py) and each of its
eight pages in Streamlit's headless test runner (streamlit.testing.v1.AppTest) from the committed results, and fails if
any raises, or if any rendered text names a real company other than MCI (the naming rule, 2026-10-19: the e-commerce
role's former name, in English or Persian). Writes results/mci_loan/v2/dashboard-check.json.

Usage (from the repo root):  .venv/Scripts/python -I dashboard/check_pages.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import streamlit  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import data  # noqa: E402
import views  # noqa: E402

OUT = HERE.parent / "results" / "mci_loan" / "v2" / "dashboard-check.json"
FORBIDDEN = re.compile(r"digikala|دیجی", re.IGNORECASE)


def texts(at: AppTest) -> list[str]:
    """Every text the page rendered: titles, headers, markdown, captions, alerts, metrics, expanders, tables, selects."""
    out = [e.value for kind in ("title", "subheader", "markdown", "caption", "info", "warning") for e in getattr(at, kind)]
    out += [f"{m.label} {m.value}" for m in at.metric] + [x.label for x in at.expander]
    out += [d.value.to_csv() for d in at.dataframe] + [str(o) for sb in at.selectbox for o in sb.options]
    return [str(x) for x in out]


def run(at: AppTest) -> dict:
    at.run(timeout=180)
    errs = [str(e.value) for e in at.exception]
    named = sorted({m for t in texts(at) for m in FORBIDDEN.findall(t)})
    return {"ok": not errs and not named, "exceptions": errs, "forbiddenNames": named, "images": len(at.get("image")),
            "dataframes": len(at.dataframe), "titles": [t.value for t in at.title]}


def main() -> None:
    res = {"app": run(AppTest.from_file(str(HERE / "app.py")))}
    for slug, _, _ in views.PAGES:
        src = (f"import sys\nsys.path.insert(0, {str(HERE)!r})\nimport views\nviews.style()\nviews.banner()\n"
               f"next(fn for s, _, fn in views.PAGES if s == {slug!r})()\n")
        res[slug] = run(AppTest.from_string(src))
    out = {"what": "the v2 dashboard's pages rendered headless, with the naming check (dashboard/check_pages.py)", "streamlit": streamlit.__version__,
           "onchainResults": data.on_available(), "pages": res, "allOk": all(r["ok"] for r in res.values())}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: (v["ok"], v["images"], v["dataframes"]) for k, v in res.items()}, ensure_ascii=False))
    if not out["allOk"]:
        sys.exit("a page failed: " + json.dumps({k: v["exceptions"] + v["forbiddenNames"] for k, v in res.items() if not v["ok"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
