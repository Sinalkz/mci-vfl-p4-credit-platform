"""MCI loan dashboard (v2): exports every chart (dashboard/charts.py) as a 300-dpi PNG with Persian labels to
docs/mci/v2/figures/ (the task's Step 6 rule), from the committed results under results/mci_loan/v2/ only. v1's figures
stay in docs/mci/figures/.

Usage (from the repo root):  .venv/Scripts/python -I dashboard/export_figures.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt  # noqa: E402

import charts  # noqa: E402

OUT = HERE.parent / "docs" / "mci" / "v2" / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, build in charts.figures_all().items():
        fig = build()
        fig.savefig(out / f"{name}.png", dpi=300)
        plt.close(fig)
        written.append(f"{name}.png")
    print(json.dumps({"dir": str(out), "figures": written}))


if __name__ == "__main__":
    main()
