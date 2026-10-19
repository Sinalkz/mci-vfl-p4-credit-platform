"""MCI loan dashboard: Persian text helpers and the fonts (Vazirmatn 33.0.3, SIL Open Font License, dashboard/static/).

- fa_digits (for numbers only): Western digits to Persian digits (۰-۹), the decimal point to the Persian decimal separator «٫» and the
  thousands separator to «٬»; Latin technical terms stay in Latin script.
- shape: Persian text for matplotlib 3.10, which does not shape Arabic script itself (as the thesis's figures/F2-2.py:
  arabic_reshaper, then the visual order with a right-to-left base). The visual order comes from python-bidi's
  pure-Python implementation (bidi.algorithm), which mirrors brackets in right-to-left runs; the default get_display of
  python-bidi 0.6 does not, so "(Home Credit)" would come out as ")Home Credit(". Neither applies the paired-bracket
  rule, so no label puts a bracketed Latin call like f(x) inside Persian text.
- num: a number for display, rounded half up, with Persian digits.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager as fm

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
FONT_REGULAR = STATIC / "Vazirmatn-Regular.ttf"
FONT_BOLD = STATIC / "Vazirmatn-Bold.ttf"
for f in (FONT_REGULAR, FONT_BOLD):
    fm.fontManager.addfont(str(f))
FONT = fm.FontProperties(fname=str(FONT_REGULAR)).get_name()  # "Vazirmatn"

_DIGITS = str.maketrans("0123456789.,-", "۰۱۲۳۴۵۶۷۸۹٫٬−")


def fa_digits(text) -> str:
    return str(text).translate(_DIGITS)


def num(x, d: int = 0, pct: bool = False, sign: bool = False) -> str:
    """x rounded half up to d decimals (x 100 and a percent sign if pct), thousands separated, Persian digits."""
    if x is None:
        return "–"
    v = Decimal(str(100 * x if pct else x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{v:,.{d}f}"
    if sign and v > 0:
        s = "+" + s
    return fa_digits(s) + ("٪" if pct else "")


def shape(text: str) -> str:
    """Persian (with any Latin runs) in visual order for matplotlib."""
    return get_display(arabic_reshaper.reshape(str(text)), base_dir="R")
