"""Round half up: the one rounding rule for every number a report, table text or figure shows.

Python's round() and format() round a half to the even neighbour (4,568.5 -> 4,568; -66.5 -> -66). The thesis rounds a
half away from zero (decimal.ROUND_HALF_UP): 4,568.5 -> 4,569, -66.5 -> -67, 0.125 -> 0.13 at 2 decimals. A number is
taken at its decimal value: a CSV cell as written ("4568.5000"), a float by its shortest repr (2.675, not the binary
2.67499999...), an int as it is. CSV columns written at 4 decimals (data1_analysis.cell) are data, not display, and
keep their format.
"""
from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal


def dec(x) -> Decimal:
    """x as a Decimal: str as written (thousands separators and spaces dropped), int exactly, float by its repr."""
    if isinstance(x, Decimal):
        return x
    if isinstance(x, str):
        return Decimal(x.strip().replace(",", ""))
    if isinstance(x, int) and not isinstance(x, bool):
        return Decimal(x)
    if hasattr(x, "dtype") and getattr(x.dtype, "kind", "") in "iu":  # numpy integers
        return Decimal(int(x))
    return Decimal(repr(float(x)))


def _q(x, nd: int) -> Decimal:
    d = dec(x).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return d.copy_abs() if d.is_zero() else d   # no "-0"


def round_half_up(x, nd: int = 0):
    """x rounded to nd decimals, halves away from zero: an int for nd = 0, else a float."""
    d = _q(x, nd)
    return int(d) if nd == 0 else float(d)


def fmt(x, nd: int = 0, sep: bool = True, sign: bool = False) -> str:
    """Fixed point with nd decimals, rounded half up; thousands separated unless sep=False; '+' on positives if sign.
    inf and nan are written as Python writes them."""
    spec = f"{'+' if sign else ''}{',' if sep else ''}.{nd}f"
    if not dec(x).is_finite():
        return format(float(dec(x)), spec)
    return format(_q(x, nd), spec)


def fmt_pct(x, nd: int = 1) -> str:
    """A share as a percentage (0.425 -> '42.5%'), rounded half up."""
    return fmt(dec(x) * 100, nd, sep=False) + "%"


def fmt_e(x, nd: int = 2) -> str:
    """Scientific notation as format(x, '.{nd}e') writes it (7.83e-05), with the mantissa rounded half up."""
    d = dec(x)
    if d.is_zero():
        return format(0.0, f".{nd}e")
    exp = d.adjusted()
    m = d.scaleb(-exp).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    if m.copy_abs() >= 10:  # 9.995 -> 10.00: one more digit to the left
        m, exp = (m / 10).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP), exp + 1
    return f"{format(m, f'.{nd}f')}e{exp:+03d}"


def format_spec(spec: str, x) -> str:
    """spec.format(x) for a one-field spec such as '{:,.1f}', '{:+,.0f}', '{:.3e}' or '{:.1%}', rounded half up; any other
    spec ('{:g}' for parameters written exactly, such as a clip bound of 4 or sigma 0.05) is passed to str.format."""
    m = re.fullmatch(r"\{:(\+?)(,?)\.(\d+)([fe%])\}", spec)
    if not m:
        return spec.format(x)
    sign, sep, nd, kind = m.groups()
    if kind == "f":
        return fmt(x, int(nd), sep=bool(sep), sign=bool(sign))
    return fmt_e(x, int(nd)) if kind == "e" else fmt_pct(x, int(nd))
