"""py/half_up.py: every reported number is rounded half up, i.e. a half goes away from zero (4,568.5 -> 4,569 and
-66.5 -> -67), unlike Python's round() and format(), which round a half to the even neighbour."""
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import half_up as hu  # noqa: E402


@pytest.mark.parametrize("x,nd,want", [
    (4568.5, 0, 4569), (-66.5, 0, -67), (0.5, 0, 1), (-0.5, 0, -1), (2.5, 0, 3), (-2.5, 0, -3),   # halves: away from 0
    (4568.4, 0, 4568), (4568.6, 0, 4569), (-66.4, 0, -66), (-66.6, 0, -67), (3, 0, 3),           # not halves
    (382.25, 1, 382.3), (-382.25, 1, -382.3), (0.125, 2, 0.13), (-0.125, 2, -0.13),               # halves, decimals
    (0.124, 2, 0.12), (183.502, 0, 184), (111.9307, 0, 112),
])
def test_round_half_up(x, nd, want):
    got = hu.round_half_up(x, nd)
    assert got == want and type(got) is (int if nd == 0 else float)


def test_python_rounds_these_halves_to_even_and_half_up_does_not():
    assert (round(4568.5), f"{4568.5:.0f}", round(-66.5), f"{2.675:.2f}") == (4568, "4568", -66, "2.67")
    assert (hu.fmt(4568.5), hu.fmt(-66.5), hu.fmt(2.675, 2)) == ("4,569", "-67", "2.68")


def test_a_number_is_taken_at_its_decimal_value():
    assert hu.dec("4568.5000") == Decimal("4568.5000")       # a CSV cell as written
    assert hu.dec(2.675) == Decimal("2.675")                 # a float by its shortest repr, not 2.67499999...
    assert hu.dec(np.float64(0.425)) == Decimal("0.425") and hu.dec(np.int64(7)) == Decimal(7)
    assert hu.dec(" -1,234.5 ") == Decimal("-1234.5")


def test_fmt_separates_thousands_signs_and_never_shows_minus_zero():
    assert hu.fmt(1516.5) == "1,517" and hu.fmt(1516.5, sep=False) == "1517"
    assert hu.fmt(-1777.5, sign=True) == "-1,778" and hu.fmt(343.6, sign=True) == "+344"
    assert hu.fmt(1802.75, 1) == "1,802.8" and hu.fmt(-4.45, 1) == "-4.5" and hu.fmt(7, 1) == "7.0"
    assert hu.fmt(-0.4) == "0" and hu.fmt(-0.04, 1) == "0.0" and hu.fmt(0, sign=True) == "+0"


def test_fmt_pct_and_fmt_e():
    assert hu.fmt_pct(0.425) == "42.5%" and hu.fmt_pct(0.4255) == "42.6%" and hu.fmt_pct(0.3, 0) == "30%"
    assert hu.fmt_e(7.829627e-05, 2) == "7.83e-05" and hu.fmt_e(0.001645, 2) == "1.65e-03"   # 1.645: a half
    assert hu.fmt_e(9.995e-06, 2) == "1.00e-05" and hu.fmt_e(-2.5e-3, 0) == "-3e-03"
    assert hu.fmt_e(6.352138814946784e-10, 1) == "6.4e-10" and hu.fmt_e(0, 1) == "0.0e+00"
    assert hu.fmt_e(0.001645, 2) != f"{0.001645:.2e}"                                        # Python: 1.64e-03


def test_format_spec_is_a_half_up_drop_in_for_one_field_specs():
    assert hu.format_spec("{:,.1f}", 1234.25) == "1,234.3" and "{:,.1f}".format(1234.25) == "1,234.2"
    assert hu.format_spec("{:+,.0f}", -1777.5) == "-1,778" and hu.format_spec("{:.3f}", 0.4255) == "0.426"
    assert hu.format_spec("{:.3e}", 7.8296e-05) == "7.830e-05" and hu.format_spec("{:.1%}", 0.4255) == "42.6%"
    assert hu.format_spec("{:g}", 0.05) == "0.05" and hu.format_spec("{:g}", 4.0) == "4"


def test_infinity_is_written_as_python_writes_it():
    assert hu.fmt(float("inf"), 1) == "inf" and hu.format_spec("{:,.1f}", float("inf")) == "inf" and hu.fmt("-inf") == "-inf"
