"""MCI loan dashboard (v2): the figure pack for the Phase 4 Word report, in docs/mci/v2/figures/report/.

1. The 17 dashboard charts (charts.figures_all) in a report version, 16 cm wide (1,890 px at 300 dpi): no in-image title
   and no footer (the simulated-organisations note, the off-chain label and the source path go to captions.md); axis
   labels, legends and Persian digits kept. A panel that only its title named gets that name as its axis label; panels
   too narrow side by side at 16 cm are stacked, the first-read (right-hand) panel on top.
2. 14-credit-policy-flow.png: the model's score, the calibrated PD (prior-corrected to application_train's default
   rate, prepared/manifest.json), the three tiers (offchain/summary.json, tierRule).
3. captions.md: every file of the folder with a one-line Persian caption and its source results file, the dashboard
   screenshots (dashboard/report_screens.mjs, run it first) included.
4. --zip PATH: the folder as a zip file.

Font: B Nazanin when installed, else the dashboard's Vazirmatn. B Nazanin draws the ASCII digits as Persian digits and has
no Latin letters, so the figures use a copy of it whose character map leaves out ASCII (U+0021-U+007E): ASCII then comes
from Times New Roman, as in Word, and Persian script from B Nazanin. The copy lives in a temporary folder for the run
only. With B Nazanin, «هٔ» is written «ۀ» (U+06C0) before shaping: arabic_reshaper drops the separate mark U+0654.

The dashboard's figures (docs/mci/v2/figures/*.png, export_figures.py) are not changed: this script restyles its own
instances of the figures after charts.py builds them.

Usage (from the repo root):  .venv/Scripts/python -I dashboard/report_figures.py [--zip <path.zip>]
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from PIL import Image  # noqa: E402

import charts  # noqa: E402
import data  # noqa: E402
import fa  # noqa: E402
from fa import fa_digits, num  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "docs" / "mci" / "v2" / "figures" / "report"
SRC = "results/mci_loan/v2"
DPI = 300
PX = 1890  # 16 cm at 300 dpi
WIDTH = (PX + 0.5) / DPI  # matplotlib truncates the pixel size, hence the half pixel
FAMILY = "B Nazanin ASCII-free"
SEPARATORS = "Vazirmatn separators"
FORBIDDEN = re.compile(r"digikala|دیجی", re.I)  # the naming rule: no real company's name but MCI

# per figure: height (inches), text scale, stacked panel order (top to bottom, by fig.axes index), axis labels for the
# panels a title named, a figure legend made again with new options, a background behind the diagram's plane headings
SPEC = {
    "01-architecture": {"h": 3.7, "k": 0.78, "heading_bg": True},
    "02-column-counts": {"h": 2.6},
    "03-quality": {"h": 3.2},
    "04-auc": {"h": 3.3},
    "05-business": {"h": 6.0, "stack": [1, 0],
                    "fig_legend": {"loc": "upper center", "bbox_to_anchor": (0.5, 1.0), "ncol": 5, "fontsize": 8,
                                   "columnspacing": 0.9, "handlelength": 1.2, "handletextpad": 0.4},
                    "ylabels": {1: "نرخ نکول در میان تأییدشدگان", 0: "تأیید اضافه در هر ۱۰٬۰۰۰ متقاضی"}},
    "06-partner-value": {"h": 6.4, "stack": [2, 1, 0],
                         "ylabels": {2: "سهم هر شریک lᵢ\nmacro-F1 × ۱۰⁴", 1: "کاهش نرخ نکول\nتأییدشدگان، واحد درصد",
                                     0: "تأیید اضافه در هر\n۱۰٬۰۰۰ متقاضی"}},
    "07-tiers": {"h": 6.2, "stack": [1, 0], "ylabels": {1: "سهم مشترکان", 0: "نرخ نکول وزن‌دهی‌شده"}},
    "08-payments-offchain": {"h": 3.3},
    "09-payouts": {"h": 3.3},
    "10-gas": {"h": 3.0},
    "11-proof-time": {"h": 3.0, "ylabels": {1: "زمان تولید هر اثبات، ثانیه", 0: "بیشینهٔ حافظهٔ اثبات‌گر، مگابایت"}},
    "12-privacy": {"h": 3.3},
    "13-settlement-run": {"h": 5.4, "stack": [1, 0],
                          "ylabels": {1: "رقم پذیرفته‌شده با اثبات\nmacro-F1 × ۱۰⁴", 0: "ادعا و سهم تلمودی\nواحد عملکرد"}},
}
K = 1.1  # B Nazanin draws smaller than Vazirmatn at the same size
PLANE_FILL = {"#1f4f6b": "#eaf4f8", "#8a3b00": "#fdf2e3"}  # charts.architecture: a heading's colour -> its plane's fill
FLOW = "14-credit-policy-flow"
SCREENS = [  # dashboard/report_screens.mjs's files: (file, caption, sources)
    ("dashboard-1-mci-alone-vs-partners.png",
     "صفحهٔ «همراه اول به‌تنهایی و با شرکا» در داشبورد: کیفیت مدل، قدرت تفکیک، شاخص‌های کسب‌وکار و ارزش هر شریک.",
     [f"{SRC}/offchain/loo_summary.csv", f"{SRC}/offchain/auc_summary.csv", f"{SRC}/offchain/business_summary.csv"]),
    ("dashboard-2-credit-tiers.png",
     "صفحهٔ «سطوح اعتبار» در داشبورد: سیاست اعتباری، سهم و نرخ نکول هر سطح، وارسی کالیبراسیون و سهم گروه‌های سنی.",
     [f"{SRC}/offchain/tiers_summary.csv", f"{SRC}/offchain/age_tiers_summary.csv", f"{SRC}/offchain/calibration.csv"]),
    ("dashboard-3a-settlement.png",
     "صفحهٔ «تسویه» در داشبورد، اجرای ۱ روی زنجیره: رقم‌های پذیرفته‌شده با اثبات، تقسیم تلمودی، پرداخت‌ها و پایستگی توکن.",
     [f"{SRC}/onchain/summary.json", f"{SRC}/onchain/payouts.csv"]),
    ("dashboard-3b-audit-trail.png",
     "صفحهٔ «ردپای حسابرسی» در داشبورد، اجرای ۱ روی زنجیره: رویدادهای تسویه، وارسی اثبات‌ها و تراکنش‌ها به تفکیک مرحله.",
     [f"{SRC}/onchain/events.csv", f"{SRC}/onchain/proofs.csv", f"{SRC}/onchain/transactions.csv"]),
]


# ---- the font -----------------------------------------------------------------------------------------------------

def nazanin_files() -> dict[str, Path]:
    """The installed B Nazanin (family "B Nazanin"), regular and bold, from matplotlib's list of the system's fonts
    (on Windows, the machine's and the user's font folders)."""
    found = {}
    for f in sorted(fm.findSystemFonts()):
        if "nazanin" not in Path(f).name.lower():
            continue
        try:
            ft = fm.get_font(f)
        except (OSError, RuntimeError):
            continue
        if ft.family_name == "B Nazanin" and ft.style_name in ("Regular", "Bold"):
            found.setdefault(ft.style_name, Path(f))
    return found


def derive(src: Path, dst: Path, family: str, keep=None, drop=()) -> None:
    """A copy of a font under another family name, its character map cut to `keep` or without `drop`; registered with
    matplotlib."""
    t = TTFont(src)
    for sub in t["cmap"].tables:
        for c in [c for c in sub.cmap if (keep is not None and c not in keep) or c in drop]:
            del sub.cmap[c]
    for rec in t["name"].names:
        if rec.nameID in (1, 16):
            rec.string = family
    t.save(dst)
    fm.fontManager.addfont(str(dst))


def use_font(tmp: Path) -> str:
    """B Nazanin's ASCII-free copies as the first font, Times New Roman for ASCII; returns the font as recorded in
    captions.md. B Nazanin lacks the Persian percent sign and separators (٪ ٫ ٬): they come from Vazirmatn, whose
    momayyez is the slanted mark the dashboard shows (Times New Roman's looks like a comma). Vazirmatn stays the
    first font when B Nazanin is not installed."""
    files = nazanin_files()
    if "Regular" not in files:
        return f"{fa.FONT}, the dashboard's font (B Nazanin is not installed)"
    for style, src in files.items():
        derive(src, tmp / f"nazanin-{style}.ttf", FAMILY, drop=range(0x21, 0x7F))
    for style, src in (("Regular", fa.FONT_REGULAR), ("Bold", fa.FONT_BOLD)):
        derive(src, tmp / f"separators-{style}.ttf", SEPARATORS, keep={0x066A, 0x066B, 0x066C})
    plt.rcParams["font.family"] = [FAMILY, SEPARATORS, "Times New Roman", fa.FONT, "DejaVu Sans"]  # per-glyph fallback, in order
    charts.shape = lambda text: fa.shape(str(text).replace("هٔ", "ۀ"))  # charts.S shapes through it
    return ("B Nazanin for Persian script; ASCII (Latin letters, ASCII digits, punctuation) from Times New Roman, as in "
            f"Word; the Persian percent sign and separators from {fa.FONT}; glyphs none of them has from DejaVu Sans")


def release_fonts() -> None:
    """matplotlib keeps the fonts it has read open (on Windows the temporary copies could not be removed otherwise)."""
    cached = getattr(fm, "_get_font", None)
    if cached is not None and hasattr(cached, "cache_clear"):
        cached.cache_clear()
    gc.collect()


# ---- the report version of a dashboard chart ----------------------------------------------------------------------

def scale_text(fig, k: float) -> None:
    """Every text k times its size; the tick labels through the axes, so ticks made at draw time follow."""
    ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            major = axis.get_major_ticks()
            if major:
                size = major[0].label1.get_fontsize()
                ticks.update(t for tk in major for t in (tk.label1, tk.label2))
                axis.set_tick_params(which="major", labelsize=size * k)
    for t in fig.findobj(Text):
        if t not in ticks:
            t.set_fontsize(t.get_fontsize() * k)


def restyle(fig, spec: dict):
    for t in list(fig.texts):  # the footer lines and the suptitle
        t.remove()
    fig._suptitle = None
    axes = list(fig.axes)
    for ax in axes:
        for loc in ("left", "center", "right"):
            ax.set_title("", loc=loc)
    for i, label in spec.get("ylabels", {}).items():
        axes[i].set_ylabel(charts.S(label))
    if "stack" in spec:
        gs = fig.add_gridspec(len(spec["stack"]), 1)
        for row, i in enumerate(spec["stack"]):
            axes[i].set_subplotspec(gs[row, 0])
    if "fig_legend" in spec:  # matplotlib lays a legend out when it makes it: a new one, the same entries
        old = fig.legends[0]
        handles, labels = old.legend_handles, [t.get_text() for t in old.get_texts()]
        old.remove()
        fig.legend(handles, labels, frameon=False, **spec["fig_legend"])
    if spec.get("heading_bg"):  # the arrows pass behind the planes' headings, not through them
        for ax in axes:
            for t in ax.texts:
                if t.get_fontweight() == "bold" and t.get_horizontalalignment() == "right":
                    t.set_bbox({"fc": PLANE_FILL[t.get_color()], "ec": "none", "pad": 1.5})
                    t.set_zorder(5)
    scale_text(fig, spec.get("k", 1.0) * K)
    fig.set_size_inches(WIDTH, spec["h"])
    top = 1.0
    if fig.legends:  # a figure legend at the top: the panels below it
        fig.canvas.draw()
        top = 1 - fig.legends[0].get_window_extent().height / fig.bbox.height - 0.01
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


# ---- the credit-policy flow ---------------------------------------------------------------------------------------

def policy_flow():
    """Model score, then the calibrated PD (cross-fitted Platt scaling on the test applicants, prior-corrected to
    application_train's default rate), then the three tiers. Sources: results/mci_loan/v2/offchain/summary.json
    (tierRule), results/mci_loan/v2/prepared/manifest.json (population.default_rate)."""
    S = charts.S
    lo, hi = data.off_summary()["tierRule"]["cuts"]
    pi = data.manifest()["population"]["default_rate"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, 2.9)
    ax.axis("off")
    ax.set_position((0, 0, 1, 1))

    def box(x, y, w, h, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color="#555", lw=1.2, shrinkA=0, shrinkB=0))

    blue, pale = "#1f6f8b", "#eaf4f8"
    box(5.0, 0.85, 1.2, 1.2, pale, blue)
    ax.text(5.6, 1.72, S("امتیاز مدل"), ha="center", va="center", fontsize=12, weight="bold", color="#1f4f6b")
    ax.text(5.6, 1.33, S("خروجی مدل برای\nهر متقاضی"), ha="center", va="center", fontsize=10, linespacing=1.4)
    box(2.72, 0.6, 2.0, 1.7, pale, blue)
    ax.text(3.72, 2.0, S("احتمال نکول کالیبره‌شده PD"), ha="center", va="center", fontsize=12, weight="bold", color="#1f4f6b")
    ax.text(3.72, 1.38, S("رگرسیون لجستیک با برازش متقاطع\nدوبخشی روی متقاضیان آزمون"), ha="center", va="center",
            fontsize=10, linespacing=1.4)
    ax.text(3.72, 0.85, S(f"تصحیح‌شده به نرخ نکول {num(pi, 2, pct=True)}"), ha="center", va="center", fontsize=10)
    arrow((4.98, 1.45), (4.74, 1.45))
    rules = [f"PD < {num(lo, 0, pct=True)}", f"{num(lo, 0, pct=True)} ≤ PD < {num(hi, 0, pct=True)}", f"PD ≥ {num(hi, 0, pct=True)}"]
    tint = {"credit4m": "#e3f2ea", "credit1m": "#fcf1de", "none": "#f8e1de"}
    for i, ((key, name), rule) in enumerate(zip(charts.TIERS, rules)):
        y = 2.05 - i * 0.93
        box(0.1, y, 2.0, 0.72, tint[key], charts.COLORS[key])
        ax.text(1.1, y + 0.49, S(name), ha="center", va="center", fontsize=11.5, weight="bold", color=charts.COLORS[key])
        ax.text(1.1, y + 0.2, rule, ha="center", va="center", fontsize=10.5)  # left to right, no Persian letters
        arrow((2.7, 1.45), (2.14, y + 0.36))
    return fig


# ---- captions -----------------------------------------------------------------------------------------------------

OFF = "خارج از زنجیره و بدون اثبات ZK"


def caption_rows() -> list[tuple[str, str, list[str]]]:
    """(file, one-line Persian caption, source files), every number read from the results."""
    o = data.off_summary()
    mci = data.name_fa("mci")
    cols = {m["key"]: len(m["columns"]) for m in data.members()}
    names = "، ".join(f"{data.name_fa(k)} {fa_digits(cols[k])}" for k in ["mci"] + data.passive_keys())
    r = o["business"]["all/BASELINE"]["approval_rate_r"]["mean"]
    pi = data.manifest()["population"]["default_rate"]
    lo, hi = o["tierRule"]["cuts"]
    rows = [
        ("01-architecture.png", f"معماری نمونهٔ وام {mci}: صفحهٔ کنترل روی زنجیره (Besu با اجماع QBFT، یک اعتبارسنج برای هر "
         "عضو، قرارداد Federation، وارسی‌گر Groth16 و تقسیم تلمودی) و صفحهٔ داده خارج از زنجیره، که داده نزد هر عضو می‌ماند.",
         [f"{SRC}/prepared/manifest.json"]),
        ("02-column-counts.png", f"شمار ستون‌های مدل هر عضو: {names} ستون.", [f"{SRC}/prepared/manifest.json", f"{SRC}/prepared/columns.json"]),
        ("03-quality.png", f"macro-F1 × ۱۰⁴ روی مجموعهٔ آزمون: {mci} به‌تنهایی، با همهٔ شرکا، بدون هر شریک و با آموزش جداگانه "
         "(C1b)؛ میانگین ده بذر، خط خطا یک انحراف معیار.", [f"{SRC}/offchain/loo_summary.csv"]),
        ("04-auc.png", f"قدرت تفکیک (ROC-AUC) هر مدل روی همهٔ متقاضیان آزمون و متقاضیان بدون سابقهٔ اعتباری؛ میانگین ده بذر؛ {OFF}.",
         [f"{SRC}/offchain/auc_summary.csv"]),
        ("05-business.png", f"شاخص‌های کسب‌وکار: نرخ نکول در میان تأییدشدگان با نرخ تأیید ثابت {mci} به‌تنهایی "
         f"({num(r, 1, pct=True)})، و تأیید اضافه در هر ۱۰٬۰۰۰ متقاضی با نرخ نکول ثابت {mci} به‌تنهایی؛ وزن‌دهی‌شده به نرخ نکول "
         f"{num(pi, 2, pct=True)} application_train؛ میانگین ده بذر؛ {OFF}.", [f"{SRC}/offchain/business_summary.csv"]),
        ("06-partner-value.png", f"ارزش هر شریک برای {mci}: سهم ترک‌یک‌عضو lᵢ = U_fed − LOOᵢ، و کاهش نرخ نکول تأییدشدگان و "
         f"تأیید اضافه در هر ۱۰٬۰۰۰ متقاضی با افزودن آن شریک؛ میانگین ده بذر؛ شاخص‌های کسب‌وکار {OFF}.",
         [f"{SRC}/offchain/loo_summary.csv", f"{SRC}/offchain/business_summary.csv"]),
        ("07-tiers.png", f"سطوح اعتبار: سهم مشترکان و نرخ نکول وزن‌دهی‌شدهٔ هر سطح، {mci} به‌تنهایی و با شرکا، برای همهٔ متقاضیان "
         f"و متقاضیان بدون سابقهٔ اعتباری؛ اعتبار چهارماهه اگر PD کمتر از {num(lo, 0, pct=True)}، یک‌ماهه اگر کمتر از "
         f"{num(hi, 0, pct=True)}، وگرنه بدون اعتبار؛ میانگین ده بذر؛ {OFF}.", [f"{SRC}/offchain/tiers_summary.csv"]),
        ("08-payments-offchain.png", f"پرداخت {mci} به هر شریک با قاعدهٔ تلمودی از استخر B = ۱٬۰۰۰٬۰۰۳ توکن، و آنچه نزد {mci} "
         "می‌ماند، در هر بذر خارج از زنجیره روی کل مجموعهٔ آزمون.", [f"{SRC}/offchain/payments.csv"]),
    ]
    if data.off_summary().get("ldpIncluded"):
        rows.append(("12-privacy.png", f"مبادلهٔ حریم خصوصی و کارایی: macro-F1 {mci} با همهٔ شرکا (U_fed) و به‌تنهایی (U_A) در "
                     "برابر ε هر انتشار، با LDP روی تعبیه‌های شرکا؛ نقطهٔ جدا بدون نوفه، فقط برش؛ میانگین ده بذر، خط خطا یک "
                     "انحراف معیار.", [f"{SRC}/offchain/ldp_eps_summary.csv"]))
    if data.on_available():
        S = data.on_summary()
        c = data.on_csv("cost_comparison.csv").set_index("quantity")
        tot = {col: num(c.loc["gas: total", f"{col}_mean"] / 1e6, 2) for col in ("loan_v2", "loan_v1", "electricity")}
        B = S["runs"][str(S["runsDone"][0])]["division"]["B"]
        rows += [
            ("09-payouts.png", f"توکن واریزشده به هر عضو در هر اجرای روی زنجیره: سهم‌های تلمودی از استخر B = {num(B)} توکن "
             "به‌اضافهٔ ودیعه‌های مشارکت بازگشتی.", [f"{SRC}/onchain/payouts.csv"]),
            ("10-gas.png", f"gas هر مرحله، میانگین پنج اجرا، نسخهٔ ۲ وام در برابر نسخهٔ ۱ و سناریوی برق پایان‌نامه؛ مجموع هر اجرا "
             f"{tot['loan_v2']} میلیون gas در نسخهٔ ۲، {tot['loan_v1']} میلیون در نسخهٔ ۱ و {tot['electricity']} میلیون در سناریوی برق.",
             [f"{SRC}/onchain/cost_comparison.csv"]),
            ("11-proof-time.png", "زمان تولید هر اثبات با rapidsnark و بیشینهٔ حافظهٔ اثبات‌گر، نسخهٔ ۲ وام در برابر نسخهٔ ۱ و "
             "سناریوی برق؛ میانگین اجراها، خط خطا یک انحراف معیار.", [f"{SRC}/onchain/cost_comparison.csv"]),
        ]
        for k in S["runsDone"]:
            d = S["runs"][str(k)]["division"]
            rows.append((f"13-settlement-run-{k}.png", f"اجرای {fa_digits(k)} روی زنجیره: شش رقم پذیرفته‌شده با اثبات، و ادعای cᵢ "
                         f"و سهم تلمودی aᵢ هر شریک؛ E = {num(d['E'])} و E′ = {num(d['E_prime'])} با سقف استخر "
                         f"⌊B/ρ⌋ = {num(d['B_over_rho'])} واحد؛ هر واحد ρ = {num(d['rho'])} توکن.", [f"{SRC}/onchain/summary.json (run {k})"]))
    rows.append((f"{FLOW}.png", f"سیاست اعتباری {mci}: امتیاز مدل، سپس احتمال نکول کالیبره‌شده (رگرسیون لجستیک با برازش "
                 f"متقاطع دوبخشی روی متقاضیان آزمون، تصحیح‌شده به نرخ نکول {num(pi, 2, pct=True)} application_train)، سپس سه "
                 "سطح اعتبار.", [f"{SRC}/offchain/summary.json (tierRule)", f"{SRC}/prepared/manifest.json (population.default_rate)"]))
    return sorted(rows, key=lambda r: r[0]) + SCREENS


def write_captions(font: str) -> list[str]:
    files = sorted(p.name for p in OUT.glob("*.png"))
    rows = [r for r in caption_rows() if r[0] in files]
    missing = sorted(set(files) - {r[0] for r in rows}) + [r[0] for r in caption_rows() if r[0] not in files]
    if missing:
        raise SystemExit(f"captions and files differ: {missing} (run dashboard/report_screens.mjs first)")
    lines = ["# Phase 4 report figures (MCI loan v2): captions", "",
             "Each file in this folder: a one-line Persian caption and its source results file. Generated by "
             "`dashboard/report_figures.py`; the dashboard screenshots by `dashboard/report_screens.mjs`.", "",
             f"- **Charts:** 16 cm wide (1,890 px at 300 dpi). Font: {font}. No in-image title or footer; the caption carries both.",
             "- **Screenshots:** 1,600 px wide, the dashboard in its light theme, each page as it opens (run 1 where the page has a run selector).",
             "- **The note the footers carried**, for the report's text: سازمان‌ها شبیه‌سازی شده‌اند: شرکا نقش‌هایی عمومی‌اند، "
             "تعریف‌شده روی مجموعه‌دادهٔ عمومی Kaggle Home Credit Default Risk؛ دادهٔ هیچ شرکتی، از جمله همراه اول، به کار نرفته است.", ""]
    for f, cap, src in rows:
        lines += [f"## {f}", "", cap, "", "منبع: " + "، ".join(f"`{s}`" for s in src), ""]
    text = "\n".join(lines)
    if FORBIDDEN.search(text):
        raise SystemExit("captions.md names a real company")
    (OUT / "captions.md").write_text(text, encoding="utf-8", newline="\n")
    return [r[0] for r in rows]


# ---- main ---------------------------------------------------------------------------------------------------------

def check_names(fig, name: str) -> None:
    """The naming rule on the drawn text (shaped, so NFKC folds the presentation forms back, in either order)."""
    for t in fig.findobj(Text):
        s = unicodedata.normalize("NFKC", t.get_text())
        if FORBIDDEN.search(s) or FORBIDDEN.search(s[::-1]):
            raise SystemExit(f"{name}: a real company's name in the figure")


def save(fig, name: str) -> dict:
    check_names(fig, name)
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    with Image.open(p) as im:
        w, h = im.size
    if w != PX:
        raise SystemExit(f"{name}: {w} px wide, not {PX}")
    return {"file": p.name, "px": [w, h]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="also zip the folder to this path")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        font = use_font(Path(tmp))
        written = []
        for name, build in charts.figures_all().items():
            spec = SPEC["13-settlement-run" if name.startswith("13-settlement-run-") else name]
            written.append(save(restyle(build(), spec), name))
        written.append(save(policy_flow(), FLOW))
        release_fonts()
    captioned = write_captions(font)
    out = {"dir": str(OUT), "font": font, "figures": written, "captions": len(captioned)}
    if args.zip:
        z = Path(args.zip)
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(OUT.iterdir()):
                zf.write(p, p.name)
        out["zip"] = {"path": str(z), "files": len(zipfile.ZipFile(z).namelist()), "bytes": z.stat().st_size}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
