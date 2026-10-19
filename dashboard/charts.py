"""MCI loan dashboard (v2): every chart, as a matplotlib figure, shown on its page and exported at 300 dpi to
docs/mci/v2/figures/ (dashboard/export_figures.py). Persian labels (Vazirmatn, shaped by fa.shape), Persian digits; Latin
technical terms in Latin script. Each function names its source files under results/mci_loan/v2/ in its docstring;
every figure carries the simulated-organisations note in its footer.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import data  # noqa: E402
from fa import FONT, fa_digits, num, shape  # noqa: E402

plt.rcParams.update({"font.family": [FONT, "DejaVu Sans"],  # Vazirmatn has no Greek letters or subscripts: fallback
                     "font.size": 10, "axes.titlesize": 11, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.left": False})
COLORS = {"alone": "#7f8c8d", "c1b": "#b2bec3", "full": "#1f6f8b", "fullj": "#3a9ec2", "bank": "#e1a23b",
          "ecommerce": "#8e5ea2", "bureau": "#1b2f45", "elec": "#95a5a6", "v1": "#b2bec3", "v2": "#1f6f8b",
          "thin": "#d35400", "credit4m": "#2e8b57", "credit1m": "#e1a23b", "none": "#c0392b"}
# no brackets next to Persian text: python-bidi places a closing bracket between a Latin run and Persian on the wrong side
NOTE = ("سازمان‌ها شبیه‌سازی شده‌اند: شرکا نقش‌هایی عمومی‌اند، تعریف‌شده روی مجموعه‌دادهٔ عمومی Kaggle Home Credit Default Risk؛ "
        "دادهٔ هیچ شرکتی به کار نرفته است")
OFFCHAIN = "خارج از زنجیره، بدون اثبات ZK"
PHASES = {"setup": "راه‌اندازی", "alignment": "هم‌ترازی", "training": "آموزش", "scoring": "امتیازدهی", "settlement": "تسویه"}
TIERS = [("credit4m", "اعتبار چهارماهه"), ("credit1m", "اعتبار یک‌ماهه"), ("none", "بدون اعتبار")]
SRC = "results/mci_loan/v2"


def S(t):  # shape for matplotlib, line by line
    return "\n".join(shape(line) for line in str(t).split("\n"))


def alone() -> str:
    return f"{data.name_fa('mci')} به‌تنهایی"


def with_all() -> str:
    return f"{data.name_fa('mci')} با همهٔ شرکا"


def without(i: int) -> str:
    """LOO_i's label: MCI with every partner but passive i."""
    return f"بدون {data.name_fa(data.passive_keys()[i - 1])}"


def wrap(t: str, width: int = 16) -> str:
    """A tick label on two lines when it is long: the break at the space nearest the middle."""
    if len(t) <= width or "\n" in t or " " not in t:
        return t
    mid = len(t) // 2
    cut = min((i for i, ch in enumerate(t) if ch == " "), key=lambda i: abs(i - mid))
    return t[:cut] + "\n" + t[cut + 1:]


def fa_axis(ax, axis="y", d=0, pct=False):
    f = FuncFormatter(lambda v, _: num(v, d, pct=pct))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def finish(fig, source: str, offchain: bool = False):
    """The footer: the simulated-organisations note and the off-chain label where it applies; below it, the source
    file(s) (Latin). ("·" is not used: in Vazirmatn it looks like the Persian zero.)"""
    fig.text(0.99, 0.045, S(NOTE + (f"؛ {OFFCHAIN}" if offchain else "")), ha="right", va="bottom", fontsize=7, color="#555")
    fig.text(0.99, 0.012, source, ha="right", va="bottom", fontsize=6.5, color="#888", family="DejaVu Sans")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def right_axis(ax):
    """RTL: the value axis on the right."""
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines["right"].set_visible(True)


def bar_labels(ax, bars, vals, d=0, pct=False, err=None, fontsize=8.5):
    """Each bar's value just above the bar, or above its error bar."""
    for i, (b, v) in enumerate(zip(bars, vals)):
        top = b.get_height() + (err[i] if err is not None else 0)
        ax.annotate(num(v, d, pct=pct), (b.get_x() + b.get_width() / 2, top), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize)


def models_all():
    """(figure, Persian label, colour) for MCI with all partners, without each one, and alone."""
    keys = data.passive_keys()
    return ([("FULL", with_all(), COLORS["full"])] + [(f"LOO_{i}", without(i), COLORS[keys[i - 1]]) for i in (1, 2, 3)]
            + [("BASELINE", alone(), COLORS["alone"])])


# ---- 1. consortium ----------------------------------------------------------------------------------------------

def architecture():
    """The two planes: the off-chain data plane (the members' data, training, the private set intersection) and the
    on-chain control plane (Besu QBFT, one validator per member; the Federation contract, the Groth16 verifier).
    Member names from results/mci_loan/v2/prepared/manifest.json."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box = lambda x, y, w, h, fc, ec: ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",  # noqa: E731
                                                                  fc=fc, ec=ec, lw=1.2))
    box(0.2, 3.4, 9.6, 2.4, "#eaf4f8", "#1f6f8b")
    ax.text(9.7, 5.55, S("صفحهٔ کنترل روی زنجیره: Besu با اجماع QBFT، یک اعتبارسنج برای هر عضو"), ha="right", fontsize=10.5, weight="bold", color="#1f4f6b")
    box(0.2, 0.2, 9.6, 2.7, "#fdf2e3", "#d35400")
    ax.text(9.7, 2.62, S("صفحهٔ داده خارج از زنجیره: داده نزد هر عضو می‌ماند"), ha="right", fontsize=10.5, weight="bold", color="#8a3b00")
    for i, (txt, sub) in enumerate([("قرارداد Federation", "پیشنهاد، ودیعه، تعهد γ و امضا"),
                                    ("وارسی‌گر Groth16", "پذیرش هر رقم macro-F1 با اثبات"),
                                    ("SettlementMath", "تقسیم تلمودی و پرداخت")]):
        x = 6.6 - i * 3.1
        box(x, 3.7, 2.8, 1.35, "white", "#1f6f8b")
        ax.text(x + 1.4, 4.65, S(txt), ha="center", fontsize=10)
        ax.text(x + 1.4, 4.1, S(sub), ha="center", fontsize=8, color="#444")
    for i, m in enumerate(data.members()):
        x = 7.4 - i * 2.35
        box(x, 0.5, 2.1, 1.7, "white", "#d35400")
        ax.text(x + 1.05, 1.75, S(m["nameFa"]), ha="center", fontsize=10 if len(m["nameFa"]) > 14 else 10.5, weight="bold")
        role = "فعال: وام‌دهنده، برچسب و پاداش" if m["role"] == "active" else "شریک منفعل: تعبیه‌ها"
        ax.text(x + 1.05, 1.25, S(role), ha="center", fontsize=8)
        ax.text(x + 1.05, 0.8, S(f"{fa_digits(len(m['columns']))} ستون"), ha="center", fontsize=8, color="#444")
        ax.add_patch(FancyArrowPatch((x + 1.05, 2.25), (x + 1.05, 3.65), arrowstyle="<->", mutation_scale=10, color="#888", lw=1))
    ax.text(9.7, 5.2, S("روی زنجیره فقط تعهدها، امضاها، رقم‌های اثبات‌شده و تسویه؛ داده و مدل خارج از زنجیره می‌مانند"),
            ha="right", fontsize=8.5, color="#333")
    return finish(fig, f"{SRC}/prepared/manifest.json")


def column_counts():
    """Each member's model columns. Source: results/mci_loan/v2/prepared/manifest.json and columns.json."""
    ms = data.members()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    vals = [len(m["columns"]) for m in ms][::-1]
    bars = ax.bar([S(m["nameFa"]) for m in ms][::-1], vals,
                  color=[COLORS["alone"] if m["role"] == "active" else COLORS[m["key"]] for m in ms][::-1])
    bar_labels(ax, bars, vals)
    right_axis(ax)
    fa_axis(ax)
    ax.set_ylabel(S("شمار ستون‌های مدل"))
    ax.set_title(S("سهم هر عضو از ستون‌ها"), loc="right")
    return finish(fig, f"{SRC}/prepared/{{manifest,columns}}.json")


# ---- 2. MCI alone and with partners -----------------------------------------------------------------------------

def quality():
    """macro-F1 x 10^4 on the test split, mean +- SD over 10 seeds. Source: results/mci_loan/v2/offchain/loo_summary.csv."""
    lo = data.off_summary()["loo"]
    key = {"FULL": "U_fed", "LOO_1": "F_LOO_1", "LOO_2": "F_LOO_2", "LOO_3": "F_LOO_3", "BASELINE": "F_C1a"}
    items = [(key[f], lab, c) for f, lab, c in models_all()]
    items.insert(-1, ("F_C1b", f"{data.name_fa('mci')}\nآموزش جداگانه، C1b", COLORS["c1b"]))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    m = [lo[k]["mean"] for k, _, _ in items]
    sd = [lo[k]["sd"] for k, _, _ in items]
    bars = ax.bar([S(wrap(t)) for _, t, _ in items][::-1], m[::-1], yerr=sd[::-1], capsize=4, color=[c for _, _, c in items][::-1])
    bar_labels(ax, bars, m[::-1], err=sd[::-1])
    ax.set_ylim(5000, max(m) + 600)
    right_axis(ax)
    fa_axis(ax)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylabel(S("macro-F1 × ۱۰⁴ روی مجموعهٔ آزمون"))
    ax.set_title(S(f"{data.name_fa('mci')} به‌تنهایی و با شرکا: میانگین ده بذر، خط خطا یک انحراف معیار"), loc="right")
    return finish(fig, f"{SRC}/offchain/loo_summary.csv")


def auc():
    """ROC-AUC (off-chain), all test rows and thin_file. Source: results/mci_loan/v2/offchain/auc_summary.csv."""
    a = data.off_summary()["auc"]
    models = [(f, lab) for f, lab, _ in models_all()]
    segs = [("all", "همهٔ متقاضیان آزمون", COLORS["full"]), ("thin_file", "بدون سابقهٔ اعتباری (thin_file)", COLORS["thin"])]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(models))[::-1]
    for j, (seg, lab, c) in enumerate(segs):
        v = [a[f"{seg}/{m}"]["mean"] for m, _ in models]
        e = [a[f"{seg}/{m}"]["sd"] for m, _ in models]
        bars = ax.bar(x + (0.2 if j == 0 else -0.2), v, 0.38, yerr=e, capsize=3, color=c, label=S(lab))
        bar_labels(ax, bars, v, d=3, err=e, fontsize=7.5)
    ax.set_xticks(x, [S(wrap(t)) for _, t in models], fontsize=8.5)
    ax.set_ylim(0.5, 0.82)
    right_axis(ax)
    fa_axis(ax, d=2)
    ax.set_ylabel("ROC-AUC")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(S("قدرت تفکیک (AUC)، میانگین ده بذر"), loc="right")
    return finish(fig, f"{SRC}/offchain/auc_summary.csv", offchain=True)


def business():
    """The primary metric (default rate among approved at MCI alone's approval rate) and the secondary one (extra
    approvals per 10,000 applicants at MCI alone's default rate), all test rows and thin_file.
    Source: results/mci_loan/v2/offchain/business_summary.csv."""
    b = data.off_summary()["business"]
    models = models_all()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    w = 0.16
    for ax, q, title, pct in ((axes[1], "default_rate_among_approved", f"نرخ نکول در میان تأییدشدگان (نرخ تأیید {alone()} ثابت)", True),
                              (axes[0], "extra_per_10000", f"تأیید اضافه در هر ۱۰٬۰۰۰ متقاضی (نرخ نکول {alone()} ثابت)", False)):
        x = np.arange(2)[::-1]
        for j, (m, lab, c) in enumerate(models):
            v = [b[f"{seg}/{m}"][q]["mean"] for seg in ("all", "thin_file")]
            e = [b[f"{seg}/{m}"][q]["sd"] for seg in ("all", "thin_file")]
            bars = ax.bar(x + (2 - j) * w, v, w * 0.95, yerr=e, capsize=2, color=c, label=S(lab))
            bar_labels(ax, bars, v, d=1 if pct else 0, pct=pct, err=e, fontsize=6.5)
        ax.set_xticks(x, [S("همهٔ متقاضیان"), S("بدون سابقهٔ اعتباری")])
        right_axis(ax)
        fa_axis(ax, d=0, pct=pct)
        ax.set_title(S(title), loc="right", fontsize=9.5)
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], loc="upper center", frameon=False, fontsize=8.5, ncol=len(models), bbox_to_anchor=(0.5, 1.0))
    fig = finish(fig, f"{SRC}/offchain/business_summary.csv", offchain=True)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    return fig


def partner_value():
    """What each partner adds to MCI: its leave-one-out contribution l_i (macro-F1 x 10^4), and the business metrics'
    change from adding it (all members against all but that partner), mean over 10 seeds.
    Sources: results/mci_loan/v2/offchain/loo_summary.csv, business_summary.csv."""
    lo = data.off_summary()["loo"]
    b = data.off_summary()["business"]
    keys = data.passive_keys()
    labels = [S(data.name_fa(k)) for k in keys][::-1]
    cols = [COLORS[k] for k in keys][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    v = [lo[f"l_{i}"]["mean"] for i in (1, 2, 3)][::-1]
    e = [lo[f"l_{i}"]["sd"] for i in (1, 2, 3)][::-1]
    bars = axes[2].bar(labels, v, 0.55, yerr=e, capsize=4, color=cols)
    bar_labels(axes[2], bars, v, d=1, err=e)
    axes[2].set_title(S("سهم هر شریک: lᵢ = U_fed − LOOᵢ"), loc="right", fontsize=9.5)
    axes[2].set_ylabel("macro-F1 × 10⁴")
    for ax, q, title, sign, pct in ((axes[1], "default_rate_among_approved", "کاهش نرخ نکول تأییدشدگان (واحد درصد)", -1, True),
                                    (axes[0], "extra_per_10000", "تأیید اضافه در هر ۱۰٬۰۰۰ متقاضی", 1, False)):
        v = [sign * (b[f"all/FULL"][q]["mean"] - b[f"all/LOO_{i}"][q]["mean"]) for i in (1, 2, 3)][::-1]
        bars = ax.bar(labels, v, 0.55, color=cols)
        bar_labels(ax, bars, v, d=2 if pct else 0, pct=pct)
        ax.set_title(S(title), loc="right", fontsize=9.5)
        fa_axis(ax, d=1 if pct else 0, pct=pct)
    for ax in axes:
        right_axis(ax)
        ax.tick_params(axis="x", labelsize=8.5)
    fa_axis(axes[2])
    return finish(fig, f"{SRC}/offchain/loo_summary.csv, business_summary.csv", offchain=True)


# ---- 3. credit tiers --------------------------------------------------------------------------------------------

def tiers():
    """The credit policy: each tier's weighted share of subscribers and its weighted default rate, MCI alone against
    MCI with all partners, all test rows and thin_file, mean over 10 seeds. Source: results/mci_loan/v2/offchain/tiers_summary.csv."""
    t = data.off_summary()["tiers"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    groups = [("all", "BASELINE"), ("all", "FULL"), ("thin_file", "BASELINE"), ("thin_file", "FULL")]
    glab = [S(f"{'همهٔ متقاضیان' if seg == 'all' else 'بدون سابقهٔ اعتباری'}\n{'به‌تنهایی' if m == 'BASELINE' else 'با شرکا'}") for seg, m in groups]
    x = np.arange(len(groups))[::-1]
    ax = axes[1]
    bottom = np.zeros(len(groups))
    for key, lab in TIERS:
        v = np.array([t[f"{seg}/{m}/{key}"]["share"]["mean"] for seg, m in groups])
        ax.bar(x, v, 0.6, bottom=bottom, color=COLORS[key], label=S(lab))
        for xi, b0, vi in zip(x, bottom, v):
            if vi > 0.04:
                ax.text(xi, b0 + vi / 2, num(vi, 0, pct=True), ha="center", va="center", fontsize=8, color="white")
        bottom += v
    ax.set_xticks(x, glab, fontsize=8.5)
    right_axis(ax)
    fa_axis(ax, pct=True)
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper center", frameon=False, ncol=3, fontsize=8.5)
    ax.set_title(S("سهم مشترکان در هر سطح اعتبار"), loc="right", fontsize=10)
    ax = axes[0]
    w = 0.26
    for j, (key, lab) in enumerate(TIERS):
        v = [t[f"{seg}/{m}/{key}"]["default_rate"]["mean"] for seg, m in groups]
        bars = ax.bar(x + (1 - j) * w, v, w * 0.95, color=COLORS[key], label=S(lab))
        bar_labels(ax, bars, v, d=1, pct=True, fontsize=7)
    ax.set_xticks(x, glab, fontsize=8.5)
    right_axis(ax)
    fa_axis(ax, pct=True)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    ax.set_title(S("نرخ نکول در هر سطح (وزن‌دهی‌شده)"), loc="right", fontsize=10)
    lo, hi = data.off_summary()["tierRule"]["cuts"]
    fig.suptitle(S(f"سیاست اعتباری: اعتبار چهارماهه اگر احتمال نکول کمتر از {num(lo, 0, pct=True)} باشد، یک‌ماهه اگر کمتر از "
                   f"{num(hi, 0, pct=True)}، وگرنه بدون اعتبار"), x=0.99, ha="right", fontsize=10.5)
    return finish(fig, f"{SRC}/offchain/tiers_summary.csv", offchain=True)


# ---- 4. what MCI pays each partner ------------------------------------------------------------------------------

def payments_offchain():
    """At the proposal's pool B (escrowed by MCI): the Talmud award of each partner in each off-chain seed, and what
    stays with MCI. Source: results/mci_loan/v2/offchain/payments.csv."""
    p = data.off_csv("payments.csv")
    seeds = sorted(p.seed.unique())
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(seeds))
    for i, key in enumerate(data.passive_keys(), 1):
        v = np.array([p[(p.seed == s) & (p.member_index == i)].tokens.iloc[0] for s in seeds]) / 1e3
        ax.bar([fa_digits(s) for s in seeds], v, bottom=bottom, color=COLORS[key], label=S(data.name_fa(key)))
        bottom += v
    keep = np.array([p[(p.seed == s) & (p.member_index == 0)].tokens.iloc[0] for s in seeds]) / 1e3
    ax.bar([fa_digits(s) for s in seeds], keep, bottom=bottom, color=COLORS["alone"], label=S(f"نزد {data.name_fa('mci')} می‌ماند"))
    right_axis(ax)
    fa_axis(ax)
    ax.invert_xaxis()
    ax.set_xlabel(S("بذر (خارج از زنجیره، کل مجموعهٔ آزمون)"))
    ax.set_ylabel(S("هزار توکن"))
    ax.set_ylim(top=(bottom + keep).max() * 1.22)
    ax.legend(loc="upper center", frameon=False, fontsize=8.5, ncol=4)
    ax.set_title(S(f"پرداخت {data.name_fa('mci')} به هر شریک از استخر B = ۱٬۰۰۰٬۰۰۳ توکن (قاعدهٔ تلمودی)"), loc="right", fontsize=10)
    return finish(fig, f"{SRC}/offchain/payments.csv")


def settlement(k: int):
    """On-chain run k: the six accepted figures, then claims c_i = max(0, U_fed - LOO_i), the estate E' = min(E, floor(B
    / rho)) and the Talmud awards. Source: results/mci_loan/v2/onchain/summary.json (run k)."""
    r = data.on_summary()["runs"][str(k)]
    d = r["division"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    F = r["figures"]
    order = [("BASELINE", "U_A"), ("LOO_3", "LOO₃"), ("LOO_2", "LOO₂"), ("LOO_1", "LOO₁"), ("F_j", "F_j"), ("F_r", "F_r")]
    keys = data.passive_keys()
    ax = axes[1]
    v = [F[n] for n, _ in order]
    bars = ax.bar([t for _, t in order], v, color=[COLORS["alone"], COLORS[keys[2]], COLORS[keys[1]], COLORS[keys[0]], COLORS["fullj"], COLORS["full"]])
    bar_labels(ax, bars, v)
    ax.set_ylim(min(v) - 600, max(v) + 300)
    right_axis(ax)
    fa_axis(ax)
    ax.set_ylabel("macro-F1 × 10⁴")
    ax.set_title(S(f"اجرای {fa_digits(k)}: رقم‌های پذیرفته‌شده با اثبات"), loc="right", fontsize=10)
    ax = axes[0]
    x = np.arange(3)[::-1]
    ax.bar(x + 0.2, d["claims"], 0.38, color="#ccc", label=S("ادعا cᵢ"))
    bars = ax.bar(x - 0.2, d["awards"], 0.38, color=COLORS["full"], label=S("سهم تلمودی aᵢ"))
    bar_labels(ax, bars, d["awards"])
    ax.set_xticks(x, [S(data.name_fa(k_)) for k_ in keys], fontsize=8.5)
    right_axis(ax)
    fa_axis(ax)
    ax.set_ylim(top=max(max(d["claims"]), 1) * 1.3)  # room for the legend
    ax.legend(loc="upper center", frameon=False, fontsize=8.5, ncol=2)
    ax.set_title(S(f"E = {num(d['E'])}، E′ = {num(d['E_prime'])}، سقف استخر {num(d['B_over_rho'])} واحد؛ هر واحد {num(d['rho'])} توکن"),
                 loc="right", fontsize=9.5)
    return finish(fig, f"{SRC}/onchain/summary.json (run {k})")


def payouts():
    """Every on-chain run's credited tokens per member (the awards, and each passive's returned deposit D_p).
    Source: results/mci_loan/v2/onchain/payouts.csv."""
    p = data.on_csv("payouts.csv")
    runs = sorted(p.run.unique())
    fig, ax = plt.subplots(figsize=(8.5, 4))
    bottom = np.zeros(len(runs))
    keys = data.passive_keys()
    for who, key, c in [("A", "mci", COLORS["alone"])] + [(f"p{i}", keys[i - 1], COLORS[keys[i - 1]]) for i in (1, 2, 3)]:
        v = np.array([int(p[(p.run == k) & (p.party == who)].credited_tokens.iloc[0]) for k in runs]) / 1e3
        ax.bar([fa_digits(k) for k in runs], v, bottom=bottom, color=c, label=S(data.name_fa(key)))
        bottom += v
    right_axis(ax)
    fa_axis(ax)
    ax.set_xlabel(S("اجرای روی زنجیره"))
    ax.set_ylabel(S("هزار توکن"))
    ax.invert_xaxis()
    ax.set_ylim(top=bottom.max() * 1.22)  # room for the legend
    ax.legend(loc="upper center", frameon=False, fontsize=8.5, ncol=4)
    ax.set_title(S("پرداخت به هر عضو در هر اجرا (B = ۱٬۰۰۰٬۰۰۳ توکن به‌اضافهٔ ودیعه‌های بازگشتی)"), loc="right", fontsize=10)
    return finish(fig, f"{SRC}/onchain/payouts.csv")


# ---- 6. cost ----------------------------------------------------------------------------------------------------

SERIES = (("loan_v2", "وام، نسخهٔ ۲", COLORS["v2"]), ("loan_v1", "وام، نسخهٔ ۱", COLORS["v1"]), ("electricity", "برق (پایان‌نامه)", COLORS["elec"]))


def gas():
    """Gas per phase, mean over the five runs: v2 against v1 and the electricity scenario (the thesis's Chapter 5 series).
    Source: results/mci_loan/v2/onchain/cost_comparison.csv."""
    c = data.on_csv("cost_comparison.csv").set_index("quantity")
    fig, ax = plt.subplots(figsize=(9, 4))
    keys = list(PHASES)[::-1]
    x = np.arange(len(keys))
    for j, (col, lab, colr) in enumerate(SERIES):
        v = [c.loc[f"gas: {k}", f"{col}_mean"] / 1e6 for k in keys]
        bars = ax.bar(x + (1 - j) * 0.27, v, 0.26, color=colr, label=S(lab))
        bar_labels(ax, bars, v, d=2, fontsize=7)
    ax.set_xticks(x, [S(PHASES[k]) for k in keys])
    right_axis(ax)
    fa_axis(ax, d=1)
    ax.set_ylabel(S("میلیون gas"))
    ax.legend(loc="upper left", frameon=False)
    t = {col: c.loc["gas: total", f"{col}_mean"] / 1e6 for col, _, _ in SERIES}
    ax.set_title(S(f"gas هر مرحله؛ مجموع: نسخهٔ ۲ {num(t['loan_v2'], 2)}، نسخهٔ ۱ {num(t['loan_v1'], 2)} و برق {num(t['electricity'], 2)} میلیون"),
                 loc="right", fontsize=9.5)
    return finish(fig, f"{SRC}/onchain/cost_comparison.csv")


def proof_time():
    """Proof time per figure (rapidsnark) and the prover's peak memory, v2 against v1 and electricity.
    Source: results/mci_loan/v2/onchain/cost_comparison.csv."""
    c = data.on_csv("cost_comparison.csv").set_index("quantity")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, q, title, scale, d in ((axes[1], "proof time per figure (ms, rapidsnark)", "زمان تولید هر اثبات (ثانیه)", 1e-3, 2),
                                   (axes[0], "prover peak memory per figure (MB)", "بیشینهٔ حافظهٔ اثبات‌گر (مگابایت)", 1, 0)):
        v = [c.loc[q, f"{col}_mean"] * scale for col, _, _ in SERIES]
        e = [c.loc[q, f"{col}_sd"] * scale for col, _, _ in SERIES]
        bars = ax.bar([S(lab) for _, lab, _ in SERIES], v, 0.5, yerr=e, capsize=4, color=[col for _, _, col in SERIES])
        bar_labels(ax, bars, v, d=d, err=e)
        right_axis(ax)
        fa_axis(ax, d=1 if d else 0)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_title(S(title), loc="right", fontsize=10)
    return finish(fig, f"{SRC}/onchain/cost_comparison.csv")


# ---- 7. privacy -------------------------------------------------------------------------------------------------

def privacy():
    """The privacy-utility curve: U_fed and U_A against epsilon per release (sigma calibrated to it), mean +- SD over
    10 seeds, and the clipping-only runs apart. Source: results/mci_loan/v2/offchain/ldp_eps_summary.csv."""
    t = data.off_csv("ldp_eps_summary.csv")
    eps = [1, 4, 16, 64]
    conds = [f"ε = {e} per release" for e in eps]
    off = "LDP off (clipping only, σ = 0)"
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = [math.log2(e) for e in eps]
    x_off = math.log2(64) + 2
    for q, lab, c, mk in (("U_fed", f"{with_all()} (U_fed)", COLORS["full"], "o"), ("BASELINE", f"{alone()} (U_A)", COLORS["alone"], "s")):
        row = lambda cond: t[(t.condition == cond) & (t.quantity == q)].iloc[0]  # noqa: E731
        m = [row(cd)["mean"] for cd in conds]
        sd = [row(cd)["sd"] for cd in conds]
        ax.errorbar(xs, m, yerr=sd, color=c, marker=mk, capsize=3, label=S(lab))
        ax.errorbar([x_off], [row(off)["mean"]], yerr=[row(off)["sd"]], color=c, marker=mk, capsize=3)
    ax.set_xticks(xs + [x_off], [fa_digits(e) for e in eps] + [S("بدون نوفه")])
    ax.axvline(math.log2(64) + 1, color="#ccc", lw=0.8)
    ax.invert_xaxis()
    right_axis(ax)
    fa_axis(ax)
    ax.set_xlabel(S("ε در هر انتشار (σ برای آن تنظیم شده است)"))
    ax.set_ylabel(S("macro-F1 × ۱۰⁴"))
    ax.legend(loc="lower left", frameon=False)
    ax.set_title(S("مبادلهٔ حریم خصوصی و کارایی (LDP روی تعبیه‌های شرکا)، میانگین ده بذر"), loc="right")
    return finish(fig, f"{SRC}/offchain/ldp_eps_summary.csv")


def figures_all() -> dict:
    """name -> builder, for the export; the on-chain figures when Step 5's results exist, the privacy figure when the
    LDP runs do."""
    out = {"01-architecture": architecture, "02-column-counts": column_counts, "03-quality": quality, "04-auc": auc,
           "05-business": business, "06-partner-value": partner_value, "07-tiers": tiers, "08-payments-offchain": payments_offchain}
    if data.off_summary().get("ldpIncluded"):
        out["12-privacy"] = privacy
    if data.on_available():
        out.update({"09-payouts": payouts, "10-gas": gas, "11-proof-time": proof_time})
        for k in data.on_summary()["runsDone"]:
            out[f"13-settlement-run-{k}"] = (lambda k=k: settlement(k))
    return out
