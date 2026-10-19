"""MCI loan dashboard (v2, MCI-centric): the eight pages, each a function. Everything shown is read from
results/mci_loan/v2/ (data.py); every chart is a dashboard/charts.py figure; numbers are shown with Persian digits.
MCI is the lender and the active member; the Bank, the E-commerce company and the Credit bureau are its partners.
"""
from __future__ import annotations

import base64

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import charts
import data
from fa import FONT_BOLD, FONT_REGULAR, fa_digits, num

BANNER = ("**سازمان‌ها شبیه‌سازی شده‌اند.** همراه اول در این سناریو وام‌دهنده است. همهٔ سازمان‌های شریک (بانک، شرکت تجارت "
          "الکترونیک و شرکت اعتبارسنجی) نقش‌هایی عمومی و شبیه‌سازی‌شده‌اند، تعریف‌شده روی یک مجموعه‌دادهٔ عمومی: "
          "Kaggle Home Credit Default Risk. دادهٔ هیچ شرکت واقعی، از جمله همراه اول، به کار نرفته است، و هیچ عددی در اینجا "
          "ارزش دادهٔ واقعی یک سازمان را نمی‌سنجد.")
OFFCHAIN = ("AUC، شاخص‌های کسب‌وکار، احتمال نکول و سطوح اعتبار **خارج از زنجیره و بدون اثبات ZK** حساب شده‌اند؛ رقم‌های "
            "macro-F1 همان‌هایی‌اند که مدار روی زنجیره اثبات می‌کند.")
NO_TELECOM = ("**دادهٔ مخابراتی در کار نیست.** مجموعه‌دادهٔ عمومی هیچ رفتار مخابراتی ندارد: نه مصرف، نه شارژ، نه قبض. پس "
              "«همراه اول به‌تنهایی» در اینجا مدل درخواست وامِ یک وام‌دهنده است، نه مدل یک اپراتور. در یک پایلوت، دادهٔ "
              "مخابراتی واقعی همراه اول به سمت خود همراه اول افزوده می‌شود؛ فدراسیون، اثبات‌ها و تسویه تغییری نمی‌کنند.")
ROLE = {"active": "فعال: وام‌دهنده، صاحب برچسب، سازندهٔ فدراسیون، تأمین‌کنندهٔ استخر پاداش", "passive": "شریک منفعل: ارسال تعبیه‌ها"}
PHASE = charts.PHASES
SRC = "results/mci_loan/v2"


def style() -> None:
    """Right to left, Vazirmatn (embedded, so the page needs no font server), Latin runs left to right."""
    faces = "".join(
        f"@font-face{{font-family:'Vazirmatn';src:url(data:font/ttf;base64,{base64.b64encode(p.read_bytes()).decode()}) "
        f"format('truetype');font-weight:{w};}}" for p, w in ((FONT_REGULAR, 400), (FONT_BOLD, 700)))
    st.markdown(f"""<style>{faces}
html, body, .stApp, .stMarkdown, p, li, h1, h2, h3, h4, label, button, input, select, textarea,
[data-testid="stSidebar"], [data-testid="stMetric"] {{font-family: 'Vazirmatn', sans-serif !important;}}
.stApp, .stMarkdown, [data-testid="stSidebar"], [data-testid="stMetric"], [data-testid="stAlert"],
[data-testid="stExpander"], [data-testid="stCaptionContainer"] {{direction: rtl; text-align: right;}}
code, pre, .ltr {{direction: ltr; unicode-bidi: embed; font-family: Consolas, monospace !important;}}
</style>""", unsafe_allow_html=True)


def banner() -> None:
    st.warning(BANNER, icon="⚠️")


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def table(df: pd.DataFrame) -> None:
    st.dataframe(df, hide_index=True)


def onchain_missing() -> bool:
    if not data.on_available():
        st.info(f"نتایج اجراهای روی زنجیره (گام ۵) هنوز در {SRC}/onchain/ نیست.")
        return True
    return False


def mci() -> str:
    return data.name_fa("mci")


def party_names() -> dict[str, str]:
    """The run's process names (A, p1-p3, arbiter, operator) to Persian member names."""
    keys = data.passive_keys()
    return {"A": mci(), **{f"p{i}": data.name_fa(keys[i - 1]) for i in (1, 2, 3)}, "arbiter": "داور", "operator": "اپراتور"}


def model_labels() -> list[tuple[str, str]]:
    """(figure, Persian label): MCI with all partners, without each partner, alone, and C1b."""
    return [(f, lab) for f, lab, _ in charts.models_all()][:-1] + [("BASELINE", charts.alone()), ("C1b", f"{mci()}، آموزش جداگانه (C1b)")]


# ---- 1 ------------------------------------------------------------------------------------------------------------

def consortium() -> None:
    st.title("کنسرسیوم")
    st.markdown(f"{mci()} به مشترکانش اعتبار کوتاه‌مدت می‌دهد و باید بازپرداخت را پیش‌بینی کند؛ پس عضو فعال است: صاحب "
                "برچسب، سازندهٔ فدراسیون و تأمین‌کنندهٔ استخر پاداش. سه شریک منفعل تعبیه می‌فرستند. هر عضو یک گرهٔ اعتبارسنج "
                f"در شبکهٔ Besu (اجماع QBFT) دارد. منبع: `{SRC}/prepared/manifest.json`.")
    rows = [{"عضو": m["nameFa"], "Member": m["name"], "نقش": ROLE[m["role"]], "ستون‌های مدل": fa_digits(len(m["columns"])),
             "جدول‌های منبع": "، ".join(m["sources"])} for m in data.members()]
    table(pd.DataFrame(rows))
    st.subheader("ستون‌های هر عضو")
    for m in data.members():
        with st.expander(f"{m['nameFa']}: {fa_digits(len(m['columns']))} ستون"):
            st.markdown(" ".join(f"`{c}`" for c in m["columns"]))
    st.subheader("دو صفحهٔ معماری")
    show(charts.architecture())
    show(charts.column_counts())


# ---- 2 ------------------------------------------------------------------------------------------------------------

def alone_vs_partners() -> None:
    st.title(f"{mci()} به‌تنهایی و با شرکا")
    st.markdown(f"میانگین ده بذر روی ۳٬۱۹۳ متقاضی آزمون (`{SRC}/offchain/`).")
    st.info(NO_TELECOM)
    lo = data.off_summary()["loo"]
    c = st.columns(4)
    for col, (k, lab) in zip(c, (("U_fed", f"{mci()} با همهٔ شرکا (U_fed)"), ("E", "افزایش با شرکا (E)"),
                                 ("F_C1b", f"{mci()}، آموزش جداگانه (C1b)"), ("F_C1a", f"{mci()} به‌تنهایی (U_A)"))):
        col.metric(lab, num(lo[k]["mean"], 1), help="macro-F1 × ۱۰⁴، میانگین ده بذر")
    show(charts.quality())
    st.info(OFFCHAIN)
    show(charts.auc())
    st.subheader("شاخص‌های کسب‌وکار")
    st.markdown(f"**اصلی:** نرخ تأیید {mci()} به‌تنهایی ثابت؛ نرخ نکول در میان تأییدشدگان. **فرعی:** نرخ نکول {mci()} "
                "به‌تنهایی ثابت؛ تأیید اضافه در هر ۱۰٬۰۰۰ متقاضی. هر دو با وزن‌دهی به نرخ نکول واقعی application_train (۸٫۰۷٪).")
    show(charts.business())
    b = data.off_summary()["business"]
    rows = []
    for seg, sl in (("all", "همهٔ متقاضیان"), ("thin_file", "بدون سابقهٔ اعتباری")):
        for m, ml in model_labels():
            v = b.get(f"{seg}/{m}")
            if v:
                rows.append({"بخش": sl, "مدل": ml, "نرخ تأیید": num(v["approval_rate_r"]["mean"], 1, pct=True),
                             "نرخ نکول تأییدشدگان": num(v["default_rate_among_approved"]["mean"], 2, pct=True),
                             "تأیید اضافه در ۱۰٬۰۰۰": num(v["extra_per_10000"]["mean"])})
    table(pd.DataFrame(rows))
    st.caption(f"منبع: {SRC}/offchain/business_summary.csv")
    st.subheader("ارزش هر شریک برای همراه اول")
    show(charts.partner_value())


# ---- 3 ------------------------------------------------------------------------------------------------------------

def tiers() -> None:
    st.title("سطوح اعتبار")
    rule = data.off_summary()["tierRule"]
    lo, hi = rule["cuts"]
    st.markdown(f"سیاست اعتباری {mci()} روی امتیاز هر مدل: احتمال نکول (PD) با رگرسیون لجستیک روی یک نیمهٔ متقاضیان آزمون "
                f"کالیبره و روی نیمهٔ دیگر گزارش می‌شود، سپس جای دو نیمه عوض و نتیجه‌ها یکی می‌شوند؛ PD به نرخ نکول واقعی "
                f"application_train تصحیح می‌شود. **اعتبار چهارماهه** اگر PD کمتر از {num(lo, 0, pct=True)}، **اعتبار یک‌ماهه** "
                f"اگر کمتر از {num(hi, 0, pct=True)}، وگرنه **بدون اعتبار**.")
    st.info(OFFCHAIN)
    show(charts.tiers())
    t = data.off_summary()["tiers"]
    names = dict(charts.TIERS)
    for seg, sl in (("all", "همهٔ متقاضیان"), ("thin_file", "بدون سابقهٔ اعتباری")):
        st.subheader(sl)
        rows = []
        for m, ml in model_labels():
            r = {"مدل": ml}
            for key in [k for k, _ in charts.TIERS] + ["granted"]:
                v = t.get(f"{seg}/{m}/{key}")
                lab = names.get(key, "کل اعتبار داده‌شده")
                r[f"{lab}: سهم"] = num(v["share"]["mean"], 1, pct=True) if v else "–"
                dr = v.get("default_rate") if v else None
                r[f"{lab}: نرخ نکول"] = num(dr["mean"], 1, pct=True) if dr else "–"
            rows.append(r)
        table(pd.DataFrame(rows))
    a = {k: t[f"all/{k}/all"] for k in ("BASELINE", "FULL")}
    st.markdown("**وارسی کالیبراسیون** (همهٔ متقاضیان آزمون): میانگین وزنی PD در برابر نرخ نکول وزنی: " +
                "؛ ".join(f"{lab} {num(a[k]['mean_pd']['mean'], 2, pct=True)} در برابر {num(a[k]['default_rate']['mean'], 2, pct=True)}"
                          for k, lab in (("BASELINE", charts.alone()), ("FULL", charts.with_all()))) + ".")
    at = data.off_summary()["age_tiers"]
    bands = list(data.off_summary()["age_bands"]["BASELINE"])
    st.subheader("سهم هر گروه سنی در اعتبار چهارماهه")
    table(pd.DataFrame([{"مدل": lab, **{fa_digits(b): num(at[f"{m}/{b}/credit4m"]["mean"], 1, pct=True) for b in bands}}
                        for m, lab in (("BASELINE", charts.alone()), ("FULL", charts.with_all()))]))
    st.caption(f"منبع: {SRC}/offchain/tiers_summary.csv، age_tiers_summary.csv، calibration.csv")


# ---- 4 ------------------------------------------------------------------------------------------------------------

def payments() -> None:
    st.title(f"پرداخت {mci()} به شرکا")
    st.markdown(f"{mci()} هنگام ساختن هر اجرا استخر B = ۱٬۰۰۰٬۰۰۳ توکن را سپرده می‌کند. ادعای هر شریک سهم ترک‌یک‌عضو "
                "آن است (lᵢ = U_fed − LOOᵢ، کف صفر)؛ دارایی E′ = min(E, ⌊B/ρ⌋) با قاعدهٔ تلمودی میان ادعاها تقسیم و هر واحد "
                f"ρ = ۵٬۰۰۰ توکن پرداخت می‌شود؛ باقی‌مانده نزد {mci()} می‌ماند.")
    pay = data.off_summary()["payments"]
    keys = data.passive_keys()
    name_en = {m["key"]: m["name"] for m in data.members()}
    c = st.columns(4)
    for col, key in zip(c, keys + ["mci"]):
        v = pay[name_en[key]]
        col.metric(data.name_fa(key) if key != "mci" else f"نزد {mci()} می‌ماند", num(v["mean"]), help="توکن، میانگین ده بذر خارج از زنجیره")
    show(charts.payments_offchain())
    st.caption(f"منبع: {SRC}/offchain/payments.csv، payments_summary.csv (کل مجموعهٔ آزمون، ده بذر)")
    if data.on_available():
        st.subheader("پرداخت‌ها در اجراهای روی زنجیره")
        p = data.on_csv("payouts.csv")
        rows = []
        for k in sorted(p.run.unique()):
            q = p[p.run == k].set_index("party")
            rows.append({"اجرا": fa_digits(k), **{data.name_fa(keys[i - 1]): num(int(q.loc[f"p{i}", "credited_tokens"])) for i in (1, 2, 3)},
                         f"{mci()}": num(int(q.loc["A", "credited_tokens"]))})
        table(pd.DataFrame(rows))
        st.caption(f"منبع: {SRC}/onchain/payouts.csv؛ هر شریک ودیعهٔ مشارکت D_p = ۱٬۰۰۱ را هم پس می‌گیرد.")
        show(charts.payouts())


# ---- 5 ------------------------------------------------------------------------------------------------------------

def settlement() -> None:
    st.title("تسویه")
    if onchain_missing():
        return
    S = data.on_summary()
    k = st.selectbox("اجرا", S["runsDone"], format_func=lambda x: f"اجرای {fa_digits(x)}")
    r = S["runs"][str(k)]
    d = r["division"]
    st.markdown(f"وضعیت نهایی **{r['finalState']}**، مسیر تسویه **{r['path']}**؛ همهٔ رقم‌ها با اثبات پذیرفته شدند: "
                f"**{'بله' if r['accepted'] else 'خیر'}**؛ بازسازی مستقل برابر زنجیره: **{'بله' if r['replayEqualsChain'] else 'خیر'}**.")
    show(charts.settlement(k))
    keys = data.passive_keys()
    table(pd.DataFrame({"شریک": [data.name_fa(x) for x in keys], "LOO (macro-F1 × ۱۰⁴)": [num(r["figures"][f"LOO_{i}"]) for i in (1, 2, 3)],
                        "سهم lᵢ = U_fed − LOOᵢ": [num(x) for x in d["l"]], "ادعا cᵢ": [num(x) for x in d["claims"]],
                        "سهم تلمودی aᵢ": [num(x) for x in d["awards"]], "توکن (ρ × aᵢ)": [num(x) for x in d["award_tokens"]]}))
    st.markdown(f"U_fed = max(F_r, F_j) = **{num(d['U_fed'])}**، U_A = **{num(r['figures']['BASELINE'])}**، "
                f"E = U_fed − U_A = **{num(d['E'])}**، E′ = min(E, ⌊B/ρ⌋) = **{num(d['E_prime'])}**؛ نزد {mci()} می‌ماند: **{num(d['to_A'])}** توکن.")
    p = data.on_csv("payouts.csv")
    q = p[p.run == k]
    table(pd.DataFrame({"عضو": [party_names()[w] for w in q.party],
                        "واریز (توکن)": [num(int(x)) for x in q.credited_tokens],
                        "برداشت (توکن)": [num(int(x)) for x in q.withdrawn_tokens], "باقی‌مانده": [num(int(x)) for x in q.owed_at_end]}))
    cons = r["conservation"]
    st.markdown(f"پایستگی توکن: ودیعه‌ها {num(cons['deposits'])} = برداشت‌ها {num(cons['withdrawnTotal'])} + موجودی قرارداد "
                f"{num(cons['contractBalance'])} — **{'برقرار' if cons['conserved'] else 'برقرار نیست'}**.")
    st.caption(f"منبع: {SRC}/onchain/summary.json، payouts.csv")


# ---- 6 ------------------------------------------------------------------------------------------------------------

def audit() -> None:
    st.title("ردپای حسابرسی")
    if onchain_missing():
        return
    S = data.on_summary()
    k = st.selectbox("اجرا", S["runsDone"], format_func=lambda x: f"اجرای {fa_digits(x)}", key="audit_run")
    who = party_names()
    ev = data.on_csv("events.csv")
    st.subheader("رویدادهای تسویه")
    e = ev[ev.run == k]
    table(pd.DataFrame({"رویداد": e.event, "بلوک": [fa_digits(int(b)) if pd.notna(b) else "" for b in e.block],
                        "طرف": [who.get(x, x) if isinstance(x, str) else "" for x in e.party], "جزئیات": e.detail.fillna("").astype(str)}))
    st.subheader("وارسی اثبات‌ها")
    pr = data.on_csv("proofs.csv")
    q = pr[pr.run == k]
    table(pd.DataFrame({"رقم": q.figure, "F": [num(x) for x in q.F], "F بازسازی‌شده": [num(x) for x in q.oracle_F],
                        "اثبات‌گر": q.prover, "زمان اثبات (ms)": [num(x) for x in q.prove_ms],
                        "وارسی محلی (ms)": [num(x, 1) for x in q.local_verify_ms], "C_model": q.c_model}))
    st.subheader("تراکنش‌ها به تفکیک مرحله")
    tx = data.on_csv("transactions.csv")
    t = tx[tx.run == k]
    ph = st.selectbox("مرحله", ["همه"] + list(PHASE) + ["test token", "infrastructure"], format_func=lambda x: PHASE.get(x, x))
    if ph != "همه":
        t = t[t.phase == ph]
    table(pd.DataFrame({"مرحله": [PHASE.get(x, x) for x in t.phase], "گام": t.step, "عضو": [who.get(x, x) for x in t.party],
                        "بلوک": [fa_digits(int(b)) for b in t.block], "gas": [num(int(g)) for g in t.gas_used], "tx hash": t.tx_hash}))
    st.caption(f"منبع: {SRC}/onchain/events.csv، proofs.csv، transactions.csv")


# ---- 7 ------------------------------------------------------------------------------------------------------------

def cost() -> None:
    st.title("هزینه")
    if onchain_missing():
        return
    st.markdown("gas هر مرحله و زمان اثبات: این نسخه در کنار نسخهٔ ۱ (بانک فعال) و سناریوی برق پایان‌نامه (سری فصل ۵، همان "
                "perBatchWaitMs = ۱۰٬۰۰۰).")
    show(charts.gas())
    show(charts.proof_time())
    c = data.on_csv("cost_comparison.csv")
    table(pd.DataFrame({"کمیت": c.quantity, "نسخهٔ ۲: میانگین": [num(x, 1) for x in c.loan_v2_mean],
                        "نسخهٔ ۱: میانگین": [num(x, 1) for x in c.loan_v1_mean], "برق: میانگین": [num(x, 1) for x in c.electricity_mean]}))
    st.caption(f"منبع: {SRC}/onchain/cost_comparison.csv")


# ---- 8 ------------------------------------------------------------------------------------------------------------

def privacy() -> None:
    st.title("مبادلهٔ حریم خصوصی و کارایی")
    if not data.off_summary().get("ldpIncluded"):
        st.info("اجراهای LDP هنوز تحلیل نشده‌اند.")
        return
    st.markdown("نوفهٔ گاوسی محلی (LDP) روی تعبیه‌های شرکا؛ σ برای ε در هر انتشار تنظیم شده است (همان کالیبراسیون پایان‌نامه).")
    show(charts.privacy())
    ld = data.off_summary()["ldp"]
    ut = ld["eps"]["utility"]
    sc = data.off_csv("ldp_scores_summary.csv")
    rows = []
    for k, lab in (("off", "بدون نوفه"), ("e64", "ε = ۶۴"), ("e16", "ε = ۱۶"), ("e4", "ε = ۴"), ("e1", "ε = ۱")):
        q = ld["eps"]["full_test"][k]
        cond = "LDP off (clipping only, σ = 0)" if k == "off" else f"ε = {k[1:]} per release"
        auc = sc[(sc.condition == cond) & (sc.model == "FULL") & (sc.quantity == "auc")]["mean"].iloc[0]
        rows.append({"شرط": lab, "U_fed": num(q["U_fed"]["mean"], 1), "U_A": num(q["BASELINE"]["mean"], 1),
                     "بذرهای U_fed > U_A": f"{fa_digits(ut[k]['seeds_above'])} از ۱۰", "AUC با همهٔ شرکا": num(auc, 3)})
    table(pd.DataFrame(rows))
    st.caption(f"منبع: {SRC}/offchain/ldp_eps_summary.csv، ldp_scores_summary.csv، summary.json")
    st.info(OFFCHAIN)


PAGES = [("consortium", "کنسرسیوم", consortium), ("alone", "همراه اول به‌تنهایی و با شرکا", alone_vs_partners),
         ("tiers", "سطوح اعتبار", tiers), ("payments", "پرداخت به شرکا", payments), ("settlement", "تسویه", settlement),
         ("audit", "ردپای حسابرسی", audit), ("cost", "هزینه", cost), ("privacy", "حریم خصوصی", privacy)]
