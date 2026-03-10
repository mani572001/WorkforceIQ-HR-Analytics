import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="WorkforceIQ – HR Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fraunces:ital,wght@0,300;0,700;0,900;1,300&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #05080f; color: #dde3f0; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#080c1a 0%,#0c1120 100%);
    border-right: 1px solid rgba(79,209,197,0.1);
}
[data-testid="stSidebar"] .stRadio > div { gap: 4px !important; }
[data-testid="stSidebar"] .stRadio label {
    color: #7a8aaa !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 13.5px !important;
    padding: 9px 14px !important; border-radius: 10px !important;
    display: flex !important; align-items: center !important;
    transition: all 0.18s !important; border: 1px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #c7d2e8 !important; background: rgba(79,209,197,0.07) !important;
    border-color: rgba(79,209,197,0.15) !important;
}

h1, h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 800 !important; color: #eef2ff !important; }
h1 { font-size: 26px !important; letter-spacing: -0.02em !important; }

.page-banner {
    background: linear-gradient(120deg, #0d1a2e 0%, #0a1525 60%, #0d1f1e 100%);
    border: 1px solid rgba(79,209,197,0.15); border-radius: 18px;
    padding: 30px 36px; margin-bottom: 28px; position: relative; overflow: hidden;
}
.page-banner::after {
    content:''; position:absolute; top:-60px; right:-60px; width:260px; height:260px;
    background: radial-gradient(circle, rgba(79,209,197,0.07) 0%, transparent 70%); pointer-events:none;
}
.page-banner-tag {
    display:inline-flex; align-items:center; gap:6px;
    background: rgba(79,209,197,0.1); border:1px solid rgba(79,209,197,0.2);
    border-radius:20px; padding:4px 14px; font-size:11.5px; color:#4fd1c5;
    font-weight:600; letter-spacing:0.04em; margin-bottom:14px;
}
.page-banner-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:24px; font-weight:800; color:#eef2ff; margin-bottom:6px; line-height:1.2; }
.page-banner-sub { font-size:14px; color:#5a6a85; max-width:620px; line-height:1.6; }

.kpi-card {
    background: linear-gradient(135deg,#0d1525 0%,#101b2e 100%);
    border:1px solid rgba(79,209,197,0.12); border-radius:16px; padding:20px 22px;
    position:relative; overflow:hidden; transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform:translateY(-2px); box-shadow:0 14px 40px rgba(79,209,197,0.1); }
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,#4fd1c5,#667eea,#764ba2);
}
.kpi-label { font-size:11.5px; font-weight:600; letter-spacing:0.07em; text-transform:uppercase; color:#4a5a75; margin-bottom:10px; }
.kpi-val   { font-family:'Plus Jakarta Sans',sans-serif; font-size:32px; font-weight:800; color:#eef2ff; line-height:1; margin-bottom:6px; }
.kpi-sub   { font-size:12px; color:#4a5a75; }
.badge-green { display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(16,185,129,0.12);color:#10b981; }
.badge-red   { display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(239,68,68,0.12);color:#ef4444; }
.badge-blue  { display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(99,102,241,0.12);color:#818cf8; }
.badge-teal  { display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(79,209,197,0.12);color:#4fd1c5; }

.sec-hd   { display:flex;align-items:center;gap:10px;margin:4px 0 18px; }
.sec-dot  { width:8px;height:8px;border-radius:50%;background:linear-gradient(135deg,#4fd1c5,#667eea);box-shadow:0 0 8px rgba(79,209,197,0.5); }
.sec-title{ font-family:'Plus Jakarta Sans',sans-serif;font-size:13px;font-weight:700;color:#8899bb;letter-spacing:0.07em;text-transform:uppercase; }

.glowline { height:1px;background:linear-gradient(90deg,transparent,rgba(79,209,197,0.3),transparent);margin:26px 0; }

.feat-card {
    background:linear-gradient(135deg,#0d1525 0%,#101b2e 100%);
    border:1px solid rgba(79,209,197,0.12); border-radius:14px; padding:22px; height:100%;
}
.feat-icon       { width:46px;height:46px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;margin-bottom:14px; }
.feat-icon-teal  { background:rgba(79,209,197,0.12); }
.feat-icon-indigo{ background:rgba(102,126,234,0.12); }
.feat-icon-amber { background:rgba(245,158,11,0.12); }
.feat-icon-rose  { background:rgba(244,63,94,0.12); }
.feat-title { font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:700;color:#d0daf0;margin-bottom:7px; }
.feat-text  { font-size:12.5px;color:#4a5a75;line-height:1.65; }
.feat-eg    { font-size:11.5px;color:#2d4a5a;margin-top:10px;padding:8px 10px;background:rgba(79,209,197,0.04);border-left:2px solid rgba(79,209,197,0.25);border-radius:0 6px 6px 0; }

[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background:#0d1525 !important; color:#dde3f0 !important;
    border-color:rgba(79,209,197,0.2) !important; border-radius:10px !important;
}
label[data-baseweb="label"] { color:#7a8aaa !important; font-size:12.5px !important; font-weight:600 !important; }
.stSlider > div > div > div > div { background:linear-gradient(90deg,#4fd1c5,#667eea) !important; }

.stButton > button {
    background:linear-gradient(135deg,#4fd1c5 0%,#667eea 100%) !important;
    color:#05080f !important; border:none !important; border-radius:12px !important;
    padding:13px 28px !important; font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:700 !important; font-size:14px !important; letter-spacing:0.01em !important;
    box-shadow:0 4px 20px rgba(79,209,197,0.25) !important; transition:all 0.2s !important; width:100% !important;
}
.stButton > button:hover { transform:translateY(-1px) !important; box-shadow:0 8px 30px rgba(79,209,197,0.35) !important; }

.res-outer { border-radius:18px; padding:30px; text-align:center; margin-bottom:8px; }
.res-label { font-size:11.5px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px; }
.res-value { font-family:'Plus Jakarta Sans',sans-serif;font-size:60px;font-weight:800;line-height:1; }
.res-note  { font-size:12px;color:#4a5a75;margin-top:8px; }

.action-item { display:flex;align-items:flex-start;gap:13px;padding:11px 0;border-bottom:1px solid rgba(79,209,197,0.06); }
.action-ico  { font-size:17px;flex-shrink:0;margin-top:1px; }
.action-txt  { font-size:13px;color:#8899bb;line-height:1.55; }

.explain-card {
    background:#080e1c;border:1px solid rgba(79,209,197,0.1);
    border-radius:14px;padding:18px 20px;margin-bottom:12px;
}
.explain-title { font-size:12px;font-weight:700;color:#4fd1c5;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:10px; }
.explain-body  { font-size:13px;color:#7a8aaa;line-height:1.6; }

.brand-wrap    { padding:20px 16px 24px;border-bottom:1px solid rgba(79,209,197,0.1);margin-bottom:22px; }
.brand-logo    { width:40px;height:40px;background:linear-gradient(135deg,#4fd1c5,#667eea);border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:20px;margin-bottom:12px; }
.brand-name    { font-family:'Plus Jakarta Sans',sans-serif;font-size:17px;font-weight:800;color:#eef2ff;line-height:1; }
.brand-tagline { font-size:11px;color:#4fd1c5;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;margin-top:3px; }

.stat-row { display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(79,209,197,0.06); }
.stat-lbl { font-size:12.5px;color:#3d4f68; }
.stat-val { font-size:12.5px;font-weight:700; }

[data-testid="stExpander"] summary { background:#0d1525 !important; border-radius:10px !important; color:#7a8aaa !important; }
hr { border-color:rgba(79,209,197,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Human-readable field name & tooltip mappings
# ============================================================
FRIENDLY = {
    "OverTime":                 ("Works Overtime?",                       "Does the employee regularly work beyond normal hours?"),
    "MaritalStatus":            ("Marital Status",                        "Employee's current marital status."),
    "DistanceFromHome":         ("Daily Commute Distance (km)",           "How far the employee travels to the office each day."),
    "JobRole":                  ("Job Title / Role",                      "The employee's current position in the company."),
    "Department":               ("Department",                            "Which business unit the employee belongs to."),
    "TotalWorkingYears":        ("Total Years of Work Experience",        "Total professional experience across all employers."),
    "JobLevel":                 ("Seniority Level  (1=Entry → 5=Executive)", "How senior is the employee in the organisation?"),
    "YearsInCurrentRole":       ("Years in Their Current Role",           "How long have they been in this exact position?"),
    "MonthlyIncome":            ("Monthly Salary ($)",                    "Gross monthly pay in US dollars."),
    "Age":                      ("Age",                                   "Employee's current age in years."),
    "YearsWithCurrManager":     ("Years Working with Current Manager",    "How long they have reported to the same manager."),
    "StockOptionLevel":         ("Stock Options Level  (0=None → 3=High)","Level of company equity/stock options the employee holds."),
    "YearsAtCompany":           ("Total Years at This Company",           "Overall tenure at the current organisation."),
    "JobInvolvement":           ("Job Involvement  (1=Low → 4=Very High)","How engaged and invested is the employee in their work?"),
    "Education":                ("Highest Education Level",               "The employee's top academic qualification."),
    "YearsSinceLastPromotion":  ("Years Since Last Promotion",            "How many years ago was their last promotion?"),
    "RelationshipSatisfaction": ("Relationship Satisfaction  (1–4)",      "How satisfied are they with workplace relationships?"),
    "EnvironmentSatisfaction":  ("Work Environment Satisfaction  (1–4)", "Happiness with the physical and cultural work environment."),
    "PercentSalaryHike":        ("Last Salary Increase (%)",              "Percentage pay rise received in the last appraisal."),
    "PerformanceRating":        ("Performance Rating  (1=Low → 4=Excellent)", "The employee's last official performance review score."),
}

def lbl(f):  return FRIENDLY.get(f, (f, ""))[0]
def tip(f):  return FRIENDLY.get(f, ("", ""))[1]

# ============================================================
# Load data
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Employee-Attrition.csv")
    drop_cols = ['EmployeeCount','Over18','StandardHours','EmployeeNumber']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

try:
    df = load_data()
    data_loaded = True
except Exception:
    data_loaded = False
    df = None

label_encoders = {}
if data_loaded:
    for col in ['BusinessTravel','Department','EducationField',
                'Gender','JobRole','MaritalStatus','OverTime','Education']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

@st.cache_data
def summary_stats(df):
    total = len(df)
    if df['Attrition'].dtype == object:
        att = (df['Attrition'] == 'Yes').mean() * 100
    else:
        att = (df['Attrition'] == 1).mean() * 100
    return total, att, df['Age'].mean(), df['MonthlyIncome'].mean(), \
           df['YearsAtCompany'].mean(), df['Department'].value_counts()

# ============================================================
# Chart helpers
# ============================================================
PANEL = '#0a0f1e'

def donut_chart(vals, labels, colors, center_val, center_sub, title):
    fig, ax = plt.subplots(figsize=(4,4), facecolor='none')
    ax.set_facecolor('none')
    ax.pie(vals, colors=colors, startangle=90,
           wedgeprops=dict(width=0.5, edgecolor='#05080f', linewidth=2.5))
    ax.text(0, 0.1, center_val, ha='center', va='center', fontsize=22, fontweight='bold', color='#eef2ff')
    ax.text(0,-0.2, center_sub, ha='center', va='center', fontsize=8.5, color='#4a5a75')
    patches = [mpatches.Patch(color=c, label=l) for c,l in zip(colors,labels)]
    ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5,-0.18),
              ncol=2, frameon=False, fontsize=8, labelcolor='#7a8aaa')
    ax.set_title(title, color='#4a5a75', fontsize=9, pad=10)
    plt.tight_layout()
    return fig

def bar_chart(cats, vals, color='#4fd1c5', title=''):
    fig, ax = plt.subplots(figsize=(5.5,3), facecolor='none')
    ax.set_facecolor(PANEL)
    bars = ax.bar(cats, vals, color=color, width=0.52, edgecolor='none', zorder=2)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.025,
                f'{v:.0f}', ha='center', va='bottom', color='#7a8aaa', fontsize=8)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#1a2640')
    ax.tick_params(colors='#4a5a75', labelsize=8)
    ax.set_title(title, color='#4a5a75', fontsize=9, pad=10)
    ax.grid(axis='y', color='#1a2640', lw=0.8, zorder=1)
    plt.tight_layout()
    return fig

def hbar_chart(labels, vals, color='#4fd1c5', title=''):
    fig, ax = plt.subplots(figsize=(9,3), facecolor='none')
    ax.set_facecolor(PANEL)
    clrs = [color if v > np.mean(vals) else '#1e2d45' for v in vals]
    ax.barh(labels[::-1], vals[::-1], color=clrs[::-1], edgecolor='none', height=0.52)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#1a2640')
    ax.tick_params(colors='#4a5a75', labelsize=8.5)
    ax.set_xlabel('Feature Importance Score', color='#4a5a75', fontsize=9)
    ax.set_title(title, color='#4a5a75', fontsize=9, pad=10)
    ax.grid(axis='x', color='#1a2640', lw=0.8)
    plt.tight_layout()
    return fig

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
        <div class="brand-logo">💼</div>
        <div class="brand-name">WorkforceIQ</div>
        <div class="brand-tagline">HR Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("##### Choose a Tool")
    page = st.radio("", [
        "🏠  Dashboard Overview",
        "🚨  Will This Employee Leave?",
        "⭐  Performance Rating Forecast",
        "🚀  Promotion Likelihood"
    ], label_visibility="collapsed")

    st.markdown("<div class='glowline'></div>", unsafe_allow_html=True)

    if data_loaded:
        total, att, avg_age, avg_inc, avg_ten, _ = summary_stats(df)
        st.markdown("##### Dataset Summary")
        for lbl_t, val, col in [
            ("Employees",    f"{total:,}",       "#4fd1c5"),
            ("Attrition Rate", f"{att:.1f}%",    "#f87171"),
            ("Avg. Age",     f"{avg_age:.0f} yrs","#a78bfa"),
            ("Avg. Salary",  f"${avg_inc:,.0f}", "#60a5fa"),
            ("Avg. Tenure",  f"{avg_ten:.1f} yrs","#34d399"),
        ]:
            st.markdown(f"""
            <div class="stat-row">
                <span class="stat-lbl">{lbl_t}</span>
                <span class="stat-val" style="color:{col}">{val}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Add `Employee-Attrition.xlsx` to the app folder to load data.")

    st.markdown("""
    <div style="font-size:11px;color:#1e2d45;text-align:center;padding:18px 0 4px">
        Random Forest · GBR · SMOTE<br><span style="color:#4fd1c5">WorkforceIQ v3.0</span>
    </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "🏠  Dashboard Overview":
    st.markdown("""
    <div class="page-banner">
        <div class="page-banner-tag">💼 HR Analytics Platform</div>
        <div class="page-banner-title">Welcome to WorkforceIQ</div>
        <div class="page-banner-sub">
            Your intelligent HR co-pilot — predict who might leave, forecast performance ratings,
            and spot employees ready for promotion using machine learning on real workforce data.
        </div>
    </div>""", unsafe_allow_html=True)

    if not data_loaded:
        st.warning("⚠️ Place `Employee-Attrition.xlsx` in the same folder as this app to unlock all features.")
        st.stop()

    total, att, avg_age, avg_inc, avg_ten, dept_counts = summary_stats(df)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(title,val,badge,note) in zip([c1,c2,c3,c4,c5],[
        ("Total Employees",       f"{total:,}",       "badge-teal", "Full dataset"),
        ("Historical Attrition",  f"{att:.1f}%",      "badge-red",  "% who left"),
        ("Avg. Monthly Salary",   f"${avg_inc:,.0f}", "badge-blue", "USD gross"),
        ("Average Age",           f"{avg_age:.0f} yrs","badge-blue","Workforce"),
        ("Average Tenure",        f"{avg_ten:.1f} yrs","badge-green","Years stayed"),
    ]):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{title}</div>
                <div class="kpi-val">{val}</div>
                <div style="margin-top:8px"><span class="{badge}">{note}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='glowline'></div>", unsafe_allow_html=True)

    cl,cm,cr = st.columns([1.2,1,1])
    with cl:
        st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Employees by Department</div></div>', unsafe_allow_html=True)
        dept_data   = dept_counts.head(5)
        dept_labels = label_encoders['Department'].inverse_transform(dept_data.index.astype(int))
        fig = bar_chart([d[:14] for d in dept_labels], dept_data.values, '#4fd1c5')
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with cm:
        st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Who Stayed vs Left?</div></div>', unsafe_allow_html=True)
        ac = df['Attrition'].value_counts()
        if set(ac.index) <= {0,1}:
            stayed, left = ac.get(0,0), ac.get(1,0)
        else:
            stayed, left = ac.get('No',0), ac.get('Yes',0)
        t = stayed+left
        fig2 = donut_chart([stayed,left],['Stayed','Left'],['#4fd1c5','#f87171'],
                           f'{left/t*100:.0f}%','Left','Attrition split')
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    with cr:
        st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Performance Distribution</div></div>', unsafe_allow_html=True)
        perf = df['PerformanceRating'].value_counts().sort_index()
        fig3 = donut_chart(perf.values,[f"Rating {i}" for i in perf.index],
                           ['#f87171','#fb923c','#60a5fa','#10b981'],
                           str(perf.index[perf.values.argmax()]),'Most common','Performance ratings')
        st.pyplot(fig3, use_container_width=True); plt.close(fig3)

    st.markdown("<div class='glowline'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">What Can This Platform Do?</div></div>', unsafe_allow_html=True)

    f1,f2,f3,f4 = st.columns(4)
    for col,(icls,ico,title,desc,eg) in zip([f1,f2,f3,f4],[
        ("feat-icon-rose","🚨","Will This Employee Leave?",
         "Predict whether a specific employee is at risk of resigning, based on 15 HR signals.",
         "e.g. An employee in Sales, working overtime, low satisfaction — model flags 78% flight risk."),
        ("feat-icon-teal","⭐","Performance Rating Forecast",
         "Forecast what performance rating an employee is likely to earn at their next appraisal.",
         "e.g. High job involvement + recent promotion + good manager = predicted Rating 4 (Excellent)."),
        ("feat-icon-indigo","🚀","Promotion Likelihood",
         "Estimate whether an employee is overdue for promotion based on tenure and performance.",
         "e.g. 5 years in role, top performer, no recent promotion — flagged 'Promotion Ready'."),
        ("feat-icon-amber","📊","Workforce Analytics",
         "Visual overview of headcount, attrition patterns, and performance spread across the org.",
         "e.g. R&D has 60% of staff but 35% attrition — a clear signal for targeted HR action."),
    ]):
        with col:
            st.markdown(f"""
            <div class="feat-card">
                <div class="feat-icon {icls}">{ico}</div>
                <div class="feat-title">{title}</div>
                <div class="feat-text">{desc}</div>
                <div class="feat-eg">💡 {eg}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='glowline'></div>", unsafe_allow_html=True)
    with st.expander("📋  View raw dataset (first 15 rows)"):
        st.dataframe(df.head(15), use_container_width=True)


# ============================================================
# PAGE: ATTRITION PREDICTION
# ============================================================
elif page == "🚨  Will This Employee Leave?":
    st.markdown("""
    <div class="page-banner">
        <div class="page-banner-tag">🚨 Attrition Risk Predictor</div>
        <div class="page-banner-title">Will This Employee Leave the Company?</div>
        <div class="page-banner-sub">
            Fill in the employee's profile below. The AI model scores their resignation risk and
            provides specific, actionable steps HR can take before it is too late.
        </div>
    </div>""", unsafe_allow_html=True)

    if not data_loaded:
        st.error("Dataset not found. Please add `Employee-Attrition.xlsx` to the app folder."); st.stop()

    with st.expander("ℹ️  What does the model look at? (Click to learn more)"):
        st.markdown("""
        The model analyses **15 key employee signals** to score attrition risk:

        | Signal | Why it predicts who leaves |
        |--------|---------------------------|
        | **Works Overtime?** | Employees burning out from overwork leave much faster |
        | **Marital Status** | Single employees tend to have higher geographic mobility |
        | **Commute Distance** | Long daily commutes steadily erode job satisfaction |
        | **Monthly Salary** | Being paid below market rate is the #1 reason people quit |
        | **Years of Experience** | Very new or very senior employees have different risk profiles |
        | **Years at Company** | Employees in their 2nd–3rd year have the highest risk window |
        | **Job Level** | Junior employees have invested less loyalty in the organisation |
        | **Manager Relationship** | A bad manager is the single biggest reason people resign |
        | **Stock Options** | Equity ownership creates strong financial retention incentive |
        | **Job Involvement** | Disengaged employees are 4× more likely to leave |
        """)

    features_a = ['OverTime','MaritalStatus','DistanceFromHome','JobRole','Department',
                  'TotalWorkingYears','JobLevel','YearsInCurrentRole','MonthlyIncome',
                  'Age','YearsWithCurrManager','StockOptionLevel','YearsAtCompany',
                  'JobInvolvement','Education']

    @st.cache_resource
    def train_attrition(n):
        df_m = df.copy()
        if df_m['Attrition'].dtype == object:
            df_m['Attrition'] = df_m['Attrition'].map({'Yes':1,'No':0})
        X_ = df_m[features_a]; y_ = df_m['Attrition']
        sc = StandardScaler(); X_s = sc.fit_transform(X_)
        sm = SMOTE(random_state=42); X_r,y_r = sm.fit_resample(X_s,y_)
        Xtr,_,ytr,_ = train_test_split(X_r,y_r,test_size=0.2,random_state=42,stratify=y_r)
        rf = RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42)
        rf.fit(Xtr,ytr); return sc,rf

    sc_a, mdl_a = train_attrition(len(df))

    st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Step 1 — Enter Employee Details</div></div>', unsafe_allow_html=True)

    user_a   = {}
    cat_a    = ['OverTime','MaritalStatus','JobRole','Department','Education']
    sld_a    = {'JobLevel':(1,5,2),'StockOptionLevel':(0,3,0),'JobInvolvement':(1,4,2)}
    num_da   = {'Age':30,'MonthlyIncome':5000,'DistanceFromHome':5,
                'TotalWorkingYears':8,'YearsInCurrentRole':3,
                'YearsWithCurrManager':3,'YearsAtCompany':5}

    c1,c2,c3 = st.columns(3)
    for i,f in enumerate(features_a):
        with [c1,c2,c3][i%3]:
            if f in cat_a:
                user_a[f] = st.selectbox(lbl(f), label_encoders[f].classes_, help=tip(f), key=f"a_{f}")
            elif f in sld_a:
                mn,mx,dv = sld_a[f]
                user_a[f] = st.slider(lbl(f), mn, mx, dv, help=tip(f), key=f"a_{f}")
            else:
                user_a[f] = st.number_input(lbl(f), min_value=0, max_value=1000000,
                                             value=num_da.get(f,0), step=1, help=tip(f), key=f"a_{f}")
    for c in cat_a:
        user_a[c] = label_encoders[c].transform([user_a[c]])[0]

    st.markdown("<br>", unsafe_allow_html=True)
    bc,_ = st.columns([1,2])
    with bc:
        go_a = st.button("🚨  Analyse Attrition Risk", use_container_width=True)

    if go_a:
        X_inp = pd.DataFrame([user_a], columns=features_a)
        prob  = mdl_a.predict_proba(sc_a.transform(X_inp))[0,1]
        pred  = int(prob >= 0.5)

        st.markdown("<div class='glowline'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Prediction Result</div></div>', unsafe_allow_html=True)

        if pred == 1:
            color,bg,border = "#f87171","rgba(239,68,68,0.07)","rgba(239,68,68,0.2)"
            verdict = "⚠️ HIGH RISK — Likely to Resign"; verdict_short = "AT RISK OF LEAVING"
            plain   = f"Based on this employee's profile, the model calculates a <strong style='color:#f87171'>{prob*100:.0f}% probability they will resign</strong>. This is well above the average and warrants immediate HR attention."
            actions = [
                ("🕐","Audit workload — excessive overtime is the single biggest burnout driver. Cap at 45 hrs/week."),
                ("💰","Benchmark their salary vs the market right now — underpayment drives 38% of voluntary exits."),
                ("🚀","Create a visible, 6-month promotion roadmap so they can see a future here."),
                ("🤝","Book a genuine career conversation with their manager this week — not a performance review."),
                ("🎯","Assign to a high-impact project — people who own meaningful work rarely leave."),
                ("📊","Run a confidential engagement check-in — surface hidden dissatisfiers before they resign."),
            ]
        else:
            color,bg,border = "#4fd1c5","rgba(79,209,197,0.07)","rgba(79,209,197,0.2)"
            verdict = "✅ LOW RISK — Likely to Stay"; verdict_short = "LIKELY TO STAY"
            plain   = f"This employee's profile matches the traits of employees who stay. The model gives only a <strong style='color:#4fd1c5'>{prob*100:.0f}% resignation probability</strong>. Continue standard engagement and development programmes."
            actions = [
                ("🏅","Maintain regular recognition — even stable employees disengage if they feel invisible."),
                ("📚","Offer a learning and development budget; employees who grow stay longer."),
                ("⚖️","Monitor overtime carefully — even low-risk employees can flip if workload spikes."),
                ("🔭","Plan their next career step now — give them a clear reason to stay for the next 2 years."),
                ("💬","Run quarterly pulse surveys — small issues caught early prevent future attrition."),
                ("🌍","Consider stretch assignments or cross-team projects to keep them challenged."),
            ]

        r1,r2 = st.columns([1,1.7])
        with r1:
            st.markdown(f"""
            <div class="res-outer" style="background:{bg};border:1px solid {border}">
                <div class="res-label" style="color:{color}">{verdict_short}</div>
                <div class="res-value" style="color:{color}">{prob*100:.0f}%</div>
                <div class="res-note">Probability of leaving the company</div>
                <div style="margin-top:14px;font-size:13px;color:{color};font-weight:600">{verdict}</div>
            </div>""", unsafe_allow_html=True)

            confidence = "Very High" if abs(prob-0.5)>0.35 else "High" if abs(prob-0.5)>0.2 else "Moderate"
            st.markdown("**Model Confidence**")
            st.progress(float(prob) if pred==1 else float(1-prob))
            st.caption(f"Confidence level: **{confidence}**")

            st.markdown(f"""
            <div class="explain-card">
                <div class="explain-title">What does this mean?</div>
                <div class="explain-body">{plain}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            head = "🚨 Immediate Retention Actions" if pred==1 else "✅ Ongoing Development Plan"
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:16px;padding:22px">
                <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;color:{color};margin-bottom:14px;font-size:15px">{head}</div>
            """, unsafe_allow_html=True)
            for ico,txt in actions:
                st.markdown(f'<div class="action-item"><span class="action-ico">{ico}</span><span class="action-txt">{txt}</span></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glowline'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Which Factors Drove This Score?</div></div>', unsafe_allow_html=True)
        imp  = mdl_a.feature_importances_
        top  = np.argsort(imp)[-8:][::-1]
        fig_i = hbar_chart([lbl(features_a[i]) for i in top], imp[top], '#4fd1c5',
                            'Top 8 features that influenced the attrition risk score')
        st.pyplot(fig_i, use_container_width=True); plt.close(fig_i)
        st.caption("Teal bars = above-average importance. These signals had the most weight in this prediction.")


# ============================================================
# PAGE: PERFORMANCE PREDICTION
# ============================================================
elif page == "⭐  Performance Rating Forecast":
    st.markdown("""
    <div class="page-banner">
        <div class="page-banner-tag">⭐ Performance Forecast</div>
        <div class="page-banner-title">What Performance Rating Will This Employee Receive?</div>
        <div class="page-banner-sub">
            Enter the employee's work profile. The model forecasts their likely performance rating
            on a scale of 1 (Needs Improvement) to 4 (Excellent) — helping managers plan appraisals.
        </div>
    </div>""", unsafe_allow_html=True)

    if not data_loaded:
        st.error("Dataset not found."); st.stop()

    with st.expander("ℹ️  How does the performance model work? (Click to learn more)"):
        st.markdown("""
        The model learns from historical appraisal data and links performance ratings to these signals:

        | Input Signal | What it reveals |
        |-------------|----------------|
        | **Last Salary Increase (%)** | High hikes usually follow high-performance cycles |
        | **Job Involvement** | Engaged, invested employees consistently outperform peers |
        | **Work Environment Satisfaction** | Happy people in positive environments do their best work |
        | **Years in Current Role** | Mastery and confidence grow over time in a stable position |
        | **Monthly Salary** | Compensation level often correlates with the performance tier |
        | **Overtime** | Chronically overworked employees often underperform due to burnout |
        | **Manager Relationship** | Good managers enable high performance; poor ones suppress it |
        """)

    features_p = ['YearsInCurrentRole','YearsWithCurrManager','YearsSinceLastPromotion',
                  'TotalWorkingYears','DistanceFromHome','RelationshipSatisfaction',
                  'EnvironmentSatisfaction','JobInvolvement','PercentSalaryHike',
                  'Age','JobLevel','Education','StockOptionLevel','MonthlyIncome','OverTime']

    @st.cache_resource
    def train_perf(n):
        X_ = df[features_p]; y_ = df['PerformanceRating']
        sc = MinMaxScaler(); X_s = sc.fit_transform(X_)
        sm = SMOTE(random_state=42); X_r,y_r = sm.fit_resample(X_s,y_)
        Xtr,_,ytr,_ = train_test_split(X_r,y_r,test_size=0.25,random_state=42)
        rf = RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42)
        rf.fit(Xtr,ytr); return sc,rf

    sc_p,mdl_p = train_perf(len(df))

    st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Step 1 — Employee Work Profile</div></div>', unsafe_allow_html=True)

    user_p = {}
    sld_p  = {'RelationshipSatisfaction':(1,4,2),'EnvironmentSatisfaction':(1,4,2),
               'JobInvolvement':(1,4,2),'JobLevel':(1,5,2),'StockOptionLevel':(0,3,0)}
    cat_p  = ['Education','OverTime']
    num_dp = {'YearsInCurrentRole':3,'YearsWithCurrManager':3,'YearsSinceLastPromotion':2,
               'TotalWorkingYears':8,'DistanceFromHome':5,'PercentSalaryHike':13,'Age':32,'MonthlyIncome':5500}

    c1,c2,c3 = st.columns(3)
    for i,f in enumerate(features_p):
        with [c1,c2,c3][i%3]:
            if f in sld_p:
                mn,mx,dv = sld_p[f]; user_p[f] = st.slider(lbl(f),mn,mx,dv,help=tip(f),key=f"p_{f}")
            elif f in cat_p:
                user_p[f] = st.selectbox(lbl(f),label_encoders[f].classes_,help=tip(f),key=f"p_{f}")
            else:
                user_p[f] = st.number_input(lbl(f),min_value=0,max_value=1000000,
                                             value=num_dp.get(f,0),step=1,help=tip(f),key=f"p_{f}")
    for c in cat_p:
        user_p[c] = label_encoders[c].transform([user_p[c]])[0]

    st.markdown("<br>", unsafe_allow_html=True)
    bp,_ = st.columns([1,2])
    with bp:
        go_p = st.button("⭐  Forecast Performance Rating", use_container_width=True)

    if go_p:
        pred_p = mdl_p.predict(sc_p.transform(pd.DataFrame([user_p],columns=features_p)))[0]
        st.markdown("<div class='glowline'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Forecast Result</div></div>', unsafe_allow_html=True)

        cfg = {
            1:("#f87171","⚠️","NEEDS IMPROVEMENT","rgba(239,68,68,0.07)","rgba(239,68,68,0.18)",
               "The model forecasts a low performance rating. This employee needs a structured improvement plan, targeted coaching and possibly a role-fit review.",
               [("🎓","Enrol in a targeted skills training programme relevant to their day-to-day role."),
                ("🤝","Pair with a senior mentor for weekly 1:1 coaching — not just annual reviews."),
                ("🔄","Review whether this role genuinely suits their strengths — consider reassignment."),
                ("📋","Build a Performance Improvement Plan (PIP) with clear, weekly measurable targets."),
                ("🧠","Check for burnout or personal pressures — low performance often has a human root cause.")]),
            2:("#fb923c","📊","BELOW EXPECTATIONS","rgba(251,146,60,0.07)","rgba(251,146,60,0.18)",
               "Performance is forecast below the company average. Targeted development, more frequent feedback, and goal-setting can close this gap within one review cycle.",
               [("📖","Provide access to online courses, workshops or internal training programmes."),
                ("🌱","Set stretch goals — achievable but slightly above their current comfort zone."),
                ("💬","Switch from annual to monthly feedback cycles to catch and fix issues early."),
                ("🌟","Connect them with a high-performer buddy for knowledge transfer and inspiration."),
                ("🔍","Diagnose root cause — is it the manager, tools, workload clarity or motivation?")]),
            3:("#60a5fa","ℹ️","MEETS EXPECTATIONS","rgba(96,165,250,0.07)","rgba(96,165,250,0.18)",
               "This employee is on track to meet their targets. With the right recognition and a growth push, they have real potential to move into the top-performer bracket.",
               [("🏅","Recognise good work publicly — consistent recognition lifts performance by up to 14%."),
                ("📚","Offer a self-directed learning budget to fuel their professional growth ambitions."),
                ("⚖️","Keep workload balanced — overloading a solid performer risks disengagement."),
                ("🔭","Identify hidden potential — assign a stretch project to see what they're capable of."),
                ("💡","Increase autonomy — let them own an end-to-end project for motivation and ownership.")]),
            4:("#10b981","🌟","HIGH PERFORMER","rgba(16,185,129,0.07)","rgba(16,185,129,0.18)",
               "Excellent performance is forecast. This is a key retention priority — reward immediately, fast-track for promotion, and leverage their expertise through mentorship before they are poached.",
               [("🏆","Reward with a merit bonus, public recognition or additional leave — do it now."),
                ("🚀","Fast-track for promotion — top performers leave within 12 months if career growth stalls."),
                ("🎓","Ask them to mentor 2–3 junior employees — multiplies their positive impact across the team."),
                ("🌍","Offer a cross-functional or leadership project to broaden experience and keep them engaged."),
                ("💎","Retain with equity stake, flexible working or a long-term incentive plan.")])
        }
        color,emoji,verdict,bg,border,plain,actions = cfg.get(pred_p,cfg[3])

        r1,r2 = st.columns([1,1.7])
        with r1:
            st.markdown(f"""
            <div class="res-outer" style="background:{bg};border:1px solid {border}">
                <div style="font-size:40px;margin-bottom:8px">{emoji}</div>
                <div class="res-label" style="color:{color}">{verdict}</div>
                <div class="res-value" style="color:{color}">{pred_p}</div>
                <div class="res-note">out of 4 — Performance Rating Scale</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("**Rating Scale — Where Do They Sit?**")
            for rv,(rn,pct) in {1:("Needs Improvement",10),2:("Below Expectations",35),
                                 3:("Meets Expectations",65),4:("High Performer",100)}.items():
                active = rv == pred_p
                bc2 = f"background:linear-gradient(90deg,{color},#667eea)" if active else "background:#1a2640"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px">
                    <div style="{bc2};border-radius:4px;height:7px;width:{pct}%;min-width:6px"></div>
                    <span style="color:{'#eef2ff' if active else '#2d3a50'};font-size:12px;
                                 font-weight:{'700' if active else '400'}">
                        {rv} — {rn}{'  ◀' if active else ''}
                    </span>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="explain-card" style="margin-top:14px">
                <div class="explain-title">In Plain English</div>
                <div class="explain-body">{plain}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            plan_titles = {1:"⚠️ Improvement Plan",2:"📊 Development Plan",
                           3:"ℹ️ Growth Plan",4:"🌟 Reward & Retain"}
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:16px;padding:22px">
                <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;
                            color:{color};margin-bottom:14px;font-size:15px">{plan_titles[pred_p]}</div>
            """, unsafe_allow_html=True)
            for ico,txt in actions:
                st.markdown(f'<div class="action-item"><span class="action-ico">{ico}</span><span class="action-txt">{txt}</span></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glowline'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Which Inputs Had the Most Influence?</div></div>', unsafe_allow_html=True)
        imp2 = mdl_p.feature_importances_; top2 = np.argsort(imp2)[-8:][::-1]
        fig_i2 = hbar_chart([lbl(features_p[i]) for i in top2], imp2[top2], '#10b981',
                             'Top 8 signals in the performance rating forecast')
        st.pyplot(fig_i2, use_container_width=True); plt.close(fig_i2)


# ============================================================
# PAGE: PROMOTION LIKELIHOOD
# ============================================================
elif page == "🚀  Promotion Likelihood":
    st.markdown("""
    <div class="page-banner">
        <div class="page-banner-tag">🚀 Promotion Predictor</div>
        <div class="page-banner-title">Is This Employee Ready for a Promotion?</div>
        <div class="page-banner-sub">
            Based on performance history, tenure and career trajectory, the model estimates whether
            this employee is overdue, approaching, or not yet ready for their next promotion —
            giving HR a data-backed starting point for succession planning.
        </div>
    </div>""", unsafe_allow_html=True)

    if not data_loaded:
        st.error("Dataset not found."); st.stop()

    with st.expander("ℹ️  How is promotion readiness calculated? (Click to learn more)"):
        st.markdown("""
        The model predicts the **likely years since last promotion** for a given employee profile.
        A low predicted value suggests the employee is due soon; a high value flags potential stagnation.

        | Key Signal | Why it matters for promotions |
        |------------|-------------------------------|
        | **Job Level** | Senior employees have longer natural promotion cycles |
        | **Performance Rating** | High performers get promoted ~2× faster on average |
        | **Years of Experience** | Breadth of experience is a promotion prerequisite |
        | **Years in Current Role** | Long tenure without promotion = likely overdue |
        | **Monthly Salary** | Pay band often determines when the next title uplift is possible |
        | **Education Level** | Higher qualifications can accelerate career ladder advancement |
        """)

    features_pr = ['JobLevel','TotalWorkingYears','YearsInCurrentRole','PerformanceRating',
                   'Education','MonthlyIncome','Age','JobInvolvement','YearsAtCompany','StockOptionLevel']

    @st.cache_resource
    def train_promo(n):
        X_ = df[features_pr]; y_ = df['YearsSinceLastPromotion']
        sc = StandardScaler(); X_s = sc.fit_transform(X_)
        Xtr,_,ytr,_ = train_test_split(X_s,y_,test_size=0.2,random_state=42)
        gbr = GradientBoostingRegressor(n_estimators=200,max_depth=4,random_state=42)
        gbr.fit(Xtr,ytr); return sc,gbr

    sc_pr,mdl_pr = train_promo(len(df))

    st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Step 1 — Employee Career Profile</div></div>', unsafe_allow_html=True)

    user_pr = {}
    sld_pr  = {'JobLevel':(1,5,2),'PerformanceRating':(1,4,3),
                'JobInvolvement':(1,4,2),'StockOptionLevel':(0,3,0)}
    cat_pr  = ['Education']
    num_dpr = {'TotalWorkingYears':8,'YearsInCurrentRole':3,'MonthlyIncome':5500,'Age':32,'YearsAtCompany':5}

    c1,c2 = st.columns(2)
    for i,f in enumerate(features_pr):
        with [c1,c2][i%2]:
            if f in sld_pr:
                mn,mx,dv = sld_pr[f]; user_pr[f] = st.slider(lbl(f),mn,mx,dv,help=tip(f),key=f"pr_{f}")
            elif f in cat_pr:
                user_pr[f] = st.selectbox(lbl(f),label_encoders[f].classes_,help=tip(f),key=f"pr_{f}")
            else:
                user_pr[f] = st.number_input(lbl(f),min_value=0,max_value=1000000,
                                              value=num_dpr.get(f,0),step=1,help=tip(f),key=f"pr_{f}")
    for c in cat_pr:
        user_pr[c] = label_encoders[c].transform([user_pr[c]])[0]

    st.markdown("<br>", unsafe_allow_html=True)
    bpr,_ = st.columns([1,2])
    with bpr:
        go_pr = st.button("🚀  Check Promotion Readiness", use_container_width=True)

    if go_pr:
        pred_yrs = max(0.0, round(float(mdl_pr.predict(
            sc_pr.transform(pd.DataFrame([user_pr],columns=features_pr)))[0]), 1))

        st.markdown("<div class='glowline'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Promotion Readiness Result</div></div>', unsafe_allow_html=True)

        if pred_yrs <= 1:
            color,bg,border = "#10b981","rgba(16,185,129,0.07)","rgba(16,185,129,0.18)"
            status = "🚀 PROMOTION READY"
            plain  = f"Only <strong style='color:#10b981'>{pred_yrs:.1f} year(s)</strong> since last promotion is predicted — this employee is likely due or already overdue. This profile is a strong candidate for immediate promotion consideration."
            actions= [("🏆","Initiate a formal promotion review in the next available HR cycle."),
                      ("💬","Have a direct career conversation — confirm their goals match open opportunities."),
                      ("📋","Prepare a promotion business case with documented achievements for leadership."),
                      ("💰","Review compensation — a title change without a meaningful pay increase often backfires."),
                      ("🌟","Announce the promotion publicly to boost team morale and signal a growth culture.")]
        elif pred_yrs <= 3:
            color,bg,border = "#60a5fa","rgba(96,165,250,0.07)","rgba(96,165,250,0.18)"
            status = "📈 APPROACHING PROMOTION WINDOW"
            plain  = f"Approximately <strong style='color:#60a5fa'>{pred_yrs:.1f} years</strong> since last promotion is forecast. This employee is entering the typical promotion window. Start preparing them now to be ready when the opportunity opens."
            actions= [("🎯","Set stretch objectives that align with the next job level's required competencies."),
                      ("📚","Identify skill gaps now and provide targeted resources to close them."),
                      ("🤝","Increase their visibility with senior leaders through high-profile project assignments."),
                      ("💬","Set clear, documented promotion criteria with a target timeline of 6–12 months."),
                      ("📊","Schedule a mid-year progress review specifically focused on promotion readiness.")] 
        else:
            color,bg,border = "#f87171","rgba(239,68,68,0.07)","rgba(239,68,68,0.18)"
            status = "⚠️ OVERDUE — CAREER STAGNATION RISK"
            plain  = f"The model estimates <strong style='color:#f87171'>{pred_yrs:.1f} years</strong> without a promotion — well above average. Employees in this situation face a significantly elevated attrition risk. Urgent action is needed to address career stagnation."
            actions= [("🚨","Urgently review the stagnation cause — is it budget, performance, manager oversight, or process?"),
                      ("💰","Consider an off-cycle salary review even if a formal promotion title is currently unavailable."),
                      ("🤝","Have an honest, empathetic career conversation — acknowledge the gap and present a real path forward."),
                      ("🔭","Explore lateral moves or expanded responsibilities as a meaningful interim step."),
                      ("📋","Flag this employee as a flight risk — this profile pattern often precedes resignation within 6 months.")]

        r1,r2 = st.columns([1,1.7])
        with r1:
            st.markdown(f"""
            <div class="res-outer" style="background:{bg};border:1px solid {border}">
                <div class="res-label" style="color:{color}">{status}</div>
                <div class="res-value" style="color:{color}">{pred_yrs:.1f}</div>
                <div class="res-note">Predicted years since last promotion</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("**Promotion Timeline — Where Do They Sit?**")
            pct = min(100, pred_yrs/10*100)
            bclr = "#10b981" if pred_yrs<=1 else "#60a5fa" if pred_yrs<=3 else "#f87171"
            st.markdown(f"""
            <div style="background:#1a2640;border-radius:8px;height:12px;overflow:hidden;margin:8px 0 4px">
                <div style="background:linear-gradient(90deg,{bclr},#667eea);height:100%;width:{pct}%;border-radius:8px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;color:#4a5a75;margin-bottom:14px">
                <span>0 yrs — Ready Now</span><span>5 yrs</span><span>10+ yrs — Overdue</span>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="explain-card">
                <div class="explain-title">What does this mean for HR?</div>
                <div class="explain-body">{plain}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            head_map = {
                "🚀 PROMOTION READY":"🚀 Promote Now — Recommended Actions",
                "📈 APPROACHING PROMOTION WINDOW":"📈 Prepare for Upcoming Promotion",
                "⚠️ OVERDUE — CAREER STAGNATION RISK":"⚠️ Urgent Career Intervention Needed",
            }
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:16px;padding:22px">
                <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;
                            color:{color};margin-bottom:14px;font-size:15px">{head_map.get(status,'')}</div>
            """, unsafe_allow_html=True)
            for ico,txt in actions:
                st.markdown(f'<div class="action-item"><span class="action-ico">{ico}</span><span class="action-txt">{txt}</span></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glowline'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hd"><div class="sec-dot"></div><div class="sec-title">Factors That Shaped the Promotion Estimate</div></div>', unsafe_allow_html=True)
        imp3 = mdl_pr.feature_importances_; top3 = np.argsort(imp3)[::-1]
        fig_i3 = hbar_chart([lbl(features_pr[i]) for i in top3], imp3[top3], '#667eea',
                             'All features — contribution to the promotion timeline estimate')
        st.pyplot(fig_i3, use_container_width=True); plt.close(fig_i3)

# python -m streamlit run streamlit_app.py