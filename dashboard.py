import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Scoring Dashboard", layout="wide")

# ==============================
# YOUR THEME / COLORS (kept)
# ==============================
APP_BG = "#E5C7AB"
CARD_BG = "#EB7100"
BORDER = "rgba(0,0,0,0.12)"
CT = "#D8D5E9"
TEXT = "#FFFFFF"
MUTED = "rgba(0,0,0,0.70)"
BLA = "#090015"
CHART_PAPER = "#313132"
CHART_PLOT = "#D7D6D4"

METRIC_LABEL_SIZE = 26
METRIC_VALUE_SIZE = 22
METRIC_LABEL_COLOR = BLA
METRIC_VALUE_COLOR = CARD_BG

BLUES = ["#031432", "#0B47A8", "#1D7AFC", "#3B8EF5", "#74A8FF", "#073072"]
HEAT_SCALE = [
    [0.00, "#073072"],
    [0.25, "#E2D6FF"],
    [0.55, "#1D7AFC"],
    [0.80, "#0B47A8"],
    [1.00, "#031432"],
]

def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {APP_BG};
            color: {TEXT};
        }}
        .block-container {{
            padding-top: 1.1rem;
        }}

        html, body, [class*="css"] {{
            color: {TEXT} !important;
            font-size: 20px !important;
        }}

        h1 {{
            font-size: 32px !important;
            font-weight: 900 !important;
            color: {TEXT} !important;
        }}
        h2 {{
            font-size: 28px !important;
            font-weight: 900 !important;
            color: {TEXT} !important;
        }}
        h3, h4 {{
            font-size: 20px !important;
            font-weight: 900 !important;
            color: {TEXT} !important;
        }}

        section[data-testid="stSidebar"] {{
            background: #241E4E;
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] * {{
            color: {TEXT} !important;
        }}

        .card {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 10px 22px rgba(0,0,0,0.06);
        }}
        .card-title {{
            font-size: 28px !important;
            text-transform: uppercase;
            font-weight: 900;
            color: {TEXT} !important;
            margin-bottom: 6px;
            opacity: .85;
        }}
        .card-value {{
            font-size: 26px !important;
            font-weight: 900;
            color: {TEXT} !important;
            line-height: 1.05;
        }}
        .card-sub {{
            font-size: 24px !important;
            color: {TEXT} !important;
            opacity: .85;
            margin-top: 6px;
        }}

        button[data-baseweb="tab"] {{
            font-weight: 900 !important;
            color: {BLA} !important;
            font-size: 20px !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            border-bottom: 4px solid {BLUES[0]} !important;
        }}

        .stDataFrame {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid {BORDER};
            background: {CARD_BG};
        }}

        div[data-testid="stMetricLabel"] > div {{
            font-size: {METRIC_LABEL_SIZE}px !important;
            color: {METRIC_LABEL_COLOR} !important;
            font-weight: 800 !important;
        }}
        div[data-testid="stMetricValue"] > div {{
            font-size: {METRIC_VALUE_SIZE}px !important;
            color: {METRIC_VALUE_COLOR} !important;
            font-weight: 900 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-value">{value}</div>
          <div class="card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================
# GOOGLE SHEETS CONFIG
# ==============================
WS_THOUGHT   = "Thought Leadership"
WS_GROWTH    = "Growth Mindset"
WS_NETWORK   = "Networking"
WS_ADVISORY  = "Advisory"
WS_INFLUENCE = "Influencingrelationship"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SPREADSHEET_KEY = st.secrets["GSHEETS_SPREADSHEET_KEY"]

PAGES = {
    "Thought Leadership": {
        "worksheet": WS_THOUGHT,
        "sections": [
            "Locally Anchored Visioning",
            "Innovation and Insight",
            "Execution Planning",
            "Cross-Functional Collaboration",
            "Follow-Through Discipline",
            "Learning-Driven Adjustment",
            "Result-Oriented Decision-Making",
        ],
        "overall_total_col": "Overall Total (0–21)",
        "overall_rank_col": "Overall Rank",
        "ai_col": "AI Suspected",
        "ai_maxscore_col": "AI_MaxScore",
        "question_titles": {},  # you already have these; ok to keep empty here
    },
    "Growth Mindset": {
        "worksheet": WS_GROWTH,
        "sections": ["Learning Agility", "Digital Savvy", "Innovation", "Contextual Intelligence"],
        "overall_total_col": "Overall Total (0–12)",
        "overall_rank_col": "Overall Rank",
        "ai_col": "AI Suspected",
        "ai_maxscore_col": "AI_MaxScore",
        "question_titles": {},
    },
    "Networking & Advocacy": {
        "worksheet": WS_NETWORK,
        "sections": [
            "Strategic Positioning & Donor Fluency",
            "Power-Aware Stakeholder Mapping",
            "Equitable Allyship & Local Fronting",
            "Coalition Governance & Convening",
            "Community-Centered Messaging",
            "Evidence-Led Learning (Local Knowledge)",
            "Influence Without Authority",
            "Risk Management & Adaptive Communication",
        ],
        "overall_total_col": "Overall Total (0–24)",
        "overall_rank_col": "Overall Rank",
        "ai_col": "AI Suspected",
        "ai_maxscore_col": "AI_MaxScore",
        "question_titles": {},
    },
    "Advisory Skills": {
        "worksheet": WS_ADVISORY,
        "sections": [
            "Strategic & analytical thinking",
            "Credibility & trustworthiness",
            "Effective communication & influence",
            "Client & stakeholder focus",
            "Fostering collaboration & partnership",
            "Ensuring relevance & impact",
            "Solution orientation & adaptability",
            "Capacity strengthening & empowerment support",
        ],
        "overall_total_col": "Overall Total (0–24)",
        "overall_rank_col": "Overall Rank",
        "ai_col": "AI Suspected",
        "ai_maxscore_col": "AI_MaxScore",
        "question_titles": {},
    },
    "Influencing Relationships": {
        "worksheet": WS_INFLUENCE,
        "sections": [
            "Strategic Positioning & Political Acumen",
            "Stakeholder Mapping & Engagement",
            "Evidence-Informed Advocacy",
            "Communication, Framing & Messaging",
            "Risk Awareness & Mitigation",
            "Coalition Building & Collaborative Action",
            "Adaptive Tactics & Channel Selection",
            "Integrity & Values-Based Influencing",
        ],
        "overall_total_col": "Overall Total (0–24)",
        "overall_rank_col": "Overall Rank",
        "ai_col": "AI Suspected",
        "ai_maxscore_col": "AI_MaxScore",
        "question_titles": {},
    },
}

def _normalize_sa_dict(raw: dict) -> dict:
    sa = dict(raw)
    if sa.get("private_key") and "\\n" in sa["private_key"]:
        sa["private_key"] = sa["private_key"].replace("\\n", "\n")
    sa.setdefault("token_uri", "https://oauth2.googleapis.com/token")
    return sa

@st.cache_resource(show_spinner=False)
def gs_client():
    sa = _normalize_sa_dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(sa, scopes=SCOPES)
    return gspread.authorize(creds)

@st.cache_data(ttl=300, show_spinner=False)
def load_sheet_df(sheet_name: str) -> pd.DataFrame:
    gc = gs_client()
    ws = gc.open_by_key(SPREADSHEET_KEY).worksheet(sheet_name)
    df = pd.DataFrame(ws.get_all_records())
    df.columns = [str(c).strip() for c in df.columns]
    return df

def drop_date_duration_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    bad = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if "duration" in cl or "date" in cl:
            bad.append(c)
    return df.drop(columns=bad, errors="ignore")

def has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c and c in df.columns:
            return c
    return None

def _clean_series(s: pd.Series) -> pd.Series:
    return s.astype(str).replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()

def safe_plot(fig, key: str):
    fig.update_layout(
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_PLOT,
        font=dict(color=TEXT, size=16),
        title_font=dict(size=20, color=TEXT, family="Arial Black"),
        legend_font=dict(color=TEXT, size=14),
        margin=dict(l=10, r=10, t=70, b=15),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.12)", zeroline=False, tickfont=dict(color=TEXT, size=14))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.12)", zeroline=False, tickfont=dict(color=TEXT, size=14))
    st.plotly_chart(fig, use_container_width=True, key=key)

# ==============================
# IMPORTANT: ROBUST STAFF ID DETECTION
# ==============================
def staff_id_column(df: pd.DataFrame) -> str | None:
    candidates = [
        # common variations
        "CARE Staff ID", "CARE Staff Id", "CARE Staff id",
        "CARE_Staff_ID", "Care_Staff", "CAREStaffID",
        "Staff ID", "Staff Id", "StaffID",
        "staff_id", "care_staff_id", "care staff id",
        # sometimes people use Name instead of ID:
        "Staff Name", "CARE Staff Name", "Name",
    ]
    return first_existing(df, candidates)

# ==============================
# INDIVIDUAL SCORE CALCS (TOTALS)
# ==============================
def section_total_from_qcols(df_in: pd.DataFrame, section: str) -> pd.Series:
    # sum Qn1..Qn4 per row, then returns that series
    cols = [f"{section}_Qn1", f"{section}_Qn2", f"{section}_Qn3", f"{section}_Qn4"]
    cols = [c for c in cols if c in df_in.columns]
    if not cols:
        return pd.Series([np.nan] * len(df_in), index=df_in.index)
    mat = df_in[cols].apply(pd.to_numeric, errors="coerce")
    return mat.sum(axis=1, min_count=1)  # NaN if all missing

def build_staff_summary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sid = staff_id_column(df)
    if not sid:
        return pd.DataFrame()

    # compute section totals on the fly (from Qn columns)
    tmp = df.copy()
    for section in cfg["sections"]:
        tmp[f"{section}__Total(0-12)"] = section_total_from_qcols(tmp, section)

    overall_total_col = cfg["overall_total_col"]
    overall_rank_col = cfg["overall_rank_col"]

    group = tmp.groupby(tmp[sid].astype(str), dropna=True)

    out = pd.DataFrame(index=group.size().index)
    out.index.name = "Staff"

    out["Rows"] = group.size().values

    if overall_total_col in tmp.columns:
        out["Overall Total (mean)"] = group[overall_total_col].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        out["Overall Total (sum)"]  = group[overall_total_col].apply(lambda s: pd.to_numeric(s, errors="coerce").sum())

    # section totals
    for section in cfg["sections"]:
        col = f"{section}__Total(0-12)"
        if col in tmp.columns:
            out[f"{section} Total (0–12)"] = group[col].mean()

    # overall rank (mode)
    if overall_rank_col in tmp.columns:
        def mode_or_blank(s):
            s2 = _clean_series(s)
            if len(s2) == 0:
                return "-"
            return str(s2.value_counts().index[0])
        out["Overall Rank (mode)"] = group[overall_rank_col].apply(mode_or_blank)

    out = out.reset_index().rename(columns={"Staff": sid})
    return out

# ==============================
# RENDER PAGE
# ==============================
def render_page(page_name: str):
    cfg = PAGES[page_name]
    ws_name = cfg["worksheet"]

    with st.spinner(f"Loading: {ws_name} ..."):
        df = load_sheet_df(ws_name)
    df = drop_date_duration_cols(df)

    with st.sidebar:
        st.markdown("## Dashboard Pages")
        st.caption("Overall visuals are unchanged. Staff analysis is added as a proper staff-level summary + profile.")

    # header cards
    cA, cB = st.columns(2)
    with cA:
        card("SECTION", ws_name, "")
    with cB:
        card("Overall Responses", f"{len(df):,}", "")

    # tabs: keep your original sections + overall + add staff tabs
    tabs = st.tabs(cfg["sections"] + ["Overall Analysis", "Staff Summary", "Individual Profile"])

    # --------------------------
    # YOUR SECTION TABS (keep your original code here)
    # (I’m not rewriting every chart again to keep this message readable)
    # --------------------------
    for i, section in enumerate(cfg["sections"]):
        with tabs[i]:
            st.subheader(section)
            st.info("Keep your existing section visuals here (avg dist, rank table, Qn charts, rubric charts, section heatmap).")

    # --------------------------
    # OVERALL ANALYSIS TAB
    # --------------------------
    with tabs[-3]:
        st.subheader("Overall Analysis")
        overall_total = cfg["overall_total_col"]
        if overall_total in df.columns:
            tot = pd.to_numeric(df[overall_total], errors="coerce")
            m1, m2 = st.columns(2)
            m1.metric("Mean Total", f"{tot.mean():.1f}" if tot.notna().any() else "-")
            m2.metric("Median Total", f"{tot.median():.1f}" if tot.notna().any() else "-")
        else:
            st.warning(f"Missing overall total column: {overall_total}")

    # --------------------------
    # STAFF SUMMARY TAB (THIS is what you said is missing)
    # --------------------------
    with tabs[-2]:
        st.subheader("Staff Summary (All Staff IDs + Their Scores)")

        sid = staff_id_column(df)
        if not sid:
            st.error("I can't find a Staff ID column in this sheet, so I can't build staff scores.")
            st.write("Columns found:", list(df.columns))
            return

        staff_summary = build_staff_summary(df, cfg)
        if staff_summary.empty:
            st.error("Staff summary could not be built (missing staff IDs or score columns).")
            return

        # quick sanity panel
        st.caption(f"Detected Staff column: **{sid}** | Staff count: **{staff_summary[sid].nunique()}**")

        # filter THIS table only
        search = st.text_input("Search Staff ID (filters only this table)", "")
        show = staff_summary.copy()
        if search.strip():
            show = show[show[sid].astype(str).str.contains(search.strip(), case=False, na=False)]

        st.dataframe(show, use_container_width=True, hide_index=True)

        # chart: top overall totals
        if "Overall Total (mean)" in staff_summary.columns:
            topn = st.slider("Top N staff to show (by Overall Total mean)", 5, 50, 15)
            chart_df = staff_summary.sort_values("Overall Total (mean)", ascending=False).head(topn)

            fig = px.bar(
                chart_df,
                x=sid,
                y="Overall Total (mean)",
                title="Top Staff by Overall Total (mean)",
            )
            fig.update_layout(xaxis_title="Staff ID", yaxis_title="Overall Total (mean)")
            safe_plot(fig, key=f"{page_name}-staff-summary-top-overall")

    # --------------------------
    # INDIVIDUAL PROFILE TAB (filtered ONLY here)
    # --------------------------
    with tabs[-1]:
        st.subheader("Individual Profile (Select ONE staff)")

        sid = staff_id_column(df)
        if not sid:
            st.error("No Staff ID column detected for this sheet.")
            st.write("Columns found:", list(df.columns))
            return

        staff_ids = (
            df[sid].astype(str)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            .dropna()
            .unique()
        )
        staff_ids = sorted(staff_ids)
        if not staff_ids:
            st.error("Staff IDs exist as a column but there are no values.")
            return

        selected = st.selectbox("Select Staff", staff_ids, key=f"{page_name}-staff-profile")
        df_staff = df[df[sid].astype(str) == str(selected)].copy()

        # show overall + section totals for THIS staff
        overall_total_col = cfg["overall_total_col"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Staff", str(selected))
        c2.metric("Rows found", f"{len(df_staff):,}")

        if overall_total_col in df_staff.columns:
            t = pd.to_numeric(df_staff[overall_total_col], errors="coerce").dropna()
            c3.metric("Overall Total (mean)", f"{t.mean():.1f}" if len(t) else "-")
        else:
            c3.metric("Overall Total (mean)", "-")

        # Section totals chart (from Qn columns)
        sec_rows = []
        for section in cfg["sections"]:
            sec_total = section_total_from_qcols(df_staff, section).dropna()
            if len(sec_total):
                sec_rows.append({"Section": section, "Section Total (0–12)": float(sec_total.mean())})

        if sec_rows:
            sec_df = pd.DataFrame(sec_rows).sort_values("Section Total (0–12)", ascending=False)
            fig = px.bar(sec_df, x="Section", y="Section Total (0–12)", title="Section Totals (Selected Staff)")
            safe_plot(fig, key=f"{page_name}-profile-section-totals")
            st.dataframe(sec_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No section totals available (missing Qn columns like Section_Qn1..Qn4).")

# ==============================
# MAIN
# ==============================
def main():
    inject_css()
    st.title("ANALYTICS DASHBOARD")

    page = st.sidebar.radio("Dashboard Pages", list(PAGES.keys()), index=0)
    render_page(page)

if __name__ == "__main__":
    main()
