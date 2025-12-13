# dashboard_tabs.py
# ------------------------------------------------------------
# Thought Leadership Dashboard (PowerBI-style tabs)
# Pulls scored data DIRECTLY from Google Sheets worksheet:
#   GSHEETS_WORKSHEET_NAME1 = "Thought Leadership"
# Uses: column charts, bar charts, donut/pie, histogram, heatmap
# EXCLUDES: dates, duration, AI_MaxScore (not used)
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Thought Leadership Dashboard", layout="wide")

# ==============================
# SHEET NAMES (as you stated)
# ==============================
GSHEETS_WORKSHEET_NAME  = "Advisory"
GSHEETS_WORKSHEET_NAME1 = "Thought Leadership"
GSHEETS_WORKSHEET_NAME2 = "Growth Mindset"
GSHEETS_WORKSHEET_NAME3 = "Networking"
GSHEETS_WORKSHEET_NAME4 = "Influencingrelationship"

# We are building THIS page from Thought Leadership:
TARGET_WORKSHEET = GSHEETS_WORKSHEET_NAME1

# ==============================
# GOOGLE SHEETS CONFIG
# ==============================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SPREADSHEET_KEY = st.secrets["GSHEETS_SPREADSHEET_KEY"]

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
    # Clean column names (keep as-is but strip spaces)
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ==============================
# DASHBOARD SECTIONS (7 + Overall tab)
# ==============================
SECTIONS = [
    "Locally Anchored Visioning",
    "Innovation and Insight",
    "Execution Planning",
    "Cross-Functional Collaboration",
    "Follow-Through Discipline",
    "Learning-Driven Adjustment",
    "Result-Oriented Decision-Making",
]

# Titles for the 4 questions per section (NO question text shown)
QUESTION_TITLES = {
    "Locally Anchored Visioning": [
        "Vision with Roots",
        "Hard-wire Local Leadership",
        "Safeguard Community Voice",
        "Trade-off for Locally Led Scale",
    ],
    "Innovation and Insight": [
        "Field-First Learning Loop",
        "Surface Contradicting Insights",
        "Balance Policy vs Experimentation",
        "Frugal Innovation Test",
    ],
    "Execution Planning": [
        "Execution Spine Ownership",
        "90-Day Plan on a Page",
        "Close Handoff Failure",
        "Drop Activities to Regain Clarity",
    ],
    "Cross-Functional Collaboration": [
        "Alignment Workshop Design",
        "Resolve MEAL vs Gender Tension",
        "Shared Principles & Adherence",
        "Co-owned Decision Structure",
    ],
    "Follow-Through Discipline": [
        "Executable Promise",
        "Light Dashboard & Escalation",
        "Partner Recovery Options",
        "Stop Update Theatre",
    ],
    "Learning-Driven Adjustment": [
        "Quarterly Pause & Reflect",
        "Hypothesis + Evidence Shift",
        "Handle Negative Findings",
        "Stop to Fund Adaptations",
    ],
    "Result-Oriented Decision-Making": [
        "Cost/Benefit Trade-off Call",
        "Decide-by-Friday Data Needs",
        "Socialize Hard Trade-off",
        "Coherence Rule for Divergence",
    ],
}

# ==============================
# HELPERS
# ==============================
def score_distribution_percent(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    s = pd.to_numeric(df[score_col], errors="coerce")
    counts = s.value_counts().reindex([0, 1, 2, 3], fill_value=0)
    total = counts.sum()
    pct = (counts / total * 100).round(1) if total else counts.astype(float)
    out = pd.DataFrame({"Score": [0, 1, 2, 3], "Percent": pct.values, "Count": counts.values})
    return out

def rubric_frequency(df: pd.DataFrame, rubric_col: str) -> pd.DataFrame:
    x = (
        df[rubric_col]
        .astype(str)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .dropna()
    )
    out = x.value_counts().reset_index()
    out.columns = ["Rubric", "Count"]
    return out

def section_heatmap_percent(df: pd.DataFrame, section: str) -> pd.DataFrame:
    rows = []
    for qn in range(1, 5):
        score_col = f"{section}_Qn{qn}"
        title = QUESTION_TITLES[section][qn - 1]
        dist = score_distribution_percent(df, score_col)
        rows.append({
            "Question": title,
            "0": dist.loc[dist["Score"] == 0, "Percent"].values[0],
            "1": dist.loc[dist["Score"] == 1, "Percent"].values[0],
            "2": dist.loc[dist["Score"] == 2, "Percent"].values[0],
            "3": dist.loc[dist["Score"] == 3, "Percent"].values[0],
        })
    h = pd.DataFrame(rows).set_index("Question")
    return h

def safe_col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

# ==============================
# RENDER ONE SECTION TAB
# ==============================
def render_section_tab(df: pd.DataFrame, section: str):
    st.subheader(section)

    # --- Section summary visuals (Avg histogram + Rank donut) ---
    avg_col  = f"{section}_Avg (0–3)"
    rank_col = f"{section}_RANK"

    c1, c2 = st.columns(2)

    with c1:
        if safe_col_exists(df, avg_col):
            fig = px.histogram(
                df, x=avg_col, nbins=10,
                title="Section Average Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Missing column: {avg_col}")

    with c2:
        if safe_col_exists(df, rank_col):
            rank_df = df[rank_col].astype(str).replace({"": np.nan, "nan": np.nan}).dropna()
            rank_df = rank_df.value_counts().reset_index()
            rank_df.columns = ["Rank", "Count"]
            fig = px.pie(rank_df, names="Rank", values="Count", hole=0.55, title="Section Rank (Donut)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Missing column: {rank_col}")

    st.divider()

    # --- For each question: Score % (column chart) + Rubric frequency (bar) ---
    for qn in range(1, 5):
        score_col  = f"{section}_Qn{qn}"
        rubric_col = f"{section}_Rubric_Qn{qn}"
        q_title    = QUESTION_TITLES[section][qn - 1]

        st.markdown(f"### {q_title}")

        colA, colB = st.columns(2)

        with colA:
            if safe_col_exists(df, score_col):
                dist = score_distribution_percent(df, score_col)
                fig = px.bar(
                    dist, x="Score", y="Percent",
                    text="Percent",
                    title="Score Distribution (%)"
                )
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(yaxis_title="Percent", xaxis_title="Score (0–3)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Missing score column: {score_col}")

        with colB:
            if safe_col_exists(df, rubric_col):
                rf = rubric_frequency(df, rubric_col)
                fig = px.bar(
                    rf, x="Count", y="Rubric",
                    orientation="h",
                    title="Rubric Frequency"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Missing rubric column: {rubric_col}")

        st.divider()

    # --- Heatmap for this section (Question x Score% 0–3) ---
    st.markdown("### Section Heatmap (Score % by Question)")
    try:
        h = section_heatmap_percent(df, section)
        fig = px.imshow(
            h,
            labels=dict(x="Score", y="Question", color="%"),
            title=f"{section} Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Heatmap failed: {e}")

# ==============================
# OVERALL TAB
# ==============================
def render_overall_tab(df: pd.DataFrame):
    st.subheader("Overall Analysis")

    # Overall Total histogram
    if safe_col_exists(df, "Overall Total (0–21)"):
        fig = px.histogram(df, x="Overall Total (0–21)", nbins=15, title="Overall Total Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing: Overall Total (0–21)")

    # Overall Rank donut + bar
    if safe_col_exists(df, "Overall Rank"):
        rank_df = df["Overall Rank"].astype(str).replace({"": np.nan, "nan": np.nan}).dropna()
        rank_df = rank_df.value_counts().reset_index()
        rank_df.columns = ["Rank", "Count"]

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(rank_df, names="Rank", values="Count", hole=0.55, title="Overall Rank (Donut)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(rank_df, x="Count", y="Rank", orientation="h", title="Overall Rank (Bar)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing: Overall Rank")

    # AI Suspected (donut)
    if safe_col_exists(df, "AI_Suspected"):
        ai_df = df["AI_Suspected"].astype(str).replace({"": np.nan, "nan": np.nan}).dropna()
        ai_df = ai_df.value_counts().reset_index()
        ai_df.columns = ["AI_Suspected", "Count"]
        fig = px.pie(ai_df, names="AI_Suspected", values="Count", hole=0.55, title="AI Suspected (Donut)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing: AI_Suspected")

# ==============================
# MAIN APP
# ==============================
def main():
    st.title("Thought Leadership Dashboard")

    with st.spinner(f"Loading Google Sheet: {TARGET_WORKSHEET} ..."):
        df = load_sheet_df(TARGET_WORKSHEET)

    # We keep Care_Staff if present; we DO NOT use Date/Duration/AI_MaxScore anyway.
    # (No need to drop columns; visuals only reference what we need.)

    # Tabs: 7 sections + Overall
    tabs = st.tabs(SECTIONS + ["Overall"])

    for i, section in enumerate(SECTIONS):
        with tabs[i]:
            render_section_tab(df, section)

    with tabs[-1]:
        render_overall_tab(df)

if __name__ == "__main__":
    main()
